"""
Script Parser — splits a raw script into segments.

Supports both text-based splitting (for pre-audio phase) and
time-based splitting (after audio timestamps are available) to
ensure images change every 3-4 seconds.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ScriptSegment:
    """A single narrative beat in the video."""
    index: int
    text: str
    image_path: str | None = None
    search_query: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    # Manual image override from script (e.g., [IMAGE: filename.jpg])
    manual_image: str | None = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def parse_script(script_text: str, mode: str = "sentence") -> list[ScriptSegment]:
    """
    Split the script into segments (pre-audio phase).

    Args:
        script_text: The full narration script.
        mode: "sentence", "paragraph", or "marker"

    Returns:
        List of ScriptSegment objects.
    """
    script_text = script_text.strip()

    # First, extract any [IMAGE: ...] directives
    image_overrides = {}
    lines = script_text.split("\n")
    clean_lines = []
    current_override = None

    for line in lines:
        override_match = re.match(r'\[IMAGE:\s*(.+?)\]', line.strip())
        if override_match:
            current_override = override_match.group(1).strip()
            continue
        if line.strip():
            if current_override:
                # The override applies to the next non-empty text line
                image_overrides[line.strip()] = current_override
                current_override = None
            clean_lines.append(line)

    clean_text = "\n".join(clean_lines)

    if mode == "marker":
        segments = _split_by_markers(clean_text)
    elif mode == "paragraph":
        segments = _split_by_paragraph(clean_text)
    else:
        segments = _split_by_sentence(clean_text)

    # Apply manual overrides
    for seg in segments:
        if seg.text.strip() in image_overrides:
            seg.manual_image = image_overrides[seg.text.strip()]

    # Filter and re-index
    segments = [s for s in segments if s.text.strip()]
    for i, seg in enumerate(segments):
        seg.index = i

    return segments


def resegment_by_time(
    segments: list[ScriptSegment],
    target_interval: float = 3.5,
    min_interval: float = 2.0,
    max_interval: float = 5.0,
) -> list[ScriptSegment]:
    """
    Re-segment AFTER audio timestamps are available to ensure
    images change every 3-4 seconds.

    Strategy:
    - Segments shorter than min_interval get merged with the next one
    - Segments longer than max_interval get split at clause boundaries
    - Result: every segment is between min_interval and max_interval duration

    Args:
        segments: Segments with start_time/end_time populated.
        target_interval: Ideal image duration in seconds.
        min_interval: Minimum allowed segment duration.
        max_interval: Maximum allowed segment duration.

    Returns:
        New list of properly-timed segments.
    """
    if not segments or segments[0].end_time == 0:
        return segments  # timestamps not populated yet

    result = []

    i = 0
    while i < len(segments):
        seg = segments[i]

        if seg.duration < min_interval and i + 1 < len(segments):
            # Too short — merge with next segment
            next_seg = segments[i + 1]
            merged = ScriptSegment(
                index=len(result),
                text=seg.text + " " + next_seg.text,
                start_time=seg.start_time,
                end_time=next_seg.end_time,
                manual_image=seg.manual_image or next_seg.manual_image,
            )
            # Replace next segment in the list so we can check its duration too
            segments[i + 1] = merged
            i += 1
            continue

        elif seg.duration > max_interval:
            # Too long — split into roughly equal parts
            num_parts = max(2, round(seg.duration / target_interval))
            part_duration = seg.duration / num_parts

            # Try to split text at sentence/clause boundaries
            sub_texts = _split_text_evenly(seg.text, num_parts)

            for j, sub_text in enumerate(sub_texts):
                sub_seg = ScriptSegment(
                    index=len(result),
                    text=sub_text,
                    start_time=seg.start_time + j * part_duration,
                    end_time=seg.start_time + (j + 1) * part_duration,
                    manual_image=seg.manual_image if j == 0 else None,
                )
                result.append(sub_seg)
            i += 1
            continue

        else:
            # Duration is good
            seg.index = len(result)
            result.append(seg)
            i += 1

    return result


def _split_text_evenly(text: str, num_parts: int) -> list[str]:
    """Split text into roughly equal parts at clause boundaries.

    Splits *before* conjunctions/commas so they stay attached to the
    clause they introduce, then merges any chunk shorter than 3 words
    into its neighbour to avoid orphan fragments like "and" or "but".
    """
    # Split BEFORE conjunctions and after punctuation — keeps the
    # conjunction with the clause it introduces.
    clauses = re.split(r'(?<=[,;])\s+|\s+(?=\band\b)|\s+(?=\bbut\b)', text)
    clauses = [c.strip() for c in clauses if c.strip()]

    # Merge any chunk with fewer than 3 words into its neighbour
    clauses = _merge_short_chunks(clauses, min_words=3)

    if len(clauses) >= num_parts:
        # Distribute clauses evenly across parts
        parts = []
        per_part = max(1, len(clauses) // num_parts)
        for i in range(0, len(clauses), per_part):
            chunk = " ".join(clauses[i:i + per_part])
            if chunk.strip():
                parts.append(chunk.strip())
        return parts[:num_parts]

    # Not enough natural split points — split by word count
    words = text.split()
    per_part = max(1, len(words) // num_parts)
    parts = []
    for i in range(0, len(words), per_part):
        chunk = " ".join(words[i:i + per_part])
        if chunk.strip():
            parts.append(chunk.strip())
    return parts[:num_parts]


def _merge_short_chunks(chunks: list[str], min_words: int = 3) -> list[str]:
    """Merge chunks with fewer than *min_words* words into an adjacent chunk."""
    if len(chunks) <= 1:
        return chunks

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1].split()) < min_words:
            # Previous chunk was too short — glue this one onto it
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Final chunk might also be too short — merge backwards
    if len(merged) >= 2 and len(merged[-1].split()) < min_words:
        merged[-2] = merged[-2] + " " + merged[-1]
        merged.pop()

    return merged


def _split_by_sentence(text: str) -> list[ScriptSegment]:
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [ScriptSegment(index=i, text=s.strip()) for i, s in enumerate(raw) if s.strip()]


def _split_by_paragraph(text: str) -> list[ScriptSegment]:
    parts = re.split(r'\n\s*\n', text)
    return [ScriptSegment(index=i, text=p.strip()) for i, p in enumerate(parts) if p.strip()]


def _split_by_markers(text: str) -> list[ScriptSegment]:
    parts = re.split(r'\[(?:SCENE|CUT)\]|---', text)
    return [ScriptSegment(index=i, text=p.strip()) for i, p in enumerate(parts) if p.strip()]


def get_full_script_text(segments: list[ScriptSegment]) -> str:
    return " ".join(seg.text for seg in segments)
