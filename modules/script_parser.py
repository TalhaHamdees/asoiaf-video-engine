"""
Script Parser — splits a raw script into segments.

Each segment becomes one "beat" in the final video:
one image, one chunk of narration, one caption group.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ScriptSegment:
    """A single narrative beat in the video."""
    index: int
    text: str
    # These get populated later in the pipeline
    image_path: str | None = None
    search_query: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def parse_script(script_text: str, mode: str = "sentence") -> list[ScriptSegment]:
    """
    Split the script into segments.

    Args:
        script_text: The full narration script.
        mode: 
            "sentence" — one segment per sentence (default, best for shorts)
            "paragraph" — one segment per paragraph
            "marker" — split on [SCENE] or [CUT] markers in the script

    Returns:
        List of ScriptSegment objects.
    """
    script_text = script_text.strip()

    if mode == "marker":
        segments = _split_by_markers(script_text)
    elif mode == "paragraph":
        segments = _split_by_paragraph(script_text)
    else:
        segments = _split_by_sentence(script_text)

    # Filter out empty segments
    segments = [s for s in segments if s.text.strip()]
    # Re-index
    for i, seg in enumerate(segments):
        seg.index = i

    return segments


def _split_by_sentence(text: str) -> list[ScriptSegment]:
    """Split on sentence boundaries (. ! ?)"""
    # This regex splits on sentence-ending punctuation while keeping the punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [
        ScriptSegment(index=i, text=s.strip())
        for i, s in enumerate(raw_sentences)
        if s.strip()
    ]


def _split_by_paragraph(text: str) -> list[ScriptSegment]:
    """Split on double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [
        ScriptSegment(index=i, text=p.strip())
        for i, p in enumerate(paragraphs)
        if p.strip()
    ]


def _split_by_markers(text: str) -> list[ScriptSegment]:
    """Split on [SCENE], [CUT], or --- markers."""
    parts = re.split(r'\[(?:SCENE|CUT)\]|---', text)
    return [
        ScriptSegment(index=i, text=p.strip())
        for i, p in enumerate(parts)
        if p.strip()
    ]


def merge_short_segments(
    segments: list[ScriptSegment],
    min_words: int = 5
) -> list[ScriptSegment]:
    """
    Merge segments that are too short (< min_words) into the previous one.
    Prevents choppy 1-second image flashes.
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        word_count = len(seg.text.split())
        if word_count < min_words and merged:
            # Append to previous segment
            merged[-1].text += " " + seg.text
        else:
            merged.append(seg)

    # Re-index
    for i, seg in enumerate(merged):
        seg.index = i

    return merged


def get_full_script_text(segments: list[ScriptSegment]) -> str:
    """Reconstruct the full script from segments."""
    return " ".join(seg.text for seg in segments)
