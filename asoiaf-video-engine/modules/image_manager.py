"""
Image Manager — Incremental Library System.

Workflow:
1. Score existing library against script segments
2. Generate a shopping list for unmatched segments
3. After user provides images, ingest them into the library with auto-tags
4. Return the final image assignments for video composition

The library grows with every video — manual effort decreases over time.
"""

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from config import ImageSearchConfig
from modules.script_parser import ScriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImageMeta:
    """Metadata for a single image in the library."""
    filepath: str
    tags: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    location: str = ""
    event: str = ""
    mood: str = ""
    concepts: list[str] = field(default_factory=list)
    quality: int = 4  # 1-5
    date_added: str = ""
    source_query: str = ""


@dataclass
class ImageAssignment:
    """Result of matching: which image goes to which segment."""
    segment_index: int
    image_path: str | None
    score: float
    source: str  # "library", "user_input", "placeholder"
    search_query: str = ""
    description: str = ""  # "Looking for" hint


# ---------------------------------------------------------------------------
# Library management
# ---------------------------------------------------------------------------

class ImageLibrary:
    """Manages the image library, metadata, and usage tracking."""

    def __init__(self, config: ImageSearchConfig):
        self.config = config
        self.library_path = Path(config.local_library_path)
        self.metadata_path = self.library_path / "metadata.json"
        self.usage_log_path = self.library_path / "usage_log.json"
        self.gaps_path = self.library_path / "gaps.json"

        self.metadata: dict[str, dict] = {}
        self.usage_log: dict[str, dict] = {}

        self._ensure_dirs()
        self._load_metadata()
        self._load_usage_log()

    def _ensure_dirs(self):
        """Create library directories if they don't exist."""
        for subdir in ["characters", "locations", "events", "concepts", "general"]:
            (self.library_path / subdir).mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load the metadata.json tag database."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            logger.info("Loaded library metadata: %d images.", len(self.metadata))
        else:
            self.metadata = {}
            logger.info("No metadata.json found — starting fresh library.")

    def _save_metadata(self):
        """Persist metadata to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _load_usage_log(self):
        """Load usage tracking log."""
        if self.usage_log_path.exists():
            with open(self.usage_log_path, "r") as f:
                self.usage_log = json.load(f)
        else:
            self.usage_log = {}

    def _save_usage_log(self):
        """Persist usage log to disk."""
        with open(self.usage_log_path, "w") as f:
            json.dump(self.usage_log, f, indent=2)

    @property
    def image_count(self) -> int:
        return len(self.metadata)

    def score_image(
        self,
        image_key: str,
        meta: dict,
        entities: dict,
        used_in_this_video: set[str],
        video_name: str = "",
    ) -> float:
        """
        Score a library image against extracted entities from a script segment.

        Entities dict expected shape:
        {
            "characters": ["jon snow", "sansa stark"],
            "location": "winterfell",
            "event": "battle of the bastards",
            "concepts": ["king in the north", "stark", "bastard"],
            "mood": "triumphant"
        }
        """
        score = 0.0

        # Hard block: never repeat in same video
        if image_key in used_in_this_video:
            return -100

        img_tags = set(t.lower() for t in meta.get("tags", []))
        img_chars = set(c.lower() for c in meta.get("characters", []))
        img_concepts = set(c.lower() for c in meta.get("concepts", []))

        # Character match (highest weight)
        for char in entities.get("characters", []):
            char_lower = char.lower()
            if char_lower in img_chars:
                score += 5
            elif any(char_lower in tag for tag in img_tags):
                score += 3  # partial match (e.g., "jon" matches tag "jon snow")

        # Event match
        seg_event = entities.get("event", "").lower()
        img_event = meta.get("event", "").lower()
        if seg_event and img_event and (seg_event in img_event or img_event in seg_event):
            score += 4
        elif seg_event:
            # Check if event terms appear in tags
            event_words = set(seg_event.split())
            tag_overlap = len(event_words & img_tags)
            if tag_overlap >= 2:
                score += 3

        # Location match
        seg_loc = entities.get("location", "").lower()
        img_loc = meta.get("location", "").lower()
        if seg_loc and img_loc and (seg_loc in img_loc or img_loc in seg_loc):
            score += 3

        # Concept/tag overlap
        for concept in entities.get("concepts", []):
            if concept.lower() in img_tags or concept.lower() in img_concepts:
                score += 2

        # Mood match
        if entities.get("mood") and meta.get("mood"):
            if entities["mood"].lower() == meta["mood"].lower():
                score += 1

        # Quality bonus
        quality = meta.get("quality", 3)
        if quality >= 4:
            score += (quality - 3)  # +1 for quality 4, +2 for quality 5

        # Recency penalty (avoid repeating across recent videos)
        usage = self.usage_log.get(image_key, {})
        recent_videos = usage.get("used_in", [])[-5:]  # last 5 videos
        if video_name and video_name in recent_videos:
            score -= 2
        elif recent_videos:
            last_used = usage.get("last_used", "")
            if last_used:
                try:
                    days_ago = (datetime.now() - datetime.fromisoformat(last_used)).days
                    if days_ago < 7:
                        score -= 2
                    elif days_ago < 14:
                        score -= 1
                except ValueError:
                    pass

        return score

    def find_best_match(
        self,
        entities: dict,
        used_in_this_video: set[str],
        threshold: float = 5.0,
        video_name: str = "",
    ) -> tuple[str | None, float]:
        """
        Find the best matching library image for given entities.

        Returns (image_key, score) or (None, 0) if nothing scores above threshold.
        """
        best_key = None
        best_score = 0

        for image_key, meta in self.metadata.items():
            score = self.score_image(image_key, meta, entities, used_in_this_video, video_name)
            if score > best_score:
                best_score = score
                best_key = image_key

        if best_score >= threshold:
            return best_key, best_score
        return None, best_score

    def record_usage(self, image_key: str, video_name: str):
        """Record that an image was used in a video."""
        if image_key not in self.usage_log:
            self.usage_log[image_key] = {
                "last_used": "",
                "used_in": [],
                "total_uses": 0,
            }

        entry = self.usage_log[image_key]
        entry["last_used"] = datetime.now().isoformat()
        if video_name not in entry["used_in"]:
            entry["used_in"].append(video_name)
        entry["total_uses"] += 1

        self._save_usage_log()

    def ingest_new_image(
        self,
        source_path: Path,
        meta: ImageMeta,
    ) -> str:
        """
        Move a new image into the library with proper naming and metadata.

        Returns the library key (relative path within library).
        """
        # Determine subfolder
        if meta.characters:
            subfolder = "characters"
        elif meta.event:
            subfolder = "events"
        elif meta.location:
            subfolder = "locations"
        elif meta.concepts:
            subfolder = "concepts"
        else:
            subfolder = "general"

        # Generate permanent filename
        permanent_name = meta.filepath if meta.filepath else source_path.stem
        permanent_name = re.sub(r'[^\w\-]', '-', permanent_name.lower())
        permanent_name = re.sub(r'-+', '-', permanent_name).strip('-')

        # Avoid collisions
        ext = source_path.suffix or ".jpg"
        dest_dir = self.library_path / subfolder
        dest_path = dest_dir / f"{permanent_name}{ext}"
        counter = 1
        while dest_path.exists():
            counter += 1
            dest_path = dest_dir / f"{permanent_name}-{counter:02d}{ext}"

        # Copy file
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

        # Build library key (relative path)
        library_key = f"{subfolder}/{dest_path.name}"

        # Store metadata
        self.metadata[library_key] = {
            "tags": meta.tags,
            "characters": meta.characters,
            "location": meta.location,
            "event": meta.event,
            "mood": meta.mood,
            "concepts": meta.concepts,
            "quality": meta.quality,
            "date_added": datetime.now().isoformat(),
            "source_query": meta.source_query,
        }

        self._save_metadata()
        logger.info("Ingested: %s → %s", source_path.name, library_key)

        return library_key


# ---------------------------------------------------------------------------
# Entity extraction (LLM-powered)
# ---------------------------------------------------------------------------

def extract_entities_batch(
    segments: list[ScriptSegment],
    anthropic_api_key: str,
) -> list[dict]:
    """
    Use Claude to extract visual entities from each script segment.

    Returns a list of entity dicts, one per segment.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — using naive extraction.")
        return _naive_extract_all(segments)

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    full_script = " ".join(seg.text for seg in segments)
    segment_texts = "\n".join(
        f"[{i}] {seg.text}" for i, seg in enumerate(segments)
    )

    prompt = f"""You are analyzing script segments for an ASOIAF (A Song of Ice and Fire / Game of Thrones) YouTube video.

FULL SCRIPT (for narrative context):
{full_script}

For each numbered segment below, extract:
- characters: array of character names mentioned or clearly implied (use full names)
- location: the primary location depicted (empty string if unclear)
- event: specific ASOIAF event name if referenced (empty string if none)
- concepts: array of visual concepts useful for image searching (dragons, battle, throne, betrayal, etc.)
- mood: one of: epic, dark, dramatic, peaceful, intense, mysterious, tragic, triumphant
- search_query: a Google Image search query (5-10 words) that an editor can paste to find matching ASOIAF fan art or fantasy artwork
- looking_for: a short human-readable description of what the ideal image would show

SEARCH QUERY RULES:
- The query will be pasted directly into Google Images — make it practical.
- Always include a VISUAL ANCHOR describing what the image should SHOW (e.g., "volcanic eruption", "dragon breathing fire", "castle siege"), not just character/place names.
- Always include an ASOIAF-specific term (character, house, location, or event name).
- End with "fantasy art" or "asoiaf art" to bias toward artwork rather than photos.
- GOOD: "Doom of Valyria volcanic eruption fantasy art", "Targaryen dragonlords riding dragons asoiaf art"
- BAD: "Their asoiaf art", "asoiaf fantasy art", "In asoiaf art", "Long Targaryens Narrow Sea asoiaf art"
- If a segment is vague on its own, use the full script context above to infer what it's about.

IMPORTANT: Understand ASOIAF context. "The Young Wolf" = Robb Stark. "The Imp" = Tyrion Lannister.
"She sailed across the Narrow Sea" in context = Daenerys Targaryen. Use your knowledge.

Return ONLY a JSON array of objects (one per segment). No markdown, no backticks, no commentary.

Segments:
{segment_texts}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        response_text = re.sub(r"```json\s*|```", "", response_text).strip()
        entities = json.loads(response_text)

        if len(entities) == len(segments):
            logger.info("Extracted entities for %d segments via LLM.", len(entities))
            return entities
        else:
            logger.warning(
                "Entity count mismatch (%d vs %d segments).",
                len(entities), len(segments),
            )
    except Exception as e:
        logger.warning("LLM entity extraction failed: %s", e)

    return _naive_extract_all(segments)



# Stopwords that get falsely picked up as entities when capitalised
# at the start of a sentence.
_STOPWORDS = {
    "the", "a", "an", "but", "and", "or", "in", "on", "at", "to", "of",
    "for", "by", "with", "from", "their", "they", "them", "he", "him",
    "his", "she", "her", "it", "its", "this", "that", "these", "those",
    "only", "long", "when", "where", "then", "there", "here", "some",
    "all", "no", "not", "yet", "so", "as", "if", "while", "before",
    "after", "once", "now", "never", "every", "each", "many", "few",
    "more", "most", "such", "what", "who", "whom", "which", "even",
    "still", "just", "much", "soon", "great", "one", "two", "three",
}

# ASOIAF entity dictionary — maps lowercase phrases to visual search hints.
ASOIAF_ENTITIES: dict[str, dict] = {
    # Houses
    "targaryens": {"type": "house", "visual": "Targaryen dragon dynasty"},
    "targaryen": {"type": "house", "visual": "Targaryen dragon dynasty"},
    "starks": {"type": "house", "visual": "Stark direwolf Winterfell"},
    "stark": {"type": "house", "visual": "Stark direwolf Winterfell"},
    "lannisters": {"type": "house", "visual": "Lannister golden lion"},
    "lannister": {"type": "house", "visual": "Lannister golden lion"},
    "baratheons": {"type": "house", "visual": "Baratheon stag"},
    "baratheon": {"type": "house", "visual": "Baratheon stag"},
    "martells": {"type": "house", "visual": "Martell Dorne sun spear"},
    "martell": {"type": "house", "visual": "Martell Dorne sun spear"},
    "tyrells": {"type": "house", "visual": "Tyrell Highgarden rose"},
    "tyrell": {"type": "house", "visual": "Tyrell Highgarden rose"},
    "greyjoys": {"type": "house", "visual": "Greyjoy Iron Islands kraken"},
    "greyjoy": {"type": "house", "visual": "Greyjoy Iron Islands kraken"},
    "arryns": {"type": "house", "visual": "Arryn Eyrie falcon"},
    "arryn": {"type": "house", "visual": "Arryn Eyrie falcon"},
    "tullys": {"type": "house", "visual": "Tully Riverrun fish"},
    "tully": {"type": "house", "visual": "Tully Riverrun fish"},
    "boltons": {"type": "house", "visual": "Bolton flayed man Dreadfort"},
    "bolton": {"type": "house", "visual": "Bolton flayed man Dreadfort"},
    "freys": {"type": "house", "visual": "Frey Twins bridge"},
    "frey": {"type": "house", "visual": "Frey Twins bridge"},
    "velaryons": {"type": "house", "visual": "Velaryon seahorse fleet"},
    "velaryon": {"type": "house", "visual": "Velaryon seahorse fleet"},
    "hightowers": {"type": "house", "visual": "Hightower Oldtown tower"},
    "hightower": {"type": "house", "visual": "Hightower Oldtown tower"},
    # Characters
    "daenerys": {"type": "character", "visual": "Daenerys Targaryen dragons"},
    "jon snow": {"type": "character", "visual": "Jon Snow Night's Watch"},
    "aegon": {"type": "character", "visual": "Aegon Targaryen conqueror"},
    "cersei": {"type": "character", "visual": "Cersei Lannister queen"},
    "tyrion": {"type": "character", "visual": "Tyrion Lannister"},
    "jaime": {"type": "character", "visual": "Jaime Lannister knight"},
    "ned": {"type": "character", "visual": "Eddard Stark Winterfell"},
    "eddard": {"type": "character", "visual": "Eddard Stark Winterfell"},
    "robb": {"type": "character", "visual": "Robb Stark Young Wolf"},
    "bran": {"type": "character", "visual": "Bran Stark three-eyed raven"},
    "arya": {"type": "character", "visual": "Arya Stark sword"},
    "sansa": {"type": "character", "visual": "Sansa Stark queen"},
    "rhaegar": {"type": "character", "visual": "Rhaegar Targaryen prince"},
    "robert": {"type": "character", "visual": "Robert Baratheon warhammer"},
    "stannis": {"type": "character", "visual": "Stannis Baratheon Dragonstone"},
    "renly": {"type": "character", "visual": "Renly Baratheon"},
    "joffrey": {"type": "character", "visual": "Joffrey Baratheon crown"},
    "petyr": {"type": "character", "visual": "Petyr Baelish Littlefinger"},
    "littlefinger": {"type": "character", "visual": "Littlefinger schemer"},
    "varys": {"type": "character", "visual": "Varys Spider"},
    "melisandre": {"type": "character", "visual": "Melisandre red priestess fire"},
    "the night king": {"type": "character", "visual": "Night King White Walker ice"},
    "viserys": {"type": "character", "visual": "Viserys Targaryen"},
    "rhaenyra": {"type": "character", "visual": "Rhaenyra Targaryen dragon"},
    "alicent": {"type": "character", "visual": "Alicent Hightower queen"},
    "daemon": {"type": "character", "visual": "Daemon Targaryen Dark Sister"},
    "balerion": {"type": "character", "visual": "Balerion Black Dread dragon"},
    "drogon": {"type": "character", "visual": "Drogon dragon fire"},
    "dragonlords": {"type": "character", "visual": "Valyrian dragonlords dragons"},
    "dragonlord": {"type": "character", "visual": "Valyrian dragonlord dragon"},
    # Locations
    "valyria": {"type": "location", "visual": "Valyria ancient empire"},
    "valyrian freehold": {"type": "location", "visual": "Valyrian Freehold empire"},
    "narrow sea": {"type": "location", "visual": "Narrow Sea ships crossing"},
    "fourteen flames": {"type": "location", "visual": "Fourteen Flames volcanoes Valyria"},
    "dragonstone": {"type": "location", "visual": "Dragonstone castle island"},
    "winterfell": {"type": "location", "visual": "Winterfell castle snow"},
    "king's landing": {"type": "location", "visual": "King's Landing city Red Keep"},
    "kings landing": {"type": "location", "visual": "King's Landing city Red Keep"},
    "the wall": {"type": "location", "visual": "Wall Night's Watch ice"},
    "castle black": {"type": "location", "visual": "Castle Black Night's Watch"},
    "iron throne": {"type": "location", "visual": "Iron Throne swords"},
    "red keep": {"type": "location", "visual": "Red Keep castle"},
    "oldtown": {"type": "location", "visual": "Oldtown Citadel Hightower"},
    "braavos": {"type": "location", "visual": "Braavos Titan free city"},
    "pentos": {"type": "location", "visual": "Pentos free city Essos"},
    "meereen": {"type": "location", "visual": "Meereen pyramid city"},
    "harrenhal": {"type": "location", "visual": "Harrenhal ruined castle"},
    "highgarden": {"type": "location", "visual": "Highgarden castle Reach"},
    "dorne": {"type": "location", "visual": "Dorne desert sun"},
    "riverrun": {"type": "location", "visual": "Riverrun castle river"},
    "the eyrie": {"type": "location", "visual": "Eyrie mountain castle"},
    "pyke": {"type": "location", "visual": "Pyke Iron Islands castle"},
    "asshai": {"type": "location", "visual": "Asshai shadow lands"},
    "essos": {"type": "location", "visual": "Essos continent"},
    "westeros": {"type": "location", "visual": "Westeros continent map"},
    "beyond the wall": {"type": "location", "visual": "Beyond the Wall snow wildlings"},
    # Events
    "doom": {"type": "event", "visual": "Doom of Valyria destruction"},
    "doom of valyria": {"type": "event", "visual": "Doom of Valyria volcanic eruption"},
    "red wedding": {"type": "event", "visual": "Red Wedding massacre"},
    "battle of the bastards": {"type": "event", "visual": "Battle of the Bastards"},
    "field of fire": {"type": "event", "visual": "Field of Fire Aegon dragons"},
    "blackwater": {"type": "event", "visual": "Battle of Blackwater wildfire"},
    "robert's rebellion": {"type": "event", "visual": "Robert's Rebellion war"},
    "dance of the dragons": {"type": "event", "visual": "Dance of Dragons civil war"},
    "long night": {"type": "event", "visual": "Long Night White Walkers darkness"},
    "conquest": {"type": "event", "visual": "Aegon's Conquest Targaryen"},
    "aegon's conquest": {"type": "event", "visual": "Aegon's Conquest three dragons"},
    # Concepts
    "dragon eggs": {"type": "concept", "visual": "dragon eggs hatching"},
    "old valyria": {"type": "concept", "visual": "Old Valyria ancient glory"},
    "valyrian steel": {"type": "concept", "visual": "Valyrian steel sword"},
    "wildfire": {"type": "concept", "visual": "wildfire green explosion"},
    "white walkers": {"type": "concept", "visual": "White Walkers Others ice"},
    "three-eyed raven": {"type": "concept", "visual": "three-eyed raven weirwood"},
    "weirwood": {"type": "concept", "visual": "weirwood tree face"},
    "direwolf": {"type": "concept", "visual": "direwolf Stark"},
    "faceless men": {"type": "concept", "visual": "Faceless Men Braavos"},
    "prophecy": {"type": "concept", "visual": "prophecy vision fire"},
    "prophetic dream": {"type": "concept", "visual": "prophetic dream vision"},
}

# Visual action/concept keywords that make search queries more useful.
_VISUAL_KEYWORDS = {
    "fire", "flame", "flames", "burn", "burned", "burning",
    "battle", "war", "fought", "siege", "sacked", "conquered",
    "dragon", "dragons", "beast", "beasts",
    "sword", "swords", "blade", "steel",
    "sailed", "ships", "fleet", "crossing", "voyage",
    "erupted", "eruption", "volcano", "volcanic", "destroyed", "destruction",
    "castle", "fortress", "tower", "city", "kingdom", "empire",
    "death", "died", "killed", "murdered", "slain", "massacre",
    "throne", "crown", "king", "queen", "prince", "princess", "lord",
    "army", "soldiers", "knights", "cavalry",
    "night", "darkness", "shadow", "ice", "snow", "winter",
    "blood", "betrayal", "revenge", "exile", "fled", "escape",
    "magic", "sorcery", "sorcerers", "ritual", "spell",
    "smoke", "ash", "ruin", "ruins", "silence",
}


def _naive_entity_extraction(
    text: str,
    neighbors: list[str] | None = None,
) -> dict:
    """ASOIAF-aware entity extraction — no API key needed.

    Args:
        text: The segment text to analyse.
        neighbors: Optional list of neighbouring segment texts (±1)
                   used as context when *text* is too short/vague.
    """
    text_lower = text.lower()

    # Also consider neighbour text for context
    context_lower = text_lower
    if neighbors:
        context_lower = " ".join(neighbors).lower() + " " + text_lower

    # 1. Match against ASOIAF entity dictionary (check multi-word first)
    matched_visuals: list[str] = []
    matched_entities: list[str] = []
    entity_types: dict[str, str] = {}  # visual -> type

    # Sort keys longest-first so multi-word phrases match before sub-words.
    # First pass: match against segment text only.
    for key in sorted(ASOIAF_ENTITIES, key=len, reverse=True):
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, text_lower):
            entry = ASOIAF_ENTITIES[key]
            visual = entry["visual"]
            if visual not in matched_visuals:
                matched_visuals.append(visual)
                matched_entities.append(key)
                entity_types[visual] = entry["type"]

    # Second pass: if segment itself yielded no entities (or is very short),
    # pull context from neighbouring segments.
    if (not matched_visuals or len(text.split()) < 5) and neighbors:
        for key in sorted(ASOIAF_ENTITIES, key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, context_lower):
                entry = ASOIAF_ENTITIES[key]
                visual = entry["visual"]
                if visual not in matched_visuals:
                    matched_visuals.append(visual)
                    matched_entities.append(key)
                    entity_types[visual] = entry["type"]
                    if len(matched_visuals) >= 3:
                        break  # Don't pull too many from context

    # 2. Extract proper nouns, filtering stopwords
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Strip leading stopwords from multi-word proper nouns, then filter
    all_entity_text = " ".join(matched_entities)
    cleaned_nouns = []
    for pn in proper_nouns:
        # Strip leading stopwords (e.g., "The Fourteen Flames" → "Fourteen Flames")
        words = pn.split()
        while words and words[0].lower() in _STOPWORDS:
            words.pop(0)
        if not words:
            continue
        clean = " ".join(words)
        if (clean.lower() not in _STOPWORDS
                and clean.lower() not in matched_entities
                and clean.lower() not in all_entity_text):
            cleaned_nouns.append(clean)
    filtered_nouns = cleaned_nouns

    # 3. Pull visual action keywords from the text
    words_in_text = set(re.findall(r'[a-z]+', text_lower))
    visual_concepts = sorted(words_in_text & _VISUAL_KEYWORDS)

    # 4. Determine structured fields
    characters = [v for v, t in entity_types.items() if t == "character"]
    location = next((v for v, t in entity_types.items() if t == "location"), "")
    event = next((v for v, t in entity_types.items() if t == "event"), "")
    concepts = [v for v, t in entity_types.items() if t in ("concept", "house")]

    # Add non-stopword proper nouns as additional characters
    for noun in filtered_nouns[:3]:
        if noun not in characters:
            characters.append(noun)

    # 5. Build search query
    query_parts: list[str] = []
    # Lead with matched entity visuals (max 2)
    for vis in matched_visuals[:2]:
        query_parts.append(vis)
    # Add any leftover proper nouns
    for noun in filtered_nouns[:2]:
        if noun.lower() not in " ".join(query_parts).lower():
            query_parts.append(noun)
    # Add visual concepts (max 2)
    for vc in visual_concepts[:2]:
        if vc not in " ".join(query_parts).lower():
            query_parts.append(vc)
    # Always end with genre anchor
    query_parts.append("asoiaf art")

    search_query = " ".join(query_parts) if len(query_parts) > 1 else "asoiaf fantasy art"

    # 6. Looking-for description
    if matched_visuals:
        looking_for = "Artwork showing: " + ", ".join(matched_visuals[:3])
    else:
        looking_for = "General ASOIAF artwork matching: " + text[:60]

    return {
        "characters": characters[:4],
        "location": location,
        "event": event,
        "concepts": concepts + visual_concepts[:3],
        "mood": "dramatic",
        "search_query": search_query,
        "looking_for": looking_for,
    }


def _naive_extract_all(segments: list[ScriptSegment]) -> list[dict]:
    """Run naive extraction on every segment with neighbour context."""
    results = []
    for i, seg in enumerate(segments):
        neighbors = []
        if i > 0:
            neighbors.append(segments[i - 1].text)
        if i + 1 < len(segments):
            neighbors.append(segments[i + 1].text)
        results.append(_naive_entity_extraction(seg.text, neighbors=neighbors))
    return results


# ---------------------------------------------------------------------------
# Auto-tagging for newly downloaded images
# ---------------------------------------------------------------------------

def auto_tag_images(
    image_paths: list[Path],
    segment_contexts: list[dict],
    anthropic_api_key: str,
) -> list[ImageMeta]:
    """
    Auto-tag newly downloaded images using LLM.

    Each image has context: the search query used and the script segment.
    The LLM generates proper metadata for the library.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — using basic tags.")
        return [
            ImageMeta(
                filepath=ctx.get("search_query", "unnamed").replace(" ", "-"),
                tags=ctx.get("search_query", "").split(),
                characters=ctx.get("characters", []),
                location=ctx.get("location", ""),
                event=ctx.get("event", ""),
                mood=ctx.get("mood", "dramatic"),
                concepts=ctx.get("concepts", []),
                source_query=ctx.get("search_query", ""),
            )
            for ctx in segment_contexts
        ]

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    # Build context for LLM
    items = []
    for path, ctx in zip(image_paths, segment_contexts):
        items.append({
            "filename": path.name,
            "search_query": ctx.get("search_query", ""),
            "script_segment": ctx.get("script_text", ""),
            "characters": ctx.get("characters", []),
        })

    prompt = f"""I have {len(items)} new ASOIAF images to add to my library. 
For each one, I'll give you the search query used and the script context.

Generate a metadata entry for each:
- permanent_name: descriptive kebab-case name (e.g., "rhaegar-lyanna-secret-wedding")
- tags: array of ALL relevant searchable terms (character names, locations, events, houses, concepts, adjectives)
- characters: array of character full names depicted
- location: primary location shown
- event: ASOIAF event name if applicable (empty string if none)
- mood: one of: epic, dark, dramatic, peaceful, intense, mysterious, tragic, triumphant
- concepts: array of abstract visual concepts (battle, betrayal, throne, dragon, etc.)
- quality: estimated 1-5 based on how specific/useful this image likely is

Return ONLY a JSON array. No markdown, no backticks.

Images:
{json.dumps(items, indent=2)}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        response_text = re.sub(r"```json\s*|```", "", response_text).strip()
        tag_data = json.loads(response_text)

        results = []
        for data, ctx in zip(tag_data, segment_contexts):
            results.append(ImageMeta(
                filepath=data.get("permanent_name", "unnamed"),
                tags=data.get("tags", []),
                characters=data.get("characters", []),
                location=data.get("location", ""),
                event=data.get("event", ""),
                mood=data.get("mood", "dramatic"),
                concepts=data.get("concepts", []),
                quality=data.get("quality", 4),
                source_query=ctx.get("search_query", ""),
            ))

        logger.info("Auto-tagged %d images via LLM.", len(results))
        return results

    except Exception as e:
        logger.warning("LLM auto-tagging failed: %s — using basic tags.", e)
        return [
            ImageMeta(
                filepath=ctx.get("search_query", "unnamed").replace(" ", "-"),
                tags=ctx.get("search_query", "").split(),
                characters=ctx.get("characters", []),
                location=ctx.get("location", ""),
                mood=ctx.get("mood", "dramatic"),
                concepts=ctx.get("concepts", []),
                source_query=ctx.get("search_query", ""),
            )
            for ctx in segment_contexts
        ]


# ---------------------------------------------------------------------------
# Shopping list generation
# ---------------------------------------------------------------------------

def generate_shopping_list(
    segments: list[ScriptSegment],
    entities_list: list[dict],
    assignments: list[ImageAssignment],
    output_path: Path,
    video_title: str = "",
) -> Path:
    """
    Generate a human-readable shopping list for unmatched segments.

    Returns path to the shopping list file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched = [a for a in assignments if a.source == "library"]
    unmatched = [a for a in assignments if a.source != "library"]

    lines = []
    lines.append("=" * 65)
    lines.append(f"  IMAGE SHOPPING LIST — \"{video_title}\"")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Total segments: {len(segments)}")
    lines.append(f"  From library: {len(matched)}  |  Need from you: {len(unmatched)}")
    lines.append("=" * 65)
    lines.append("")

    if matched:
        lines.append("FROM LIBRARY (no action needed):")
        for a in matched:
            seg = segments[a.segment_index]
            lines.append(f"  ✅ Seg {a.segment_index + 1:02d} → {Path(a.image_path).name}")
        lines.append("")

    if unmatched:
        lines.append("NEED FROM YOU:")
        lines.append("  Download each image and save to input/images/ with the")
        lines.append("  segment number as filename (01.jpg, 02.jpg, etc.)")
        lines.append("")

        for a in unmatched:
            seg = segments[a.segment_index]
            entities = entities_list[a.segment_index] if a.segment_index < len(entities_list) else {}
            num = a.segment_index + 1

            lines.append(f"  ❌ {num:02d}. Search: \"{a.search_query}\"")
            lines.append(f"       Script: \"{seg.text[:80]}{'...' if len(seg.text) > 80 else ''}\"")
            if a.description:
                lines.append(f"       Looking for: {a.description}")
            lines.append("")
    else:
        lines.append("🎉 All segments covered from library! No images needed.")
        lines.append("   Generating video automatically...")
        lines.append("")

    lines.append("=" * 65)
    if unmatched:
        lines.append("  After downloading, run:  python main.py --continue")
    lines.append("=" * 65)

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Also print to console (handle Windows encoding)
    try:
        print("\n" + text + "\n")
    except UnicodeEncodeError:
        print("\n" + text.encode("ascii", errors="replace").decode("ascii") + "\n")

    logger.info("Shopping list saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_images_for_segments(
    segments: list[ScriptSegment],
    config: ImageSearchConfig,
    video_name: str = "",
    input_folder: Path = Path("input/images"),
    match_threshold: float = 5.0,
) -> tuple[list[ImageAssignment], list[dict], bool]:
    """
    Main entry point for the image system.

    Phase 1: Score library, generate shopping list for gaps.
    Phase 2: After user provides images, assign everything.

    Args:
        segments: Script segments with timing.
        config: Image search config.
        video_name: Name of this video (for usage tracking).
        input_folder: Where user drops downloaded images.
        match_threshold: Minimum score to accept a library match.

    Returns:
        (assignments, entities_list, all_covered)
        - assignments: list of ImageAssignment for each segment
        - entities_list: extracted entities per segment
        - all_covered: True if every segment has an image
    """
    library = ImageLibrary(config)

    logger.info("Library has %d images.", library.image_count)

    # Step 1: Extract entities for each segment
    entities_list = []
    if config.anthropic_api_key:
        entities_list = extract_entities_batch(segments, config.anthropic_api_key)
    else:
        entities_list = _naive_extract_all(segments)

    # Step 2: Score library against each segment
    assignments = []
    used_in_this_video: set[str] = set()

    for i, (seg, entities) in enumerate(zip(segments, entities_list)):
        best_key, best_score = library.find_best_match(
            entities, used_in_this_video, match_threshold, video_name
        )

        if best_key:
            full_path = str(library.library_path / best_key)
            assignments.append(ImageAssignment(
                segment_index=i,
                image_path=full_path,
                score=best_score,
                source="library",
                search_query=entities.get("search_query", ""),
            ))
            used_in_this_video.add(best_key)
            logger.info(
                "  Seg %02d: MATCHED %s (score=%.1f)",
                i + 1, best_key, best_score,
            )
        else:
            assignments.append(ImageAssignment(
                segment_index=i,
                image_path=None,
                score=best_score,
                source="needed",
                search_query=entities.get("search_query", ""),
                description=entities.get("looking_for", ""),
            ))
            logger.info(
                "  Seg %02d: NO MATCH (best=%.1f) — needs: %s",
                i + 1, best_score, entities.get("search_query", "?"),
            )

    # Step 3: Check if user has provided images in input folder
    input_folder = Path(input_folder)
    if input_folder.exists():
        _pickup_user_images(assignments, input_folder, segments, entities_list)

    # Check if all covered
    all_covered = all(a.image_path is not None for a in assignments)

    # Record usage for library images
    for a in assignments:
        if a.source == "library" and a.image_path:
            rel_key = str(Path(a.image_path).relative_to(library.library_path))
            library.record_usage(rel_key, video_name)

    return assignments, entities_list, all_covered


def _pickup_user_images(
    assignments: list[ImageAssignment],
    input_folder: Path,
    segments: list[ScriptSegment],
    entities_list: list[dict],
):
    """Check input folder for user-provided images matching segment numbers."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for a in assignments:
        if a.image_path is not None:
            continue  # already has an image

        seg_num = a.segment_index + 1

        # Look for files like 01.jpg, 01.png, etc.
        for ext in extensions:
            candidate = input_folder / f"{seg_num:02d}{ext}"
            if candidate.exists():
                a.image_path = str(candidate)
                a.source = "user_input"
                logger.info("  Seg %02d: Picked up user image %s", seg_num, candidate.name)
                break


def ingest_user_images_to_library(
    assignments: list[ImageAssignment],
    entities_list: list[dict],
    segments: list[ScriptSegment],
    config: ImageSearchConfig,
):
    """
    After video generation, ingest newly provided images into the permanent library.
    Auto-tags them via LLM and moves them from input/ to the library.
    """
    library = ImageLibrary(config)

    # Find images that came from user input (not already in library)
    new_images = []
    new_contexts = []

    for a in assignments:
        if a.source == "user_input" and a.image_path:
            new_images.append(Path(a.image_path))
            entities = entities_list[a.segment_index] if a.segment_index < len(entities_list) else {}
            new_contexts.append({
                "search_query": a.search_query or entities.get("search_query", ""),
                "script_text": segments[a.segment_index].text if a.segment_index < len(segments) else "",
                "characters": entities.get("characters", []),
                "location": entities.get("location", ""),
                "event": entities.get("event", ""),
                "mood": entities.get("mood", "dramatic"),
                "concepts": entities.get("concepts", []),
            })

    if not new_images:
        logger.info("No new images to ingest.")
        return

    logger.info("Auto-tagging %d new images...", len(new_images))

    # Auto-tag via LLM
    if config.anthropic_api_key:
        meta_list = auto_tag_images(new_images, new_contexts, config.anthropic_api_key)
    else:
        meta_list = [
            ImageMeta(
                filepath=ctx.get("search_query", "unnamed").replace(" ", "-"),
                tags=ctx.get("search_query", "").split(),
                characters=ctx.get("characters", []),
                location=ctx.get("location", ""),
                mood=ctx.get("mood", "dramatic"),
                concepts=ctx.get("concepts", []),
                source_query=ctx.get("search_query", ""),
            )
            for ctx in new_contexts
        ]

    # Ingest each image
    for img_path, meta in zip(new_images, meta_list):
        library.ingest_new_image(img_path, meta)

    logger.info(
        "Library updated: %d images total (added %d new).",
        library.image_count, len(new_images),
    )

    # Log gaps for subjects that are running low
    _check_and_log_gaps(library)


def _check_and_log_gaps(library: ImageLibrary):
    """Check for overused images and log suggestions."""
    overused = []
    for key, usage in library.usage_log.items():
        if usage.get("total_uses", 0) >= 5:
            meta = library.metadata.get(key, {})
            chars = meta.get("characters", [])
            if chars:
                overused.append({
                    "image": key,
                    "characters": chars,
                    "total_uses": usage["total_uses"],
                    "suggestion": f"Consider adding more images for: {', '.join(chars)}",
                })

    if overused:
        gaps_path = library.library_path / "gaps.json"
        with open(gaps_path, "w") as f:
            json.dump(overused, f, indent=2)
        logger.info(
            "⚠️  %d images are heavily reused. See gaps.json for suggestions.",
            len(overused),
        )
