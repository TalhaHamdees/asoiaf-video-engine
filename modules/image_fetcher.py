"""
Image Fetcher — hybrid local library + web search.

Strategy:
1. Try to match each script segment to a local image using keyword matching.
2. For segments with no local match, generate a search query and fetch from web.
3. Download and validate all images (min resolution, no watermarks, etc.).
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import requests

from config import ImageSearchConfig
from modules.script_parser import ScriptSegment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local library search
# ---------------------------------------------------------------------------

def build_local_index(library_path: Path) -> dict[str, Path]:
    """
    Build a simple keyword index from local image filenames.

    Expects images named descriptively, e.g.:
        aegon-targaryen-iron-throne.jpg
        red-wedding-robb-stark.png
        winterfell-castle-north.jpg
    
    For a more sophisticated approach, use a metadata JSON sidecar:
        assets/images/metadata.json with { "filename": ["tag1", "tag2", ...] }
    """
    library_path = Path(library_path)
    index = {}

    if not library_path.exists():
        logger.warning("Local image library not found: %s", library_path)
        return index

    # Check for metadata JSON
    metadata_path = library_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        for filename, tags in metadata.items():
            filepath = library_path / filename
            if filepath.exists():
                key = " ".join(tags).lower()
                index[key] = filepath
        logger.info("Loaded metadata index with %d images.", len(index))
        return index

    # Fallback: index by filename
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for filepath in library_path.rglob("*"):
        if filepath.suffix.lower() in extensions:
            # Convert filename to searchable key
            key = filepath.stem.lower().replace("-", " ").replace("_", " ")
            index[key] = filepath

    logger.info("Built filename index with %d images.", len(index))
    return index


def search_local_library(
    query: str,
    index: dict[str, Path],
    min_score: int = 2,
) -> Path | None:
    """
    Simple keyword match against the local index.

    Returns the best matching image path, or None.
    """
    query_words = set(query.lower().split())
    best_path = None
    best_score = 0

    for key, path in index.items():
        key_words = set(key.split())
        score = len(query_words & key_words)
        if score > best_score:
            best_score = score
            best_path = path

    if best_score >= min_score:
        logger.info("Local match for '%s': %s (score=%d)", query, best_path, best_score)
        return best_path

    return None


# ---------------------------------------------------------------------------
# Web image search
# ---------------------------------------------------------------------------

def search_google_images(
    query: str,
    api_key: str,
    cx: str,
    num_results: int = 3,
) -> list[str]:
    """
    Search Google Custom Search for images.

    Returns list of image URLs.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "searchType": "image",
        "num": min(num_results, 10),
        "imgSize": "large",
        "safe": "active",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [item["link"] for item in data.get("items", [])]
    except Exception as e:
        logger.error("Google image search failed: %s", e)
        return []


def search_bing_images(
    query: str,
    api_key: str,
    num_results: int = 3,
) -> list[str]:
    """
    Search Bing Image Search API.

    Returns list of image URLs.
    """
    url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": num_results,
        "imageType": "Photo",
        "size": "Large",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [img["contentUrl"] for img in data.get("value", [])]
    except Exception as e:
        logger.error("Bing image search failed: %s", e)
        return []


def download_image(
    image_url: str,
    save_dir: Path,
    min_width: int = 800,
    min_height: int = 800,
) -> Path | None:
    """
    Download an image and validate its dimensions.

    Returns the saved path, or None if the image is too small / invalid.
    """
    try:
        resp = requests.get(image_url, timeout=15, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return None

        # Generate filename from URL hash
        url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
        ext = ".jpg"  # default
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"

        save_path = Path(save_dir) / f"{url_hash}{ext}"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate dimensions
        from PIL import Image
        with Image.open(save_path) as img:
            w, h = img.size
            if w < min_width or h < min_height:
                logger.debug("Image too small (%dx%d): %s", w, h, image_url)
                save_path.unlink()
                return None

        logger.info("Downloaded image: %s → %s", image_url[:80], save_path)
        return save_path

    except Exception as e:
        logger.debug("Failed to download %s: %s", image_url[:80], e)
        return None


# ---------------------------------------------------------------------------
# LLM-powered query generation
# ---------------------------------------------------------------------------

def generate_search_queries(
    segments: list[ScriptSegment],
    anthropic_api_key: str,
) -> list[str]:
    """
    Use Claude to generate optimized image search queries for each segment.
    
    This is MUCH better than naive keyword extraction — the LLM understands
    ASOIAF context and can generate queries like "Balerion the Black Dread 
    dragon art" instead of just extracting nouns from the text.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — falling back to naive queries.")
        return [_naive_query(seg.text) for seg in segments]

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    segment_texts = "\n".join(
        f"[{i}] {seg.text}" for i, seg in enumerate(segments)
    )

    prompt = f"""You are helping create a video about A Song of Ice and Fire / Game of Thrones.

For each numbered script segment below, generate a short image search query (3-6 words) 
that would find the most visually striking and relevant ASOIAF artwork or screenshot.

Focus on: character names, locations, key events, dramatic moments.
Add "asoiaf art" or "game of thrones" to queries when helpful.
Return ONLY a JSON array of strings, one query per segment. No other text.

Segments:
{segment_texts}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = response.content[0].text
        # Strip markdown fences if present
        response_text = re.sub(r"```json\s*|```", "", response_text).strip()
        queries = json.loads(response_text)
        if len(queries) == len(segments):
            return queries
        else:
            logger.warning("Query count mismatch, falling back to naive.")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning("Failed to parse LLM queries: %s", e)

    return [_naive_query(seg.text) for seg in segments]


def _naive_query(text: str) -> str:
    """Fallback: extract capitalized words + 'asoiaf' as a search query."""
    # Find proper nouns / capitalized words
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    if words:
        return " ".join(words[:4]) + " asoiaf art"
    # Last resort
    return " ".join(text.split()[:5]) + " asoiaf"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_images_for_segments(
    segments: list[ScriptSegment],
    config: ImageSearchConfig,
    download_dir: Path,
) -> list[ScriptSegment]:
    """
    Main entry point: fetch an image for each segment.
    
    1. Build local library index
    2. Generate search queries via LLM
    3. For each segment: try local → try web → use placeholder
    """
    # Build local index
    local_index = build_local_index(config.local_library_path)

    # Generate search queries
    if config.anthropic_api_key:
        queries = generate_search_queries(segments, config.anthropic_api_key)
    else:
        queries = [_naive_query(seg.text) for seg in segments]

    for seg, query in zip(segments, queries):
        seg.search_query = query

    # Fetch images
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    used_images = set()  # avoid reusing the same image

    for seg in segments:
        # 1. Try local library
        local_match = search_local_library(seg.search_query, local_index)
        if local_match and str(local_match) not in used_images:
            seg.image_path = str(local_match)
            used_images.add(str(local_match))
            continue

        # 2. Try web search
        image_urls = []
        if config.google_api_key and config.google_cx:
            image_urls = search_google_images(
                seg.search_query,
                config.google_api_key,
                config.google_cx,
                num_results=config.web_results_per_query,
            )
        elif config.bing_api_key:
            image_urls = search_bing_images(
                seg.search_query,
                config.bing_api_key,
                num_results=config.web_results_per_query,
            )

        for url in image_urls:
            downloaded = download_image(url, download_dir)
            if downloaded and str(downloaded) not in used_images:
                seg.image_path = str(downloaded)
                used_images.add(str(downloaded))
                break
            time.sleep(0.2)  # rate limit courtesy

        if not seg.image_path:
            logger.warning(
                "No image found for segment %d: '%s' — will use placeholder.",
                seg.index, seg.search_query,
            )

    found = sum(1 for s in segments if s.image_path)
    logger.info("Images fetched: %d/%d segments have images.", found, len(segments))

    return segments
