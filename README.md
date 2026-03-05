# ASOIAF Short-Form Video Engine

An end-to-end automation pipeline that transforms narration scripts into fully edited short-form videos (YouTube Shorts / Reels / TikTok) featuring AI-generated voiceover, curated ASOIAF artwork, Ken Burns cinematic transitions, and synchronized burned-in captions.

Built for the **Fictopia** content brand — drop a script, get a publish-ready video.

---

## Demo Pipeline

```
Script (text)
    │
    ├──► Script Parser ──► segmented narrative beats
    ├──► ElevenLabs TTS ──► voiceover audio + word timestamps
    │         │
    │         └──► Silence Removal ──► tight, short-form pacing
    │
    ├──► Whisper Transcription ──► word-level alignment
    │
    ├──► Image Fetcher (local library + web search) ──► matched visuals
    │
    └──► Video Composer
              ├── Ken Burns pan/zoom effects
              ├── 9:16 vertical framing (1080×1920)
              ├── SRT / highlighted captions
              └──► Final MP4 output
```

---

## Features

- **ElevenLabs TTS** with native word-level timestamps (Whisper as fallback)
- **Smart silence removal** — strips dead air while preserving natural pacing
- **Timestamp remapping** — adjusts word timings after silence removal so captions stay in sync
- **Hybrid image sourcing** — local curated library with Google/Bing web search fallback
- **LLM-powered search queries** — uses Claude API to generate contextually accurate image queries from script segments
- **8 Ken Burns effects** — zoom in/out, pan 4 directions, and combo movements cycled per segment
- **Two caption modes** — SRT burn-in via FFmpeg (fast, reliable) or word-by-word highlight rendering via Pillow
- **Configurable everything** — voice settings, audio thresholds, caption styling, video encoding, all in one config file

---

## Quick Start

### Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [ElevenLabs](https://elevenlabs.io/) API key + voice ID

### Installation

```bash
git clone https://github.com/TalhaHamdees/asoiaf-video-engine.git
cd asoiaf-video-engine

pip install -r requirements.txt
```

### Configuration

Open `config.py` and set your API keys:

```python
# Required
api_key: str = "your-elevenlabs-api-key"
voice_id: str = "your-voice-id"

# Optional (improves image fetching)
google_api_key: str = "your-google-cse-key"
google_cx: str = "your-search-engine-id"
anthropic_api_key: str = "your-anthropic-key"
```

### Add Images (Recommended)

Drop curated ASOIAF artwork into `assets/images/` with descriptive filenames:

```
assets/images/
├── aegon-targaryen-conquest.jpg
├── red-wedding-robb-stark.png
├── winterfell-castle-north.jpg
└── ...
```

Or create `assets/images/metadata.json` for tag-based matching:

```json
{
  "aegon-conquest.jpg": ["aegon", "targaryen", "conquest", "dragon", "balerion"],
  "red-wedding.png": ["red wedding", "stark", "frey", "walder"]
}
```

### Run

```bash
# From a script file
python main.py --script my_script.txt --output my_video.mp4

# From inline text
python main.py --script-text "The Doom of Valyria destroyed an empire that had lasted five thousand years..."

# With options
python main.py --script script.txt --caption-style srt --voice-id pNInz6obpgDQGcFmaJgB --debug
```

---

## Project Structure

```
asoiaf-video-engine/
├── main.py                  # Pipeline orchestrator — runs all 7 steps
├── config.py                # Centralized configuration (API keys, styles, thresholds)
├── requirements.txt         # Python dependencies
├── modules/
│   ├── script_parser.py     # Splits script into segments (sentence/paragraph/marker modes)
│   ├── tts_generator.py     # ElevenLabs TTS with word-level timestamp extraction
│   ├── audio_processor.py   # Silence removal, normalization, timestamp remapping
│   ├── transcriber.py       # Whisper transcription + segment-to-audio alignment
│   ├── image_fetcher.py     # Local library search + LLM query gen + web image download
│   └── video_composer.py    # Ken Burns effects, caption rendering, MoviePy assembly
├── assets/
│   ├── images/              # Curated ASOIAF image library
│   ├── fonts/               # Caption fonts (Montserrat-Bold recommended)
│   └── music/               # Background tracks (future feature)
├── output/                  # Final rendered videos
└── temp/                    # Intermediate files (auto-cleaned between runs)
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `script_parser` | Parse script into segments, merge short ones (<5 words) |
| 2 | `tts_generator` | Generate voiceover via ElevenLabs (with timestamps if available) |
| 3 | `audio_processor` | Remove silence, normalize volume, remap timestamps |
| 4 | `transcriber` | Whisper fallback for timestamps + align segments to audio |
| 5 | `image_fetcher` | Match images from local library or fetch via web search |
| 6 | `video_composer` | Assemble clips with Ken Burns effects at 1080×1920 @ 30fps |
| 7 | `video_composer` | Generate SRT and burn captions via FFmpeg |

---

## Configuration Reference

| Section | Key Settings |
|---------|-------------|
| **Video** | 1080×1920 @ 30fps, h264 @ 8Mbps, 0.3s crossfade |
| **Ken Burns** | Zoom range 1.0–1.15×, max pan 80px, 8 effect variations |
| **Audio** | Silence threshold -40 dBFS, min silence 400ms, normalize to -20 dBFS |
| **Captions** | Montserrat Bold 60px, white text + black stroke, gold highlight, 3 words/group |
| **Whisper** | Base model, English, CPU (set `device: "cuda"` for GPU) |

---

## Roadmap

- [x] End-to-end pipeline (script → video)
- [x] ElevenLabs TTS with word timestamps
- [x] Silence removal with timestamp remapping
- [x] Ken Burns cinematic transitions
- [x] SRT caption burn-in
- [ ] Word-by-word highlighted captions (Hormozi style)
- [ ] Crossfade transitions between image segments
- [ ] Background music layer with voice ducking
- [ ] AI-generated images per scene (Flux/DALL-E) for copyright safety
- [ ] Batch processing (multiple scripts → multiple videos)
- [ ] Auto-upload to YouTube via API
- [ ] A/B thumbnail generation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video composition | MoviePy 2.x |
| Encoding | FFmpeg (h264/AAC) |
| Audio processing | pydub |
| Transcription | faster-whisper |
| Caption rendering | Pillow + FFmpeg subtitles filter |
| TTS | ElevenLabs API |
| Image search | Google CSE / Bing Image Search |
| Query generation | Claude API (Anthropic SDK) |

---

## License

MIT
