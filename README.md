# ASOIAF Short-Form Video Engine

Automated pipeline that turns narration scripts into short-form vertical videos (1080x1920) with AI voiceover, intelligent image matching, Ken Burns transitions, and burned-in captions — built for ASOIAF / Game of Thrones content.

## How It Works

The engine uses a **two-phase workflow** with a self-growing image library:

```
Phase 1: prepare     Script → TTS → Audio cleanup → Entity extraction → Library matching → Shopping list
                      ↓ (you download missing images)
Phase 2: continue    Pick up images → Ken Burns video → SRT captions → Auto-tag & ingest to library
```

Every image you provide gets **auto-tagged by Claude** and added to the permanent library. Early videos require manual image sourcing, but as the library grows, coverage increases automatically — by ~30 videos, most content is fully automated.

## Quick Start

### Prerequisites

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [ElevenLabs](https://elevenlabs.io/) API key (required for TTS)
- [Anthropic](https://console.anthropic.com/) API key (recommended for smart entity extraction)

### Setup

```bash
git clone https://github.com/TalhaHamdees/asoiaf-video-engine.git
cd asoiaf-video-engine
pip install -r requirements.txt
```

Edit `config.py` with your API keys:

```python
class ElevenLabsConfig:
    api_key: str = "your-elevenlabs-key"
    voice_id: str = "your-voice-id"

class ImageSearchConfig:
    anthropic_api_key: str = "your-anthropic-key"  # optional but recommended
```

### Usage

**Phase 1** — Generate voiceover and get your image shopping list:

```bash
python main.py prepare --script script.txt --title "The Doom of Valyria"
```

**Between phases** — Check `output/shopping_list.txt`, download the listed images, and save them to `input/images/` as `01.jpg`, `02.jpg`, etc.

**Phase 2** — Build the video:

```bash
python main.py continue
```

**One-shot** — If the library already covers everything:

```bash
python main.py run --script script.txt --title "The Doom of Valyria"
```

## Project Structure

```
├── main.py                  # Pipeline orchestrator & CLI
├── config.py                # All configuration (API keys, video settings, thresholds)
├── requirements.txt         # Python dependencies
├── modules/
│   ├── script_parser.py     # Script → timed segments
│   ├── tts_generator.py     # ElevenLabs TTS with word timestamps
│   ├── audio_processor.py   # Silence removal & normalization
│   ├── transcriber.py       # Whisper fallback for word timing
│   ├── image_manager.py     # Library system, entity extraction, scoring, auto-tagging
│   └── video_composer.py    # Ken Burns effects, video assembly, SRT captions
├── input/images/            # Drop downloaded images here (01.jpg, 02.jpg, ...)
├── assets/
│   ├── images/              # Permanent image library (auto-managed)
│   ├── fonts/               # Caption fonts (Montserrat-Bold.ttf)
│   └── music/               # Background music (future)
├── temp/                    # Intermediate files (auto-created)
└── output/                  # Final videos & shopping lists
```

## Image Library System

The core innovation is an **incremental image library** with LLM-powered tagging:

- **Entity extraction** — Claude analyzes each script segment to identify characters, locations, events, mood, and visual concepts
- **Scoring algorithm** — Library images are scored against segment entities (character match: +5, event: +4, location: +3, concept: +2, mood: +1) with recency penalties to avoid repetition
- **Auto-tagging** — New images are tagged by Claude with rich metadata before being ingested
- **Usage tracking** — `usage_log.json` tracks reuse frequency; `gaps.json` flags overused images needing more variety

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| Video resolution | 1080x1920 | Vertical short-form format |
| Image interval | 3.5s | Target duration per image |
| Match threshold | 5.0 | Minimum score to accept a library match |
| Ken Burns zoom | 1.0–1.15 | Zoom range for motion effect |
| Caption font | Montserrat-Bold | SRT caption font |
| Whisper model | base | Speech-to-text model size |

## CLI Reference

```bash
# Phase 1: Prepare audio + shopping list
python main.py prepare --script <file> --title <name> [--auto] [--voice-id <id>]

# Phase 2: Build video from provided images
python main.py continue

# Full pipeline (prepare + continue)
python main.py run --script <file> --title <name> [--voice-id <id>]

# Options
--debug          Enable debug logging
--auto           Auto-continue if library covers all segments
--voice-id       Override ElevenLabs voice ID
```

## Dependencies

| Package | Purpose |
|---------|---------|
| moviepy | Video composition and concatenation |
| Pillow | Image loading, resizing, cropping |
| pydub | Audio manipulation and silence detection |
| numpy | Array operations for Ken Burns effects |
| requests | ElevenLabs API calls |
| faster-whisper | Word-level speech-to-text timestamps |
| anthropic | Claude API for entity extraction and auto-tagging |

**System requirement:** FFmpeg must be installed and available on PATH.

## License

MIT
