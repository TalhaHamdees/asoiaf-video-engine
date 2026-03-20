# ASOIAF Short-Form Video Engine

Automated pipeline: script → voiceover → images → Ken Burns transitions → captions → video.

## Two-Phase Workflow

```bash
# Phase 1: Generate voiceover + get your image shopping list
python main.py prepare --script my_script.txt --title "The Doom of Valyria"

# Check output/shopping_list.txt — it tells you which images to find
# Download them into input/images/ as 01.jpg, 02.jpg, etc.

# Phase 2: Build the video (also auto-tags new images into library)
python main.py continue
```

The library grows with every video. Early on you'll download 15+ images per video. After 30+ videos, the library handles everything automatically.

## Quick Setup

```bash
pip install -r requirements.txt
sudo apt install ffmpeg
# Edit config.py with your API keys (ElevenLabs required, Anthropic recommended)
```

## Folder Structure

```
input/images/        ← Drop downloaded images here (01.jpg, 02.jpg, ...)
assets/images/       ← Permanent library (auto-managed, grows over time)
output/              ← Final videos + shopping lists
temp/                ← Intermediate files
```

## Commands

```bash
python main.py prepare --script script.txt --title "Video Title"  # Phase 1
python main.py continue                                           # Phase 2
python main.py run --script script.txt --title "Video Title"      # Both phases (when library covers everything)
```
