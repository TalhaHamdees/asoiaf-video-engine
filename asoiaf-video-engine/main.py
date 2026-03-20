#!/usr/bin/env python3
"""
ASOIAF Short-Form Video Engine — CLI

Usage:
    python main.py prepare --script script.txt --title "The Doom of Valyria"
    python main.py continue
    python main.py prepare --script script.txt --title "Jon Snow" --auto
"""

import argparse
import logging
import sys
from pathlib import Path

from config import Config
from pipeline import phase_prepare, phase_continue, MissingImagesError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def main():
    parser = argparse.ArgumentParser(
        description="ASOIAF Short-Form Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline phase")

    prep = subparsers.add_parser("prepare", help="Phase 1: Generate audio + shopping list")
    prep_input = prep.add_mutually_exclusive_group(required=True)
    prep_input.add_argument("--script", type=str, help="Path to script text file")
    prep_input.add_argument("--script-text", type=str, help="Script text directly")
    prep.add_argument("--title", type=str, default="video", help="Video title/name")
    prep.add_argument("--auto", action="store_true",
                       help="Auto-continue to video if library covers all segments")
    prep.add_argument("--voice-id", type=str, help="ElevenLabs voice ID override")

    subparsers.add_parser("continue", help="Phase 2: Build video from images")

    oneshot = subparsers.add_parser("run", help="Full run (prepare + continue)")
    oneshot_input = oneshot.add_mutually_exclusive_group(required=True)
    oneshot_input.add_argument("--script", type=str, help="Path to script text file")
    oneshot_input.add_argument("--script-text", type=str, help="Script text directly")
    oneshot.add_argument("--title", type=str, default="video", help="Video title/name")
    oneshot.add_argument("--voice-id", type=str, help="ElevenLabs voice ID override")

    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Config()

    if hasattr(args, 'voice_id') and args.voice_id:
        config.elevenlabs.voice_id = args.voice_id

    if not config.elevenlabs.api_key:
        logger.error("Set ELEVENLABS_API_KEY in your .env file.")
        sys.exit(1)

    if args.command in ("prepare", "run"):
        if args.script:
            script_path = Path(args.script)
            if not script_path.exists():
                logger.error("Script not found: %s", script_path)
                sys.exit(1)
            script_text = script_path.read_text(encoding="utf-8")
        else:
            script_text = args.script_text

        auto = args.command == "run" or getattr(args, 'auto', False)
        result = phase_prepare(script_text, config, args.title, auto_continue=auto)

        if result["shopping_list_path"]:
            _safe_print(f"\n[INFO] Shopping list saved: {result['shopping_list_path']}")
            _safe_print(f"   Download images -> {config.input_dir}/")
            _safe_print(f"   Then run: python main.py continue\n")
        elif result["all_covered"] and not auto:
            _safe_print("\n[OK] All segments covered from library!")
            _safe_print("Run `python main.py continue` to generate the video.\n")

        if args.command == "run":
            try:
                output = phase_continue(config)
                _safe_print(f"\n[OK] Video generated: {output}\n")
            except MissingImagesError as e:
                _safe_print(f"\n[WARN] {e}")
                sys.exit(1)

    elif args.command == "continue":
        try:
            output = phase_continue(config)
            _safe_print(f"\n[OK] Video generated: {output}\n")
        except MissingImagesError as e:
            _safe_print(f"\n[WARN] {e}")
            for a in e.missing:
                _safe_print(f"   Expected: {config.input_dir}/{a.segment_index + 1:02d}.jpg")
            _safe_print("\nPlease add the missing images and run `python main.py continue` again.")
            sys.exit(1)


if __name__ == "__main__":
    main()
