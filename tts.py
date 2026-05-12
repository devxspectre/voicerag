from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from voice import build_synthesizer


def main() -> None:
    load_dotenv()
    args = parse_args()
    synthesizer = build_synthesizer(
        checkpoint_dir=args.checkpoints,
        language=args.language,
        speaker=args.speaker,
        reference_audio=args.reference_audio,
        device=args.device,
    )
    path = synthesizer.synthesize(args.text, args.output)
    print(f"Wrote {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speech with Dia.")
    parser.add_argument("text", nargs="?", default="Hello from the RAG voice agent.")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--language", default="English")
    parser.add_argument("--speaker", default="friendly")
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
