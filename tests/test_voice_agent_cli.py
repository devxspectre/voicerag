import sys
from pathlib import Path

from voice_agent import parse_args


def test_voice_agent_uses_low_latency_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["voice_agent.py", "resume_remote.pdf"])

    args = parse_args()

    assert args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert args.whisper_model == "tiny"
    assert args.whisper_compute_type == "int8"
    assert args.vad_silence_seconds == 0.5
    assert args.vad_max_seconds == 12
    assert args.pdf == Path("resume_remote.pdf")
