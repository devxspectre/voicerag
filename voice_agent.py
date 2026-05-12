from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from core import HierarchicalChunker, RagPipeline
from main import (
    build_embedder,
    extract_pdf_pages,
    normalize_embedding_model,
    print_sources,
)
from router.llm_router import DeepSeekRouter
from voice import (
    AudioPlayer,
    FasterWhisperTranscriber,
    MicrophoneRecorder,
    VoiceAgent,
    build_synthesizer,
)


LOGGER = logging.getLogger("voice_agent")


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pages = extract_pdf_pages(args.pdf.resolve())
    if not pages:
        raise SystemExit(f"No extractable text found in {args.pdf}")

    answer_generator = None
    if not args.no_llm:
        answer_generator = DeepSeekRouter(model_tier=args.model_tier).generate

    pipeline = RagPipeline(
        embedder=build_embedder(args.embedder, args.embedding_model),
        answer_generator=answer_generator,
    )
    indexed = pipeline.index_pages(
        pages=pages,
        tenant_id=args.tenant_id,
        source_file=str(args.pdf.resolve()),
        document_id=args.document_id,
        chunker=HierarchicalChunker(
            parent_token_limit=args.parent_tokens,
            child_token_limit=args.child_tokens,
            child_overlap_tokens=args.child_overlap,
        ),
    )
    LOGGER.info(
        "Indexed %s parent chunk(s) and %s child chunk(s)",
        indexed.parent_count,
        indexed.child_count,
    )

    agent = VoiceAgent(
        pipeline=pipeline,
        tenant_id=args.tenant_id,
        top_k=args.top_k,
        transcriber=FasterWhisperTranscriber(
            model_size=args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            language=args.stt_language,
        ),
        synthesizer=build_synthesizer(
            checkpoint_dir=args.openvoice_checkpoints,
            language=args.tts_language,
            speaker=args.tts_speaker,
            reference_audio=args.reference_audio,
            device=args.tts_device,
        ),
        history_window=args.history_turns,
    )

    if args.mic:
        microphone_loop(agent=agent, args=args)
        return

    if args.audio is None:
        raise SystemExit("Provide an audio file or use --mic.")

    result = agent.answer_audio(args.audio.resolve(), args.output.resolve())

    print(f"\nYou> {result.question}")
    print(f"\nMirinda> {result.answer}")
    print(f"\nAudio> {result.audio_path}")
    print_sources(result.citations)


def microphone_loop(agent: VoiceAgent, args: argparse.Namespace) -> None:
    recorder = MicrophoneRecorder(
        sample_rate=args.mic_sample_rate,
        channels=args.mic_channels,
        device=args.mic_device,
    )
    player = AudioPlayer(device=args.playback_device)
    args.turn_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_vad:
        print("\nMirinda is listening. Speak naturally; press Ctrl+C to stop.")
    else:
        print(
            "\nMicrophone conversation started. Press Enter to record a turn; "
            "type 'q' then Enter to quit."
        )
    turn = 1
    while True:
        input_path = args.turn_dir / f"turn-{turn:03d}-question.wav"
        output_path = args.turn_dir / f"turn-{turn:03d}-answer.wav"
        if args.auto_vad:
            print(f"\nTurn {turn}> Listening...")
            try:
                recorder.record_until_silence(
                    input_path,
                    speech_threshold=args.vad_threshold,
                    silence_seconds=args.vad_silence_seconds,
                    min_speech_seconds=args.vad_min_speech_seconds,
                    max_seconds=args.vad_max_seconds,
                    pre_roll_seconds=args.vad_pre_roll_seconds,
                )
            except TimeoutError as exc:
                print(f"Mirinda> {exc}")
                continue
        else:
            command = input(f"\nTurn {turn}> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                return
            print(f"Recording for {args.record_seconds:g} seconds...")
            recorder.record(input_path, seconds=args.record_seconds)

        try:
            result = agent.answer_audio(input_path, output_path)
        except ValueError as exc:
            print(f"Mirinda> {exc}")
            continue

        print(f"\nYou> {result.question}")
        print(f"\nMirinda> {result.answer}")
        print(f"\nAudio> {result.audio_path}")
        print_sources(result.citations)

        if args.play:
            player.play(result.audio_path)
        turn += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the PDF a spoken question and receive a spoken answer.",
    )
    parser.add_argument("pdf", type=Path, help="PDF to index.")
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Input audio file containing a question. Omit when using --mic.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/voice-answer.wav"),
        help="Output WAV path for the spoken answer.",
    )
    parser.add_argument("--tenant-id", default=os.getenv("DEFAULT_TENANT", "local"))
    parser.add_argument("--document-id")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=int(os.getenv("VOICE_HISTORY_TURNS", "6")),
        help="Recent conversation turns to include for follow-up questions.",
    )
    parser.add_argument(
        "--embedder",
        choices=["transformers", "mistral", "hash"],
        default=os.getenv("VOICE_EMBEDDER", "transformers"),
        help="Embedding backend. Use transformers for local Hugging Face models.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Embedding model name for transformers or Mistral backends.",
    )
    parser.add_argument("--model-tier", choices=["fast", "smart"], default="fast")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--parent-tokens", type=int, default=1600)
    parser.add_argument("--child-tokens", type=int, default=500)
    parser.add_argument("--child-overlap", type=int, default=75)
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "tiny"))
    parser.add_argument("--whisper-device", default=os.getenv("WHISPER_DEVICE", "auto"))
    parser.add_argument(
        "--whisper-compute-type",
        default=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )
    parser.add_argument("--stt-language", default=os.getenv("STT_LANGUAGE"))
    parser.add_argument(
        "--openvoice-checkpoints",
        type=Path,
        default=Path(os.getenv("OPENVOICE_CHECKPOINT_DIR", "checkpoints")),
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=_reference_audio_default(),
        help=(
            "Reference voice clip for Mirinda. Defaults to "
            "MIRINDA_VOICE_REFERENCE_AUDIO or OPENVOICE_REFERENCE_AUDIO."
        ),
    )
    parser.add_argument("--tts-language", default=os.getenv("TTS_LANGUAGE", "English"))
    parser.add_argument("--tts-speaker", default=os.getenv("TTS_SPEAKER", "friendly"))
    parser.add_argument("--tts-device", default=os.getenv("TTS_DEVICE"))
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Start an interactive microphone conversation loop.",
    )
    parser.add_argument(
        "--auto-vad",
        action="store_true",
        help="Listen continuously and automatically stop each turn after silence.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=float(os.getenv("MIC_RECORD_SECONDS", "6")),
        help="Seconds to record for each microphone turn.",
    )
    parser.add_argument(
        "--turn-dir",
        type=Path,
        default=Path(os.getenv("VOICE_TURN_DIR", "outputs/voice-turns")),
        help="Directory for per-turn microphone and answer audio.",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=int(os.getenv("MIC_SAMPLE_RATE", "16000")),
    )
    parser.add_argument(
        "--mic-channels",
        type=int,
        default=int(os.getenv("MIC_CHANNELS", "1")),
    )
    parser.add_argument("--mic-device", default=os.getenv("MIC_DEVICE"))
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=float(os.getenv("VAD_THRESHOLD", "0.02")),
        help="RMS level that starts/continues speech detection.",
    )
    parser.add_argument(
        "--vad-silence-seconds",
        type=float,
        default=float(os.getenv("VAD_SILENCE_SECONDS", "0.5")),
        help="Silence duration that ends an automatic turn.",
    )
    parser.add_argument(
        "--vad-min-speech-seconds",
        type=float,
        default=float(os.getenv("VAD_MIN_SPEECH_SECONDS", "0.4")),
        help="Minimum detected speech before auto-stop can trigger.",
    )
    parser.add_argument(
        "--vad-max-seconds",
        type=float,
        default=float(os.getenv("VAD_MAX_SECONDS", "12")),
        help="Maximum length for one automatic microphone turn.",
    )
    parser.add_argument(
        "--vad-pre-roll-seconds",
        type=float,
        default=float(os.getenv("VAD_PRE_ROLL_SECONDS", "0.25")),
        help="Audio kept from just before speech starts.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play each synthesized answer through the default speaker.",
    )
    parser.add_argument("--playback-device", default=os.getenv("PLAYBACK_DEVICE"))
    args = parser.parse_args()
    if args.auto_vad:
        args.mic = True
    if (
        args.embedder == "mistral"
        and args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    ):
        args.embedding_model = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
    args.embedding_model = normalize_embedding_model(args.embedding_model)
    return args


def _reference_audio_default() -> Path | None:
    value = os.getenv("MIRINDA_VOICE_REFERENCE_AUDIO") or os.getenv(
        "OPENVOICE_REFERENCE_AUDIO"
    )
    return Path(value) if value else None


if __name__ == "__main__":
    main()
