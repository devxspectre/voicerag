from pathlib import Path

from core.chunker import Chunk, ChunkPosition
from core.pipeline import RagPipeline
from voice.agent import VoiceAgent, _spoken_answer
from voice.stt import TranscriptionResult


class StaticEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


class StaticTranscriber:
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            text="What is the invoice identifier?",
            language="en",
            duration=1.25,
        )


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def synthesize(self, text: str, output_path: Path) -> Path:
        self.calls.append((text, output_path))
        return output_path


def make_chunk(chunk_id: str, parent_id: str | None, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id="tenant-a",
        source_file="docs/source.pdf",
        document_id="doc-1",
        position=ChunkPosition(page=1, chunk_index=0),
        parent_id=parent_id,
        text=text,
    )


def test_voice_agent_transcribes_queries_and_synthesizes_answer(tmp_path: Path) -> None:
    synthesizer = RecordingSynthesizer()
    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        answer_generator=lambda _prompt: "The invoice identifier is ZXQ-7781.",
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "The invoice identifier is ZXQ-7781.")],
        [make_chunk("child-1", "parent-1", "invoice identifier ZXQ-7781")],
    )
    audio_path = tmp_path / "question.wav"
    output_path = tmp_path / "answer.wav"
    audio_path.write_bytes(b"placeholder")

    result = VoiceAgent(
        pipeline=pipeline,
        transcriber=StaticTranscriber(),
        synthesizer=synthesizer,
        tenant_id="tenant-a",
    ).answer_audio(audio_path, output_path)

    assert result.question == "What is the invoice identifier?"
    assert result.answer == "The invoice identifier is ZXQ-7781."
    assert result.audio_path == output_path
    assert result.citations[0].child_id == "child-1"
    spoken_text, spoken_path = synthesizer.calls[0]
    assert spoken_text.startswith("The invoice identifier is ZXQ-7781.")
    assert "page 1" in spoken_text
    assert spoken_path == output_path


def test_voice_agent_keeps_recent_conversation_context(tmp_path: Path) -> None:
    prompts: list[str] = []

    def answer_generator(prompt: str) -> str:
        prompts.append(prompt)
        return "Answer."

    pipeline = RagPipeline(
        embedder=StaticEmbedder(),
        answer_generator=answer_generator,
    )
    pipeline.index_chunks(
        [make_chunk("parent-1", None, "The invoice identifier is ZXQ-7781.")],
        [make_chunk("child-1", "parent-1", "invoice identifier ZXQ-7781")],
    )
    audio_path = tmp_path / "question.wav"
    audio_path.write_bytes(b"placeholder")
    agent = VoiceAgent(
        pipeline=pipeline,
        transcriber=StaticTranscriber(),
        synthesizer=RecordingSynthesizer(),
        tenant_id="tenant-a",
        history_window=2,
    )

    agent.answer_audio(audio_path, tmp_path / "answer-1.wav")
    agent.answer_audio(audio_path, tmp_path / "answer-2.wav")

    assert "Recent conversation:" not in prompts[0]
    assert "Recent conversation:" in prompts[1]
    assert "Current question:" in prompts[1]
    assert "What is the invoice identifier?" in prompts[1]


def test_spoken_answer_keeps_not_found_direct() -> None:
    text = _spoken_answer("I don't see that in the document.", [])

    assert text == "I don't see that in the document."
