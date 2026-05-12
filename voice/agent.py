from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.pipeline import Citation, RagPipeline
from voice.stt import SpeechTranscriber, TranscriptionResult
from voice.tts import SpeechSynthesizer


@dataclass(frozen=True)
class VoiceAgentResult:
    question: str
    answer: str
    audio_path: Path
    transcription: TranscriptionResult
    citations: list[Citation]


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str


class VoiceAgent:
    def __init__(
        self,
        pipeline: RagPipeline,
        transcriber: SpeechTranscriber,
        synthesizer: SpeechSynthesizer,
        tenant_id: str,
        top_k: int = 5,
        history_window: int = 6,
    ) -> None:
        self.pipeline = pipeline
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.tenant_id = tenant_id
        self.top_k = top_k
        self.history_window = history_window
        self.history: list[ConversationTurn] = []

    def answer_audio(self, audio_path: Path, output_path: Path) -> VoiceAgentResult:
        transcription = self.transcriber.transcribe(audio_path)
        question = transcription.text.strip()
        if not question:
            raise ValueError("No speech was transcribed from the input audio")

        result = self.pipeline.query(
            tenant_id=self.tenant_id,
            question=self._question_with_history(question),
            top_k=self.top_k,
        )
        answer = result.answer or result.context or "I could not find relevant context."
        spoken_answer = _spoken_answer(answer, result.citations)
        spoken_path = self.synthesizer.synthesize(spoken_answer, output_path)
        self._remember(question, answer)

        return VoiceAgentResult(
            question=question,
            answer=answer,
            audio_path=spoken_path,
            transcription=transcription,
            citations=result.citations,
        )

    def _question_with_history(self, question: str) -> str:
        if not self.history or self.history_window < 1:
            return question

        turns = self.history[-self.history_window :]
        transcript = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in turns
        )
        return (
            "Use the recent conversation only to resolve follow-up references. "
            "Still answer from the retrieved document context.\n\n"
            f"Recent conversation:\n{transcript}\n\n"
            f"Current question:\n{question}"
        )

    def _remember(self, question: str, answer: str) -> None:
        if self.history_window < 1:
            return

        self.history.append(ConversationTurn(question=question, answer=answer))
        if len(self.history) > self.history_window:
            del self.history[: len(self.history) - self.history_window]


def _spoken_answer(answer: str, citations: list[Citation]) -> str:
    cleaned = " ".join(answer.split())
    if not cleaned:
        return "I don't see that in the document."
    if _is_not_found(cleaned):
        return cleaned

    source = _spoken_source(citations)
    if source:
        return f"{cleaned} I found that on {source}."
    return cleaned


def _is_not_found(answer: str) -> bool:
    return "i don't see that in the document" in answer.lower()


def _spoken_source(citations: list[Citation]) -> str | None:
    if not citations:
        return None

    position = citations[0].position
    page = position.get("page") or position.get("page_start")
    if page:
        return f"page {page}"
    section = position.get("section")
    if section:
        return f"the {section} section"
    return None
