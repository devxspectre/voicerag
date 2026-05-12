from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
from typing import Protocol


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str, output_path: Path) -> Path:
        ...


class OpenVoiceSynthesizer:
    def __init__(
        self,
        checkpoint_dir: Path | str | None = None,
        language: str = "English",
        speaker: str = "friendly",
        reference_audio: Path | str | None = None,
        device: str | None = None,
        encode_message: str = "@MyShell",
    ) -> None:
        self.checkpoint_dir = Path(
            checkpoint_dir or os.getenv("OPENVOICE_CHECKPOINT_DIR", "checkpoints")
        )
        self.language = language
        self.speaker = speaker
        self.reference_audio = _reference_audio_path(reference_audio)
        self.device = device
        self.encode_message = encode_message
        self._tts_model = None
        self._tone_color_converter = None
        self._source_se = None
        self._target_se = None

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        if self.reference_audio:
            tmp_path = output_path.with_name(f"{output_path.stem}.base.wav")
            self.tts_model.tts(
                text,
                str(tmp_path),
                speaker=self.speaker,
                language=self.language,
            )
            self.tone_color_converter.convert(
                audio_src_path=str(tmp_path),
                src_se=self.source_se,
                tgt_se=self.target_se,
                output_path=str(output_path),
                message=self.encode_message,
            )
        else:
            self.tts_model.tts(
                text,
                str(output_path),
                speaker=self.speaker,
                language=self.language,
            )
        return output_path

    @property
    def tts_model(self):
        if self._tts_model is None:
            from openvoice.api import BaseSpeakerTTS

            config_path = self.checkpoint_dir / "base_speakers" / "EN" / "config.json"
            checkpoint_path = (
                self.checkpoint_dir / "base_speakers" / "EN" / "checkpoint.pth"
            )
            self._tts_model = BaseSpeakerTTS(str(config_path), device=self._device())
            self._tts_model.load_ckpt(str(checkpoint_path))
        return self._tts_model

    @property
    def tone_color_converter(self):
        if self._tone_color_converter is None:
            from openvoice.api import ToneColorConverter

            config_path = self.checkpoint_dir / "converter" / "config.json"
            checkpoint_path = self.checkpoint_dir / "converter" / "checkpoint.pth"
            self._tone_color_converter = ToneColorConverter(
                str(config_path),
                device=self._device(),
            )
            self._tone_color_converter.load_ckpt(str(checkpoint_path))
        return self._tone_color_converter

    @property
    def source_se(self):
        if self._source_se is None:
            import torch

            speaker_key = f"en-{self.speaker}"
            source_path = (
                self.checkpoint_dir / "base_speakers" / "ses" / f"{speaker_key}.pth"
            )
            self._source_se = torch.load(
                str(source_path),
                map_location=self._device(),
                weights_only=False,
            )
        return self._source_se

    @property
    def target_se(self):
        if self._target_se is None:
            if self.reference_audio is None:
                raise RuntimeError("reference_audio is required for voice cloning")
            from openvoice import se_extractor

            self._target_se, _audio_name = se_extractor.get_se(
                str(self.reference_audio),
                self.tone_color_converter,
                target_dir=str(self.checkpoint_dir / "processed"),
                vad=True,
            )
        return self._target_se

    def _device(self) -> str:
        if self.device:
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


class DiaSynthesizer:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int | None = None,
        reference_audio: Path | str | None = None,
        reference_transcript: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "DIA_MODEL",
            "nari-labs/Dia-1.6B-0626",
        )
        self.device = device or os.getenv("DIA_DEVICE")
        self.max_new_tokens = max_new_tokens or int(
            os.getenv("DIA_MAX_NEW_TOKENS", "512")
        )
        self.reference_audio = _dia_reference_audio_path(reference_audio)
        self.reference_transcript = reference_transcript or os.getenv(
            "DIA_REFERENCE_TRANSCRIPT"
        )
        self.seed = seed if seed is not None else int(os.getenv("DIA_SEED", "42"))
        self._processor = None
        self._model = None

    def synthesize(self, text: str, output_path: Path) -> Path:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        inputs, audio_prompt_len = self._inputs(text)
        self._seed()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        if audio_prompt_len is None:
            audio = self.processor.batch_decode(outputs)
        else:
            audio = self.processor.batch_decode(
                outputs,
                audio_prompt_len=audio_prompt_len,
            )
        self.processor.save_audio(audio, str(output_path))
        return output_path

    def _inputs(self, text: str):
        if self.reference_audio is None:
            return (
                self.processor(
                    text=[_dia_text(text)],
                    padding=True,
                    return_tensors="pt",
                ).to(self._device()),
                None,
            )

        if not self.reference_transcript:
            raise RuntimeError(
                "Dia voice conditioning needs DIA_REFERENCE_TRANSCRIPT when "
                "DIA_REFERENCE_AUDIO is set."
            )

        import soundfile as sf

        audio, _sample_rate = sf.read(self.reference_audio, dtype="float32")
        prompt = f"{_dia_text(self.reference_transcript)} {_dia_text(text)}"
        inputs = self.processor(
            text=[prompt],
            audio=audio,
            padding=True,
            return_tensors="pt",
        ).to(self._device())
        return inputs, self.processor.get_audio_prompt_len(
            inputs["decoder_attention_mask"]
        )

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            from transformers import DiaForConditionalGeneration

            self._model = DiaForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self._device())
        return self._model

    def _device(self) -> str:
        if self.device:
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _seed(self) -> None:
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


class SystemSpeechSynthesizer:
    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice or os.getenv("MIRINDA_SYSTEM_VOICE") or _system_voice()

    def synthesize(self, text: str, output_path: Path) -> Path:
        if platform.system() != "Darwin":
            raise RuntimeError("macOS system speech requires macOS.")
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        audio_path = _system_audio_path(output_path)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["say", "-v", self.voice, "-o", str(audio_path), text],
            check=True,
        )
        return audio_path


def build_synthesizer(
    engine: str | None = None,
    *,
    checkpoint_dir: Path | str | None = None,
    language: str = "English",
    speaker: str = "friendly",
    reference_audio: Path | str | None = None,
    device: str | None = None,
) -> SpeechSynthesizer:
    normalized = (engine or os.getenv("TTS_ENGINE", "system")).strip().lower()
    if normalized == "system":
        return SystemSpeechSynthesizer()
    if normalized == "dia":
        return DiaSynthesizer(device=device, reference_audio=reference_audio)
    if normalized == "openvoice":
        return OpenVoiceSynthesizer(
            checkpoint_dir=checkpoint_dir,
            language=language,
            speaker=speaker,
            reference_audio=reference_audio,
            device=device,
        )

    raise ValueError("tts engine must be one of: system, dia, openvoice")


def _reference_audio_path(reference_audio: Path | str | None) -> Path | None:
    value = (
        reference_audio
        or os.getenv("MIRINDA_VOICE_REFERENCE_AUDIO")
        or os.getenv("OPENVOICE_REFERENCE_AUDIO")
    )
    return Path(value) if value else None


def _dia_reference_audio_path(reference_audio: Path | str | None) -> Path | None:
    value = (
        reference_audio
        or os.getenv("DIA_REFERENCE_AUDIO")
        or os.getenv("MIRINDA_VOICE_REFERENCE_AUDIO")
    )
    return Path(value) if value else None


def _system_audio_path(output_path: Path) -> Path:
    if output_path.suffix.lower() in {".aiff", ".aif", ".m4a"}:
        return output_path
    return output_path.with_suffix(".aiff")


def _system_voice() -> str:
    installed = _installed_system_voices()
    for voice in ("Samantha", "Ava", "Susan", "Victoria", "Karen"):
        if voice in installed:
            return voice
    return "Samantha"


def _installed_system_voices() -> set[str]:
    if platform.system() != "Darwin":
        return set()
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.strip()
    }


def _dia_text(text: str) -> str:
    stripped = " ".join(text.split())
    if stripped.startswith("[S1]") or stripped.startswith("[S2]"):
        return stripped
    return f"[S1] {stripped}"
