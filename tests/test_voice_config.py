from pathlib import Path

from voice.tts import (
    DiaSynthesizer,
    OpenVoiceSynthesizer,
    SystemSpeechSynthesizer,
    _dia_text,
    _system_audio_path,
    build_synthesizer,
)


def test_build_synthesizer_defaults_to_system_voice() -> None:
    synthesizer = build_synthesizer()

    assert isinstance(synthesizer, SystemSpeechSynthesizer)


def test_dia_uses_reference_audio_env(monkeypatch) -> None:
    monkeypatch.setenv("DIA_REFERENCE_AUDIO", "voices/nari.wav")
    monkeypatch.setenv("DIA_REFERENCE_TRANSCRIPT", "Hello, I am Mirinda.")

    synthesizer = DiaSynthesizer()

    assert synthesizer.reference_audio == Path("voices/nari.wav")
    assert synthesizer.reference_transcript == "Hello, I am Mirinda."


def test_dia_wraps_single_speaker_text() -> None:
    assert _dia_text("Hello there.") == "[S1] Hello there."
    assert _dia_text("[S1] Hello there.") == "[S1] Hello there."


def test_openvoice_defaults_to_mirinda_friendly_voice() -> None:
    synthesizer = OpenVoiceSynthesizer()

    assert synthesizer.speaker == "friendly"


def test_system_audio_path_uses_aiff_for_wav_request() -> None:
    assert _system_audio_path(Path("answer.wav")) == Path("answer.aiff")
