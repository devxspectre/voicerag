from __future__ import annotations

import math
from pathlib import Path


class MicrophoneRecorder:
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

    def record(self, output_path: Path, seconds: float) -> Path:
        if seconds <= 0:
            raise ValueError("seconds must be greater than zero")

        import sounddevice as sd
        import soundfile as sf

        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = int(seconds * self.sample_rate)
        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        sf.write(output_path, audio, self.sample_rate)
        return output_path

    def record_until_silence(
        self,
        output_path: Path,
        *,
        speech_threshold: float = 0.02,
        silence_seconds: float = 1.0,
        min_speech_seconds: float = 0.4,
        max_seconds: float = 30.0,
        pre_roll_seconds: float = 0.25,
        block_seconds: float = 0.1,
    ) -> Path:
        if speech_threshold <= 0:
            raise ValueError("speech_threshold must be greater than zero")
        if silence_seconds <= 0:
            raise ValueError("silence_seconds must be greater than zero")
        if min_speech_seconds <= 0:
            raise ValueError("min_speech_seconds must be greater than zero")
        if max_seconds <= min_speech_seconds:
            raise ValueError("max_seconds must be greater than min_speech_seconds")
        if block_seconds <= 0:
            raise ValueError("block_seconds must be greater than zero")

        import sounddevice as sd
        import soundfile as sf

        output_path.parent.mkdir(parents=True, exist_ok=True)
        block_frames = max(1, int(self.sample_rate * block_seconds))
        silence_blocks = max(1, math.ceil(silence_seconds / block_seconds))
        min_speech_blocks = max(1, math.ceil(min_speech_seconds / block_seconds))
        max_blocks = max(1, math.ceil(max_seconds / block_seconds))
        pre_roll_blocks = max(0, math.ceil(pre_roll_seconds / block_seconds))

        pre_roll = []
        captured = []
        speaking = False
        speech_blocks = 0
        quiet_blocks = 0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        ) as stream:
            for _block_index in range(max_blocks):
                block, _overflowed = stream.read(block_frames)
                is_speech = rms(block) >= speech_threshold

                if not speaking:
                    pre_roll.append(block.copy())
                    if len(pre_roll) > pre_roll_blocks:
                        pre_roll.pop(0)
                    if not is_speech:
                        continue

                    speaking = True
                    captured.extend(pre_roll)
                    pre_roll.clear()

                captured.append(block.copy())
                if is_speech:
                    speech_blocks += 1
                    quiet_blocks = 0
                else:
                    quiet_blocks += 1

                if (
                    speech_blocks >= min_speech_blocks
                    and quiet_blocks >= silence_blocks
                ):
                    break

        if not captured:
            raise TimeoutError("No speech detected before max_seconds elapsed")

        import numpy as np

        audio = np.concatenate(captured, axis=0)
        sf.write(output_path, audio, self.sample_rate)
        return output_path


def rms(audio) -> float:
    import numpy as np

    values = np.asarray(audio, dtype="float32")
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


class AudioPlayer:
    def __init__(self, device: int | str | None = None) -> None:
        self.device = device

    def play(self, audio_path: Path) -> None:
        import sounddevice as sd
        import soundfile as sf

        audio, sample_rate = sf.read(audio_path, dtype="float32")
        sd.play(audio, sample_rate, device=self.device)
        sd.wait()
