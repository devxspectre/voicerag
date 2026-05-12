# RAG Voice Agent

This project indexes a PDF, retrieves relevant chunks with hybrid dense/BM25
search, and can answer either text or spoken questions.

## Text CLI

```bash
python main.py sample.pdf --embedder transformers --embedding-model BAAI/bge-m3 \
  -q "What is this document about?"
```

Use `--embedder mistral` only if you want API-backed Mistral embeddings. Local
Hugging Face models such as `BAAI/bge-m3` do not need an embeddings API key.

## Evals

Run the predefined eval cases in `evals/cases.jsonl`:

```bash
uv run python -m evals.run resume_remote.pdf
```

Include DeepSeek judge grading:

```bash
uv run python -m evals.run resume_remote.pdf --eval-judge
```

Chat with the PDF while judging each answer live:

```bash
uv run python -m evals.chat resume_remote.pdf
```

Batch evals write `outputs/evals/results.jsonl` and
`outputs/evals/summary.json`. Live eval chat appends turns to
`outputs/evals/live_results.jsonl`. RAG observability traces are written to
`outputs/observability/traces.jsonl`.

## Voice CLI

```bash
python voice_agent.py sample.pdf question.wav -o outputs/answer.wav
```

For a continued microphone conversation over a PDF:

```bash
python voice_agent.py resume_remote.pdf --mic --play
```

Press Enter to record each turn. By default each turn records 6 seconds; change
that with `--record-seconds 10`. The PDF is indexed once, and recent turns are
kept in memory for follow-up questions.

For hands-free turn detection, use automatic voice activity detection:

```bash
python voice_agent.py resume_remote.pdf --auto-vad --play
```

Mirinda will listen continuously, start recording when your voice crosses the
RMS threshold, and stop the turn after silence. Tune noisy rooms with
`--vad-threshold`, `--vad-silence-seconds`, and `--vad-max-seconds`.

The voice CLI defaults to `tiny` Whisper with `int8`,
`sentence-transformers/all-MiniLM-L6-v2` embeddings, short VAD silence, and
macOS system speech for TTS.

The voice path is:

1. `faster-whisper` transcribes the input audio.
2. The configured embedding model creates vectors for indexing and retrieval.
3. The existing RAG pipeline generates an answer.
4. macOS system speech writes the spoken answer to an audio file.

Useful environment variables:

```bash
DEEPSEEK_API_KEY=...
MIRINDA_SYSTEM_VOICE=Samantha
WHISPER_MODEL=tiny
WHISPER_COMPUTE_TYPE=int8
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TTS_ENGINE=system
VAD_SILENCE_SECONDS=0.5
VAD_MAX_SECONDS=12
```

Mirinda defaults to a macOS female system voice preference, starting with
`Samantha` when installed. Override it with `MIRINDA_SYSTEM_VOICE` if you prefer
another `say` voice from `say -v ?`.


uv run python -m evals.run resume_remote.pdf
uv run python -m evals.run resume_remote.pdf --eval-judge
uv run python -m evals.chat resume_remote.pdf
