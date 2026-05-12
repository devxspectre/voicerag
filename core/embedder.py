from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Iterable, Sequence
from typing import Protocol

from dotenv import load_dotenv


class Embedder(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [list(map(float, embedding)) for embedding in embeddings]


class TransformersTextEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._model

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()


class HashingEmbedder:
    """Small deterministic embedder for tests and offline smoke checks."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        self.dimensions = dimensions

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_l2_normalize(self._embed_one(text)) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return vector


class MistralAIEmbedder:
    def __init__(
        self,
        model_name: str = "mistral-embed",
        api_key: str | None = None,
        batch_size: int = 32,
    ) -> None:
        load_dotenv()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_name = model_name
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.batch_size = batch_size
        self._client = None

        if not self.api_key:
            raise RuntimeError("No Mistral API key found. Set MISTRAL_API_KEY.")

    @property
    def client(self):
        if self._client is None:
            from mistralai import Mistral

            self._client = Mistral(api_key=self.api_key)
        return self._client

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        values = list(texts)
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=batch,
            )
            embeddings.extend(
                list(map(float, item.embedding))
                for item in sorted(response.data, key=lambda item: item.index)
            )
        return embeddings


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def _l2_normalize(vector: Iterable[float]) -> list[float]:
    values = list(vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _mean_pool(token_embeddings, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
