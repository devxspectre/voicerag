from core.embedder import MistralAIEmbedder


class EmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class EmbeddingResponse:
    def __init__(self, data: list[EmbeddingItem]) -> None:
        self.data = data


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, model: str, inputs: list[str]) -> EmbeddingResponse:
        self.calls.append((model, inputs))
        return EmbeddingResponse(
            [
                EmbeddingItem(index=index, embedding=[float(index), float(len(text))])
                for index, text in reversed(list(enumerate(inputs)))
            ]
        )


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddings()


def test_mistral_embedder_batches_and_preserves_response_order() -> None:
    client = FakeClient()
    embedder = MistralAIEmbedder(
        model_name="mistral-embed",
        api_key="test-key",
        batch_size=2,
    )
    embedder._client = client

    embeddings = embedder.embed_texts(["a", "bb", "ccc"])

    assert embeddings == [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]]
    assert client.embeddings.calls == [
        ("mistral-embed", ["a", "bb"]),
        ("mistral-embed", ["ccc"]),
    ]
