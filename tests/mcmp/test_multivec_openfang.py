import numpy as np
import pytest

from embeddinggemma.multivec import build_multivec_index


class OpenFangEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return np.ones((len(texts), 3072), dtype=np.float32)


def test_multivec_uses_the_gateway_embedding_interface_without_local_kwargs():
    model = OpenFangEmbeddingModel()

    embeddings, chunk_ids = build_multivec_index(["# file: sample.py\nline one\nline two"], model)

    assert len(model.calls) == 1
    assert embeddings.shape[1] == 3072
    assert chunk_ids.shape[0] == embeddings.shape[0]


def test_multivec_rejects_a_gateway_response_count_mismatch():
    class MissingVectorModel:
        def encode(self, texts):
            return np.ones((max(0, len(texts) - 1), 3072), dtype=np.float32)

    with pytest.raises(RuntimeError, match="response count does not match.*expected"):
        build_multivec_index(["# file: sample.py\nline one\nline two"], MissingVectorModel())
