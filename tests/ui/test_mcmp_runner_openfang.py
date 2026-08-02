import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def test_diverse_result_selection_propagates_embedding_gateway_failure():
    from embeddinggemma.ui.mcmp_runner import select_diverse_results

    class UnavailableEmbedder:
        def encode(self, _texts):
            raise RuntimeError("OpenFang unreachable")

    class Retriever:
        embedding_model = UnavailableEmbedder()

    with pytest.raises(RuntimeError, match="OpenFang unreachable"):
        select_diverse_results(
            [{"content": "one", "metadata": {}, "relevance_score": 1.0}],
            Retriever(),
            top_k=1,
            alpha=0.7,
            dedup_tau=0.9,
            per_folder_cap=1,
        )
