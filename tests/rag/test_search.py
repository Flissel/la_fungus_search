def test_hybrid_search_merging_and_scoring(monkeypatch):
    from embeddinggemma.rag import search

    class DummyNode:
        def __init__(self, text, node_id="n"):
            self.text = text
            self.node_id = node_id
            self.metadata = {"source": node_id}

        def get_content(self):
            return self.text

    class Retriever:
        def __init__(self, nodes):
            self._nodes = nodes

        def retrieve(self, query):
            return self._nodes

    class Index:
        def as_retriever(self, similarity_top_k=10, node_postprocessors=None):
            # First call from hybrid_search: semantic
            if node_postprocessors is None:
                return Retriever([DummyNode("foo bar"), DummyNode("bar baz")])
            # Keyword path: still return some nodes
            return Retriever([DummyNode("bar qux")])

    results = search.hybrid_search(Index(), query="bar", top_k=2, alpha=0.5)
    assert isinstance(results, list) and len(results) <= 2
    assert all("hybrid_score" in r for r in results)


