class DummyCollections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n}) for n in names]


class DummyClient:
    def __init__(self):
        self.created = []
        self.deleted = []
        self._size = 128
        self._exists = False

    def get_collections(self):
        return DummyCollections(["other"] if not self._exists else ["codebase"])

    def get_collection(self, collection_name):
        vectors = type("V", (), {"size": self._size})
        params = type("P", (), {"vectors": vectors})
        config = type("Cfg", (), {"params": params})
        return type("Info", (), {"config": config})

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self._exists = True

    def delete_collection(self, name):
        self.deleted.append(name)
        self._exists = False


def test_ensure_collection_creates_when_missing(monkeypatch):
    from embeddinggemma.rag.vectorstore import ensure_collection
    client = DummyClient()
    ensure_collection(client, "codebase", desired_dim=64)
    assert client.created, "should create collection when missing"


def test_ensure_collection_recreate_on_dim_mismatch(monkeypatch):
    from embeddinggemma.rag.vectorstore import ensure_collection
    client = DummyClient()
    client._exists = True
    client._size = 256
    ensure_collection(client, "codebase", desired_dim=64)
    assert client.deleted and client.created


