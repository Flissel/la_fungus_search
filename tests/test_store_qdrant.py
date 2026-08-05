from __future__ import annotations

import os

import numpy as np
import pytest

from embeddinggemma import store_qdrant


def test_collection_name_ist_stabil():
    a = store_qdrant.collection_for("C:\\Users\\User\\Desktop\\Laura")
    b = store_qdrant.collection_for("c:/Users/User/Desktop/Laura/")
    assert a == b
    assert a.startswith("fungus_Laura_")


def test_gleichnamige_ordner_kollidieren_nicht():
    a = store_qdrant.collection_for("C:\\a\\Laura")
    b = store_qdrant.collection_for("C:\\b\\Laura")
    assert a != b


class FakeClient:
    """Minimal-Double: haelt Punkte im Speicher, spricht die benutzte Qdrant-API."""

    def __init__(self):
        self.collections: dict[str, list] = {}
        self.dims: dict[str, int] = {}

    def get_collections(self):
        class R:
            pass
        r = R()
        r.collections = [type("C", (), {"name": n})() for n in self.collections]
        return r

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = []
        self.dims[collection_name] = vectors_config.size

    def get_collection(self, collection_name):
        dim = self.dims[collection_name]
        return type("I", (), {"config": type("C", (), {"params": type("P", (), {
            "vectors": type("V", (), {"size": dim})()})()})()})()

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)
        self.dims.pop(collection_name, None)

    def upsert(self, collection_name, points):
        self.collections.setdefault(collection_name, []).extend(points)

    def scroll(self, collection_name, limit, offset=None, with_vectors=True, with_payload=True):
        punkte = self.collections.get(collection_name, [])
        start = offset or 0
        teil = punkte[start:start + limit]
        naechster = start + limit if start + limit < len(punkte) else None
        return teil, naechster


def test_save_load_roundtrip():
    c = FakeClient()
    texte = ["def a(): pass", "class B: pass", "print('x')"]
    embs = np.arange(9, dtype=np.float32).reshape(3, 3)
    n = store_qdrant.save(c, "fungus_test_0000", texte, embs)
    assert n == 3
    geladen = store_qdrant.load(c, "fungus_test_0000")
    assert geladen is not None
    texte_zurueck, embs_zurueck = geladen
    assert texte_zurueck == texte
    assert np.allclose(embs_zurueck, embs)


def test_load_unbekannter_collection_gibt_none():
    assert store_qdrant.load(FakeClient(), "gibtsnicht") is None


@pytest.mark.skipif(not os.environ.get("QDRANT_URL"),
                    reason="Integration nur mit laufender Instanz (QDRANT_URL setzen)")
def test_echte_instanz_antwortet():
    assert store_qdrant.verbunden(os.environ["QDRANT_URL"]) is True
