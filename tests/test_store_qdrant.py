from __future__ import annotations

import os

import numpy as np
import pytest

from embeddinggemma import store_qdrant


def test_collection_name_ist_stabil():
    a = store_qdrant.collection_for("C:\\Users\\User\\Desktop\\Laura")
    b = store_qdrant.collection_for("c:/Users/User/Desktop/Laura/")
    assert a == b
    assert a.startswith("fungus_laura_")


def test_gleichnamige_ordner_kollidieren_nicht():
    a = store_qdrant.collection_for("C:\\a\\Laura")
    b = store_qdrant.collection_for("C:\\b\\Laura")
    assert a != b


def test_schreibweisen_desselben_ordners_ergeben_einen_namen():
    pfade = [
        "C:\\Users\\User\\Desktop\\Laura",
        "c:/Users/User/Desktop/Laura/",
        "C:\\Users\\User\\Desktop\\laura",
        "C:\\Users\\User\\Desktop\\LAURA",
    ]
    namen = {store_qdrant.collection_for(p) for p in pfade}
    assert len(namen) == 1


class FakeClient:
    """Minimal-Double: haelt Punkte im Speicher, spricht die benutzte Qdrant-API.

    upsert() normalisiert Vektoren auf Einheitslaenge, so wie es eine
    echte COSINE-Collection in Qdrant beim Speichern tut (empirisch gegen
    eine echte Instanz auf Port 6339 geprueft) - Nullvektoren bleiben
    dabei unveraendert Null, da Qdrant sie ebenfalls nicht normiert.
    """

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
        gespeichert = self.collections.setdefault(collection_name, [])
        for p in points:
            vek = np.asarray(p.vector, dtype=np.float64)
            n = float(np.linalg.norm(vek))
            p.vector = (vek / n).tolist() if n > 0 else vek.tolist()
            gespeichert.append(p)

    def scroll(self, collection_name, limit, offset=None, with_vectors=True, with_payload=True):
        punkte = self.collections.get(collection_name, [])
        start = offset or 0
        teil = punkte[start:start + limit]
        naechster = start + limit if start + limit < len(punkte) else None
        return teil, naechster


def test_save_load_roundtrip():
    c = FakeClient()
    texte = ["def a(): pass", "class B: pass", "print('x')"]
    # Normen bewusst ungleich 1 (5.0, 1.0, 2.828) - deckt genau die
    # COSINE-Normalisierung ab, die FakeClient.upsert jetzt nachbildet.
    embs = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 2.0]], dtype=np.float32)
    n = store_qdrant.save(c, "fungus_test_0000", texte, embs)
    assert n == 3
    geladen = store_qdrant.load(c, "fungus_test_0000")
    assert geladen is not None
    texte_zurueck, embs_zurueck = geladen
    assert texte_zurueck == texte
    assert np.allclose(embs_zurueck, embs)


def test_load_unbekannter_collection_gibt_none():
    assert store_qdrant.load(FakeClient(), "gibtsnicht") is None


def test_load_ohne_norm_in_payload_verwendet_vektor_unveraendert():
    """Rueckwaertskompatibel: Punkte aus einer Fassung ohne Norm-Feld
    duerfen load() nicht abstuerzen lassen - der (unkorrigierte) Vektor
    wird dann einfach unveraendert zurueckgegeben."""
    from qdrant_client.http.models import PointStruct
    c = FakeClient()
    c.collections["alt"] = [PointStruct(id=0, vector=[0.6, 0.8, 0.0], payload={"content": "x", "idx": 0})]
    texte, mat = store_qdrant.load(c, "alt")
    assert texte == ["x"]
    assert np.allclose(mat, [[0.6, 0.8, 0.0]])


@pytest.mark.skipif(not os.environ.get("QDRANT_URL"),
                    reason="Integration nur mit laufender Instanz (QDRANT_URL setzen)")
def test_echte_instanz_antwortet():
    assert store_qdrant.verbunden(os.environ["QDRANT_URL"]) is True


@pytest.mark.skipif(not os.environ.get("QDRANT_URL"),
                    reason="Integration nur mit laufender Instanz (QDRANT_URL setzen)")
def test_echte_instanz_roundtrip_mit_normen_ungleich_1():
    """Regressionstest gegen eine echte Instanz: Qdrant normiert COSINE-Vektoren
    beim Speichern auf Einheitslaenge - save()/load() muessen trotzdem exakt
    die eingegebenen (nicht normierten) Embeddings liefern."""
    client = store_qdrant.make_client(os.environ["QDRANT_URL"])
    name = "fungus_test_roundtrip_norm_DELETE_ME"
    try:
        texte = ["def a(): pass", "class B: pass", "print('x')"]
        embs = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 2.0]], dtype=np.float32)
        n = store_qdrant.save(client, name, texte, embs)
        assert n == 3
        geladen = store_qdrant.load(client, name)
        assert geladen is not None
        texte_zurueck, embs_zurueck = geladen
        assert texte_zurueck == texte
        assert np.allclose(embs_zurueck, embs, atol=1e-5)
    finally:
        try:
            client.delete_collection(name)
        except Exception:
            pass
