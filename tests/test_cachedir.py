from __future__ import annotations

import os

from embeddinggemma.cachedir import cache_dir, resolve_for_codebase


def test_ohne_collection_bleibt_der_alte_pfad(monkeypatch):
    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    assert cache_dir("/repo") == os.path.join("/repo", ".fungus_cache")


def test_collection_bekommt_ein_unterverzeichnis(monkeypatch):
    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    assert cache_dir("/repo", "fungus_Laura_abc") == os.path.join("/repo", ".fungus_cache", "fungus_Laura_abc")


def test_env_gewinnt_ueber_default(monkeypatch):
    monkeypatch.setenv("FUNGUS_CACHE_DIR", os.path.join("/woanders"))
    assert cache_dir("/repo", "c1") == os.path.join("/woanders", "c1")


def test_resolve_ohne_qdrant_liefert_leere_collection_und_alten_pfad(monkeypatch):
    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    monkeypatch.delenv("FUNGUS_STORE", raising=False)
    collection, ziel = resolve_for_codebase("/repo", "/irgendwo/eine_codebase")
    assert collection == ""
    assert ziel == os.path.join("/repo", ".fungus_cache")


def test_resolve_mit_qdrant_liefert_collection_for_und_passendes_unterverzeichnis(monkeypatch):
    from embeddinggemma.store_qdrant import collection_for

    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    monkeypatch.setenv("FUNGUS_STORE", "qdrant")
    erwartet = collection_for("/irgendwo/eine_codebase")
    collection, ziel = resolve_for_codebase("/repo", "/irgendwo/eine_codebase")
    assert collection == erwartet
    assert ziel == os.path.join("/repo", ".fungus_cache", erwartet)
