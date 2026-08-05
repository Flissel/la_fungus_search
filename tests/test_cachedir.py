from __future__ import annotations

import os

from embeddinggemma.cachedir import cache_dir


def test_ohne_collection_bleibt_der_alte_pfad(monkeypatch):
    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    assert cache_dir("/repo") == os.path.join("/repo", ".fungus_cache")


def test_collection_bekommt_ein_unterverzeichnis(monkeypatch):
    monkeypatch.delenv("FUNGUS_CACHE_DIR", raising=False)
    assert cache_dir("/repo", "fungus_Laura_abc") == os.path.join("/repo", ".fungus_cache", "fungus_Laura_abc")


def test_env_gewinnt_ueber_default(monkeypatch):
    monkeypatch.setenv("FUNGUS_CACHE_DIR", os.path.join("/woanders"))
    assert cache_dir("/repo", "c1") == os.path.join("/woanders", "c1")
