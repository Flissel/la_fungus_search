import pytest
import importlib.metadata
from selbstbild import sprachen
from selbstbild.sprachen import sprache_fuer, parser_fuer, GRAMMATIK_VERSION

@pytest.mark.parametrize("pfad,erwartet", [
    ("a/b.py", "python"), ("x.ts", "typescript"), ("x.tsx", "tsx"),
    ("x.js", "javascript"), ("x.mjs", "javascript"), ("x.sql", "sql"),
    ("docker-compose.yml", "yaml"), ("x.yaml", "yaml"), ("x.json", "json"),
    ("x.ps1", "powershell"), ("x.sh", "bash"), ("x.rs", "rust"), ("x.go", "go"),
    (".env.example", "dotenv"), ("README.md", "markdown"), ("bild.png", None),
    (".env.local", "dotenv"), (".envrc", None), (".env", None), (".environment", None),
])
def test_sprache_fuer(pfad, erwartet):
    assert sprache_fuer(pfad) == erwartet

@pytest.mark.parametrize("s", ["python", "typescript", "tsx", "javascript", "rust", "go", "yaml", "json", "bash"])
def test_parser_parst(s):
    baum = parser_fuer(s).parse(b"x")
    assert baum.root_node is not None


def test_grammatik_version_ist_gepinnt():
    assert GRAMMATIK_VERSION == "1.22.1"
    assert importlib.metadata.version("tree-sitter-language-pack") == GRAMMATIK_VERSION


def test_fehlende_grammatik_laedt_nicht_nach(monkeypatch):
    monkeypatch.setattr(sprachen._pack, "downloaded_languages", lambda: [])
    sprachen.parser_fuer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="grammatiken_bereitstellen"):
            parser_fuer("python")
    finally:
        sprachen.parser_fuer.cache_clear()


def test_grammatiken_bereitstellen_ruft_download(monkeypatch):
    gerufen = []
    monkeypatch.setattr(sprachen._pack, "download", lambda n: gerufen.append(list(n)) or len(n))
    sprachen.grammatiken_bereitstellen()
    assert set(gerufen[0]) == set(sprachen.SPRACHEN)
