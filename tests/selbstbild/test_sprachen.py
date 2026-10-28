import pytest
from selbstbild.sprachen import sprache_fuer, parser_fuer

@pytest.mark.parametrize("pfad,erwartet", [
    ("a/b.py", "python"), ("x.ts", "typescript"), ("x.tsx", "tsx"),
    ("x.js", "javascript"), ("x.mjs", "javascript"), ("x.sql", "sql"),
    ("docker-compose.yml", "yaml"), ("x.yaml", "yaml"), ("x.json", "json"),
    ("x.ps1", "powershell"), ("x.sh", "bash"), ("x.rs", "rust"), ("x.go", "go"),
    (".env.example", "dotenv"), ("README.md", "markdown"), ("bild.png", None),
])
def test_sprache_fuer(pfad, erwartet):
    assert sprache_fuer(pfad) == erwartet

@pytest.mark.parametrize("s", ["python", "typescript", "tsx", "javascript", "rust", "go", "yaml", "json", "bash"])
def test_parser_parst(s):
    baum = parser_fuer(s).parse(b"x")
    assert baum.root_node is not None
