"""Skill-Erzeugung "neuer Space" (Stufe 3, T10): Regeln aller Repos, Vorbilder aus dem Stand, Determinismus, CLI."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from selbstbild.__main__ import main
from selbstbild.skill import skill_neuer_space

REGELN_AUSSEN = (
    "name: aussen\n"
    "struktur:\n"
    "  - art: kein_space_im_root\n"
    "    ort: \"./\"\n"
    "    pflicht: []\n"
    "    verboten:\n"
    "      - \"spaces/*/\"\n"
    "      - \"*/capabilities.yaml\"\n"
    "    hinweis: \"Spaces gehoeren nach sub/spaces/<name>/\"\n"
    "  - art: space\n"
    "    ort: \"sub/spaces/*/\"\n"
    "    pflicht: [main.py, tests/]\n"
    "    verboten: []\n"
    "    hinweis: \"Spaces gehoeren nach sub/spaces/<name>/\"\n"
)

# (Pfad im Submodul, Lebendigkeit)
DATEIEN = [
    ("spaces/alpha/main.py", 0.5), ("spaces/alpha/tests/test_a.py", 0.5),
    ("spaces/beta/main.py", 0.9), ("spaces/beta/tests/test_b.py", 0.9), ("spaces/beta/x.py", 0.9),
    ("spaces/gamma/main.py", 0.2), ("spaces/gamma/tests/test_g.py", 0.2),
    ("spaces/delta/main.py", 0.1), ("spaces/delta/tests/test_d.py", 0.1),
    ("spaces/eps/main.py", 1.0),  # ohne tests/: faellt raus
    ("spaces/_contract/main.py", 1.0), ("spaces/_contract/tests/t.py", 1.0),  # Unterstrich: raus
    ("spaces/__pycache__/main.py", 1.0), ("spaces/__pycache__/tests/t.py", 1.0),
    ("spaces/zeta/main.py", 0.5), ("spaces/zeta/tests/test_z.py", 0.5),  # gleiche Summe wie alpha
]


def _stand(ordner: Path) -> Path:
    ordner.mkdir(parents=True)
    with closing(sqlite3.connect(ordner / "suche.sqlite")) as c:
        c.execute("CREATE TABLE eintrag (id TEXT PRIMARY KEY, art TEXT NOT NULL, repo TEXT, pfad TEXT, zeile INTEGER,"
                  " blob TEXT, lebendigkeit REAL, tot TEXT, bereich TEXT, warnung TEXT)")
        c.execute("INSERT INTO eintrag VALUES ('r', 'datei', 'aussen', 'README.md', NULL, NULL, 0.7, NULL, NULL, NULL)")
        for i, (pfad, leb) in enumerate(DATEIEN):
            c.execute("INSERT INTO eintrag VALUES (?, 'datei', 'aussen/sub', ?, NULL, NULL, ?, NULL, NULL, NULL)",
                      (f"d{i}", pfad, leb))
            c.execute("INSERT INTO eintrag VALUES (?, 'symbol', 'aussen/sub', ?, 1, NULL, NULL, 'lebendig', NULL, NULL)",
                      (f"s{i}", pfad))
        c.commit()
    return ordner


@pytest.fixture
def wurzel(tmp_path: Path) -> Path:
    w = tmp_path / "aussen"
    w.mkdir()
    (w / "selbstbild.yaml").write_bytes(REGELN_AUSSEN.encode("utf-8"))
    return w


@pytest.fixture(autouse=True)
def heim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "leer-home"
    monkeypatch.setenv("SELBSTBILD_HOME", str(h))
    return h


def test_skill_enthaelt_ort_und_pflicht(wurzel: Path):
    text = skill_neuer_space(wurzel)
    assert text.startswith("---\nname: neuer-space\ndescription:")
    kopf = text.split("---")[1]
    for stichwort in ("neuen Space anlegen", "Space erstellen", "new space"):
        assert stichwort in kopf
    assert "`sub/spaces/*/`" in text
    assert "Spaces gehoeren nach sub/spaces/<name>/" in text
    assert "`main.py`" in text and "`tests/`" in text
    assert "Nie hier anlegen" in text and "`spaces/*/`" in text and "`*/capabilities.yaml`" in text
    for werkzeug in ("suchen", "nachbarn", "bereich", "regeln", "lebendigkeit", "tot", "doku"):
        assert f"selbstbild_{werkzeug}" in text
    assert text.count("\n1. ") == 1 and "\n4. " in text and "\n5. " not in text


def test_skill_ohne_feste_pflichtteile(tmp_path: Path):
    w = tmp_path / "x"
    w.mkdir()
    (w / "selbstbild.yaml").write_bytes(
        b"struktur:\n  - art: space\n    ort: s/*/\n    pflicht: []\n    verboten: []\n")
    text = skill_neuer_space(w)
    assert "keine festen Pflichtdateien" in text
    assert "## Vorbilder" not in text


def test_skill_vorbilder_aus_stand(wurzel: Path, tmp_path: Path):
    stand = _stand(tmp_path / "stand")
    text = skill_neuer_space(wurzel, stand)
    abschnitt = text.split("## Vorbilder")[1].split("\n## ")[0]
    pfade = [z.split("`")[1] for z in abschnitt.splitlines() if z.startswith("- `")]
    # Summe der Lebendigkeit: beta 2.7, alpha 1.0 = zeta 1.0 (Gleichstand nach Name), gamma 0.4, delta 0.2
    assert pfade == ["sub/spaces/beta/", "sub/spaces/alpha/", "sub/spaces/zeta/"]
    assert "eps" not in abschnitt and "_contract" not in abschnitt and "__pycache__" not in abschnitt


def test_skill_ohne_stand(wurzel: Path, tmp_path: Path):
    text = skill_neuer_space(wurzel)
    assert "## Vorbilder" not in text
    leer = tmp_path / "leer"
    leer.mkdir()
    assert skill_neuer_space(wurzel, leer) == text
    assert "`sub/spaces/*/`" in text


def test_skill_stand_ohne_treffer(wurzel: Path, tmp_path: Path):
    stand = tmp_path / "stand2"
    stand.mkdir()
    with closing(sqlite3.connect(stand / "suche.sqlite")) as c:
        c.execute("CREATE TABLE eintrag (id TEXT PRIMARY KEY, art TEXT NOT NULL, repo TEXT, pfad TEXT, zeile INTEGER,"
                  " blob TEXT, lebendigkeit REAL, tot TEXT, bereich TEXT, warnung TEXT)")
        c.commit()
    assert "## Vorbilder" not in skill_neuer_space(wurzel, stand)


def test_skill_deterministisch(wurzel: Path, tmp_path: Path):
    stand = _stand(tmp_path / "stand")
    a = skill_neuer_space(wurzel, stand)
    b = skill_neuer_space(wurzel, stand)
    assert a == b
    assert "\r" not in a and a.endswith("\n")


def test_skill_regeln_aus_untermodul(tmp_path: Path):
    w = tmp_path / "w"
    (w / "sub").mkdir(parents=True)
    (w / "selbstbild.yaml").write_bytes(b"name: w\n")
    (w / "sub" / "selbstbild.yaml").write_bytes(
        b"struktur:\n  - art: space\n    ort: spaces/*/\n    pflicht: [a.yaml]\n    hinweis: hier\n")
    text = skill_neuer_space(w)
    assert "`sub/spaces/*/`" in text and "`a.yaml`" in text and "hier" in text


def test_cli_skill_schreibt_und_ueberschreibt_nur_mit_flag(wurzel: Path, tmp_path: Path, capsys):
    ziel = tmp_path / "out" / "SKILL.md"
    assert main(["skill", "neuer-space", str(wurzel), str(ziel)]) == 0
    assert ziel.read_bytes() == skill_neuer_space(wurzel).encode("utf-8")
    ziel.write_bytes(b"alt")
    assert main(["skill", "neuer-space", str(wurzel), str(ziel)]) == 2
    assert ziel.read_bytes() == b"alt"
    assert "--ueberschreiben" in capsys.readouterr().err
    assert main(["skill", "neuer-space", str(wurzel), str(ziel), "--ueberschreiben"]) == 0
    assert ziel.read_bytes() != b"alt"
