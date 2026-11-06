from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import yaml

from selbstbild.funde.goldsatz_tor import (
    antworten_uebernehmen,
    goldsatz_bleibt,
    tor_tot,
    unklare_eintraege,
    urteile_uebernehmen,
)

FIXTURE = Path(__file__).parent / "daten" / "goldsatz_fixture.yaml"


def _gold(tmp_path: Path, ja: int, nein: int, offen: int = 0, doppelt_ja: int = 0) -> Path:
    zeilen = ["tot:"]
    for u in ["ja"] * ja + ["nein"] * nein + ["offen"] * offen:
        zeilen += ["- gruende:", "  - alt_90", "  symbol: s", f"  urteil: {u}"]
    zeilen.append("doppelt:")
    for _ in range(doppelt_ja):
        zeilen += ["- art: x", "  urteil: ja"]
    p = tmp_path / "g.yaml"
    p.write_text("\n".join(zeilen) + "\n", encoding="utf-8")
    return p


def test_tor_geschlossen_zu_wenig(tmp_path: Path) -> None:
    t = tor_tot(_gold(tmp_path, 29, 0, 5))
    assert (t.offen, t.beurteilt, t.ja) == (False, 29, 29)
    assert t.grund.startswith("gesperrt")


def test_tor_offen(tmp_path: Path) -> None:
    t = tor_tot(_gold(tmp_path, 27, 3))
    assert t.offen and t.beurteilt == 30 and t.ja == 27
    assert t.grund.startswith("offen")


def test_tor_geschlossen_quote(tmp_path: Path) -> None:
    t = tor_tot(_gold(tmp_path, 26, 4))
    assert not t.offen and "26/30" in t.grund


def test_tor_zaehlt_nur_tot(tmp_path: Path) -> None:
    t = tor_tot(_gold(tmp_path, 10, 0, doppelt_ja=50))
    assert t.beurteilt == 10


def test_tor_echte_datei_gesperrt() -> None:
    assert not tor_tot(FIXTURE).offen


def _kopie(tmp_path: Path) -> Path:
    p = tmp_path / "g.yaml"
    shutil.copyfile(FIXTURE, p)
    return p


def test_urteile_uebernehmen_erhaelt_kommentare(tmp_path: Path) -> None:
    p = _kopie(tmp_path)
    vorher = p.read_bytes()
    diff = urteile_uebernehmen(p, {"tot-00": "ja", "tot-03": "nein", "dop-01": "ja"})
    nachher = p.read_bytes()
    assert "+  urteil: ja" in diff
    a, b = vorher.decode().splitlines(), nachher.decode().splitlines()
    assert len(a) == len(b)
    geaendert = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(geaendert) == 3 and all(x.strip().startswith("urteil:") for x, _ in geaendert)
    assert b"commit: 1e409b4" in nachher and nachher.startswith(vorher[:200])
    t = tor_tot(p)
    assert (t.beurteilt, t.ja) == (2, 1)


def test_urteile_uebernehmen_gleich_schreibt_nichts(tmp_path: Path) -> None:
    p = _kopie(tmp_path)
    vorher = p.read_bytes()
    assert urteile_uebernehmen(p, {"tot-00": "offen"}) == ""
    assert p.read_bytes() == vorher


def test_urteile_uebernehmen_crlf(tmp_path: Path) -> None:
    p = tmp_path / "g.yaml"
    p.write_bytes(FIXTURE.read_bytes().replace(b"\n", b"\r\n"))
    urteile_uebernehmen(p, {"tot-01": "ja"})
    roh = p.read_bytes()
    assert roh.count(b"\n") == roh.count(b"\r\n")
    assert tor_tot(p).ja == 1


@pytest.mark.parametrize("urteile", [{"tot-99": "ja"}, {"dop-15": "ja"}, {"xyz-1": "ja"}, {"tot-00": "vielleicht"}])
def test_urteile_uebernehmen_ungueltig(tmp_path: Path, urteile: dict[str, str]) -> None:
    p = _kopie(tmp_path)
    vorher = p.read_bytes()
    with pytest.raises(ValueError):
        urteile_uebernehmen(p, {"tot-00": "ja", **urteile})
    assert p.read_bytes() == vorher


def _gold_sym(tmp_path: Path, urteile: list[str], extra: dict[int, str] | None = None) -> Path:
    zeilen = ["tot:"]
    for i, u in enumerate(urteile):
        zeilen += ["- gruende:", "  - alt_90", f"  symbol: sym{i}", f"  urteil: {u}"]
        if extra and i in extra:
            zeilen.append(extra[i])
    zeilen += ["doppelt:", "- art: x", "  urteil: ja"]
    p = tmp_path / "gs.yaml"
    p.write_bytes(("\n".join(zeilen) + "\n").encode("utf-8"))
    return p


def test_unklar_zaehlt_nicht_als_beurteilt(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["ja"] * 3 + ["unklar"] * 2)
    t = tor_tot(p)
    assert t.beurteilt == 3 and t.ja == 3


def test_urteile_uebernehmen_erlaubt_unklar(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["offen"])
    assert urteile_uebernehmen(p, {"tot-00": "unklar"}) != ""
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["tot"][0]["urteil"] == "unklar"


def test_unklare_eintraege_und_bleibt(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["ja", "unklar", "offen", "unklar"], {0: "  bleibt: true"})
    liste = unklare_eintraege(p)
    assert [i for i, _ in liste] == ["tot-01", "tot-03"]
    assert liste[0][1]["symbol"] == "sym1"
    assert goldsatz_bleibt(p) == frozenset({"sym0"})


def test_antworten_bleibt_fuegt_zeile_ein(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["unklar", "unklar"])
    vorher = p.read_bytes().decode().splitlines(keepends=True)
    diff = antworten_uebernehmen(p, {"sym1": ("ja", True)})
    assert diff != ""
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert doc["tot"][1]["bleibt"] is True and doc["tot"][1]["urteil"] == "ja"
    assert "bleibt" not in doc["tot"][0] and doc["tot"][0]["urteil"] == "unklar"
    erwartet = list(vorher)
    i = len(erwartet) - 1 - erwartet[::-1].index("  urteil: unklar\n")
    erwartet[i:i + 1] = ["  urteil: ja\n", "  bleibt: true\n"]
    assert p.read_bytes().decode().splitlines(keepends=True) == erwartet


def test_antworten_bleibt_false_entfernt_zeile(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["unklar"], {0: "  bleibt: true"})
    diff = antworten_uebernehmen(p, {"sym0": ("ja", False)})
    assert diff != ""
    e = yaml.safe_load(p.read_text(encoding="utf-8"))["tot"][0]
    assert "bleibt" not in e and e["urteil"] == "ja"


def test_antworten_crlf_bleibt_erhalten(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_bytes(b"tot:\r\n- symbol: a\r\n  urteil: unklar\r\n")
    antworten_uebernehmen(p, {"a": ("ja", True)})
    assert p.read_bytes() == b"tot:\r\n- symbol: a\r\n  urteil: ja\r\n  bleibt: true\r\n"


@pytest.mark.parametrize(("roh", "erwartet"), [
    (b"tot:\n- symbol: a\n  urteil: unklar", b"tot:\n- symbol: a\n  urteil: ja\n  bleibt: true"),
    (b"tot:\r\n- symbol: a\r\n  urteil: unklar", b"tot:\r\n- symbol: a\r\n  urteil: ja\r\n  bleibt: true"),
])
def test_antworten_bleibt_ohne_schluss_umbruch(tmp_path: Path, roh: bytes, erwartet: bytes) -> None:
    """R17-1: steht `urteil:` als letzte Zeile ohne Umbruch, bekommt die eingefuegte `bleibt:`-Zeile eine eigene."""
    p = tmp_path / "ende.yaml"
    p.write_bytes(roh)
    assert antworten_uebernehmen(p, {"a": ("ja", True)}) != ""
    assert p.read_bytes() == erwartet
    e = yaml.safe_load(p.read_text(encoding="utf-8"))["tot"][0]
    assert e["urteil"] == "ja" and e["bleibt"] is True


def test_antworten_uebernehmen_idempotent(tmp_path: Path) -> None:
    import os

    p = _gold_sym(tmp_path, ["unklar", "unklar"])
    antworten_uebernehmen(p, {"sym0": ("ja", True), "sym1": ("nein", False)})
    os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    inhalt = p.read_bytes()
    assert antworten_uebernehmen(p, {"sym0": ("ja", True), "sym1": ("nein", False)}) == ""
    assert p.read_bytes() == inhalt
    assert p.stat().st_mtime_ns == 1_000_000_000


def test_unbekanntes_symbol_valueerror(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["unklar"])
    inhalt = p.read_bytes()
    with pytest.raises(ValueError):
        antworten_uebernehmen(p, {"gibt-es-nicht": ("ja", False)})
    assert p.read_bytes() == inhalt


def test_antworten_ungueltiges_urteil_valueerror(tmp_path: Path) -> None:
    p = _gold_sym(tmp_path, ["unklar"])
    with pytest.raises(ValueError):
        antworten_uebernehmen(p, {"sym0": ("vielleicht", False)})
