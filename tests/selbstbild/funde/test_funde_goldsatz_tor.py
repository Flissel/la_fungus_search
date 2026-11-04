from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from selbstbild.funde.goldsatz_tor import tor_tot, urteile_uebernehmen

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
