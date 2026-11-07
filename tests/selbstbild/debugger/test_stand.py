"""Tests fuer `selbstbild.debugger.stand`."""
from pathlib import Path

from selbstbild.debugger.stand import DebuggerStand, FundStand, stand_laden, stand_speichern


def test_stand_rundreise(tmp_path: Path) -> None:
    p = tmp_path / "stand.json"
    s = DebuggerStand(
        start="2026-11-07T12:00:00Z",
        seit="2026-11-07T12:05:00Z",
        verarbeitet=[1, 2, 3],
        funde={"42": FundStand(repo="Flissel/Vibemind_V1", branch="debugger/42-x", sitzung="s1", pr=7,
                               zustand="pr", versuche={"fix": 2})},
        gemeldet=["x"],
    )
    stand_speichern(p, s)
    assert stand_laden(p) == s
    assert not list(tmp_path.glob("*.tmp"))


def test_standardwerte() -> None:
    s = DebuggerStand()
    assert (s.start, s.seit, s.verarbeitet, s.funde, s.gemeldet) == (None, None, [], {}, [])
    f = FundStand(repo="a/b", branch="debugger/1-x")
    assert (f.sitzung, f.pr, f.zustand, f.versuche) == (None, None, "offen", {})


def test_stand_fehlt_ergibt_leer(tmp_path: Path) -> None:
    s = stand_laden(tmp_path / "nix.json")
    assert s.start is None and s.verarbeitet == [] and s.gemeldet == ["stand_fehlt"]


def test_stand_kaputt_ergibt_leer(tmp_path: Path) -> None:
    p = tmp_path / "stand.json"
    p.write_text("{kaputt", encoding="utf-8")
    s = stand_laden(p)
    assert s.start is None and s.funde == {} and s.gemeldet == ["stand_kaputt"]
    assert p.read_text(encoding="utf-8") == "{kaputt"


def test_verarbeitet_kappt_auf_5000(tmp_path: Path) -> None:
    s = DebuggerStand()
    s.verarbeitet_hinzu(range(1, 5006))
    assert len(s.verarbeitet) == 5000 and s.verarbeitet[0] == 6 and s.verarbeitet[-1] == 5005
    s2 = DebuggerStand(verarbeitet=list(range(7000)))
    p = tmp_path / "stand.json"
    stand_speichern(p, s2)
    geladen = stand_laden(p)
    assert len(geladen.verarbeitet) == 5000 and geladen.verarbeitet[-1] == 6999
