"""doku_loop: Rowboat-Notizen, Pruefliste und Urteile-Rueckweg (Task 17).

Notizen gehen in einen tmp-Wissensordner, Urteile laufen gegen das tmp-Git-Fixture aus `test_doku_tick`
(Judge, Budget, Torwaechter und Manifest-Bau sind Fakes). Der echte Wissensordner wird nie angefasst.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import rowboat as rb  # noqa: E402
from selbstbild.doku import stand as st  # noqa: E402
from selbstbild.doku import tick as tk  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402
from selbstbild.doku.graph import Graph, Kante, Knoten, hop_zustand_neu, kante_id  # noqa: E402
from selbstbild.doku.typen import Pruefpunkt  # noqa: E402
from selbstbild.doku.umschreiben import LedgerEintrag  # noqa: E402

from test_doku_tick import (  # noqa: E402,F401
    AUSSAGE, KETTEN, NEUER_TEXT, FakeLaufzeit, _anstoss, _commit, _git, _head, _jetzt, _schreib, _seed, _start,
    _status, _text, _tick, _urteil, fx,
)

LAUF = "2026-10-28T12:00:00#7"
JETZT = datetime(2026, 10, 28, 13, 0, tzinfo=timezone.utc)


# ---------- Graph-Fixture fuer Notizen ----------

def _hop(kette, nr, von, nach, *, belege=(), mech="HTTP POST `/api/x`"):
    zustand = hop_zustand_neu()
    zustand["gemessen"] = {"wert": "laeuft", "zeit": "2026-10-28T00:00:00+00:00"}
    return Kante(kante_id("hop", von, nach, kette, nr), "hop", von, nach, "seed", tuple(belege),
                 {"kette": kette, "nr": str(nr), "mechanismus": mech, "zustand": zustand, "zustand_text": "läuft",
                  "von_nach": "x"})


def _graph(*, mech="HTTP POST `/api/x`", extra_dienst=None):
    g = Graph(commit_aussen="abc123", commit_submodul="def456")
    g.knoten_hinzu(Knoten("space:a", "space", "a", "a", "sp/a"))
    g.knoten_hinzu(Knoten("dienst:b", "dienst", "b", None, None, {"port": "8130"}))
    g.knoten_hinzu(Knoten("dienst:c", "dienst", "c", None, None, {"port": "8140"}))
    g.knoten_hinzu(Knoten("komponente:Client", "komponente", "Client"))
    g.knoten_hinzu(Knoten("kette:K1", "kette", "Fixture", None, None,
                          {"titel_roh": "Fixture-Kette", "zustand_roh": "läuft", "reisst_bei_roh": "—"}))
    g.knoten_hinzu(Knoten("kette:K2", "kette", "Zweite", None, None, {"zustand_roh": "bricht"}))
    beleg = Beleg("symbol", "sp/a/client.py", "client.senden", "d1", "c1", 7)
    g.kante_hinzu(_hop("K1", 1, "space:a", "dienst:b", belege=[beleg], mech=mech))
    g.kante_hinzu(_hop("K1", 2, "dienst:b", "komponente:Client"))
    g.kante_hinzu(_hop("K2", 1, "dienst:c", "dienst:b"))
    g.kante_hinzu(Kante("verbindet:dienst:c->dienst:b", "verbindet", "dienst:c", "dienst:b", "extraktor", (),
                        {"env": ["B_URL"], "dateien": ["infra/swarm/vibemind-stack.yml"]}))
    if extra_dienst is not None:
        g.knoten_hinzu(Knoten(f"dienst:{extra_dienst}", "dienst", extra_dienst))
        g.kante_hinzu(_hop("K2", 2, "dienst:b", f"dienst:{extra_dienst}"))
    return g


def _ledger_eintrag(lauf=LAUF):
    return LedgerEintrag(lauf=lauf, zeit="2026-10-28T12:00:00", aussage_id=f"{KETTEN}#Ketten#abcd1234",
                         datei=KETTEN, alt="Alter Satz.", neu="Neuer Satz.",
                         beleg=Beleg("symbol", "sp/a/client.py", "client.schicken", "d2", "c2", 7),
                         urteil="veraltet", quelle="llm", basis_commit="b" * 40)


def _punkt(**kw):
    d = {"art": "hop_beleg_verschwunden", "text": "K1/1: symbol client.senden in sp/a/client.py ist verschwunden",
         "datei": KETTEN, "zeile": 12, "aussage_id": "K1/1", "lauf": LAUF}
    d.update(kw)
    return d


def _schreiben(wissen, graph=None, liste=None, ledger=None, lauf=LAUF, **kw):
    kw.setdefault("commits", {LAUF: "f" * 40})
    kw.setdefault("spitzen", {KETTEN: "f" * 40})
    kw.setdefault("manifest_pfad", wissen.parent / "notizen-manifest.json")
    return rb.notizen_schreiben(graph or _graph(), liste if liste is not None else [_punkt()],
                                ledger if ledger is not None else [_ledger_eintrag()], wissen, "Doku", lauf, **kw)


def _lesen(p: Path) -> str:
    return p.read_bytes().decode("utf-8")


# ---------- Notizen ----------

def test_notizen_schreiben_struktur_und_wikilinks(tmp_path):
    wissen = tmp_path / "wissen"
    geschrieben = _schreiben(wissen)
    doku = wissen / "Doku"
    erwartet = {doku / "Ketten" / "K1.md", doku / "Ketten" / "K2.md", doku / "Spaces" / "a.md",
                doku / "Dienste" / "b.md", doku / "Dienste" / "c.md", doku / "Prüfliste.md", doku / "Index.md"}
    assert set(geschrieben) == erwartet
    k1 = _lesen(doku / "Ketten" / "K1.md")
    kopf = k1.split("---\n")[1]
    for schluessel in ("erzeugt_von: doku-loop", 'kette: "K1"', 'zustand: "läuft"', 'commit: "abc123"',
                       f'lauf: "{LAUF}"'):
        assert schluessel in kopf, schluessel
    assert "| Nr | Von | Nach |" in k1
    assert "[[Doku/Spaces/a]]" in k1 and "[[Doku/Dienste/b]]" in k1
    assert "Client" in k1 and "[[Doku/Komponente" not in k1  # nur Dienste und Spaces werden verlinkt
    assert "`sp/a/client.py:7`" in k1
    assert "HTTP POST `/api/x`" in k1 and "(gemessen: läuft)" in k1
    b = _lesen(doku / "Dienste" / "b.md")
    assert "[[Doku/Ketten/K1]]" in b and "[[Doku/Ketten/K2]]" in b and "[[Doku/Dienste/c]]" in b and "8130" in b
    assert "[[Doku/Ketten/K1]]" in _lesen(doku / "Spaces" / "a.md")
    index = _lesen(doku / "Index.md")
    for link in ("[[Doku/Prüfliste]]", "[[Doku/Ketten/K1]]", "[[Doku/Ketten/K2]]", "[[Doku/Spaces/a]]",
                 "[[Doku/Dienste/b]]", "[[Doku/Dienste/c]]"):
        assert link in index, link
    pl = _lesen(doku / "Prüfliste.md")
    aid = f"{KETTEN}#Ketten#abcd1234"
    for teil in (LAUF, aid, "Alter Satz.", "Neuer Satz.", "client.schicken", "f" * 10, "veraltet",
                 "hop_beleg_verschwunden", "K1/1", "[[Doku-Urteile/Urteile]]",
                 f"{LAUF} {aid} ok", f"{LAUF} {aid} zurueck", f"{LAUF} {aid} korrektur: "):
        assert teil in pl, teil
    assert f"{LAUF} K1/1 ok" in pl and f"{LAUF} K1/1 korrektur" not in pl  # keine Aussage-ID: nur ok
    assert "## Angewandte Urteile (0)" in pl and "## Struktur-Hinweise (0)" in pl
    # Liste nach einer Leerzeile (M6), sonst rendern manche Leser sie als Fliesstext
    assert "\n\n- `ok`: passt" in pl
    for p in erwartet:
        assert rb.vom_loop(p)


def test_notizen_nur_geaenderte_dateien(tmp_path):
    wissen = tmp_path / "wissen"
    _schreiben(wissen)
    doku = wissen / "Doku"
    alle = sorted(doku.rglob("*.md"))
    zeiten = {p: p.stat().st_mtime_ns for p in alle}
    for p in alle:
        os.utime(p, (1_000_000, 1_000_000))
    # anderer Lauf, gleicher Inhalt: nichts geschrieben (Lauf/Commit im Kopf zaehlen nicht)
    assert _schreiben(wissen, lauf="2026-10-28T12:30:00#8") == []
    assert all(p.stat().st_mtime == 1_000_000 for p in alle) and zeiten
    # ein Hop aendert sich: nur die Kette (und was sie zeigt) wird neu geschrieben
    neu = _schreiben(wissen, graph=_graph(mech="HTTP GET `/api/y`"), lauf="2026-10-28T12:40:00#9")
    assert neu == [doku / "Ketten" / "K1.md"]
    assert 'lauf: "2026-10-28T12:40:00#9"' in _lesen(doku / "Ketten" / "K1.md")
    assert f'lauf: "{LAUF}"' in _lesen(doku / "Ketten" / "K2.md")


def test_notizen_loeschen_nur_mit_marker_und_nur_in_doku(tmp_path):
    wissen = tmp_path / "wissen"
    _schreiben(wissen, graph=_graph(extra_dienst="weg"))
    doku = wissen / "Doku"
    assert (doku / "Dienste" / "weg.md").is_file()
    hand = doku / "Dienste" / "hand.md"
    hand.write_bytes(b"# von Hand\n")
    fremd_mit_marker = wissen / "Notes" / "x.md"
    fremd_mit_marker.parent.mkdir(parents=True)
    fremd_mit_marker.write_bytes(b"---\nerzeugt_von: doku-loop\n---\n# x\n")
    # eine von Hand angelegte Datei an der Stelle einer Notiz wird nie ueberschrieben
    c = doku / "Dienste" / "c.md"
    c.write_bytes(b"# c von Hand\n")
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, graph=_graph(), pruefpunkte=pp)
    assert [(p.art, p.datei) for p in pp] == [("notiz_von_hand", "Doku/Dienste/c.md")]
    assert not (doku / "Dienste" / "weg.md").exists()
    assert hand.read_bytes() == b"# von Hand\n"
    assert c.read_bytes() == b"# c von Hand\n"
    assert fremd_mit_marker.is_file()
    assert {p.parent.name for p in wissen.rglob("*.md") if rb.vom_loop(p)} <= {"Doku", "Ketten", "Spaces", "Dienste",
                                                                               "Notes"}
    assert not any(p.name.endswith(".tmp") for p in wissen.rglob("*"))


def test_notizen_dateinamen_windows_sicher(tmp_path):
    wissen = tmp_path / "wissen"
    g = _graph()
    for name in ("Supabase marketing.*", 'a<b>:c"d/e\\f|g?h', "punkt. ", "CON", "B"):
        g.knoten_hinzu(Knoten(f"dienst:{name}", "dienst", name))
        g.kante_hinzu(_hop("K2", f"x{len(g.kanten)}", "dienst:c", f"dienst:{name}"))
    _schreiben(wissen, graph=g)
    dienste = wissen / "Doku" / "Dienste"
    namen = sorted(p.name for p in dienste.iterdir())
    for n in namen:
        stamm = n[:-3]
        assert not set('<>:"/\\|?*') & set(stamm), n
        assert not stamm.endswith((".", " ")), n
        assert stamm.upper() != "CON"
    assert all(p.parent == dienste for p in dienste.rglob("*.md"))
    assert len(namen) == 7  # b, c, 5 neue; "B" kollidiert unter Windows mit "b" und bekommt einen Zusatz
    k2 = _lesen(wissen / "Doku" / "Ketten" / "K2.md")
    for n in namen:
        assert f"[[Doku/Dienste/{n[:-3]}]]" in k2, n


# ---------- Urteile lesen ----------

def test_urteile_lesen_formen_und_unlesbar(tmp_path):
    aid = f"{KETTEN}#Ketten > K1 · Fixture#1234abcd"
    pfad = tmp_path / "Urteile.md"
    pfad.write_bytes("\n".join([
        "---",
        "titel: Urteile",
        "---",
        "# Urteile",
        "",
        "> Form: <lauf> <aussage_id> ok|zurueck|korrektur: <text>",
        "<!-- Kommentar -->",
        f"{LAUF} {aid} ok",
        f"- `{LAUF} K1/1 zurueck`",
        f"{LAUF} {aid} korrektur: Der Client ruft `x` auf. Mit korrektur: im Text.",
        f"{LAUF} {aid} OK",
        "Quatsch ohne Form",
        "Quatsch ohne Form",
        f"{LAUF} {aid} korrektur:   ",
        f"{LAUF} nur-zwei",
        "",
    ]).encode("utf-8"))
    unlesbar: list[Pruefpunkt] = []
    urteile = rb.urteile_lesen(pfad, unlesbar=unlesbar)
    assert urteile == [
        rb.Urteil(LAUF, aid, "ok", None),
        rb.Urteil(LAUF, "K1/1", "zurueck", None),
        rb.Urteil(LAUF, aid, "korrektur", "Der Client ruft `x` auf. Mit korrektur: im Text."),
        rb.Urteil(LAUF, aid, "ok", None),
    ]
    assert [p.art for p in unlesbar] == ["urteil_unlesbar"] * 3  # die doppelte Zeile nur einmal
    assert all(p.zeile is not None for p in unlesbar)
    assert rb.urteile_lesen(tmp_path / "fehlt.md") == []


# ---------- Urteile anwenden (tmp-Git) ----------

def _anwenden(fx, urteile, *, drift=None, lauf="2026-10-28T13:00:00#20", offene=(), ledger=None):
    return rb.urteile_anwenden(urteile, fx.repo, ledger or fx.repo / "docs/graph/ledger.jsonl",
                               rb.erledigt_pfad(fx.konfig), lauf=lauf, jetzt=JETZT,
                               drift=drift or (lambda repo: set()), doku=[KETTEN], offene_punkte=list(offene))


def _geschrieben_vom_loop(fx):
    _start(fx)
    erg = _tick(fx, laufzeit=FakeLaufzeit(_urteil(), _urteil()))
    assert erg.commit and erg.umgeschrieben == 1
    e = json.loads((fx.repo / "docs/graph/ledger.jsonl").read_text("utf-8").splitlines()[-1])
    return erg, e


def test_urteile_anwenden_zurueck_revertiert_nur_an_der_spitze(fx):
    erg, e = _geschrieben_vom_loop(fx)
    vorher = _head(fx.repo)
    pp = _anwenden(fx, [rb.Urteil(e["lauf"], e["aussage_id"], "zurueck", None)])
    assert pp == []
    assert _head(fx.repo) != vorher
    assert AUSSAGE in _text(fx) and NEUER_TEXT not in _text(fx)
    assert set(_git(fx.repo, "show", "--name-only", "--format=", "HEAD").split()) == {KETTEN}
    assert _status(fx.repo) == []


def test_urteile_anwenden_zurueck_nicht_an_der_spitze_kommt_in_die_pruefliste(fx):
    erg, e = _geschrieben_vom_loop(fx)
    _schreib(fx.repo, KETTEN, _text(fx) + "\nNachtrag von Hand.\n")
    _commit(fx.repo, "hand", KETTEN)
    vorher, text = _head(fx.repo), _text(fx)
    pp = _anwenden(fx, [rb.Urteil(e["lauf"], e["aussage_id"], "zurueck", None)])
    assert [p.art for p in pp] == ["zurueck_nicht_moeglich"]
    assert pp[0].aussage_id == e["aussage_id"] and erg.commit[:7] in pp[0].text
    assert _head(fx.repo) == vorher and _text(fx) == text
    # Ruling 54: ohne Ledger-Eintrag fuer den Lauf kein Bezug
    pp2 = _anwenden(fx, [rb.Urteil("gibt-es-nicht#1", e["aussage_id"], "zurueck", None)])
    assert [p.art for p in pp2] == ["urteil_ohne_bezug"] and _head(fx.repo) == vorher


def _aid_aussage(fx):
    return next(a.id for a in rb.aussagen_ohne_belege(KETTEN, _text(fx)) if a.text == AUSSAGE)


def test_urteile_anwenden_korrektur_schreibt_und_committet(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    vorher = _head(fx.repo)
    korrigiert = "Der Client ruft `senden` in `sp/a/client.py:7` auf; Dienst b nimmt den Auftrag an."
    pp = _anwenden(fx, [rb.Urteil(LAUF, aid, "korrektur", korrigiert)], offene=[_punkt(aussage_id=aid)])
    assert pp == []
    assert korrigiert in _text(fx) and AUSSAGE not in _text(fx)
    betreff, *rest = _git(fx.repo, "log", "-1", "--format=%B").strip().split("\n")
    assert betreff == f"docs(loop): Korrektur aus Review {aid}"
    lauf_zeile = next(z for z in rest if z.startswith("Lauf: "))
    assert _git(fx.repo, "rev-parse", "HEAD~1").strip() == vorher
    assert set(_git(fx.repo, "show", "--name-only", "--format=", "HEAD").split()) == {KETTEN,
                                                                                     "docs/graph/ledger.jsonl"}
    e = json.loads((fx.repo / "docs/graph/ledger.jsonl").read_text("utf-8").splitlines()[-1])
    assert e["quelle"] == "review" and e["urteil"] == "korrektur" and e["neu"] == korrigiert and e["alt"] == AUSSAGE
    assert e["lauf"] == lauf_zeile[len("Lauf: "):] and e["basis_commit"] == vorher
    assert tk.ledger_commit_von(fx.repo, e["lauf"]) == _head(fx.repo)
    assert _status(fx.repo) == []


def test_korrektur_vom_strukturwaechter_abgelehnt(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    vorher, text = _head(fx.repo), _text(fx)
    pp = _anwenden(fx, [rb.Urteil(LAUF, aid, "korrektur", "# Neue Ueberschrift statt Aussage")],
                   offene=[_punkt(aussage_id=aid)])
    assert [p.art for p in pp] == ["korrektur_abgelehnt"] and pp[0].aussage_id == aid
    assert _head(fx.repo) == vorher and _text(fx) == text and _status(fx.repo) == []
    # unbekannte Aussage (offener Punkt, aber nicht mehr im Text)
    weg = f"{KETTEN}#Gibt es nicht#00000000"
    pp2 = _anwenden(fx, [rb.Urteil(LAUF, weg, "korrektur", "Ein Satz.")], offene=[_punkt(aussage_id=weg)])
    assert [p.art for p in pp2] == ["korrektur_abgelehnt"] and _head(fx.repo) == vorher


def test_korrektur_torwaechter_blockiert_setzt_zurueck(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    vorher, text = _head(fx.repo), _text(fx)
    aufrufe = []

    def drift(repo):
        aufrufe.append(1)
        return set() if len(aufrufe) == 1 else {("KETTEN", "neu kaputt")}

    pp = _anwenden(fx, [rb.Urteil(LAUF, aid, "korrektur", "Ein ganz neuer Satz.")], drift=drift,
                   offene=[_punkt(aussage_id=aid)])
    assert [p.art for p in pp] == ["torwaechter_blockiert"]
    assert _head(fx.repo) == vorher and _text(fx) == text and _status(fx.repo) == []


def test_urteil_wird_nicht_zweimal_angewandt(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    urteil = rb.Urteil(LAUF, aid, "korrektur", "Ein ganz neuer Satz über den Client.")
    offen = [_punkt(aussage_id=aid)]
    assert _anwenden(fx, [urteil, urteil], offene=offen) == []
    nach_erstem = _head(fx.repo)
    assert _git(fx.repo, "rev-list", "--count", "HEAD").strip() == "3"  # start, seed, eine Korrektur
    assert _anwenden(fx, [urteil], offene=offen) == []
    assert _head(fx.repo) == nach_erstem
    erledigt = rb.erledigt_laden(rb.erledigt_pfad(fx.konfig))
    assert len(erledigt) == 1
    # der Vermerk liegt lokal, nie im Repo
    assert not str(rb.erledigt_pfad(fx.konfig)).startswith(str(fx.repo))


def test_urteil_ok_blendet_den_punkt_in_der_pruefliste_aus(fx, tmp_path):
    _seed(fx)
    assert _anwenden(fx, [rb.Urteil(LAUF, "K1/1", "ok", None)], offene=[_punkt()]) == []
    vermerke = rb.vermerke_laden(rb.erledigt_pfad(fx.konfig))
    wissen = tmp_path / "w"
    _schreiben(wissen, liste=[_punkt(), _punkt(aussage_id="K1/2", text="anderer")], ledger=[], vermerke=vermerke)
    pl = _lesen(wissen / "Doku" / "Prüfliste.md")
    assert "K1/2" in pl and f"{LAUF} K1/1 ok" not in pl
    assert "## Angewandte Urteile (1)" in pl and "· ok · Lauf" in pl


# ---------- Tick-Verdrahtung ----------

def _urteile_datei(fx, *zeilen):
    pfad = fx.konfig.rowboat_wissen / fx.konfig.rowboat_urteile
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_bytes(("\n".join(["# Urteile", *zeilen]) + "\n").encode("utf-8"))
    return pfad


def test_tick_wendet_urteile_an_und_schreibt_notizen(fx):
    _erg, e = _geschrieben_vom_loop(fx)
    # Korrektur einer Umschreibung des Loops: die Aussage hat inzwischen die id des neuen Texts
    _urteile_datei(fx, f"{e['lauf']} {e['aussage_id']} korrektur: Ein ganz neuer Satz über den Client.")
    _anstoss(fx)
    erg = _tick(fx)
    assert NEUER_TEXT not in _text(fx)
    assert "Ein ganz neuer Satz über den Client." in _text(fx)
    assert any(z.startswith("docs(loop): Korrektur aus Review")
               for z in _git(fx.repo, "log", "--format=%s").splitlines())
    doku = fx.konfig.rowboat_wissen / "Doku"
    assert (doku / "Index.md").is_file() and (doku / "Ketten" / "K1.md").is_file()
    assert (doku / "Prüfliste.md").is_file()
    assert erg.uebersprungen is None or erg.uebersprungen == "keine_aenderung"
    assert _status(fx.repo) == []
    # die Urteile-Datei liest der Loop nur
    assert (fx.konfig.rowboat_wissen / fx.konfig.rowboat_urteile).read_bytes().startswith(b"# Urteile\n")
    # zweiter Tick: nichts erneut angewandt
    kopf = _head(fx.repo)
    _anstoss(fx)
    _tick(fx)
    assert not any(z.startswith("docs(loop): Korrektur") for z in
                   _git(fx.repo, "log", f"{kopf}..HEAD", "--format=%s").splitlines())


def test_tick_urteile_warten_bei_fremden_aenderungen(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    _urteile_datei(fx, f"{LAUF} {aid} korrektur: Ein ganz neuer Satz über den Client.")
    _schreib(fx.repo, KETTEN, _text(fx) + "\nfremd\n")
    vorher = _head(fx.repo)
    _anstoss(fx)
    erg = _tick(fx, erzwingen=True)
    assert erg.uebersprungen.startswith("fremde_aenderungen")
    assert _head(fx.repo) == vorher and "Ein ganz neuer Satz" not in _text(fx)
    assert rb.erledigt_laden(rb.erledigt_pfad(fx.konfig)) == {}
    assert not (fx.konfig.rowboat_wissen / "Doku").exists()


def test_tick_keine_aenderung_schreibt_notizen_nur_bei_neuer_pruefliste(fx):
    _seed(fx)
    _anstoss(fx)
    _tick(fx)
    doku = fx.konfig.rowboat_wissen / "Doku"
    assert (doku / "Index.md").is_file()
    pl = doku / "Prüfliste.md"
    os.utime(pl, (1_000_000, 1_000_000))
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen == "keine_aenderung" and pl.stat().st_mtime == 1_000_000
    _urteile_datei(fx, "das ist kein Urteil")
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen == "keine_aenderung"
    assert "urteil_unlesbar" in _lesen(pl) and "das ist kein Urteil" in _lesen(pl)


def test_tick_sperre_schreibt_keine_notizen(fx):
    _seed(fx)
    _anstoss(fx)
    with st.sperre(st.lokaler_ordner(fx.konfig)) as frei:
        assert frei
        erg = _tick(fx)
    assert erg.uebersprungen == "laeuft_schon"
    assert not (fx.konfig.rowboat_wissen / "Doku").exists()


def test_tick_zurueck_ueber_urteile_datei_und_render_hashes(fx):
    erg, e = _geschrieben_vom_loop(fx)
    _urteile_datei(fx, f"{e['lauf']} {e['aussage_id']} zurueck")
    _anstoss(fx)
    erg2 = _tick(fx)
    assert AUSSAGE in _text(fx)
    assert any(z.startswith("revert(loop):") for z in _git(fx.repo, "log", "--format=%s").splitlines())
    # der Revert ist kein Hand-Edit
    assert not any(p["art"] == "hand_edit" for p in tk.pruefliste_laden(fx.konfig)["offen"])
    # M2: die vom Revert geaenderten Bloecke werden im selben Lauf neu gerendert (kein haengender Hand-Edit)
    assert erg2.uebersprungen is None
    stand = st.stand_laden(fx.konfig)
    from selbstbild.doku import render_ketten
    bloecke = render_ketten.bloecke_finden(_text(fx))
    assert all(render_ketten.block_hash(bloecke[b].inhalt) == h for b, h in stand.render_hashes.items() if b in bloecke)
    assert _status(fx.repo) == []
    # ein voller Lauf danach rendert neu, ohne den Revert als Hand-Edit zu melden
    _schreib(fx.repo, "sp/a/client.py", (fx.repo / "sp/a/client.py").read_text("utf-8") + "\n# danach\n")
    _commit(fx.repo, "danach", "sp/a/client.py")
    _anstoss(fx)
    _tick(fx, laufzeit=FakeLaufzeit())
    assert not any(p["art"] == "hand_edit" for p in tk.pruefliste_laden(fx.konfig)["offen"])
    assert AUSSAGE in _text(fx) and _status(fx.repo) == []



def test_ende_zu_ende_umbenennung_bis_notiz(fx):
    """Abnahme C.9.4 + C.9.7 (Ruling 56): Umbenennung -> Commit -> ein Tick -> Aussage umgeschrieben mit neuem
    Anker, Commit, Notiz -> `zurueck` -> Revert-Commit -> naechster Tick bleibt ruhig und konsistent."""
    _start(fx)  # geseedet, `senden` -> `schicken` committet, Marker gesetzt
    vorher = _head(fx.repo)
    lauf = FakeLaufzeit(_urteil(), _urteil())  # zweimal `veraltet` (Zwei-Befragungen-Regel)
    erg = _tick(fx, laufzeit=lauf)
    # ein Tick: Signal -> Judge (2 Aufrufe) -> Umschreiben -> Commit
    assert erg.signale >= 1 and erg.llm_aufrufe == 2 and len(lauf.aufrufe) == 2 and erg.umgeschrieben == 1
    assert erg.commit == _head(fx.repo) != vorher
    assert NEUER_TEXT in _text(fx) and AUSSAGE not in _text(fx)
    e = json.loads((fx.repo / "docs/graph/ledger.jsonl").read_text("utf-8").splitlines()[-1])
    assert e["beleg"]["anker"] == "client.schicken" and e["lauf"] == erg.lauf
    graph = json.loads((fx.repo / "docs/graph/graph.json").read_text("utf-8"))
    anker = {b["anker"] for k in graph["kanten"] if k["art"] == "belegt_durch" for b in k["belege"]}
    assert "client.schicken" in anker and "client.senden" not in anker  # Beleg zeigt auf den neuen Anker
    # Notiz in Rowboat im selben Tick
    pl = fx.konfig.rowboat_wissen / "Doku" / "Prüfliste.md"
    notiz = _lesen(pl)
    assert "## Umschreibungen des Loops (1)" in notiz and NEUER_TEXT in notiz and e["lauf"] in notiz
    assert f"{e['lauf']} {e['aussage_id']} zurueck" in notiz  # an der Spitze: `zurueck` wird angeboten
    # Rueckweg: `zurueck` aus Urteile.md -> Revert-Commit
    _urteile_datei(fx, f"{e['lauf']} {e['aussage_id']} zurueck")
    _anstoss(fx)
    erg2 = _tick(fx, laufzeit=FakeLaufzeit())
    assert erg2.llm_aufrufe == 0
    nach = _git(fx.repo, "log", "--reverse", "--format=%H %s", f"{erg.commit}..HEAD").splitlines()
    assert nach[0].split(" ", 1)[1].startswith("revert(loop):")  # Revert direkt auf den Loop-Commit ...
    assert _git(fx.repo, "rev-parse", f"{nach[0].split()[0]}~1").strip() == erg.commit
    # ... und im selben Tick nachgefuehrt: Bloecke neu gerendert, Aussage-Belege im Graph angepasst
    assert erg2.commit == _head(fx.repo) and len(nach) == 2 and nach[1].split(" ", 1)[1].startswith("docs(loop):")
    assert AUSSAGE in _text(fx) and NEUER_TEXT not in _text(fx)
    assert _status(fx.repo) == []
    assert "## Angewandte Urteile (1)" in _lesen(pl)
    # naechster Tick ohne Code-Aenderung: kein Judge, kein Commit, Urteil nicht erneut, kein Hand-Edit
    kopf = _head(fx.repo)
    _anstoss(fx)
    lauf3 = FakeLaufzeit(_urteil(), _urteil())
    erg3 = _tick(fx, laufzeit=lauf3)
    assert lauf3.aufrufe == [] and erg3.llm_aufrufe == 0 and erg3.commit is None
    assert _head(fx.repo) == kopf and _status(fx.repo) == [] and AUSSAGE in _text(fx)
    assert not any(p["art"] == "hand_edit" for p in tk.pruefliste_laden(fx.konfig)["offen"])


def test_urteile_pfad_unter_kanal_wird_nie_gelesen(tmp_path):
    pfad = tmp_path / "Kanal" / "Urteile.md"
    pfad.parent.mkdir()
    pfad.write_bytes(f"{LAUF} K1/1 ok\n".encode("utf-8"))
    with pytest.raises(ValueError):
        rb.urteile_pfad(tmp_path, "Kanal/Urteile.md")
    assert rb.urteile_pfad(tmp_path, "Doku-Urteile/Urteile.md") == tmp_path / "Doku-Urteile" / "Urteile.md"


# ---------- Fix-Runde 1 ----------

def test_urteil_ohne_bezug_wird_abgelehnt(fx):
    """Ruling 54: ok nur mit Ledger-Eintrag oder offenem Punkt; zurueck nur mit Ledger-Eintrag; korrektur nur mit
    Ledger-Eintrag oder offenem Punkt mit Aussage-ID."""
    _seed(fx)
    aid = _aid_aussage(fx)
    vorher = _head(fx.repo)
    pp = _anwenden(fx, [rb.Urteil(LAUF, "K9/9", "ok", None),
                        rb.Urteil(LAUF, "K1/1", "zurueck", None),
                        rb.Urteil(LAUF, aid, "korrektur", "Ein Satz."),
                        rb.Urteil(LAUF, "K1/1", "korrektur", "Ein Satz.")],
                   offene=[_punkt()])
    assert [p.art for p in pp] == ["urteil_ohne_bezug"] * 4
    assert _head(fx.repo) == vorher and AUSSAGE in _text(fx)
    assert rb.erledigt_laden(rb.erledigt_pfad(fx.konfig)) == {}


def test_zurueck_findet_den_commit_nur_mit_genau_diesem_lauf(fx, tmp_path):
    erg, e = _geschrieben_vom_loop(fx)
    kurz = e["lauf"][:-1]  # Praefix des echten Laufs: `--grep` haette den Loop-Commit gefunden
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes((json.dumps({**e, "lauf": kurz}, ensure_ascii=False) + "\n").encode("utf-8"))
    vorher = _head(fx.repo)
    pp = _anwenden(fx, [rb.Urteil(kurz, e["aussage_id"], "zurueck", None)], ledger=ledger)
    assert [p.art for p in pp] == ["zurueck_nicht_moeglich"] and "genau" in pp[0].text
    assert _head(fx.repo) == vorher and NEUER_TEXT in _text(fx)
    assert rb.lauf_commits(fx.repo)[e["lauf"]] == erg.commit and kurz not in rb.lauf_commits(fx.repo)


def test_zurueck_mit_neuem_drift_ist_nicht_blockierend(fx):
    _erg, e = _geschrieben_vom_loop(fx)
    aufrufe = []

    def drift(repo):
        aufrufe.append(1)
        return set() if len(aufrufe) == 1 else {("KETTEN", "Revert kaputt")}

    pp = _anwenden(fx, [rb.Urteil(e["lauf"], e["aussage_id"], "zurueck", None)], drift=drift)
    assert [p.art for p in pp] == ["zurueck_drift"] and "Revert kaputt" in pp[0].text
    assert AUSSAGE in _text(fx) and _status(fx.repo) == []  # der Revert bleibt
    v = rb.vermerke_laden(rb.erledigt_pfad(fx.konfig))[-1]
    assert v["ergebnis"] == "angewandt" and v["commit"] == _head(fx.repo) and v["datei"] == KETTEN


def test_pruefliste_zurueck_nur_an_der_spitze_und_mit_zurueckgenommene(tmp_path):
    wissen = tmp_path / "wissen"
    e1 = _ledger_eintrag()
    e2 = LedgerEintrag(**{**e1.__dict__, "aussage_id": f"{KETTEN}#Ketten#ffff0000", "alt": "A2.", "neu": "N2."})
    # M3: der Loop-Commit ist nicht mehr der letzte von KETTEN.md -> kein `zurueck`
    _schreiben(wissen, ledger=[e1, e2], spitzen={KETTEN: "a" * 40})
    pl = _lesen(wissen / "Doku" / "Prüfliste.md")
    assert f"{LAUF} {e1.aussage_id} ok" in pl and f"{LAUF} {e1.aussage_id} zurueck" not in pl
    assert "nicht mehr der letzte Commit" in pl
    # M1 + Item 3: ein `zurueck` auf e1 nimmt e2 mit; beide verschwinden, die Anzeige nennt e2
    vermerke = [{"lauf": LAUF, "aussage_id": e1.aussage_id, "art": "zurueck", "text": None, "ergebnis": "angewandt",
                 "zeit": "2026-10-28T13:00:00", "commit": "c" * 40, "datei": KETTEN},
                {"lauf": "x", "aussage_id": "y", "art": "korrektur", "text": "Neuer Text.", "ergebnis": "abgelehnt",
                 "zeit": "2026-10-28T13:01:00", "commit": None, "datei": KETTEN}]
    struktur = [{"art": "anker_unsicher", "text": "zu grob", "datei": KETTEN, "zeile": 3, "aussage_id": None}]
    _schreiben(wissen, ledger=[e1, e2], vermerke=vermerke, struktur=struktur)
    pl = _lesen(wissen / "Doku" / "Prüfliste.md")
    assert "## Umschreibungen des Loops (0)" in pl
    assert "## Angewandte Urteile (1)" in pl and "· zurueck · Lauf" in pl and "c" * 10 in pl
    assert f"mit zurückgenommen: `{e1.aussage_id}`, `{e2.aussage_id}`" in pl
    assert "## Struktur-Hinweise (1)" in pl and "- anker_unsicher: zu grob (`docs/spaces/KETTEN.md:3`)" in pl


def test_notiz_kopie_und_bearbeitung_bleiben_stehen(tmp_path):
    wissen = tmp_path / "wissen"
    _schreiben(wissen, graph=_graph(extra_dienst="weg"))
    doku = wissen / "Doku"
    # Kopie einer Notiz (mit Marker) an einer Stelle, die der Loop nie geschrieben hat
    kopie = doku / "Dienste" / "kopie.md"
    kopie.write_bytes((doku / "Dienste" / "b.md").read_bytes())
    # bearbeitete Notiz, deren Thema verschwindet
    weg = doku / "Dienste" / "weg.md"
    weg.write_bytes(weg.read_bytes() + "Notiz von Hand.\n".encode("utf-8"))
    # bearbeitete Notiz, deren Thema bleibt und sich aendert
    k1 = doku / "Ketten" / "K1.md"
    k1.write_bytes(k1.read_bytes() + "Von Hand.\n".encode("utf-8"))
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, graph=_graph(mech="HTTP GET `/api/neu`"), pruefpunkte=pp)
    assert kopie.is_file() and weg.is_file() and k1.read_bytes().endswith("Von Hand.\n".encode("utf-8"))
    assert "/api/neu" not in _lesen(k1)
    assert sorted(p.datei for p in pp) == ["Doku/Dienste/weg.md", "Doku/Ketten/K1.md"]
    assert all(p.art == "notiz_von_hand" for p in pp)
    assert "Von Hand angelegte oder geänderte Notizen" in _lesen(doku / "Index.md").replace("\n", " ")


def test_ohne_manifest_uebernimmt_nur_soll_mit_bekanntem_lauf_und_loescht_nichts(tmp_path):
    wissen = tmp_path / "wissen"
    _schreiben(wissen, graph=_graph(extra_dienst="weg"), manifest_pfad=None)  # Notizen ohne Manifest (wie die 124)
    m = tmp_path / "neu-manifest.json"
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, graph=_graph(mech="HTTP GET `/api/neu`"), manifest_pfad=m, pruefpunkte=pp,
               bekannte_laeufe={LAUF})
    k1 = wissen / "Doku" / "Ketten" / "K1.md"
    assert "/api/neu" in _lesen(k1)  # Marker + bekannter Lauf -> uebernommen und aktualisiert
    weg = wissen / "Doku" / "Dienste" / "weg.md"
    assert weg.is_file()  # die Uebernahme loescht nie ...
    assert [(p.art, p.datei) for p in pp] == [("notiz_von_hand", "Doku/Dienste/weg.md")]  # ... sondern meldet
    dateien = json.loads(m.read_text("utf-8"))["dateien"]
    assert "Ketten/K1.md" in dateien and "Dienste/weg.md" not in dateien
    # danach gilt das Manifest: weg.md bleibt (nie vom Manifest erfasst), nichts Neues zu melden ausser weg.md nicht
    pp2: list[Pruefpunkt] = []
    _schreiben(wissen, graph=_graph(), manifest_pfad=m, pruefpunkte=pp2)
    assert weg.is_file() and pp2 == []


def test_uebernahme_kopie_bleibt_und_handaenderung_wird_gemeldet(tmp_path):
    """Probe des Reviews: Kopie mit Marker und bearbeitete Notiz bei fehlendem Manifest."""
    wissen = tmp_path / "wissen"
    _schreiben(wissen, graph=_graph(), manifest_pfad=None)
    doku = wissen / "Doku"
    kopie = doku / "Dienste" / "b Kopie.md"
    kopie.write_bytes((doku / "Dienste" / "b.md").read_bytes() + "meine Notiz\n".encode("utf-8"))
    k1 = doku / "Ketten" / "K1.md"
    k1.write_bytes(k1.read_bytes() + "Von Hand.\n".encode("utf-8"))
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, graph=_graph(mech="HTTP GET `/api/neu`"), manifest_pfad=tmp_path / "m.json", pruefpunkte=pp)
    assert kopie.is_file() and "Von Hand." in _lesen(k1) and "/api/neu" not in _lesen(k1)
    assert sorted(p.datei for p in pp) == ["Doku/Dienste/b Kopie.md", "Doku/Ketten/K1.md"]
    assert {p.art for p in pp} == {"notiz_von_hand"}


def test_kaputtes_manifest_schreibt_nichts(tmp_path):
    wissen, m = tmp_path / "wissen", tmp_path / "m.json"
    _schreiben(wissen, graph=_graph(), manifest_pfad=m)
    doku = wissen / "Doku"
    kopie = doku / "Dienste" / "b Kopie.md"
    kopie.write_bytes((doku / "Dienste" / "b.md").read_bytes())
    m.write_text("{kaputt", "utf-8")
    vorher = {p: p.read_bytes() for p in doku.rglob("*.md")}
    pp: list[Pruefpunkt] = []
    assert _schreiben(wissen, graph=_graph(mech="HTTP GET `/api/neu`"), manifest_pfad=m, pruefpunkte=pp) == []
    assert {p: p.read_bytes() for p in doku.rglob("*.md")} == vorher and kopie.is_file()
    assert [p.art for p in pp] == ["notizen_manifest_kaputt"]
    assert m.read_text("utf-8") == "{kaputt"  # nicht stillschweigend ersetzt


def test_handgeaenderte_pruefliste_bekommt_eine_loop_fassung_daneben(tmp_path):
    wissen, m = tmp_path / "wissen", tmp_path / "m.json"
    _schreiben(wissen, manifest_pfad=m)
    doku = wissen / "Doku"
    pl = doku / "Prüfliste.md"
    pl.write_bytes(pl.read_bytes() + "Meine Notizen.\n".encode("utf-8"))
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, manifest_pfad=m, pruefpunkte=pp, liste=[_punkt(), _punkt(aussage_id="K2/1", text="neu")])
    assert "Meine Notizen." in _lesen(pl) and "K2/1" not in _lesen(pl)
    loop = doku / "Prüfliste (Loop).md"
    assert "K2/1" in _lesen(loop) and rb.vom_loop(loop)
    index = _lesen(doku / "Index.md")
    assert "[[Doku/Prüfliste (Loop)]]" in index and "`Prüfliste.md` ist von Hand geändert" in index
    assert [(p.art, p.datei) for p in pp] == [("notiz_von_hand", "Doku/Prüfliste.md")]
    # Hand-Datei weg -> wieder die normale Pruefliste, die Loop-Fassung verschwindet
    pl.unlink()
    _schreiben(wissen, manifest_pfad=m)
    assert pl.is_file() and not loop.exists() and "[[Doku/Prüfliste]]" in _lesen(doku / "Index.md")
    # Index von Hand geaendert -> Index (Loop).md, die Pruefliste nennt ihn
    ix = doku / "Index.md"
    ix.write_bytes(ix.read_bytes() + "Hand.\n".encode("utf-8"))
    _schreiben(wissen, manifest_pfad=m)
    assert (doku / "Index (Loop).md").is_file() and "Hand." in _lesen(ix)
    assert "[[Doku/Index (Loop)]]" in _lesen(pl)


def _junction(link: Path, ziel: Path) -> bool:
    import subprocess
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(ziel)], capture_output=True, timeout=30)
    return r.returncode == 0 and link.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Junctions nur unter Windows")
def test_junction_im_notizordner_wird_nie_verfolgt(tmp_path):
    wissen = tmp_path / "wissen"
    draussen = tmp_path / "draussen"
    draussen.mkdir()
    fremd = draussen / "b.md"
    fremd.write_bytes(b"---\nerzeugt_von: doku-loop\n---\n# fremd\n")
    (wissen / "Doku").mkdir(parents=True)
    if not _junction(wissen / "Doku" / "Dienste", draussen):
        pytest.skip("Junction nicht anlegbar")
    pp: list[Pruefpunkt] = []
    _schreiben(wissen, manifest_pfad=None, pruefpunkte=pp)  # auch die Uebernahme folgt der Junction nicht
    assert fremd.read_bytes() == b"---\nerzeugt_von: doku-loop\n---\n# fremd\n"
    assert sorted(p.name for p in draussen.iterdir()) == ["b.md"]
    assert (wissen / "Doku" / "Ketten" / "K1.md").is_file()
    # der ganze Notizordner als Junction: nichts geschrieben
    w2 = tmp_path / "w2"
    w2.mkdir()
    if not _junction(w2 / "Doku", draussen):
        pytest.skip("Junction nicht anlegbar")
    assert _schreiben(w2) == [] and sorted(p.name for p in draussen.iterdir()) == ["b.md"]


def test_tick_urteile_fehler_wird_pruefpunkt_und_tick_laeuft_weiter(fx, monkeypatch):
    _seed(fx)
    _urteile_datei(fx, f"{LAUF} K1/1 ok")

    def kaputt(*a, **kw):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(rb, "urteile_anwenden", kaputt)
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen in (None, "keine_aenderung")
    assert any(p["art"] == "urteile_fehler" and "kaputt" in p["text"] for p in tk.pruefliste_laden(fx.konfig)["offen"])
    assert (fx.konfig.rowboat_wissen / "Doku" / "Prüfliste.md").is_file()


def test_tick_hysterese_messung_gilt_erst_bei_zweiter_gleicher_lesung(fx):
    """Ruling 55a: ein Wechsel von `gemessen` wird erst committet, wenn zwei messende Laeufe ihn gleich lesen."""
    from test_doku_tick import _code_aenderung
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": True}}))
    _seed(fx)
    assert "g:läuft" in _text(fx)
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": False}}))
    _code_aenderung(fx, 1)
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.commit is None and "g:läuft" in _text(fx)
    assert set(st.stand_laden(fx.konfig).gemessen_ausstehend.values()) == {"bricht"}
    # ein Flackern zurueck: nichts committet, nichts mehr ausstehend
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": True}}))
    _code_aenderung(fx, 2)
    _anstoss(fx)
    assert _tick(fx).commit is None and st.stand_laden(fx.konfig).gemessen_ausstehend == {}
    # zweimal gleich gelesen -> gilt
    _schreib(fx.repo, "logs/w.json", json.dumps({"stand": {"probe": False}}))
    for i in (3, 4):
        _code_aenderung(fx, i)
        _anstoss(fx)
        erg = _tick(fx)
    assert erg.commit is not None and "g:bricht" in _text(fx)
    assert st.stand_laden(fx.konfig).gemessen_ausstehend == {}


def test_tick_revert_verwirft_render_hashes_auch_wenn_ein_spaeteres_urteil_wirft(fx, monkeypatch):
    _erg, e = _geschrieben_vom_loop(fx)
    _urteile_datei(fx, f"{e['lauf']} {e['aussage_id']} zurueck", f"{e['lauf']} {e['aussage_id']} korrektur: Satz.")

    def kaputt(self, u, e):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(rb._Anwender, "korrektur", kaputt)
    _anstoss(fx)
    erg2 = _tick(fx)
    assert AUSSAGE in _text(fx) and erg2.uebersprungen is None  # Revert + Render im selben Lauf
    assert any(p["art"] == "urteil_fehler" for p in tk.pruefliste_laden(fx.konfig)["offen"])
    assert not any(p["art"] == "hand_edit" for p in tk.pruefliste_laden(fx.konfig)["offen"])
    assert _status(fx.repo) == []
