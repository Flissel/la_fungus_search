"""Task 3 (Stufe 4): Anstoss bei bewegtem HEAD (D6) und Trockenlauf `doku tick --trocken`.

Gleiches Fixture-Repo wie `test_doku_tick.py` (git init unter tmp_path, Fake-Manifest-Bauer, kein echtes LLM, kein
echtes VibeMind). Der Messbaum gehoert im Betrieb dem alten Loop; im Test spielt `messbaum.bereitstellen` den alten Loop.
"""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

import test_doku_tick as tt
from test_doku_tick import (  # noqa: F401  (Fixture `fx`)
    CLIENT, KETTEN, FakeBudget, FakeLaufzeit, FakeManifestBauer, _anstoss, _code_aenderung, _commit, _git, _head,
    _jetzt, _schreib, _seed, _tick, _umbenennen, fx,
)
from selbstbild.doku import __main__ as cli
from selbstbild.doku import messbaum
from selbstbild.doku import stand as st
from selbstbild.doku import tick as tk


# ---------- Hilfen ----------

def _schnappschuss(wurzel: Path) -> dict[str, str]:
    """Pfad -> sha256 der Bytes (Verzeichnisse `<dir>`) und `st_mtime_ns`, ueber den ganzen Baum inkl. `.git`."""
    aus: dict[str, str] = {}
    for p in sorted(Path(wurzel).rglob("*")):
        rel = p.relative_to(wurzel).as_posix()
        try:
            mtime = p.stat().st_mtime_ns
            inhalt = "<dir>" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            mtime, inhalt = 0, "<unlesbar>"
        aus[rel] = f"{inhalt}@{mtime}"
    return aus


def _zustand(fx) -> dict[str, object]:
    # die Beobachtung darf selbst nichts schreiben: `git status` frischt sonst den Index auf (racy-git)
    vorher = os.environ.get("GIT_OPTIONAL_LOCKS")
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        worktrees = _git(fx.repo, "worktree", "list", "--porcelain")
        status = _git(fx.repo, "status", "--porcelain=v1", "--untracked-files=all")
    finally:
        if vorher is None:
            os.environ.pop("GIT_OPTIONAL_LOCKS", None)
        else:
            os.environ["GIT_OPTIONAL_LOCKS"] = vorher
    return {"repo": _schnappschuss(fx.repo), "lokal": _schnappschuss(fx.lokal), "worktrees": worktrees,
            "status": status}


def _unveraendert(fx, vorher) -> None:
    nachher = _zustand(fx)
    diff = [f"{sek}:{k}" for sek in ("repo", "lokal") for k in sorted(set(vorher[sek]) | set(nachher[sek]))
            if vorher[sek].get(k) != nachher[sek].get(k)]
    assert not diff, diff
    assert nachher["worktrees"] == vorher["worktrees"] and nachher["status"] == vorher["status"]


def _alter_loop(fx) -> None:
    """Der alte Loop hat zuletzt auf dem aktuellen HEAD getickt: Messbaum + Stand stehen auf HEAD."""
    messbaum.bereitstellen(fx.konfig)


def _trocken(fx, **kw):
    return tk.tick(fx.konfig, jetzt=_jetzt(), manifest_bauer=FakeManifestBauer(), trocken=True, **kw)


@pytest.fixture
def ohne_judge(monkeypatch):
    """Der Judge darf im Trockenlauf nie geladen werden: Import scheitert, falls doch."""
    import selbstbild.doku as paket
    monkeypatch.delattr(paket, "judge", raising=False)
    monkeypatch.setitem(sys.modules, "selbstbild.doku.judge", None)


# ---------- D6: bewegter HEAD ----------

def test_head_bewegt_stoesst_an(fx):
    _seed(fx)
    _anstoss(fx)
    _tick(fx)  # mit Marker: Stand zieht auf den HEAD nach (der Seed-Commit liegt nach dem gespeicherten Stand)
    stand = st.stand_laden(fx.konfig)
    assert stand is not None and stand.letzter_commit_aussen == _head(fx.repo)
    jetzt = datetime_nach(stand, minuten=1)
    assert not fx.marker.exists()
    koepfe = tk.anstoss_koepfe(fx.repo)
    assert koepfe == {"": _head(fx.repo)}
    # kein neuer Commit: kein Anstoss (letzter Tick vor 1 Minute, kein Marker)
    assert not tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe=koepfe)
    assert _tick(fx, jetzt=jetzt).uebersprungen == "kein_anstoss"
    # neuer Commit: Anstoss ohne Marker
    neu = _code_aenderung(fx, 1)
    koepfe = tk.anstoss_koepfe(fx.repo)
    assert koepfe == {"": neu}
    assert tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe=koepfe)
    assert not tk.anstoss_noetig(fx.konfig, stand, jetzt)  # ohne koepfe wie vorher
    bauer = FakeManifestBauer()
    erg = _tick(fx, jetzt=jetzt, bauer=bauer)
    assert erg.uebersprungen != "kein_anstoss" and bauer.gebaut == ["b"]
    assert st.stand_laden(fx.konfig).letzter_commit_aussen == neu


def test_submodul_kopf_weicht_ab_stoesst_an(fx):
    stand = st.Stand(letzter_commit_aussen="a" * 40, letzter_tick="2026-10-28T12:00:00+00:00",
                     submodul_commits={"vibemind-os": "b" * 40})
    jetzt = datetime_nach(stand, minuten=1)
    gleich = {"": "a" * 40, "vibemind-os": "b" * 40}
    assert not tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe=gleich)
    assert tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe={**gleich, "vibemind-os": "c" * 40})
    assert tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe={**gleich, "vibemind-os/spaces/x": "d" * 40})
    assert tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe={**gleich, "": "e" * 40})


def test_git_fehler_beim_koepfe_lesen_stuerzt_nicht_ab(fx, tmp_path):
    kein_repo = tmp_path / "leer"
    kein_repo.mkdir()
    assert tk.anstoss_koepfe(kein_repo) is None
    assert tk.anstoss_koepfe(tmp_path / "gibt-es-nicht") is None
    stand = st.Stand(letzter_commit_aussen="a" * 40, letzter_tick="2026-10-28T12:00:00+00:00")
    assert not tk.anstoss_noetig(fx.konfig, stand, datetime_nach(stand, minuten=1), koepfe=None)


def datetime_nach(stand, *, minuten: int):
    from datetime import datetime
    return datetime.fromisoformat(stand.letzter_tick) + timedelta(minutes=minuten)


# ---------- Trockenlauf ----------

def test_trocken_schreibt_nichts(fx, ohne_judge):
    _seed(fx)
    _umbenennen(fx)
    _anstoss(fx)
    _alter_loop(fx)
    vorher = _zustand(fx)
    lauf, budget = FakeLaufzeit(), FakeBudget()
    erg = _trocken(fx, laufzeit=lauf, budget=budget)
    assert erg.uebersprungen is None
    assert erg.llm_aufrufe == 0 and lauf.aufrufe == [] and budget.pruefungen == [] and budget.buchungen == []
    assert erg.signale >= 1  # Beleg `senden` ist verschwunden: ginge im echten Tick an den Judge
    assert erg.commit is None and erg.umgeschrieben == 0 and erg.pruefliste == 0
    assert erg.spaces == ["a", "b"]
    assert len(erg.graph_digest) == 64
    _unveraendert(fx, vorher)  # Repo (inkl. .git), lokaler Ordner (Marker, Stand, Messbaum) unveraendert
    assert fx.marker.exists()


def test_trocken_fragt_den_anstoss_nicht_und_ist_wiederholbar(fx, ohne_judge):
    _seed(fx)
    _umbenennen(fx)
    _alter_loop(fx)
    assert not fx.marker.exists()
    erste = _trocken(fx)
    zweite = _trocken(fx)
    assert erste.uebersprungen is None and erste.signale >= 1
    assert (erste.signale, erste.graph_digest) == (zweite.signale, zweite.graph_digest)


def test_trocken_graph_digest_ist_der_stabile_digest_des_schlanken_graphen(fx):
    from selbstbild.doku.graph import Graph, schlank
    _seed(fx)
    _alter_loop(fx)
    erg = _trocken(fx)
    assert erg.signale == 0
    # ohne Codeaenderung seit dem Seed gleicht der gemessene Graph dem committeten
    committet = Graph.laden(fx.repo / "docs/graph/graph.json")
    assert erg.graph_digest == schlank(committet).digest_stabil() == committet.digest_stabil()


def test_trocken_messbaum_veraltet(fx, ohne_judge):
    _seed(fx)
    _alter_loop(fx)
    _code_aenderung(fx, 1)  # neuer HEAD, der alte Loop hat noch nicht nachgezogen
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen == "messbaum_veraltet" and erg.graph_digest == "" and erg.llm_aufrufe == 0
    _unveraendert(fx, vorher)
    assert "messbaum-" in " ".join(vorher["lokal"])  # Messbaum steht noch auf dem alten Commit


def test_trocken_ohne_messbaum(fx):
    _seed(fx)
    shutil.rmtree(fx.lokal)  # kein Messbaum, kein Stand
    erg = _trocken(fx)
    assert erg.uebersprungen == "messbaum_veraltet"
    assert not fx.lokal.exists()  # nichts angelegt


def test_trocken_worktree_auf_falschem_commit_gilt_als_veraltet(fx):
    _seed(fx)
    _alter_loop(fx)
    gitdir = messbaum._gitdir(messbaum.messbaum_pfad(fx.konfig))
    (gitdir / "HEAD").write_text("0" * 40 + "\n", encoding="utf-8")
    vorher = _zustand(fx)
    assert _trocken(fx).uebersprungen == "messbaum_veraltet"
    _unveraendert(fx, vorher)


def test_trocken_frische_sperre_alter_loop_laeuft(fx):
    _seed(fx)
    _alter_loop(fx)
    sperre = fx.lokal / "VibeMind" / "doku_loop" / st.SPERRE_DATEI
    sperre.write_text("1234 0 abcd", encoding="utf-8")
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen == "alter_loop_laeuft" and erg.llm_aufrufe == 0
    _unveraendert(fx, vorher) and sperre.read_text(encoding="utf-8") == "1234 0 abcd"


def test_trocken_veraltete_sperre_zaehlt_nicht_und_bleibt(fx):
    _seed(fx)
    _alter_loop(fx)
    sperre = fx.lokal / "VibeMind" / "doku_loop" / st.SPERRE_DATEI
    sperre.write_text("1234 0 abcd", encoding="utf-8")
    alt = time.time() - st.SPERRE_VERALTET_S - 60
    os.utime(sperre, (alt, alt))
    vorher = _zustand(fx)
    assert _trocken(fx).uebersprungen is None
    assert sperre.exists()  # nicht uebernommen, nicht entfernt
    _unveraendert(fx, vorher)


def test_vorhanden_lesend_aendert_nichts_und_liefert_den_messbaum(fx):
    _seed(fx)
    # der Seed hat den Messbaum vor seinem eigenen (Doku-)Commit angelegt: T3-c
    assert messbaum.vorhanden_lesend(fx.konfig).hinweis == "messbaum_hinter_head_nur_eigene_dateien"
    echt = messbaum.bereitstellen(fx.konfig)
    vorher = _zustand(fx)
    mb = messbaum.vorhanden_lesend(fx.konfig)
    assert mb is not None and mb.wurzel == echt.wurzel and mb.commits == echt.commits == {"": _head(fx.repo)}
    assert mb.konfig.repo == echt.wurzel and mb.nicht_messbar == ()
    _code_aenderung(fx, 1)
    assert messbaum.vorhanden_lesend(fx.konfig) is None
    assert _zustand(fx)["lokal"] == vorher["lokal"]


def test_trocken_ruft_bereitstellen_nie(fx, monkeypatch):
    _seed(fx)
    _alter_loop(fx)

    def verboten(*a, **kw):
        raise AssertionError("Trockenlauf darf den Messbaum nicht bereitstellen")

    monkeypatch.setattr(messbaum, "bereitstellen", verboten)
    monkeypatch.setattr(messbaum, "_anlegen", verboten)
    assert _trocken(fx).uebersprungen is None


def test_trocken_ohne_seed(fx):
    erg = _trocken(fx)
    assert erg.uebersprungen == "kein_seed"


# ---------- CLI ----------

def test_cli_trocken_gibt_json_auf_stdout(fx, ohne_judge, capsys):
    _seed(fx)
    _umbenennen(fx)
    _alter_loop(fx)
    vorher = _zustand(fx)
    w = {"manifest_bauer": FakeManifestBauer(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick", "--trocken"], werkzeuge=w) == 0
    aus = capsys.readouterr()
    daten = json.loads(aus.out)  # stdout ist nur das JSON
    assert set(daten) == {f.name for f in dataclasses.fields(tk.TickErgebnis)}
    assert daten["signale"] >= 1 and daten["llm_aufrufe"] == 0 and daten["commit"] is None
    assert len(daten["graph_digest"]) == 64
    assert "Trockenlauf" not in aus.out and not (fx.repo / "logs").exists()
    _unveraendert(fx, vorher)


def test_cli_trocken_gilt_auch_auf_der_universellen_vorgabe(tmp_path, capsys):
    """Die Sperre `tick` auf der Vorgabe ohne Torwaechter gilt nicht: der Trockenlauf schreibt nichts."""
    wurzel = tmp_path / "wurzel"
    wurzel.mkdir()
    assert cli.main(["tick", str(wurzel)]) == 2
    capsys.readouterr()
    assert cli.main(["tick", str(wurzel), "--trocken"]) == 0
    assert json.loads(capsys.readouterr().out)["uebersprungen"] in ("kein_seed", "messbaum_veraltet")


def test_cli_trocken_setzt_git_optional_locks_nur_waehrend_des_laufs(fx, monkeypatch):
    _seed(fx)
    _alter_loop(fx)
    monkeypatch.delenv("GIT_OPTIONAL_LOCKS", raising=False)
    gesehen: list[str | None] = []
    echt = tk.unsaubere_dateien

    def spaeh(repo):
        gesehen.append(os.environ.get("GIT_OPTIONAL_LOCKS"))
        return echt(repo)

    monkeypatch.setattr(tk, "unsaubere_dateien", spaeh)
    _trocken(fx)
    assert gesehen == ["0"] and "GIT_OPTIONAL_LOCKS" not in os.environ


# ---------- Fix-Runde 1 ----------

def _kopf_ergebnis(fx):
    return json.loads(Path(fx.lokal / "VibeMind" / "doku_loop").joinpath(f"stand-{st.repo_schluessel(fx.repo)}.json")
                      .read_text(encoding="utf-8"))


def test_drossel_bleibt_nach_nicht_fortschreibendem_stand(fx):
    """T3-b: dreckige KETTEN.md + neuer Commit -> `fremde_aenderungen` einmal; danach `kein_anstoss` bis zum
    Intervall oder einem neuen Commit (der Stand merkt sich die gesehenen Koepfe, auch ohne fortzuschreiben)."""
    _seed(fx)
    _anstoss(fx)
    _tick(fx)  # zieht den Stand nach dem Seed-Commit nach
    _code_aenderung(fx, 1)
    (fx.repo / KETTEN).write_bytes((fx.repo / KETTEN).read_bytes() + b"von Hand\n")
    stand = st.stand_laden(fx.konfig)
    t0 = datetime_nach(stand, minuten=1)
    erste = _tick(fx, jetzt=t0)
    assert erste.uebersprungen == f"fremde_aenderungen:{KETTEN}"
    gesehen = st.stand_laden(fx.konfig)
    assert gesehen.koepfe_gesehen == {"": _head(fx.repo)}
    assert gesehen.letzter_commit_aussen != _head(fx.repo)  # nicht fortgeschrieben
    for minuten in (1, 2, 6):
        assert _tick(fx, jetzt=t0 + timedelta(minutes=minuten)).uebersprungen == "kein_anstoss"
    _code_aenderung(fx, 2)  # neuer Commit stoesst wieder an
    assert _tick(fx, jetzt=t0 + timedelta(minutes=7)).uebersprungen == f"fremde_aenderungen:{KETTEN}"
    assert _tick(fx, jetzt=t0 + timedelta(minutes=8)).uebersprungen == "kein_anstoss"
    # das Intervall greift weiterhin
    assert _tick(fx, jetzt=t0 + timedelta(minutes=40)).uebersprungen == f"fremde_aenderungen:{KETTEN}"


def test_stand_ohne_neues_feld_laedt_und_faellt_auf_die_zeiger_zurueck(fx):
    alt = {"letzter_commit_aussen": "a" * 40, "letzter_tick": "2026-10-28T12:00:00+00:00", "lauf_nr": 3,
           "unbekannter_schluessel": 1}
    stand = st.Stand.from_dict(alt)
    assert stand.koepfe_gesehen == {} and stand.letzter_commit_aussen == "a" * 40
    assert "koepfe_gesehen" in stand.to_dict()
    jetzt = datetime_nach(stand, minuten=1)
    assert not tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe={"": "a" * 40})
    assert tk.anstoss_noetig(fx.konfig, stand, jetzt, koepfe={"": "b" * 40})
    neu = st.Stand.from_dict({**alt, "koepfe_gesehen": {"": "b" * 40}})
    assert not tk.anstoss_noetig(fx.konfig, neu, jetzt, koepfe={"": "b" * 40})  # gesehen schlaegt den Zeiger
    assert tk.anstoss_noetig(fx.konfig, neu, jetzt, koepfe={"": "a" * 40})


def test_anstoss_koepfe_lazy_und_einmal(fx, monkeypatch):
    _seed(fx)
    _anstoss(fx)
    _tick(fx)
    aufrufe = {"koepfe": 0, "sub": 0}
    echt_k, echt_s = tk.anstoss_koepfe, messbaum.submodul_koepfe

    def koepfe(repo):
        aufrufe["koepfe"] += 1
        return echt_k(repo)

    def sub(repo, *a, **kw):
        aufrufe["sub"] += 1
        return echt_s(repo, *a, **kw)

    monkeypatch.setattr(tk, "anstoss_koepfe", koepfe)
    monkeypatch.setattr(messbaum, "submodul_koepfe", sub)
    stand = st.stand_laden(fx.konfig)
    t0 = datetime_nach(stand, minuten=1)
    _code_aenderung(fx, 1)
    _tick(fx, jetzt=t0)  # HEAD bewegt: einmal gelesen, im Tick wiederverwendet
    assert aufrufe == {"koepfe": 1, "sub": 1}
    aufrufe.update(koepfe=0, sub=0)
    _anstoss(fx)  # Marker: Anstoss ohne Git; der Tick liest die Koepfe selbst (einmal)
    _tick(fx, jetzt=t0 + timedelta(minutes=1))
    assert aufrufe["koepfe"] == 0 and aufrufe["sub"] == 1


def test_trocken_messbaum_hinter_head_nur_eigene_dateien(fx, ohne_judge):
    _seed(fx)
    _alter_loop(fx)
    a = _head(fx.repo)
    graph = fx.repo / "docs/graph/graph.json"
    graph.write_bytes(graph.read_bytes() + b"\n")
    b = _commit(fx.repo, "docs(loop): eigener Commit", "docs/graph/graph.json")
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen is None and len(erg.graph_digest) == 64
    assert erg.messbaum_commit == a != b and erg.hinweis == "messbaum_hinter_head_nur_eigene_dateien"
    _unveraendert(fx, vorher)
    # auch KETTEN.md zaehlt als eigene Datei
    ketten = fx.repo / KETTEN
    ketten.write_bytes(ketten.read_bytes() + b"\n")
    _commit(fx.repo, "docs(loop): ketten", KETTEN)
    assert _trocken(fx).hinweis == "messbaum_hinter_head_nur_eigene_dateien"
    # ein Quellcode-Commit macht ihn veraltet
    _code_aenderung(fx, 1)
    erg = _trocken(fx)
    assert erg.uebersprungen == "messbaum_veraltet" and erg.messbaum_commit == "" and erg.hinweis == ""
    assert messbaum.vorhanden_lesend(fx.konfig) is None


def test_trocken_gleicher_kopf_hat_keinen_hinweis(fx):
    _seed(fx)
    _alter_loop(fx)
    erg = _trocken(fx)
    assert erg.messbaum_commit == _head(fx.repo) and erg.hinweis == ""


def test_trocken_meldet_fremde_aenderungen_lesend(fx, ohne_judge):
    _seed(fx)
    _alter_loop(fx)
    ketten = fx.repo / KETTEN
    ketten.write_bytes(ketten.read_bytes() + b"von Hand\n")
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen == f"fremde_aenderungen:{KETTEN}" and erg.llm_aufrufe == 0
    _unveraendert(fx, vorher)


def test_trocken_unerwartete_ausnahme_wird_ergebnis(fx, monkeypatch, capsys):
    _seed(fx)
    _alter_loop(fx)

    def kaputt(*a, **kw):
        raise RuntimeError("Dateiinhalt darf nicht in die Ausgabe")

    monkeypatch.setattr(tk.bauen, "graph_bauen", kaputt)
    vorher = _zustand(fx)
    erg = _trocken(fx)
    assert erg.uebersprungen == "tick_fehler:RuntimeError"
    _unveraendert(fx, vorher)
    w = {"manifest_bauer": FakeManifestBauer(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick", "--trocken"], werkzeuge=w) == 0
    aus = capsys.readouterr()
    assert json.loads(aus.out)["uebersprungen"] == "tick_fehler:RuntimeError"
    assert "Dateiinhalt" not in aus.out + aus.err
