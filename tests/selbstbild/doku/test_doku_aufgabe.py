"""Stufe 4, Task 10: die Windows-Aufgabe `VibeMind Doku-Loop` umschalten und zurueck (`doku aufgabe`).

Nie die echte Aufgabe aendern (gefaelschtes Scheduler-Backend), nie den echten lokalen Zustand oder den echten
Rowboat-Ordner beschreiben (alles unter tmp_path). Gelesen wird die echte Aufgabe nur in einem `braucht_vm`-Test.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from doku_vm import VORLAGE, braucht_vm, laden as vm_laden
from selbstbild.doku import __main__ as cli
from selbstbild.doku import aufgabe as ag
from selbstbild.doku import rowboat as rb
from selbstbild.doku import rowboat_erzaehlt as rz
from selbstbild.doku.stand import repo_schluessel
from test_doku_rowboat_erzaehlt import GUT, LAUF, _doku, _g

ALT_ARGS = '--headless cmd /c "scripts\\doku-loop.cmd tick >> logs\\doku-loop.log 2>&1"'
PY = r"C:\venv mit leer\Scripts\python.exe"
T1 = datetime(2026, 11, 3, 18, 0, 0)
T2 = datetime(2026, 11, 3, 18, 30, 0)
T3 = datetime(2026, 11, 4, 9, 0, 0)
T4 = datetime(2026, 11, 4, 9, 30, 0)


class FakeBackend:
    """Gefaelschter Scheduler: haelt die Aktionen im Speicher, merkt jedes Setzen."""

    def __init__(self, aktion, *, fehler_beim_setzen=False, setzt_trotzdem=False):
        self.aktionen = [aktion]
        self.gesetzt: list[ag.Aktion] = []
        self.fehler_beim_setzen, self.setzt_trotzdem = fehler_beim_setzen, setzt_trotzdem
        self.wirkung = None  # Fix-Runde 1 (Minor 7): "nichts" oder eine andere Aktion -> Rueckmeldung weicht ab
        self.beim_setzen = None  # Rueckruf vor dem Setzen (Reihenfolge-Pruefung)

    def lesen(self, aufgabe):
        assert aufgabe == ag.AUFGABE
        return list(self.aktionen)

    def setzen(self, aufgabe, aktion):
        assert aufgabe == ag.AUFGABE
        self.gesetzt.append(aktion)
        if self.beim_setzen is not None:
            self.beim_setzen(aktion)
        if self.wirkung == "nichts":
            return
        if isinstance(self.wirkung, ag.Aktion):
            self.aktionen = [self.wirkung]
            return
        if self.fehler_beim_setzen:
            if self.setzt_trotzdem:
                self.aktionen = [aktion]
            raise ag.AufgabeFehler("Set-ScheduledTask: Zugriff verweigert")
        self.aktionen = [aktion]


def _baum(wurzel: Path, *, ohne=(), mtime=False) -> dict[str, object]:
    """rel -> Bytes (Datei, optional mit mtime) oder "<dir>"; Eintraege der obersten Ebene in `ohne` fehlen."""
    aus: dict[str, object] = {}
    if not wurzel.exists():
        return aus
    for p in sorted(wurzel.rglob("*")):
        rel = p.relative_to(wurzel).as_posix()
        if any(rel == o or rel.startswith(o + "/") or (o.endswith("*") and rel.startswith(o[:-1])) for o in ohne):
            continue
        if p.is_dir():
            aus[rel] = "<dir>"
        else:
            aus[rel] = (p.read_bytes(), p.stat().st_mtime_ns) if mtime else p.read_bytes()
    return aus


def _alles(tmp_path: Path) -> dict[str, object]:
    return _baum(tmp_path, mtime=True)


def _seite(titel: str, text: str = "Ein erzählter Text.") -> str:
    return f"---\nerzeugt_von: selbstbild\ngrundlage: \"abc\"\n---\n# {titel}\n\n{text}\n"


@pytest.fixture()
def umg(tmp_path, monkeypatch):
    """Zustand und Rowboat-Ordner wie beim alten Loop (nur tmp_path), Aufgabe mit der alten Aktion."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lokal"))
    repo = tmp_path / "repo"
    (repo / "logs").mkdir(parents=True)
    ordner = tmp_path / "lokal" / "VibeMind" / "doku_loop"
    k = dataclasses.replace(vm_laden(), repo=repo, marker=ordner / "anstoss", rowboat_wissen=tmp_path / "wissen",
                            rowboat_notizen="Doku")
    h = repo_schluessel(repo)
    ordner.mkdir(parents=True)
    (ordner / f"stand-{h}.json").write_text('{"lauf_nr": 7}\n', encoding="utf-8")
    (ordner / f"graph-voll-{h}.json").write_bytes(b"{" + b"x" * 5000 + b"}")
    (ordner / "urteile.jsonl").write_text('{"a": 1}\n', encoding="utf-8")
    (ordner / f"pruefliste-{h}.json").write_text('{"offen": []}\n', encoding="utf-8")
    (ordner / f"messbaum-{h}").mkdir()
    (ordner / f"messbaum-{h}" / "worktree.txt").write_text("aussen\n", encoding="utf-8")
    (ordner / f"messbaum-{h}.json").write_text("{}\n", encoding="utf-8")
    (ordner / "messbaum-hooks-leer").mkdir()
    (ordner / "vibemind-claude-invocation-doku-loop").mkdir()
    for i, p in enumerate(sorted(ordner.iterdir())):  # alte, unterschiedliche mtimes
        if p.is_file():
            os.utime(p, ns=(1_700_000_000_000_000_000 + i * 10**9, 1_700_000_000_000_000_000 + i * 10**9))
    manifest = rb.manifest_pfad(k)
    rb.notizen_schreiben(_g(), [], [], k.rowboat_wissen, "Doku", LAUF, manifest_pfad=manifest)  # der alte Loop
    alt = ag.Aktion("conhost.exe", ALT_ARGS, str(repo))
    return SimpleNamespace(k=k, ordner=ordner, repo=repo, h=h, basis=k.rowboat_wissen / "Doku", manifest=manifest,
                           alt=alt, backend=FakeBackend(alt), tmp=tmp_path)


def _um(u, *, trocken=False, jetzt=T1, aus=None):
    zeilen: list[str] = []
    rc = ag.umschalten(u.k, backend=u.backend, python=PY, trocken=trocken, jetzt=jetzt,
                       aus=aus or zeilen.append, import_pruefen=lambda python, wurzel: None)
    return rc, "\n".join(zeilen)


def _zu(u, *, trocken=False, jetzt=T2):
    zeilen: list[str] = []
    rc = ag.zurueck(u.k, backend=u.backend, trocken=trocken, jetzt=jetzt, aus=zeilen.append)
    return rc, "\n".join(zeilen)


def _manifest(u) -> dict[str, str]:
    return json.loads(u.manifest.read_text(encoding="utf-8"))["dateien"]


def _neuer_loop_schreibt_seiten(u, seiten: dict[str, str]) -> None:
    """Fake fuer den neuen Loop: eigene Seiten schreiben und ins gemeinsame Manifest eintragen."""
    m = _manifest(u)
    for rel, text in seiten.items():
        p = u.basis / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
        m[rel] = rb._stabil_hash(text)
    rb._manifest_schreiben(u.manifest, m)


EIGENE = {
    "Was ist VibeMind.md": _seite("Was ist VibeMind"),
    "Abläufe/K1.md": _seite("K1"),
    "Mitbauen/Neuen Space erstellen mit Claude Code.md": _seite("Neuen Space erstellen mit Claude Code"),
    "Zustand.md": _seite("Zustand"),
}


# ---------- Aktion: exakte Zeichenkette und Einordnung ----------

def test_neue_aktion_exakt():
    a = ag.neue_aktion(r"C:\Users\User\ClaudeWork\wt-fungus-gate1\.venv-selbstbild\Scripts\python.exe",
                       r"C:\Users\User\Desktop\Vibemind_V1")
    assert a == ag.Aktion(
        "conhost.exe",
        '--headless cmd /c ""C:\\Users\\User\\ClaudeWork\\wt-fungus-gate1\\.venv-selbstbild\\Scripts\\python.exe" '
        '-m selbstbild doku tick "C:\\Users\\User\\Desktop\\Vibemind_V1" >> '
        '"C:\\Users\\User\\Desktop\\Vibemind_V1\\logs\\doku-loop.log" 2>&1"',
        r"C:\Users\User\Desktop\Vibemind_V1")


@pytest.mark.parametrize("schlecht", ['C:\\a"b\\python.exe', "C:\\%TEMP%\\python.exe", "C:\\a&b\\python.exe",
                                      "C:\\a^b\\python.exe", "C:\\a|b\\python.exe", "C:\\a<b\\python.exe"])
def test_neue_aktion_lehnt_cmd_sonderzeichen_ab(schlecht):
    with pytest.raises(ValueError):
        ag.neue_aktion(schlecht, r"C:\w")
    with pytest.raises(ValueError):
        ag.neue_aktion(r"C:\p\python.exe", schlecht.replace("python.exe", "wurzel"))


def test_einordnen():
    wd = r"C:\Users\User\Desktop\Vibemind_V1"
    assert ag.einordnen([ag.Aktion("conhost.exe", ALT_ARGS, wd)]) == "alt"
    assert ag.einordnen([ag.neue_aktion(PY, wd)]) == "neu"
    assert ag.einordnen([ag.Aktion("C:\\Windows\\System32\\conhost.exe", ALT_ARGS, wd)]) == "alt"
    assert ag.einordnen([ag.Aktion("cmd.exe", "/c echo", wd)]) == "unbekannt"
    assert ag.einordnen([ag.Aktion("conhost.exe", ALT_ARGS, wd)] * 2) == "unbekannt"
    assert ag.einordnen([]) == "unbekannt"


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe gibt es nur unter Windows")
def test_cmd_zerlegt_die_aktion_wie_gedacht(tmp_path):
    """Die exakte Zeichenkette hinter `--headless` durch das echte cmd: Python startet mit der Wurzel als einem
    Argument (Leerzeichen im Pfad), stdout und stderr landen im Log. Die Wurzel hat kein selbstbild.yaml, also bricht
    der Tick vor jedem Schreiben ab (universelle Vorgabe, Exit 2)."""
    wurzel = tmp_path / "mit leer zeichen" / "Vibe Mind"
    (wurzel / "logs").mkdir(parents=True)
    a = ag.neue_aktion(sys.executable, wurzel)
    kopf = "--headless cmd /c "
    assert a.arguments.startswith(kopf)
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "SELBSTBILD_VIBEMIND")}
    env["LOCALAPPDATA"] = str(tmp_path / "lokal")
    lauf = subprocess.run("cmd /c " + a.arguments[len(kopf):], cwd=wurzel, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    log = (wurzel / "logs" / "doku-loop.log").read_text(encoding="utf-8")
    assert lauf.returncode == 2
    assert f"Repo {wurzel.resolve()}" in log and "universellen Vorgabe" in log
    assert not (tmp_path / "lokal").exists()


# ---------- Brief Step 1-3 ----------

def test_umschalten_und_zurueck(umg):
    u = umg
    zustand_vorher = _baum(u.ordner, mtime=True)
    rowboat_vorher = _baum(u.basis, mtime=True)
    rc, text = _um(u)
    assert rc == 0, text
    neu = ag.neue_aktion(PY, u.repo)
    assert u.backend.aktionen == [neu] and u.backend.gesetzt == [neu]
    sicherung = u.ordner.parent / "doku_loop-sicherung-20261103-180000"
    assert sicherung.is_dir()
    assert _baum(sicherung, mtime=True) == _baum(u.ordner, ohne=("messbaum-*", ag.ALT_DATEI), mtime=True)
    assert not any(n.startswith("messbaum") for n in os.listdir(sicherung))
    gemerkt = json.loads((u.ordner / ag.ALT_DATEI).read_text(encoding="utf-8"))
    assert gemerkt["aktion"] == {"Execute": "conhost.exe", "Arguments": ALT_ARGS, "WorkingDirectory": str(u.repo)}

    # der neue Loop arbeitet: Zustand geaendert, eigene Seiten geschrieben, Messbaum bewegt
    (u.ordner / f"stand-{u.h}.json").write_text('{"lauf_nr": 9, "neu": true}\n', encoding="utf-8")
    (u.ordner / "urteile.jsonl").write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    (u.ordner / f"graph-voll-{u.h}.json").unlink()
    (u.ordner / "nur-vom-neuen-loop.json").write_bytes(b"{}\n")
    (u.ordner / "vibemind-claude-invocation-doku-loop" / "rest.txt").write_text("x", encoding="utf-8")
    (u.ordner / f"messbaum-{u.h}" / "worktree.txt").write_text("weiter\n", encoding="utf-8")
    _neuer_loop_schreibt_seiten(u, EIGENE)

    rc, text = _zu(u)
    assert rc == 0, text
    assert u.backend.aktionen == [u.alt]
    # Zustand byte- und mtime-gleich (Messbaum gehoert dem Loop und bleibt, das Archiv ist neu)
    assert _baum(u.ordner, ohne=("messbaum-*", ag.ARCHIV), mtime=True) == \
        _baum(sicherung, mtime=True) == {k: v for k, v in zustand_vorher.items() if not k.startswith("messbaum")}
    assert (u.ordner / f"messbaum-{u.h}" / "worktree.txt").read_text(encoding="utf-8") == "weiter\n"
    assert _baum(u.basis, mtime=True) == rowboat_vorher
    # nichts geloescht: was nur der neue Loop hatte, liegt im Nachher-Ordner
    nachher = _baum(u.ordner.parent / "doku_loop-nachher-20261103-183000")
    assert json.loads(nachher.pop(ag.ALT_DATEI))["aktion"]["Arguments"] == ALT_ARGS
    assert nachher == {"nur-vom-neuen-loop.json": b"{}\n", "vibemind-claude-invocation-doku-loop": "<dir>",
                       "vibemind-claude-invocation-doku-loop/rest.txt": b"x"}
    archiv = _baum(u.ordner / ag.ARCHIV)
    assert {k: v for k, v in archiv.items() if k.startswith("seiten/") and v != "<dir>"} == \
        {f"seiten/{rel}": t.encode("utf-8") for rel, t in EIGENE.items()}
    assert json.loads(archiv["manifest.json"])["dateien"] == {rel: rb._stabil_hash(t) for rel, t in EIGENE.items()}
    assert sicherung.is_dir()  # die Sicherung bleibt liegen


def test_umschalten_bei_sperre_abgelehnt(umg):
    u = umg
    (u.ordner / "tick.lock").write_text("4711 1.0 abcd", encoding="utf-8")
    vorher = _alles(u.tmp)
    rc, text = _um(u)
    assert rc == 2 and "tick.lock" in text
    assert u.backend.gesetzt == [] and _alles(u.tmp) == vorher
    # auch der Rueckweg nicht
    (u.ordner / ag.ALT_DATEI).write_text(json.dumps({"aktion": u.alt.als_dict()}), encoding="utf-8")
    (u.ordner.parent / "doku_loop-sicherung-20261103-180000").mkdir()
    u.backend.aktionen = [ag.neue_aktion(PY, u.repo)]
    vorher = _alles(u.tmp)
    rc, text = _zu(u)
    assert rc == 2 and "tick.lock" in text
    assert u.backend.gesetzt == [] and _alles(u.tmp) == vorher


def test_veraltete_sperre_haelt_nicht_auf(umg):
    u = umg
    lock = u.ordner / "tick.lock"
    lock.write_text("4711 1.0 abcd", encoding="utf-8")
    os.utime(lock, (1_000_000, 1_000_000))
    rc, text = _um(u)
    assert rc == 0, text
    assert not lock.exists()  # uebernommen und wieder freigegeben
    assert "tick.lock" not in os.listdir(u.ordner.parent / "doku_loop-sicherung-20261103-180000")


def test_trocken_zeigt_nur(umg):
    u = umg
    vorher = _alles(u.tmp)
    rc, text = _um(u, trocken=True)
    assert rc == 0, text
    assert ag.neue_aktion(PY, u.repo).arguments in text
    assert "doku_loop-sicherung-20261103-180000" in text and "stand-" in text and "aufgabe-alt.json" in text
    gesichert = [z.split("sichern: ", 1)[1] for z in text.splitlines() if "sichern: " in z]
    assert f"stand-{u.h}.json" in gesichert and not any(g.startswith("messbaum") for g in gesichert)
    assert u.backend.gesetzt == [] and _alles(u.tmp) == vorher

    # Rueckweg trocken nach echtem Umschalten: zeigt Aktion, Archiv-Eintraege und Dateien, schreibt nichts
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    (u.basis / "Zustand.md").write_text(_seite("Zustand", "Von Hand."), encoding="utf-8")
    vorher, gesetzt = _alles(u.tmp), list(u.backend.gesetzt)
    rc, text = _zu(u, trocken=True)
    assert rc == 0, text
    assert ALT_ARGS in text and "Abläufe/K1.md" in text and "Was ist VibeMind.md" in text
    assert "Zustand.md" in text and "von Hand" in text and "aufgabe-alt.json" in text
    assert u.backend.gesetzt == gesetzt and _alles(u.tmp) == vorher


def test_trocken_meldet_frische_sperre(umg):
    u = umg
    (u.ordner / "tick.lock").write_text("4711 1.0 abcd", encoding="utf-8")
    vorher = _alles(u.tmp)
    rc, text = _um(u, trocken=True)
    assert rc == 0 and "frisch" in text and _alles(u.tmp) == vorher


# ---------- Rueckweg: eigene Seiten, Hand-Edits, Manifest ----------

def test_bearbeitete_erzaehlte_seite_ueberlebt_zurueck(umg):
    u = umg
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    hand = _seite("K1", "Von Hand ergänzt.")
    (u.basis / "Abläufe" / "K1.md").write_text(hand, encoding="utf-8")
    vorher = (u.basis / "Abläufe" / "K1.md").stat().st_mtime_ns
    rc, text = _zu(u)
    assert rc == 0, text
    p = u.basis / "Abläufe" / "K1.md"
    assert p.read_text(encoding="utf-8") == hand and p.stat().st_mtime_ns == vorher
    assert "Abläufe/K1.md" in text and "von Hand" in text
    assert not (u.basis / "Mitbauen").exists()  # leer geworden -> weg
    assert (u.basis / "Abläufe").is_dir()  # nicht leer -> bleibt
    assert not (u.ordner / ag.ARCHIV / "seiten" / "Abläufe" / "K1.md").exists()


def test_alte_space_notiz_und_index_ohne_signalflut(umg):
    """Review-Fokus 4: nach dem Rueckweg laeuft der alte Loop ohne `notiz_von_hand`-Flut. Die vom neuen Loop
    erzaehlte Space-Seite (ueber der alten Notiz) wird archiviert und entfernt; eine vom neuen Loop neu geschriebene
    deterministische Notiz (hier Index.md) bleibt dem Loop: das zurueckgespielte Manifest traegt ihren jetzigen Hash."""
    u = umg
    assert _um(u)[0] == 0
    index = u.basis / "Index.md"
    index_neu = index.read_text(encoding="utf-8") + "\n[[Doku/Zustand]] – neu vom neuen Loop\n"
    _neuer_loop_schreibt_seiten(u, {"Spaces/a.md": _seite("Space a"), "Index.md": index_neu})
    rc, text = _zu(u)
    assert rc == 0, text
    assert not (u.basis / "Spaces" / "a.md").exists()
    assert (u.ordner / ag.ARCHIV / "seiten" / "Spaces" / "a.md").is_file()
    assert index.read_text(encoding="utf-8") == index_neu  # bleibt, bis der alte Loop ihn neu schreibt
    assert _manifest(u)["Index.md"] == rb._stabil_hash(index_neu)
    pp: list = []
    rb.notizen_schreiben(_g(), [], [], u.k.rowboat_wissen, "Doku", LAUF, manifest_pfad=u.manifest, pruefpunkte=pp)
    assert pp == []
    assert "erzeugt_von: doku-loop" in (u.basis / "Spaces" / "a.md").read_text(encoding="utf-8")
    assert "neu vom neuen Loop" not in index.read_text(encoding="utf-8")


def test_fehlversuche_bleiben_beim_zurueck(umg):
    u = umg
    assert _um(u)[0] == 0
    fehl = u.ordner / f"seiten-fehlversuche-{u.h}.json"
    fehl.write_text('{"version": 1, "seiten": {}}\n', encoding="utf-8")
    assert _zu(u)[0] == 0
    assert fehl.read_text(encoding="utf-8") == '{"version": 1, "seiten": {}}\n'


def test_umschalten_nach_zurueck_spielt_archiv_zurueck(umg):
    """Kreislauf mit echten Seitenlaeufen: der neue Loop erzaehlt (Fake-Abo), zurueck archiviert, der alte Loop
    schreibt seine Notizen, erneutes Umschalten legt die archivierten Seiten samt Manifest-Eintraegen zurueck -- der
    naechste Seitenlauf findet alles `aktuell` und fragt 0-mal."""
    u = umg
    plan = rz.seiten_plan(_g(), _doku())
    assert {a.pfad for a in plan} >= {"Was ist VibeMind.md", "Spaces/a.md", "Abläufe/K1.md"}

    def seitenlauf(erzaehlen):
        s = rb.schreiber_fuer(u.k.rowboat_wissen, "Doku", u.manifest)
        e = rz.seiten_lauf(plan, s, erzaehlen=erzaehlen, max_aufrufe=99)
        pp = list(e.pruefpunkte)
        rb.notizen_schreiben(_g(), [], [], u.k.rowboat_wissen, "Doku", LAUF, manifest_pfad=u.manifest,
                             zusatz=e.zusatz, zusatz_ausgefallen=e.ausgefallen, spaces=False, pruefpunkte=pp,
                             vorhandene_seiten=e.vorhanden)
        return e, pp

    aufrufe: list[str] = []

    def erz(a):
        aufrufe.append(a.pfad)
        return rz.seite_bauen(a, GUT), "ok"

    assert _um(u)[0] == 0
    e, pp = seitenlauf(erz)
    assert sorted(e.erzaehlt) == sorted(a.pfad for a in plan) and pp == []
    erzaehlt = {a.pfad: (u.basis / a.pfad).read_bytes() for a in plan}
    eintraege = {a.pfad: _manifest(u)[a.pfad] for a in plan}

    assert _zu(u)[0] == 0
    assert not any((u.basis / a.pfad).exists() for a in plan if not a.pfad.startswith("Spaces/"))
    pp = []
    rb.notizen_schreiben(_g(), [], [], u.k.rowboat_wissen, "Doku", LAUF, manifest_pfad=u.manifest, pruefpunkte=pp)
    assert pp == []  # der alte Loop laeuft ohne Hand-Meldungen
    assert "erzeugt_von: doku-loop" in (u.basis / "Spaces" / "a.md").read_text(encoding="utf-8")

    rc, text = _um(u, jetzt=T3)
    assert rc == 0, text
    assert {a.pfad: (u.basis / a.pfad).read_bytes() for a in plan} == erzaehlt
    assert {a.pfad: _manifest(u)[a.pfad] for a in plan} == eintraege
    assert not (u.ordner / ag.ARCHIV).exists()  # alles zurueckgelegt
    aufrufe.clear()
    e, pp = seitenlauf(erz)
    assert aufrufe == [] and e.aufrufe == 0 and set(e.zustaende.values()) == {"aktuell"} and pp == []

    # und noch einmal zurueck: wieder archiviert, die Sicherung von T3 gilt
    rc, text = _zu(u, jetzt=T4)
    assert rc == 0, text
    assert "doku_loop-sicherung-20261104-090000" in text


def test_archiv_ueberschreibt_keine_handarbeit(umg):
    u = umg
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, {"Was ist VibeMind.md": _seite("Was ist VibeMind")})
    assert _zu(u)[0] == 0
    hand = "# Meine eigene Seite\n"
    (u.basis / "Was ist VibeMind.md").write_text(hand, encoding="utf-8")
    rc, text = _um(u, jetzt=T3)
    assert rc == 0, text
    assert (u.basis / "Was ist VibeMind.md").read_text(encoding="utf-8") == hand
    assert "Was ist VibeMind.md" not in _manifest(u)
    assert (u.ordner / ag.ARCHIV / "seiten" / "Was ist VibeMind.md").is_file()  # bleibt im Archiv
    assert "von Hand" in text


# ---------- Fehler mitten im Ablauf ----------

def test_fehler_beim_setzen_laesst_nur_die_sicherung(umg):
    u = umg
    u.backend.fehler_beim_setzen = True
    vorher_zustand = _baum(u.ordner, mtime=True)
    vorher_rowboat = _baum(u.basis, mtime=True)
    rc, text = _um(u)
    assert rc == 1 and "Zugriff verweigert" in text
    assert u.backend.aktionen == [u.alt]
    assert _baum(u.ordner, mtime=True) == vorher_zustand  # auch keine aufgabe-alt.json
    assert _baum(u.basis, mtime=True) == vorher_rowboat
    assert (u.ordner.parent / "doku_loop-sicherung-20261103-180000").is_dir()
    # ein zweiter Versuch geht durch und legt eine neue Sicherung an
    u.backend.fehler_beim_setzen = False
    assert _um(u, jetzt=T3)[0] == 0
    assert (u.ordner.parent / "doku_loop-sicherung-20261104-090000").is_dir()


def test_fehler_nach_dem_setzen_bleibt_rueckholbar(umg):
    """Wirft das Setzen, obwohl die Aktion schon neu ist, bleibt aufgabe-alt.json, und `zurueck` stellt alles her."""
    u = umg
    u.backend.fehler_beim_setzen, u.backend.setzt_trotzdem = True, True
    vorher_zustand = _baum(u.ordner, mtime=True)
    rc, text = _um(u)
    assert rc == 1 and ag.ALT_DATEI in text
    assert (u.ordner / ag.ALT_DATEI).is_file()
    u.backend.fehler_beim_setzen = False
    rc, text = _zu(u)
    assert rc == 0, text
    assert u.backend.aktionen == [u.alt]
    assert _baum(u.ordner, ohne=("messbaum-*",), mtime=True) == \
        {k: v for k, v in vorher_zustand.items() if not k.startswith("messbaum")}


def test_zurueck_ohne_gemerkte_aktion_oder_sicherung(umg):
    u = umg
    u.backend.aktionen = [ag.neue_aktion(PY, u.repo)]
    vorher = _alles(u.tmp)
    rc, text = _zu(u)
    assert rc == 2 and ag.ALT_DATEI in text and _alles(u.tmp) == vorher and u.backend.gesetzt == []
    (u.ordner / ag.ALT_DATEI).write_text(json.dumps({"aktion": u.alt.als_dict()}), encoding="utf-8")
    vorher = _alles(u.tmp)
    rc, text = _zu(u)
    assert rc == 2 and "Sicherung" in text and _alles(u.tmp) == vorher and u.backend.gesetzt == []


def test_umschalten_nur_von_alt(umg):
    u = umg
    u.backend.aktionen = [ag.neue_aktion(PY, u.repo)]
    vorher = _alles(u.tmp)
    assert _um(u)[0] == 2 and _alles(u.tmp) == vorher
    u.backend.aktionen = [ag.Aktion("cmd.exe", "/c echo", str(u.repo))]
    assert _um(u)[0] == 2 and _alles(u.tmp) == vorher
    u.backend.aktionen = [dataclasses.replace(u.alt, working_directory=r"C:\anderes\repo")]
    assert _um(u)[0] == 2 and _alles(u.tmp) == vorher and u.backend.gesetzt == []


def test_zurueck_lehnt_fremde_aktion_ab(umg):
    u = umg
    assert _um(u)[0] == 0
    u.backend.aktionen = [ag.Aktion("cmd.exe", "/c echo", str(u.repo))]
    vorher, n = _alles(u.tmp), len(u.backend.gesetzt)
    rc, _ = _zu(u)
    assert rc == 2 and _alles(u.tmp) == vorher and len(u.backend.gesetzt) == n


def test_umschalten_prueft_den_import(umg):
    u = umg
    zeilen: list[str] = []
    vorher = _alles(u.tmp)
    rc = ag.umschalten(u.k, backend=u.backend, python=PY, jetzt=T1, aus=zeilen.append,
                       import_pruefen=lambda python, wurzel: "exit:1")
    assert rc == 2 and "import" in "\n".join(zeilen) and _alles(u.tmp) == vorher


# ---------- Pruefen ----------

def test_pruefen(umg):
    u = umg
    vorher = _alles(u.tmp)
    zeilen: list[str] = []
    assert ag.pruefen(u.k, backend=u.backend, aus=zeilen.append) == 0
    text = "\n".join(zeilen)
    assert "Loop: alt" in text and ALT_ARGS in text and "Sicherung: keine" in text and "Archiv" in text
    assert "Sperre" in text and "frei" in text and _alles(u.tmp) == vorher
    assert _um(u)[0] == 0
    zeilen.clear()
    ag.pruefen(u.k, backend=u.backend, aus=zeilen.append)
    text = "\n".join(zeilen)
    assert "Loop: neu" in text and "doku_loop-sicherung-20261103-180000" in text and "vorhanden" in text
    u.backend.aktionen = [ag.Aktion("cmd.exe", "/c echo", "C:\\")]
    zeilen.clear()
    ag.pruefen(u.k, backend=u.backend, aus=zeilen.append)
    assert "Loop: unbekannt" in "\n".join(zeilen)


# ---------- PowerShell-Backend (ohne echtes PowerShell) ----------

class FakePs:
    def __init__(self, ausgabe=None, rc=0, fehlertext=b""):
        self.ausgabe, self.rc, self.fehlertext = ausgabe, rc, fehlertext
        self.aufrufe: list = []

    def __call__(self, argv, **kw):
        self.aufrufe.append((argv, kw))
        if self.ausgabe is not None:
            Path(kw["env"]["VM_AUS"]).write_text(self.ausgabe, encoding="utf-8")
        kw["stderr"].write(self.fehlertext)
        return subprocess.CompletedProcess(argv, self.rc)


def test_powershell_backend_lesen_und_setzen():
    roh = json.dumps([{"Execute": "conhost.exe", "Arguments": ALT_ARGS, "WorkingDirectory": "C:\\w"}])
    ps = FakePs(roh)
    b = ag.PowerShellBackend(laufzeit=ps)
    assert b.lesen(ag.AUFGABE) == [ag.Aktion("conhost.exe", ALT_ARGS, "C:\\w")]
    neu = ag.neue_aktion(PY, "C:\\w")
    b.setzen(ag.AUFGABE, neu)
    for argv, kw in ps.aufrufe:
        assert argv[:4] == ["powershell", "-NoProfile", "-NonInteractive", "-Command"] and len(argv) == 5
        assert '"' not in argv[4]  # Werte nur ueber die Umgebung, nie in der Befehlszeile
        assert kw["stdin"] is subprocess.DEVNULL and kw["stdout"] is subprocess.DEVNULL
        assert kw["stderr"] not in (subprocess.PIPE, None) and 0 < kw["timeout"] <= 120
        assert kw["env"]["VM_AUFGABE"] == ag.AUFGABE and "capture_output" not in kw
    lesen, setzen = ps.aufrufe[0][0][4], ps.aufrufe[1][0][4]
    assert "Get-ScheduledTask" in lesen and "Set-ScheduledTask" not in lesen
    assert "Set-ScheduledTask" in setzen and "New-ScheduledTaskAction" in setzen
    env = ps.aufrufe[1][1]["env"]
    assert (env["VM_EXECUTE"], env["VM_ARGUMENTS"], env["VM_WD"]) == (neu.execute, neu.arguments,
                                                                        neu.working_directory)


def test_powershell_backend_fehler():
    with pytest.raises(ag.AufgabeFehler, match="nicht gefunden"):
        ag.PowerShellBackend(laufzeit=FakePs(rc=1, fehlertext="Aufgabe nicht gefunden\r\n".encode())).lesen("x")

    def zu_lang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    with pytest.raises(ag.AufgabeFehler, match="Zeitlimit"):
        ag.PowerShellBackend(laufzeit=zu_lang).setzen("x", ag.Aktion("a", "b", "c"))
    with pytest.raises(ag.AufgabeFehler):
        ag.PowerShellBackend(laufzeit=FakePs("kein json")).lesen("x")


@pytest.mark.skipif(os.name != "nt", reason="PowerShell-ScheduledTasks nur unter Windows")
def test_setzen_skript_baut_die_aktion_unveraendert():
    """Das echte Setzen-Skript bis vor `Set-ScheduledTask`: `New-ScheduledTaskAction` baut die Aktion nur im Speicher
    (keine Aufgabe wird beruehrt). Statt zu setzen, schreibt die Probe die gebaute Aktion zurueck -- Execute,
    Arguments (mit allen Anfuehrungszeichen) und WorkingDirectory muessen byte-gleich ankommen."""
    setzen = "Set-ScheduledTask -TaskName $env:VM_AUFGABE -Action $neu | Out-Null "
    assert setzen in ag._PS_SETZEN
    probe = ag._PS_SETZEN.replace(setzen, (
        "$j = ConvertTo-Json -Compress -InputObject ([ordered]@{ Execute = [string]$neu.Execute; "
        "Arguments = [string]$neu.Arguments; WorkingDirectory = [string]$neu.WorkingDirectory }); "
        "[IO.File]::WriteAllText($env:VM_AUS, $j, (New-Object System.Text.UTF8Encoding $false)) "))
    neu = ag.neue_aktion(r"C:\Users\User\ClaudeWork\wt-fungus-gate1\.venv-selbstbild\Scripts\python.exe",
                         r"C:\Users\User\Desktop\Vibemind_V1")
    roh = ag.PowerShellBackend()._ps(probe, {"VM_AUFGABE": "nie-gesetzt", "VM_EXECUTE": neu.execute,
                                             "VM_ARGUMENTS": neu.arguments, "VM_WD": neu.working_directory})
    assert ag.Aktion.aus_dict(json.loads(roh)) == neu


@braucht_vm
@pytest.mark.skipif(os.name != "nt", reason="Aufgabenplanung nur unter Windows")
def test_echte_aufgabe_nur_lesen():
    """Liest die echte Aufgabe (Get-ScheduledTask), aendert nichts."""
    aktionen = ag.PowerShellBackend().lesen(ag.AUFGABE)
    assert ag.einordnen(aktionen) in ("alt", "neu")


# ---------- CLI ----------

def test_cli_aufgabe(umg):
    u = umg
    vorlage = VORLAGE.read_text(encoding="utf-8")
    ersatz = {"rowboat_wissen:": f"rowboat_wissen: {u.k.rowboat_wissen.as_posix()}",
              "marker:": f"marker: {u.k.marker.as_posix()}"}
    vorlage = "\n".join(next((v for k, v in ersatz.items() if z.startswith(k)), z) for z in vorlage.splitlines())
    eingerueckt = "".join(f"  {z}\n" if z.strip() else "\n" for z in vorlage.splitlines())
    (u.repo / "selbstbild.yaml").write_text("name: probe\ndoku:\n" + eingerueckt, encoding="utf-8")
    w = {"aufgabe_backend": u.backend, "import_pruefen": lambda python, wurzel: None, "jetzt": lambda: T1}
    assert cli.main(["aufgabe", str(u.repo), "--pruefen"], werkzeuge=w) == 0
    vorher = _alles(u.tmp)
    assert cli.main(["aufgabe", str(u.repo), "--umschalten", "--trocken", "--python", PY], werkzeuge=w) == 0
    assert _alles(u.tmp) == vorher and u.backend.gesetzt == []
    assert cli.main(["aufgabe", str(u.repo), "--umschalten", "--python", PY], werkzeuge=w) == 0
    assert u.backend.aktionen == [ag.neue_aktion(PY, u.repo.resolve())]
    w["jetzt"] = lambda: T2
    assert cli.main(["aufgabe", str(u.repo), "--zurueck"], werkzeuge=w) == 0
    assert u.backend.aktionen == [u.alt]
    with pytest.raises(SystemExit):
        cli.main(["aufgabe", str(u.repo)], werkzeuge=w)  # ein Modus ist Pflicht


def test_cli_umschalten_braucht_doku_abschnitt(umg):
    u = umg
    (u.repo / "selbstbild.yaml").write_text("name: probe\n", encoding="utf-8")
    w = {"aufgabe_backend": u.backend, "import_pruefen": lambda python, wurzel: None}
    assert cli.main(["aufgabe", str(u.repo), "--umschalten", "--trocken", "--python", PY], werkzeuge=w) == 2
    assert u.backend.gesetzt == []



# =====================================================================================================================
# Fix-Runde 1
# =====================================================================================================================

import importlib  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402

from doku_vm import HAT_VM, REPO  # noqa: E402
from selbstbild.doku.graph import Graph, Kante, Knoten, kante_id  # noqa: E402

ALT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "alter_loop" / "doku_loop"
ERZEUGT = "2026-11-03T12:00:00+01:00"
EREIGNIS = "ereignis:a:a.start"


def _alter_graph(monkeypatch):
    """`graph`-Modul des ALTEN Loops, per Dateipfad geladen (nie ueber sys.path): mit SELBSTBILD_VIBEMIND aus dem
    aeusseren Repo, sonst die Fixture-Kopie. Ohne .pyc (das aeussere Repo bleibt unberuehrt)."""
    wurzel = REPO / "scripts" / "doku_loop" if HAT_VM else ALT_FIXTURE
    name = "_alter_doku_loop_vm" if HAT_VM else "_alter_doku_loop_fx"
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, wurzel / "__init__.py",
                                                      submodule_search_locations=[str(wurzel)])
        paket = importlib.util.module_from_spec(spec)
        sys.modules[name] = paket
        spec.loader.exec_module(paket)
    return importlib.import_module(name + ".graph")


def _graph_mit_ereignis() -> Graph:
    g = Graph({}, {}, "c" * 40, "d" * 40)
    g.knoten_hinzu(Knoten("space:a", "space", "a"))
    g.knoten_hinzu(Knoten("space:b", "space", "b"))
    g.knoten_hinzu(Knoten(EREIGNIS, "ereignis", "a.start", "a", "reg.yml", {"werkzeug": "starten"}))
    g.kante_hinzu(Kante(kante_id("ereignis_von", EREIGNIS, "space:a"), "ereignis_von", EREIGNIS, "space:a",
                        "extraktor"))
    g.kante_hinzu(Kante(kante_id("verbindet", "space:a", "space:b"), "verbindet", "space:a", "space:b", "seed"))
    # bekannte Kantenart, aber ein Endpunkt faellt weg -> muss mit
    g.kante_hinzu(Kante(kante_id("belegt_durch", "space:b", EREIGNIS), "belegt_durch", "space:b", EREIGNIS,
                        "extraktor"))
    return g


def _graph_ohne_ereignis() -> Graph:
    g = Graph({}, {}, "c" * 40, "d" * 40)
    g.knoten_hinzu(Knoten("space:a", "space", "a"))
    g.knoten_hinzu(Knoten("space:b", "space", "b"))
    g.kante_hinzu(Kante(kante_id("verbindet", "space:a", "space:b"), "verbindet", "space:a", "space:b", "seed"))
    return g


@pytest.fixture()
def umg_git(umg, git_repo):
    """Wie `umg`, aber die Wurzel ist ein Git-Repo auf dem Loop-Zweig mit committetem graph.json, das einen
    `ereignis`-Knoten des neuen Loops traegt; kein Torwaechter."""
    u = umg
    assert git_repo.wurzel == u.repo
    git_repo.git("checkout", "-q", "-b", u.k.zweig or "main")
    _graph_mit_ereignis().speichern(u.repo / "docs" / "graph" / "graph.json", ERZEUGT)
    (u.repo / ".gitignore").write_bytes(b"logs/\n")
    git_repo.commit("seed")
    u.k = dataclasses.replace(u.k, torwaechter=())
    u.git = git_repo
    return u


def _ohne_git(tmp: Path) -> dict[str, object]:
    return {k: v for k, v in _alles(tmp).items() if not k.startswith("repo/.git/") and k != "repo/.git"}


# ---------- T10-d: alte Typen ----------

def test_alte_typen_aus_ast_sonst_kopie(tmp_path):
    typen, quelle = ag.alte_typen(tmp_path)  # keine Datei -> eingebaute Kopie
    assert quelle.startswith("Kopie")
    assert typen == ag.ALTE_TYPEN_KOPIE
    ziel = tmp_path / "scripts" / "doku_loop" / "graph.py"
    ziel.parent.mkdir(parents=True)
    shutil.copyfile(ALT_FIXTURE / "graph.py", ziel)
    typen, quelle = ag.alte_typen(tmp_path)
    assert quelle.startswith("ast") and typen == ag.ALTE_TYPEN_KOPIE  # die Kopie stimmt mit dem alten Loop
    ziel.write_text(ziel.read_text(encoding="utf-8").replace('"komponente",', ""), encoding="utf-8")
    typen, _ = ag.alte_typen(tmp_path)
    assert "komponente" not in typen["KNOTEN_TYPEN"] and "kette" in typen["KNOTEN_TYPEN"]
    ziel.write_text("KNOTEN_TYPEN = frozenset(irgendwas())\n", encoding="utf-8")  # kein Literal -> Kopie
    assert ag.alte_typen(tmp_path)[1].startswith("Kopie")


@braucht_vm
def test_fixture_gleich_dem_alten_loop():
    echt = (REPO / "scripts" / "doku_loop" / "graph.py").read_bytes().replace(b"\r\n", b"\n")
    assert (ALT_FIXTURE / "graph.py").read_bytes().replace(b"\r\n", b"\n") == echt  # autocrlf-fest
    typen, quelle = ag.alte_typen(REPO)
    assert typen == ag.ALTE_TYPEN_KOPIE and quelle.startswith("ast")


def test_graph_bereinigen_rein(monkeypatch):
    alt_mod = _alter_graph(monkeypatch)
    text = ag._graph_json_text(_graph_mit_ereignis(), ERZEUGT)
    with pytest.raises(ValueError, match="ereignis"):
        alt_mod.Graph.aus_text(text)  # genau das liesse jeden alten Tick sterben
    plan = ag.graph_bereinigen_plan(text.encode("utf-8"), ag.ALTE_TYPEN_KOPIE)
    assert plan.knoten_weg == {"ereignis": 1} and plan.kanten_weg == {"ereignis_von": 1, "belegt_durch": 1}
    assert plan.neu == ag._graph_json_text(_graph_ohne_ereignis(), ERZEUGT)
    g = alt_mod.Graph.aus_text(plan.neu)
    assert set(g.knoten) == {"space:a", "space:b"} and len(g.kanten) == 1
    assert ag.graph_bereinigen_plan(plan.neu.encode("utf-8"), ag.ALTE_TYPEN_KOPIE).neu is None  # nichts mehr zu tun


# ---------- T10-d im Rueckweg ----------

def test_zurueck_bereinigt_graph_und_committet(umg_git, monkeypatch):
    u = umg_git
    alt_mod = _alter_graph(monkeypatch)
    graph = u.repo / "docs" / "graph" / "graph.json"
    kopf_vorher = u.git.git("rev-parse", "HEAD")
    assert _um(u)[0] == 0
    rc, text = _zu(u)
    assert rc == 0, text
    assert u.backend.aktionen == [u.alt]
    g = alt_mod.Graph.aus_text(graph.read_text(encoding="utf-8"))  # der alte Loop liest ihn wieder
    assert set(g.knoten) == {"space:a", "space:b"}
    assert u.git.git("log", "-1", "--format=%s") == ag.GRAPH_BETREFF
    assert u.git.git("rev-parse", "HEAD~1") == kopf_vorher
    assert u.git.git("show", "--name-only", "--format=", "HEAD").splitlines() == ["docs/graph/graph.json"]
    assert "Lauf: rueckweg-20261103-183000" in u.git.git("log", "-1", "--format=%B")
    assert u.git.git("status", "--porcelain") == ""
    assert "ereignis (1)" in text and "ereignis_von (1)" in text
    assert ag.graph_bereinigen_plan(graph.read_bytes(), ag.ALTE_TYPEN_KOPIE).neu is None


def test_zurueck_verweigert_bei_fremder_aenderung_am_graph(umg_git):
    u = umg_git
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    graph = u.repo / "docs" / "graph" / "graph.json"
    graph.write_bytes(graph.read_bytes().replace(b'"name": "b"', b'"name": "b-von-hand"'))
    vorher, gesetzt, kopf = _ohne_git(u.tmp), list(u.backend.gesetzt), u.git.git("rev-parse", "HEAD")
    rc, text = _zu(u)
    assert rc == 2 and "graph.json" in text and "fremd" in text
    assert _ohne_git(u.tmp) == vorher and u.backend.gesetzt == gesetzt
    assert u.backend.aktionen == [ag.neue_aktion(PY, u.repo)] and u.git.git("rev-parse", "HEAD") == kopf
    rc, text = _zu(u, trocken=True)
    assert rc == 2 and "fremd" in text


def test_zurueck_verweigert_auf_falschem_zweig(umg_git):
    u = umg_git
    assert _um(u)[0] == 0
    u.git.git("checkout", "-q", "-b", "anderer-zweig")
    vorher = _ohne_git(u.tmp)
    rc, text = _zu(u)
    assert rc == 2 and "falscher_zweig" in text and _ohne_git(u.tmp) == vorher
    assert u.backend.aktionen == [ag.neue_aktion(PY, u.repo)]


def test_pruefen_und_trocken_nennen_unbekannte_arten(umg_git):
    u = umg_git
    zeilen: list[str] = []
    ag.pruefen(u.k, backend=u.backend, aus=zeilen.append)
    text = "\n".join(zeilen)
    assert "ereignis (1)" in text and "ereignis_von (1)" in text and "--zurueck" in text
    assert _um(u)[0] == 0
    vorher, kopf = _ohne_git(u.tmp), u.git.git("rev-parse", "HEAD")
    rc, text = _zu(u, trocken=True)
    assert rc == 0, text
    assert "ereignis (1)" in text and ag.GRAPH_BETREFF in text
    assert _ohne_git(u.tmp) == vorher and u.git.git("rev-parse", "HEAD") == kopf


# ---------- Reihenfolge: c, b, Graph, zuletzt a ----------

def test_zurueck_setzt_die_alte_aktion_zuletzt(umg_git):
    u = umg_git
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    (u.ordner / f"stand-{u.h}.json").write_text('{"lauf_nr": 99}\n', encoding="utf-8")
    beobachtet = []

    def pruefen(aktion):
        beobachtet.append((aktion == u.alt,
                           not (u.basis / "Zustand.md").exists(),
                           (u.ordner / f"stand-{u.h}.json").read_text(encoding="utf-8") == '{"lauf_nr": 7}\n',
                           b"ereignis" not in (u.repo / "docs" / "graph" / "graph.json").read_bytes(),
                           u.git.git("log", "-1", "--format=%s") == ag.GRAPH_BETREFF))
    u.backend.beim_setzen = pruefen
    assert _zu(u)[0] == 0
    assert beobachtet == [(True, True, True, True, True)]


@pytest.mark.parametrize("ort", ["_seiten_archivieren", "_zurueckspielen", "_graph_bereinigen"])
def test_fehler_vor_dem_setzen_laesst_den_neuen_loop_und_ist_wiederholbar(umg_git, monkeypatch, ort):
    u = umg_git
    zustand_vorher = _baum(u.ordner, ohne=("messbaum-*",), mtime=True)
    rowboat_vorher = _baum(u.basis, mtime=True)
    assert _um(u)[0] == 0
    neu = ag.neue_aktion(PY, u.repo)
    _neuer_loop_schreibt_seiten(u, EIGENE)
    original = getattr(ag, ort)

    def kaputt(*a, **kw):
        raise OSError("Platte voll")
    monkeypatch.setattr(ag, ort, kaputt)
    rc, text = _zu(u)
    assert rc == 1 and "erneut" in text
    assert u.backend.aktionen == [neu] and u.backend.gesetzt == [neu]  # der neue Loop bleibt geplant
    assert (u.ordner / ag.ALT_DATEI).is_file()
    monkeypatch.setattr(ag, ort, original)
    rc, text = _zu(u, jetzt=T3)
    assert rc == 0, text
    assert u.backend.aktionen == [u.alt]
    assert _baum(u.ordner, ohne=("messbaum-*", ag.ARCHIV), mtime=True) == zustand_vorher
    assert _baum(u.basis, mtime=True) == rowboat_vorher
    assert b"ereignis" not in (u.repo / "docs" / "graph" / "graph.json").read_bytes()


def test_graph_commit_scheitert_setzt_nicht(umg_git, monkeypatch):
    """Der gemeinsame Schreibvorgang setzt zurueck (hier: Torwaechter nachher kaputt): exit 1, alte Aktion NICHT
    gesetzt, graph.json unveraendert und sauber."""
    u = umg_git
    assert _um(u)[0] == 0
    from selbstbild.doku import umschreiben as us

    aufrufe = [0]

    def drift(repo):
        aufrufe[0] += 1
        if aufrufe[0] > 1:
            raise us.TorwaechterFehler("waechter kaputt")
        return set()
    monkeypatch.setattr(us, "drift_fuer", lambda konfig: drift)
    graph_vorher = (u.repo / "docs" / "graph" / "graph.json").read_bytes()
    rc, text = _zu(u)
    assert rc == 1 and "Torwächter" in text
    assert u.backend.aktionen == [ag.neue_aktion(PY, u.repo)]
    assert (u.repo / "docs" / "graph" / "graph.json").read_bytes() == graph_vorher
    assert u.git.git("status", "--porcelain") == ""


# ---------- Minor 1, 2: Manifest nach jeder Seite ----------

def _ersetzen_scheitert_beim(monkeypatch, n):
    original, zaehler = ag._ersetzen, [0]

    def ersetzen(quelle, ziel):
        zaehler[0] += 1
        if zaehler[0] == n:
            raise OSError("Platte voll")
        original(quelle, ziel)
    monkeypatch.setattr(ag, "_ersetzen", ersetzen)
    return original


def _stimmig(u):
    """Keine Seite ohne Eintrag, kein Eintrag ohne Seite (im Notizordner wie im Archiv)."""
    m = _manifest(u)
    for rel in EIGENE:
        assert (rel in m) == (u.basis / rel).is_file(), rel
    archiv = rb._manifest_laden(u.ordner / ag.ARCHIV / ag.ARCHIV_MANIFEST)
    archiv = archiv if isinstance(archiv, dict) else {}
    for rel in archiv:
        assert (u.ordner / ag.ARCHIV / ag.ARCHIV_SEITEN / rel).is_file(), rel
    return m, archiv


def test_archivieren_halb_bleibt_stimmig(umg, monkeypatch):
    u = umg
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    original = _ersetzen_scheitert_beim(monkeypatch, 2)
    assert _zu(u)[0] == 1
    m, archiv = _stimmig(u)
    assert len(archiv) == 1 and not (set(archiv) & set(m))
    monkeypatch.setattr(ag, "_ersetzen", original)
    assert _zu(u, jetzt=T3)[0] == 0
    m, archiv = _stimmig(u)
    assert set(archiv) == set(EIGENE)


def test_zuruecklegen_halb_bleibt_stimmig(umg, monkeypatch):
    u = umg
    assert _um(u)[0] == 0
    _neuer_loop_schreibt_seiten(u, EIGENE)
    assert _zu(u)[0] == 0
    _ersetzen_scheitert_beim(monkeypatch, 2)
    rc, text = _um(u, jetzt=T3)
    assert rc == 1
    m, archiv = _stimmig(u)
    assert len([r for r in EIGENE if r in m]) == 1 and len(archiv) == 3 and not (set(archiv) & set(m))


# ---------- Minor 3: die gemerkte Sicherung ----------

def test_zurueck_nimmt_die_gemerkte_sicherung(umg):
    u = umg
    zustand_vorher = _baum(u.ordner, ohne=("messbaum-*",), mtime=True)
    assert _um(u)[0] == 0
    juenger = u.ordner.parent / "doku_loop-sicherung-20261103-190000"
    juenger.mkdir()
    (juenger / "falsch.json").write_text("{}", encoding="utf-8")
    rc, text = _zu(u, jetzt=T3)
    assert rc == 0, text
    assert "Warnung" in text and "doku_loop-sicherung-20261103-180000" in text and juenger.name in text
    assert _baum(u.ordner, ohne=("messbaum-*", ag.ARCHIV), mtime=True) == zustand_vorher


def test_zurueck_ohne_die_gemerkte_sicherung_nimmt_die_neueste_mit_warnung(umg):
    u = umg
    assert _um(u)[0] == 0
    gemerkt = u.ordner.parent / "doku_loop-sicherung-20261103-180000"
    gemerkt.rename(u.ordner.parent / "doku_loop-sicherung-20261103-175900")
    rc, text = _zu(u, jetzt=T3)
    assert rc == 0, text
    assert "Warnung" in text and "fehlt" in text


# ---------- Minor 4: welches Python ----------

def test_umschalten_ohne_python_nimmt_das_feste(umg):
    u = umg
    gesehen: list[str] = []
    zeilen: list[str] = []
    rc = ag.umschalten(u.k, backend=u.backend, jetzt=T1, aus=zeilen.append, trocken=True,
                       import_pruefen=lambda python, wurzel: gesehen.append(python))
    assert rc == 0 and gesehen == [ag.STANDARD_PYTHON]
    assert ag.neue_aktion(ag.STANDARD_PYTHON, u.repo).arguments in "\n".join(zeilen)
    assert ag.STANDARD_PYTHON == r"C:\Users\User\ClaudeWork\wt-fungus-gate1\.venv-selbstbild\Scripts\python.exe"


def test_umschalten_mit_fehlendem_python_abgelehnt(umg):
    u = umg
    vorher = _alles(u.tmp)
    zeilen: list[str] = []
    rc = ag.umschalten(u.k, backend=u.backend, python=str(u.tmp / "gibt es nicht" / "python.exe"), jetzt=T1,
                       aus=zeilen.append)  # echte Import-Pruefung
    assert rc == 2 and "python" in "\n".join(zeilen).lower() and _alles(u.tmp) == vorher
    assert u.backend.gesetzt == []


def test_cli_ohne_python_nimmt_das_feste_nicht_sys_executable(umg):
    u = umg
    vorlage = VORLAGE.read_text(encoding="utf-8")
    ersatz = {"rowboat_wissen:": f"rowboat_wissen: {u.k.rowboat_wissen.as_posix()}",
              "marker:": f"marker: {u.k.marker.as_posix()}"}
    vorlage = "\n".join(next((v for k, v in ersatz.items() if z.startswith(k)), z) for z in vorlage.splitlines())
    (u.repo / "selbstbild.yaml").write_text("name: probe\ndoku:\n" + "".join(
        f"  {z}\n" if z.strip() else "\n" for z in vorlage.splitlines()), encoding="utf-8")
    gesehen: list[str] = []
    w = {"aufgabe_backend": u.backend, "import_pruefen": lambda python, wurzel: gesehen.append(python)}
    assert cli.main(["aufgabe", str(u.repo), "--umschalten", "--trocken"], werkzeuge=w) == 0
    assert gesehen == [ag.STANDARD_PYTHON]


# ---------- Minor 5: Aufraeumfehler der Temp-Ordner ----------

def test_temp_aufraeumfehler_wird_aufgabefehler(monkeypatch):
    class Ordner:
        def __init__(self, *a, **kw):
            import tempfile
            self._echt = tempfile.mkdtemp(prefix="selbstbild-aufgabe-test-")

        def __enter__(self):
            return self._echt

        def __exit__(self, *exc):
            shutil.rmtree(self._echt, ignore_errors=True)
            raise PermissionError("Datei in Benutzung")

    monkeypatch.setattr(ag.tempfile, "TemporaryDirectory", Ordner)
    roh = json.dumps([{"Execute": "conhost.exe", "Arguments": ALT_ARGS, "WorkingDirectory": "C:\\w"}])
    with pytest.raises(ag.AufgabeFehler, match="Temp"):
        ag.PowerShellBackend(laufzeit=FakePs(roh)).lesen(ag.AUFGABE)


# ---------- Minor 7: Rueckmeldung weicht ohne Ausnahme ab ----------

def test_rueckmeldung_weicht_ab_setzen_wirkt_nicht(umg):
    u = umg
    u.backend.wirkung = "nichts"
    vorher = _baum(u.ordner, mtime=True)
    rc, text = _um(u)
    assert rc == 1 and "andere Aktion" in text
    assert u.backend.aktionen == [u.alt] and _baum(u.ordner, mtime=True) == vorher  # ohne aufgabe-alt.json
    assert (u.ordner.parent / "doku_loop-sicherung-20261103-180000").is_dir()


def test_rueckmeldung_weicht_ab_dritte_aktion(umg):
    u = umg
    dritte = ag.Aktion("conhost.exe", '--headless cmd /c "normalisiert"', str(u.repo))
    u.backend.wirkung = dritte
    rc, text = _um(u)
    assert rc == 1 and u.backend.aktionen == [dritte]
    gemerkt = json.loads((u.ordner / ag.ALT_DATEI).read_text(encoding="utf-8"))
    assert ag.Aktion.aus_dict(gemerkt["aktion"]) == u.alt  # die alte Aktion bleibt gemerkt
    assert (u.ordner.parent / "doku_loop-sicherung-20261103-180000").is_dir()


def test_zurueck_rueckmeldung_weicht_ab(umg):
    u = umg
    assert _um(u)[0] == 0
    u.backend.wirkung = "nichts"
    rc, text = _zu(u)
    assert rc == 1 and u.backend.aktionen == [ag.neue_aktion(PY, u.repo)]
    assert (u.ordner / ag.ALT_DATEI).is_file()  # wiederholbar
    u.backend.wirkung = None
    assert _zu(u, jetzt=T3)[0] == 0 and u.backend.aktionen == [u.alt]
