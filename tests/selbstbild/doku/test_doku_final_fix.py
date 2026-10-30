"""doku_loop: Final-Fix-Welle (Ruling 57) -- Rowboat-Schalter (I4) und Minors 13, 15, 16, 19, 22.

Wie die Tick-Tests gegen ein `git init`-Fixture unter tmp_path; Judge, Budget und Torwaechter sind Fakes, der echte
claude-Aufruf und die echten Budget-Dateien werden nie angefasst.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden  # noqa: E402,F401
from selbstbild.doku import __main__ as cli  # noqa: E402
from selbstbild.doku import anker  # noqa: E402
from selbstbild.doku import judge as jg  # noqa: E402
from selbstbild.doku import ketten_seed as ks  # noqa: E402
from selbstbild.doku import konfig as konfig_modul  # noqa: E402
from selbstbild.doku import rowboat as rb  # noqa: E402
from selbstbild.doku import stand as st  # noqa: E402
from selbstbild.doku import tick as tk  # noqa: E402
from selbstbild.doku.anker import Beleg  # noqa: E402
from test_doku_judge import NOCH_WAHR, _frage  # noqa: E402
from test_doku_judge import FakeBudget as JudgeBudget  # noqa: E402
from test_doku_judge import FakeLaufzeit as JudgeLaufzeit  # noqa: E402
from test_doku_rowboat import LAUF, _aid_aussage, _anwenden, _punkt  # noqa: E402
from test_doku_tick import (  # noqa: E402,F401
    AUSSAGE, FakeBudget, FakeLaufzeit, FakeManifestBauer, _anstoss, _commit, _head, _jetzt, _konfig_mit, _schreib,
    _seed, _text, _tick, _UrteileZaehler, fx,
)

SCRIPTS = Path(tk.__file__).resolve().parents[2]  # src/ mit dem Paket selbstbild


def _lauf(fx):
    return tk._Lauf(fx.konfig, jetzt=_jetzt(), erzwingen=False, laufzeit=FakeLaufzeit(), ohne_llm=False, budget=None,
                    drift=lambda r: set(), manifest_bauer=FakeManifestBauer(), nach_tick=None,
                    melden=lambda z: None)


# ---------- I4: Rowboat-Notizen per Schalter ----------

def test_i4_rowboat_notizen_per_schalter_aus(fx, monkeypatch, capsys):
    _konfig_mit(fx, rowboat_notizen_aktiv=False)
    _seed(fx)
    urteile = _UrteileZaehler(monkeypatch)
    _anstoss(fx)
    erg = _tick(fx)
    assert erg.uebersprungen in (None, "keine_aenderung") and urteile.n >= 1  # Urteile werden gelesen
    doku = fx.konfig.rowboat_wissen / fx.konfig.rowboat_notizen
    assert not doku.exists()
    w = {"budget": FakeBudget(), "aufgabe_abfragen": lambda: "Bereit"}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "status"], werkzeuge=w) == 0
    assert "Rowboat-Notizen: aus" in capsys.readouterr().out
    _konfig_mit(fx, rowboat_notizen_aktiv=True)
    _anstoss(fx)
    _tick(fx)
    assert doku.is_dir()
    assert cli.main(["--konfig", str(fx.konfig_pfad), "status"], werkzeuge=w) == 0
    assert "Rowboat-Notizen: an" in capsys.readouterr().out


@braucht_vm
def test_i4_echte_konfig_hat_die_notizen_aus():
    k = vm_laden()
    assert k.rowboat_notizen_aktiv is False
    assert "1aac875c" in (REPO / "config" / "doku_loop.yaml").read_text(encoding="utf-8")


# ---------- Minor 13: Rotation ----------

def test_m13_log_rotation_ohne_offenen_strom(tmp_path):
    log = tmp_path / "logs" / "doku-loop.log"
    log.parent.mkdir()
    log.write_bytes(b"x" * 100)
    log.with_name("doku-loop.log.1").write_bytes(b"ganz alt")
    assert cli.log_rotieren(log, max_bytes=50) == "umbenannt"
    assert not log.exists() and log.with_name("doku-loop.log.1").read_bytes() == b"x" * 100
    log.write_bytes(b"y" * 10)
    assert cli.log_rotieren(log, max_bytes=50) is None and log.read_bytes() == b"y" * 10
    assert cli.log_rotieren(tmp_path / "fehlt.log", max_bytes=1) is None


def test_m13_log_rotation_waehrend_die_aufgabe_ins_log_umleitet(tmp_path):
    """Die Aufgabe haengt mit `>> logs\\doku-loop.log 2>&1` an: das Log ist waehrend des Laufs offen und laesst sich
    unter Windows nicht umbenennen. Der Einstieg kopiert es nach `.1` und kuerzt es ueber den eigenen Strom."""
    log = tmp_path / "doku-loop.log"
    log.write_bytes(b"alter Lauf\n" * 20)
    skript = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); from pathlib import Path; "
              f"from selbstbild.doku.__main__ import log_rotieren; "
              f"print(log_rotieren(Path({str(log)!r}), max_bytes=50)); print('neuer Lauf')")
    with open(log, "ab") as strom:
        lauf = subprocess.run([sys.executable, "-c", skript], stdout=strom, stderr=subprocess.STDOUT, timeout=60)
    assert lauf.returncode == 0
    neu = log.read_bytes()
    assert neu.replace(b"\r\n", b"\n") == b"gekuerzt\nneuer Lauf\n" and b"\0" not in neu
    assert log.with_name("doku-loop.log.1").read_bytes() == b"alter Lauf\n" * 20


def test_m13_tick_rotiert_das_log_vor_dem_lauf(fx, monkeypatch):
    gesehen = []
    monkeypatch.setattr(cli, "log_rotieren", lambda pfad, **kw: gesehen.append(pfad))
    w = {"budget": FakeBudget(), "laufzeit": FakeLaufzeit(), "drift": lambda r: set(),
         "manifest_bauer": FakeManifestBauer(), "jetzt": _jetzt}
    assert cli.main(["--konfig", str(fx.konfig_pfad), "tick"], werkzeuge=w) == 0
    assert gesehen == [fx.repo / "logs" / "doku-loop.log"]


def test_m13_urteils_cache_wird_gekuerzt(fx, monkeypatch):
    """Regel (Docstring `urteile_kuerzen`): je Schluessel zaehlt nur der letzte Eintrag; von den uebrigen verliert die
    aeltere Haelfte alle Eintraege ohne Urteil; die juengere Haelfte bleibt ganz."""
    k = fx.konfig
    for i in range(40):
        j = i % 10
        eintrag = {"urteil": "noch_wahr", "neuer_text": None, "beleg_anker": None, "begruendung": "b",
                   "zeit": "z"} if i % 2 == 0 else {"grund": "uneinig", "zeit": "z"}
        st.urteil_merken(k, f"a{j}", ("symbol", "f.py", "x"), f"alt{j}", f"neu{i}", eintrag)
    pfad = st.urteile_pfad(k)
    vorher = pfad.read_bytes()
    assert st.urteile_kuerzen(k, max_bytes=len(vorher) + 1) == 0 and pfad.read_bytes() == vorher
    assert st.urteile_kuerzen(k, max_bytes=100) == 32
    zeilen = [json.loads(z) for z in pfad.read_text(encoding="utf-8").splitlines()]
    assert [z["neu_digest"] for z in zeilen] == [f"neu{i}" for i in (30, 32, 34, 35, 36, 37, 38, 39)]
    assert st.urteil_holen(k, "a0", ("symbol", "f.py", "x"), "alt0", "neu30") is not None  # gebrauchtes Urteil
    assert st.urteil_holen(k, "a1", ("symbol", "f.py", "x"), "alt1", "neu31") is None  # alter Fehlschlag weg
    assert st.urteil_holen(k, "a5", ("symbol", "f.py", "x"), "alt5", "neu35") is not None
    aufrufe = []
    monkeypatch.setattr(tk, "urteile_kuerzen", lambda konfig: aufrufe.append(konfig) or 0)
    _tick(fx)
    assert aufrufe == [k]


# ---------- Minor 15: Judge-Quelltext fuer yaml_key und datei ----------

def test_m15_judge_quelltext_yaml_nur_knoten_datei_nur_ausschnitt(fx):
    _schreib(fx.repo, "tools/gross.txt", "".join(f"zeile_{i} = {i}\n" for i in range(1, 201)))
    _schreib(fx.repo, "cfg/x.yaml", "a:\n  b:\n    c: 1\n  geheim: nicht_relevant\n"
                                    + "".join(f"x{i}: {i}\n" for i in range(100)))
    head = _commit(fx.repo, "dateien", "tools/gross.txt", "cfg/x.yaml")
    lauf = _lauf(fx)
    lauf.kopf_aussen = head
    y = Beleg("yaml_key", "cfg/x.yaml", "a.b", "d", head, 2)
    for alt in (False, True):
        q = lauf.quelltext(y, alt=alt)
        assert '"c": 1' in q and "a.b" in q and "nicht_relevant" not in q and "x99" not in q
    d = Beleg("datei", "tools/gross.txt", "", "d", head, 100)
    for alt in (False, True):
        q = lauf.quelltext(d, alt=alt)
        assert "zeile_60 = 60\n" in q and "zeile_100 = 100\n" in q and "zeile_140 = 140\n" in q
        assert "zeile_59 = 59\n" not in q and "zeile_141 = 141\n" not in q and "zeile_1 = 1\n" not in q
    assert "zeile_200 = 200" in lauf.quelltext(Beleg("datei", "tools/gross.txt", "", "d", head, None), alt=False)
    assert lauf.quelltext(Beleg("yaml_key", "cfg/x.yaml", "a.fehlt", "d", head, 2), alt=False) == "(Knoten fehlt)"


# ---------- Minor 16: Korrektur aus Urteile.md ----------

@pytest.mark.parametrize("text", [
    "Der Client " + "sehr " * 50 + "lang.",  # mehr als doppelt so lang wie die alte Aussage
    "Der Client ruft <script>alert(1)</script> auf.",
    "Siehe [hier](javascript:alert(1)) beim Client.",
    "Bild <img src=x onerror=alert(1)> beim Client.",
    "Rahmen <IFRAME src=x> beim Client.",
])
def test_m16_korrektur_laenge_und_unsicherer_inhalt(fx, text):
    _seed(fx)
    aid = _aid_aussage(fx)
    vorher = _head(fx.repo)
    pp = _anwenden(fx, [rb.Urteil(LAUF, aid, "korrektur", text)], offene=[_punkt(aussage_id=aid)])
    assert [p.art for p in pp] == ["korrektur_abgelehnt"]
    assert _head(fx.repo) == vorher and AUSSAGE in _text(fx)
    vermerke = rb.vermerke_laden(rb.erledigt_pfad(fx.konfig))
    assert vermerke and vermerke[-1].get("ergebnis") == "abgelehnt"


def test_m16_korrektur_hoechstens_2000_zeichen(fx):
    _seed(fx)
    aid = _aid_aussage(fx)
    aussage = next(a for a in rb.aussagen_ohne_belege("docs/spaces/KETTEN.md", _text(fx)) if a.id == aid)
    lang = dataclasses.replace(aussage, text="x" * 1500)
    assert rb.korrektur_pruefen(lang, "Der Client " + "y" * 2100) is not None
    assert rb.korrektur_pruefen(lang, "Der Client schickt den Auftrag.") is None


# ---------- Minor 19: Judge nur ueber die Abo-Lane ----------

def test_m19_judge_umgebung_ohne_api_schluessel(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lokal"))
    weg = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
           "CLAUDE_CODE_USE_VERTEX")
    for name in weg:
        monkeypatch.setenv(name, "gesetzt")
    monkeypatch.setenv("DOKU_LOOP_TEST_BLEIBT", "ja")
    konfig = dataclasses.replace(vm_laden(), claude_bin=Path("C:/nirgends/claude.exe"))
    lauf = JudgeLaufzeit(NOCH_WAHR)
    antwort, grund = jg.befragen(_frage(), konfig, laufzeit=lauf, budget=JudgeBudget())
    assert grund == "ok"
    _, kw = lauf.aufrufe[0]
    assert not {k.upper() for k in kw["env"]} & set(weg)
    assert kw["env"]["DOKU_LOOP_TEST_BLEIBT"] == "ja"
    assert os.environ["ANTHROPIC_API_KEY"] == "gesetzt"  # die eigene Umgebung bleibt unberuehrt


# ---------- Minor 21: Konfig ----------

@pytest.mark.parametrize("eintrag,schluessel", [({"wurzel": "sp/a"}, "mcp_muster"),
                                                ({"mcp_muster": "werkzeuge"}, "wurzel")])
def test_m21_fehlender_space_schluessel_ist_konfigfehler(fx, eintrag, schluessel, capsys):
    import yaml

    roh = yaml.safe_load(fx.konfig_pfad.read_text(encoding="utf-8"))
    roh["spaces"]["a"] = eintrag
    fx.konfig_pfad.write_text(yaml.safe_dump(roh, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(konfig_modul.KonfigFehler, match=f"Space a: Schluessel {schluessel} fehlt"):
        konfig_modul.laden(fx.konfig_pfad)
    assert cli.main(["--konfig", str(fx.konfig_pfad), "status"]) == 2
    assert f"Schluessel {schluessel} fehlt" in capsys.readouterr().err


# ---------- Minor 22: Belegpfade mit `..` oder absolut ----------

def test_m22_unsichere_belegpfade_werden_vor_dem_lesen_abgelehnt(fx):
    for p in ("../x.py", "a/../../x.py", "/etc/passwd", "C:/x.py", "C:x.py", "a\\..\\..\\x.py", "\\\\srv\\x.py", ""):
        assert not anker.pfad_sicher(p), p
    for p in ("a/b.py", "docs/spaces/KETTEN.md", "a..b/c.py", "x..py", "./a.py"):
        assert anker.pfad_sicher(p), p
    (fx.repo.parent / "aussen.txt").write_text("geheim\n", encoding="utf-8")
    b = Beleg("datei", "../aussen.txt", "", "d", "c", 1)
    with pytest.raises(anker.UnsichererPfad):
        anker.beleg_pruefen(b, {}, fx.repo)
    stelle = ks.Stelle("../aussen.txt", 1, None, "voll", "`../aussen.txt:1`")
    v = ks.stelle_verankern_mit_grund(stelle, fx.konfig, {}, {}, "c")
    assert v.beleg is None and v.pruefpunkt == "pfad_unsicher"
    assert ks._voll_aufloesen("../aussen.txt", fx.repo, None) == "../aussen.txt"
    lauf = _lauf(fx)
    lauf.versioniert = frozenset({"../aussen.txt"})
    assert lauf.urteil(b) == "beleg_ok"  # kein Signal
    assert [p.art for p in lauf.offen] == ["beleg_pfad_unsicher"]


def test_m22_aussage_mit_pfad_nach_aussen_wird_nicht_verankert(fx):
    (fx.repo.parent / "aussen.py").write_text("def geheim():\n    return 1\n", encoding="utf-8")
    _schreib(fx.repo, "docs/spaces/KETTEN.md", _text(fx) + "\nDer Dienst liest `../aussen.py:1` beim Start und "
                                                            "meldet das Ergebnis.\n")
    _commit(fx.repo, "aussage nach aussen", "docs/spaces/KETTEN.md")
    _seed(fx)
    from selbstbild.doku.graph import Graph

    belege = {b.file for k in Graph.laden(fx.repo / "docs/graph/graph.json").kanten.values() for b in k.belege}
    assert not any(".." in f for f in belege)
    liste = tk.pruefliste_laden(fx.konfig)
    assert any(p["art"] == "pfad_unsicher" for p in liste["struktur"] + liste["offen"])
