from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from selbstbild.analyse.doppelt import Doppelung
from selbstbild.analyse.ergebnis import Analyse, speichern
from selbstbild.analyse.lader import ANALYSE_VERSION
from selbstbild.analyse.lauf import parameter
from selbstbild.analyse.tot import TotFund
from selbstbild.doku.anker import Beleg
from selbstbild.doku.extraktoren.ereignisse import Ereignis, abgleich
from selbstbild.doku.konfig import aus_regeln
from selbstbild.doku.pruefliste import pruefliste_pfad
from selbstbild.funde.goldsatz_tor import ToterCodeTor
from selbstbild.funde.sammeln import NICHT_PRODUKTIV, abweichend_ist_werkzeug, funde_sammeln, ist_nicht_produktiv
from selbstbild.funde.typen import fund_schluessel
from selbstbild.speicher import pfad_kennung

OFFEN = ToterCodeTor(True, 30, 28, "offen: 28 von 30 bestätigt")
ZU = ToterCodeTor(False, 0, 0, "gesperrt: 0/0 beurteilt, mindestens 30 nötig")
COMMIT = "c0ffee"


@pytest.fixture
def umgebung(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    heim = tmp_path / "heim"
    monkeypatch.setenv("SELBSTBILD_HOME", str(heim))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lokal"))
    wurzel = tmp_path / "repo"
    wurzel.mkdir()
    (wurzel / "selbstbild.yaml").write_text(
        "name: aussen\ndoku:\n  spaces: {}\n  ausschluss_namen: []\n  ausschluss_pfade: []\n  ketten_md: ''\n"
        "  compose_live: []\n  sql_wurzeln: ['.']\n  dienste_yaml: ''\n  waechter_zustandsdatei: ''\n"
        "  waechter_zuordnung: {}\n  tick_minuten: 5\n  marker: marker\n  agent: ''\n  judge_modell: sonnet\n"
        "  claude_bin: claude\n  rowboat_wissen: w\n  rowboat_notizen: n\n  rowboat_urteile: u\n",
        encoding="utf-8")
    return wurzel, heim


def _pruefliste(wurzel: Path, offen: list[dict], struktur: list[dict]) -> None:
    pfad = pruefliste_pfad(aus_regeln(wurzel))
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps({"offen": offen, "struktur": struktur}), encoding="utf-8")


def _stand(wurzel: Path, heim: Path, *, tot=(), dop=(), commit: str = COMMIT, fremdcode=()) -> Path:
    ordner = heim / f"aussen-{pfad_kennung(wurzel)}"
    ordner.mkdir(parents=True)
    c = sqlite3.connect(str(ordner / "graph.sqlite"))
    c.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    for k, v in (("commit", commit), ("gemessen_um", "t0")):
        c.execute("INSERT INTO meta VALUES (?, ?)", (json.dumps(k), json.dumps(v)))
    c.commit()
    c.close()
    a = Analyse(commit=commit, version=ANALYSE_VERSION, parameter=parameter(False, fremdcode=fremdcode), bereiche=(), pagerank_top=(),
                betweenness_top=(), hotspots=(), tot=tuple(tot), doppelungen=tuple(dop), kolonie=None, luecken=())
    speichern(ordner, a, graph_gemessen_um="t0")
    return ordner


def _item(art: str, datei: str | None, aid: str | None, text: str = "x", lauf: str = "l1") -> dict:
    return {"art": art, "text": text, "datei": datei, "zeile": 3, "aussage_id": aid, "lauf": lauf}


def _sammeln(wurzel: Path, tor: ToterCodeTor = ZU):
    return funde_sammeln(wurzel, doku_konfig=aus_regeln(wurzel), tor=tor)


def test_doku_schwere(umgebung) -> None:
    wurzel, _ = umgebung
    _pruefliste(wurzel, [], [
        _item("ereignis_quellen_abweichend", "a/reg.yaml", "ereignis:x:ev1",
              "X: `ev1` hat je Quelle ein anderes Werkzeug: `a` (q1), `b` (q2)"),
        _item("ereignis_quellen_abweichend", "a/reg.yaml", "ereignis:x:ev2", "X: `ev2` steht nur in q1, nicht in q2"),
        _item("ereignis_nur_doku", "docs/x.md", "docs/x.md#a.b"),
        _item("space_ohne_doku", None, "space:foo"),
        _item("beleg_untracked", "z.py", None),
    ])
    funde, _ = _sammeln(wurzel)
    nach = {f.anker: f.schwere for f in funde}
    assert nach["ereignis_quellen_abweichend|a/reg.yaml|ereignis:x:ev1"] == "hoch"
    assert nach["ereignis_quellen_abweichend|a/reg.yaml|ereignis:x:ev2"] == "mittel"
    assert nach["ereignis_nur_doku|docs/x.md|docs/x.md#a.b"] == "mittel"
    assert nach["space_ohne_doku||space:foo"] == "mittel"
    assert len(funde) == 4 and all(f.art == "doku_widerspruch" for f in funde)
    assert all(f.code_repo == "aussen" for f in funde)


def test_abweichend_text_stimmt_mit_echtem_abgleich() -> None:
    """Der Klassifizierer liest den Text von `abgleich`; aendert sich dieser, faellt dieser Test."""
    def ev(space: str, name: str, w: str, q: str) -> Ereignis:
        return Ereignis(space, name, w, q, Beleg("yaml_key", "f.yaml", "a", "d", "c", 1))

    anders = abgleich([ev("s", "n.e", "a", "q1"), ev("s", "n.e", "b", "q2")], {}, {"s": "docs/s.md"})
    fehlt = abgleich([ev("s", "n.e", "a", "q1"), ev("s", "m.e", "a", "q2")], {}, {"s": "docs/s.md"})
    a = [p for p in anders if p.art == "ereignis_quellen_abweichend"]
    b = [p for p in fehlt if p.art == "ereignis_quellen_abweichend"]
    assert a and all(abweichend_ist_werkzeug(p.text) for p in a)
    assert b and not any(abweichend_ist_werkzeug(p.text) for p in b)


def test_judge_zwei_laeufe(umgebung) -> None:
    wurzel, _ = umgebung
    _pruefliste(wurzel, [
        _item("judge_uneinig", "d.md", "d.md#aaa", "t1", "l1"),
        _item("judge_uneinig", "d.md", "d.md#aaa", "t2", "l2"),
        _item("judge_unklar", "d.md", "d.md#bbb", "t", "l1"),
        _item("judge_unklar", "d.md", "d.md#bbb", "t-anders", "l1"),
    ], [])
    funde, _ = _sammeln(wurzel)
    assert [(f.anker, f.schwere) for f in funde] == [("judge_uneinig|d.md|d.md#aaa", "mittel")]


def test_anker_posix(umgebung) -> None:
    wurzel, _ = umgebung
    _pruefliste(wurzel, [], [_item("ereignis_nur_doku", "docs\\x.md", "docs/x.md#a.b")])
    a = _sammeln(wurzel)[0][0]
    _pruefliste(wurzel, [], [_item("ereignis_nur_doku", "docs/x.md", "docs/x.md#a.b")])
    b = _sammeln(wurzel)[0][0]
    assert a.anker == b.anker == "ereignis_nur_doku|docs/x.md|docs/x.md#a.b"
    assert fund_schluessel(a) == fund_schluessel(b)


def test_anker_posix_symbol_und_doppelung(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim, tot=[TotFund("symbol:r:src\\a.py#f", "sicher_tot", (), None)],
           dop=[_dop("datei:r:a\\x.py", "datei:r:a/y.py", "datei:r:a/z.py")])
    funde, _ = _sammeln(wurzel, OFFEN)
    assert sorted(f.anker for f in funde) == sorted(
        ["symbol:r:src/a.py#f", "datei:r:a/x.py|datei:r:a/y.py|datei:r:a/z.py"])


def test_code_repo_submodul_praefix(umgebung) -> None:
    wurzel, _ = umgebung
    (wurzel / ".gitmodules").write_text('[submodule "v"]\n\tpath = vibemind-os\n\turl = x\n', encoding="utf-8")
    (wurzel / "vibemind-os").mkdir()
    (wurzel / "vibemind-os" / ".gitmodules").write_text('[submodule "c"]\n\tpath = coding-engine\n\turl = y\n',
                                                         encoding="utf-8")
    _pruefliste(wurzel, [], [
        _item("ereignis_nur_doku", "vibemind-os/coding-engine/docs/a.md", "k1"),
        _item("ereignis_nur_doku", "vibemind-os/spaces/b.md", "k2"),
        _item("ereignis_nur_doku", "docs/c.md", "k3"),
    ])
    funde, _ = _sammeln(wurzel)
    repos = {f.anker.split("|")[2]: f.code_repo for f in funde}
    assert repos == {"k1": "aussen/vibemind-os/coding-engine", "k2": "aussen/vibemind-os", "k3": "aussen"}


def _dop(*m: str, a: float = 1.0) -> Doppelung:
    return Doppelung("gleiche_datei", tuple(m), a)


def test_doppelung_regeln(umgebung) -> None:
    wurzel, heim = umgebung
    prod = [f"datei:vib/vibemind-os/x:src/a{i}.py" for i in range(3)]
    tests = [f"datei:vib:tests/t{i}.py" for i in range(3)]
    _stand(wurzel, heim, dop=[
        _dop(*prod),
        _dop(*tests),
        _dop(prod[0], prod[1]),
        _dop(*prod, "datei:vib:src/extra.py", a=0.95),
        _dop(prod[0], tests[0], "datei:vib:vendor/z.py"),
    ])
    funde, _ = _sammeln(wurzel)
    assert all(f.art == "doppelung" and f.schwere != "hoch" for f in funde)
    assert sorted(f.schwere for f in funde) == ["mittel", "mittel", "niedrig", "niedrig", "niedrig"]
    nach = {f.anker: f for f in funde}
    assert nach["|".join(sorted(tests))].schwere == "niedrig"
    assert nach["|".join(sorted(prod))].code_repo == "vib/vibemind-os/x"
    assert nach["|".join(sorted(tests))].code_repo == "vib"


def test_doppelung_tests_niedrig(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim, dop=[_dop("datei:r:tests/a.py", "datei:r:src/test_b.py", "datei:r:examples/c.py")])
    (f,), _ = _sammeln(wurzel)
    assert f.schwere == "niedrig"


def test_nicht_produktiv_tuple() -> None:
    assert isinstance(NICHT_PRODUKTIV, tuple)
    for p in ("a/test/x.py", "a/tests/x.py", "a/__tests__/x.ts", "e/examples/x.py", "e/example/x.py",
              "vendor/x.go", "n/node_modules/x.js", "t/third_party/x.c", "f/fixtures/x.json",
              "a/test_x.py", "a/x_test.py", "a/x.spec.ts", "a/x.test.ts", "a\\tests\\x.py"):
        assert ist_nicht_produktiv(p), p
    for p in ("src/app.py", "src/testing_tools.py", "src/contest/x.py", "a/latest.py"):
        assert not ist_nicht_produktiv(p), p


def test_tot_gesperrt(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim, tot=[TotFund("symbol:r:a.py#f", "sicher_tot", ("g",), None)])
    funde, notizen = _sammeln(wurzel, ZU)
    assert [f for f in funde if f.art == "tot"] == []
    assert any(n.startswith("toter Code: gesperrt") for n in notizen)


def test_tot_offen(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim, tot=[
        TotFund("symbol:vib/vibemind-os/ce:src/a.py#f", "sicher_tot", ("kein Aufruf",), None),
        TotFund("symbol:r:b.py#g", "wahrscheinlich_tot", (), None),
        TotFund("symbol:r:c.py#h", "verdaechtig", (), None),
    ])
    funde, notizen = _sammeln(wurzel, OFFEN)
    (f,) = funde
    assert f.art == "tot" and f.schwere == "mittel"
    assert f.anker == "symbol:vib/vibemind-os/ce:src/a.py#f"
    assert f.code_repo == "vib/vibemind-os/ce"
    assert not any("gesperrt" in n for n in notizen)


def test_ohne_analyse_kein_absturz(umgebung) -> None:
    wurzel, _ = umgebung
    funde, notizen = _sammeln(wurzel, OFFEN)
    assert funde == [] and any("Analyse" in n for n in notizen)


def test_stand_unter(umgebung, tmp_path: Path) -> None:
    wurzel, heim = umgebung
    ordner = _stand(wurzel, heim, tot=[TotFund("symbol:r:a.py#f", "sicher_tot", (), None)])
    anders = tmp_path / "anders"
    anders.mkdir()
    ordner.rename(anders / ordner.name)
    funde, _ = funde_sammeln(wurzel, doku_konfig=aus_regeln(wurzel), stand_unter=anders, tor=OFFEN)
    assert len(funde) == 1


def test_deterministisch(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim, tot=[TotFund(f"symbol:r:a{i}.py#f", "sicher_tot", (), None) for i in (3, 1, 2)],
           dop=[_dop("datei:r:b.py", "datei:r:a.py", "datei:r:c.py"), _dop("datei:r:x.py", "datei:r:y.py")])
    _pruefliste(wurzel, [], [_item("space_ohne_doku", None, "space:b"), _item("space_ohne_doku", None, "space:a")])
    e1 = _sammeln(wurzel, OFFEN)
    e2 = _sammeln(wurzel, OFFEN)
    assert e1 == e2 and len(e1[0]) == 7
    rang = ["hoch", "mittel", "niedrig"].index
    assert [rang(f.schwere) for f in e1[0]] == sorted(rang(f.schwere) for f in e1[0])


def test_pruefliste_fehlt_gibt_hinweis(umgebung) -> None:
    wurzel, _ = umgebung
    funde, notizen = _sammeln(wurzel)
    assert funde == []
    assert any(n.startswith("quelle-unlesbar:doku") and "doku-pruefliste fehlt" in n for n in notizen)


def test_pruefliste_kaputt_gibt_hinweis(umgebung) -> None:
    wurzel, _ = umgebung
    pfad = pruefliste_pfad(aus_regeln(wurzel))
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text("{kaputt", encoding="utf-8")
    funde, notizen = _sammeln(wurzel)
    assert funde == []
    assert any(n.startswith("quelle-unlesbar:doku") and "doku-pruefliste unlesbar" in n for n in notizen)
    pfad.write_text("[1, 2]", encoding="utf-8")  # gueltiges JSON, falsche Form
    assert any("unlesbar" in n for n in _sammeln(wurzel)[1])


def test_pruefliste_ok_kein_hinweis(umgebung) -> None:
    wurzel, _ = umgebung
    _pruefliste(wurzel, [], [])
    assert not any(n.startswith("quelle-unlesbar:doku") for n in _sammeln(wurzel)[1])


def test_analyse_fehlt_hat_praefix(umgebung) -> None:
    wurzel, _ = umgebung
    assert any(n.startswith("quelle-unlesbar:analyse") for n in _sammeln(wurzel, OFFEN)[1])


def test_analyse_kaputte_datei_kein_absturz(umgebung) -> None:
    wurzel, heim = umgebung
    ordner = _stand(wurzel, heim)
    (ordner / "analyse.sqlite").write_bytes(b"das ist keine sqlite-datei" * 50)
    funde, notizen = _sammeln(wurzel, OFFEN)
    assert funde == [] and any(n.startswith("quelle-unlesbar:analyse") for n in notizen)


def test_analyse_blob_falsche_form_kein_absturz(umgebung) -> None:
    wurzel, heim = umgebung
    ordner = _stand(wurzel, heim)
    c = sqlite3.connect(str(ordner / "analyse.sqlite"))
    c.execute("UPDATE lauf SET daten=?", (json.dumps({"commit": COMMIT, "version": 1}).encode(),))
    c.commit()
    c.close()
    funde, notizen = _sammeln(wurzel, OFFEN)  # KeyError im Lader
    assert funde == [] and any(n.startswith("quelle-unlesbar:analyse") for n in notizen)


def test_fehlerhafte_eintraege_uebersprungen(umgebung) -> None:
    wurzel, heim = umgebung
    _stand(wurzel, heim,
           tot=[TotFund(None, "sicher_tot", (), None), TotFund("symbol:r:ok.py#f", "sicher_tot", (), None)],  # type: ignore[arg-type]
           dop=[Doppelung("gleiche_datei", (1, 2, 3), 1.0), _dop("datei:r:a.py", "datei:r:b.py", "datei:r:c.py")])  # type: ignore[arg-type]
    funde, notizen = _sammeln(wurzel, OFFEN)
    assert sorted(f.art for f in funde) == ["doppelung", "tot"]
    assert any("1 fehlerhafte Doppelungs-Einträge" in n for n in notizen)
    assert any("1 fehlerhafte Tot-Einträge" in n for n in notizen)
    assert all(n.startswith("quelle-unlesbar:") for n in notizen if "fehlerhafte" in n)


def test_analyse_mit_fremdcode_regel_im_commit_gefunden(umgebung) -> None:
    """Die Regel am Graph-Commit gehoert zum Digest: der Leser findet die Analyse, kein "veraltet"."""
    wurzel, heim = umgebung
    with (wurzel / "selbstbild.yaml").open("a", encoding="utf-8") as f:
        f.write("fremdcode: ['lib/**']\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.hooksPath=NUL"]
    for args in (["init", "-q"], ["add", "selbstbild.yaml"], ["commit", "-q", "-m", "regeln"]):
        subprocess.run([*git, *args], cwd=wurzel, check=True, capture_output=True)
    sha = subprocess.run([*git, "rev-parse", "HEAD"], cwd=wurzel, check=True, capture_output=True,
                         text=True).stdout.strip()
    _stand(wurzel, heim, dop=[_dop("datei:r:src/a.py", "datei:r:src/b.py")], commit=sha, fremdcode=["lib/**"])
    funde, notizen = _sammeln(wurzel)
    assert [f.art for f in funde] == ["doppelung"]
    assert not any("veraltet" in n for n in notizen)
