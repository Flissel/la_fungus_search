"""Doku-Loop als selbstbild.doku verpflanzt (Stufe 4, Task 1): Manifest-Digest und Budget-Wächter ohne LLM."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from doku_vm import REPO, VORLAGE, braucht_vm, laden as vm_laden
from selbstbild.doku import anker, graph, judge, korpus, konfig as konfig_modul
from selbstbild.doku.anker import Beleg
from selbstbild.doku.aussagen import Aussage


@braucht_vm
def test_doku_manifest_digest_gleich():
    """Für 20 per Seed gezogene `symbol`-Belege aus docs/graph/graph.json liefert der verpflanzte Digest-Weg
    denselben Digest, der im Graphen steht (echtes VibeMind, nur lesend): innerhalb der Space-Wurzeln über das
    gepinnte Manifest (`stand_aus_manifest`), sonst über `symbol_digest_aus_datei`."""
    pfad = REPO / "docs" / "graph" / "graph.json"
    if not pfad.is_file():
        pytest.skip("kein docs/graph/graph.json")
    konfig = vm_laden()
    g = graph.Graph.laden(pfad)
    belege = sorted(
        {(b.file, b.anker, b.digest) for k in g.kanten.values() for b in k.belege if b.art == "symbol"}
    )
    assert len(belege) >= 20
    commit = korpus.commit_von(REPO / "vibemind-os")
    staende: dict[str, dict[tuple[str, str, str], str]] = {}
    for file, ank, digest in random.Random(20261030).sample(belege, 20):
        raum = next((s for s in konfig.spaces.values() if file.startswith(s.wurzel + "/")), None)
        if raum is None:
            assert anker.symbol_digest_aus_datei(REPO, file, ank) == digest, (file, ank)
            continue
        if raum.name not in staende:
            staende[raum.name] = anker.stand_aus_manifest(korpus.manifest_bauen(raum, konfig, commit), REPO)
        assert staende[raum.name].get(("symbol", file, ank)) == digest, (file, ank)


def test_doku_manifest_ist_im_paket():
    assert korpus.manifest_verfuegbar()
    assert korpus.Manifest.__module__ == "selbstbild.doku._manifest_gate2"


def _frage() -> judge.JudgeFrage:
    beleg = Beleg("symbol", "a.py", "a.f", "0" * 64, "c" * 40, 1)
    aussage = Aussage(id="x.md#T#12345678", datei="x.md", ueberschrift="T", zeile_start=1, zeile_ende=1,
                      text="f tut etwas.", belege=(beleg,), pruefpunkte=())
    return judge.JudgeFrage(aussage=aussage, beleg=beleg, quelltext_alt="a", quelltext_neu="b", nachbarn=())


def test_doku_budget_modul_fehlt_kein_llm(tmp_path):
    k = konfig_modul.laden(VORLAGE)
    k = type(k)(**{**k.__dict__, "repo": tmp_path, "budget_modul": "scripts/gibt_es_nicht.py"})

    def laufzeit(*a, **kw):
        raise AssertionError("subprocess darf nie aufgerufen werden")

    assert judge.befragen(_frage(), k, laufzeit=laufzeit) == (None, "budget:modul_fehlt")
    assert judge.zweimal_einig(_frage(), k, laufzeit=laufzeit) == (None, "budget:modul_fehlt")


def test_doku_budget_modul_kaputt_kein_llm(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "claude_budget.py").write_text("raise RuntimeError('kaputt')\n", encoding="utf-8")
    k = konfig_modul.laden(VORLAGE)
    k = type(k)(**{**k.__dict__, "repo": tmp_path})
    assert judge.budget_laden(k) == (None, "budget:modul_fehlt")


def test_doku_budget_modul_wird_per_pfad_geladen(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "claude_budget.py").write_text("ZEICHEN = 'geladen'\n", encoding="utf-8")
    k = konfig_modul.laden(VORLAGE)
    k = type(k)(**{**k.__dict__, "repo": tmp_path})
    modul, grund = judge.budget_laden(k)
    assert grund is None and modul.ZEICHEN == "geladen"
    assert modul.__name__.startswith("selbstbild_doku_budget_")


def test_doku_konfig_vorgaben_fuer_budget_und_torwaechter():
    k = konfig_modul.laden(VORLAGE)
    assert k.budget_modul == "scripts/claude_budget.py"
    assert k.torwaechter == (".venv/Scripts/python.exe", "scripts/docs_drift_waechter.py")
