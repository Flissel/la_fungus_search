"""Suche (Stufe 3, T3): FTS5-Stichwort, Struktur-Erweiterung und RRF-Fusion ueber `suche.sqlite`."""
from __future__ import annotations

from pathlib import Path

from selbstbild.analyse.bereiche import Bereich
from selbstbild.analyse.ergebnis import Analyse, speichern
from selbstbild.analyse.lader import ANALYSE_VERSION
from selbstbild.analyse.lauf import parameter
from selbstbild.analyse.tot import TotFund
from selbstbild.suche.abfrage import Treffer, anfrage_tokens, nachbarn, suchen
from selbstbild.suche.index import index_bauen
from selbstbild.typen import Beleg, Kante, Knoten

FREIGEBEN = "symbol:repo:app/versand.py#newsletter_freigeben"
AUFRUFER = "symbol:repo:app/cron.py#taeglich"
DOKU = "doku_seite:repo:docs/newsletter.md"
DATEI = "datei:repo:app/versand.py"
LEBT = "symbol:repo:app/b.py#pruefe_rechnung"
TOT = "symbol:repo:app/a.py#pruefe_rechnung"


def _repo(git_repo) -> str:
    git_repo.datei("app/versand.py", "def newsletter_freigeben():\n    return 1\n")
    git_repo.datei("app/cron.py", "def taeglich():\n    return 2\n")
    git_repo.datei("app/a.py", "def pruefe_rechnung():\n    return 3\n")
    git_repo.datei("app/b.py", "def pruefe_rechnung():\n    return 3\n")
    git_repo.datei("docs/newsletter.md", "# Versand\n\nDer Newsletter wird freigegeben, sobald es passt.\n")
    return git_repo.commit("init")


def _sym(pfad: str, name: str, zeile: int = 1) -> Knoten:
    return Knoten(f"symbol:repo:{pfad}#{name}", "symbol", name, Beleg("repo", pfad, zeile, name, "d"))


def _knoten() -> list[Knoten]:
    return [
        Knoten(DATEI, "datei", "app/versand.py", None, (("lebendigkeit", "0.5"),)),
        Knoten("datei:repo:app/cron.py", "datei", "app/cron.py", None),
        _sym("app/versand.py", "newsletter_freigeben"),
        _sym("app/cron.py", "taeglich"),
        _sym("app/a.py", "pruefe_rechnung"),
        _sym("app/b.py", "pruefe_rechnung"),
        Knoten(DOKU, "doku_seite", "docs/newsletter.md", None),
    ]


def _kanten() -> list[Kante]:
    return [Kante("ruft", AUFRUFER, FREIGEBEN, True, None),
            Kante("importiert", DATEI, "datei:repo:app/cron.py", False, None)]


def _stand(git_repo, graph_stand, mit_tot: bool = False) -> Path:
    sha = _repo(git_repo)
    stand = graph_stand(_knoten(), _kanten(), commit=sha)
    if mit_tot:
        speichern(stand, Analyse(
            commit=sha, version=ANALYSE_VERSION, parameter=parameter(False),
            bereiche=(Bereich("b1", "Versand", ("datei:repo:app/a.py",), 0.5),), pagerank_top=(),
            betweenness_top=(), hotspots=(), tot=(TotFund(TOT, "sicher_tot", ("kein Aufrufer",), None),),
            doppelungen=(), kolonie=None, luecken=()), graph_gemessen_um="")
    index_bauen(git_repo.wurzel, stand)
    return stand


def test_stichwort_trifft(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    treffer = suchen(stand, "wie wird der Newsletter freigegeben")
    ids = [t.id for t in treffer]
    assert DOKU in ids and FREIGEBEN in ids
    assert all(isinstance(t, Treffer) for t in treffer)
    t = next(t for t in treffer if t.id == FREIGEBEN)
    assert (t.art, t.pfad, t.zeile) == ("symbol", "app/versand.py", 1) and "stichwort" in t.herkunft


def test_struktur_bringt_aufrufer(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    treffer = {t.id: t for t in suchen(stand, "newsletter_freigeben")}
    assert treffer[FREIGEBEN].herkunft == ("stichwort",)
    assert AUFRUFER in treffer and treffer[AUFRUFER].herkunft == ("struktur",)
    assert treffer[AUFRUFER].wert < treffer[FREIGEBEN].wert


def test_tot_gedaempft_mit_markierung(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand, mit_tot=True)
    treffer = suchen(stand, "pruefe rechnung")
    ids = [t.id for t in treffer]
    assert LEBT in ids and TOT in ids  # nie unterdrueckt
    assert ids.index(LEBT) < ids.index(TOT)
    assert next(t for t in treffer if t.id == TOT).tot == "sicher_tot"
    assert next(t for t in treffer if t.id == LEBT).tot is None


def test_deterministisch(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand, mit_tot=True)
    erst = suchen(stand, "pruefe rechnung newsletter")
    assert suchen(stand, "pruefe rechnung newsletter") == erst
    assert suchen(stand, "newsletter", top_k=1) == suchen(stand, "newsletter")[:1]


def test_leere_anfrage(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    assert anfrage_tokens("wie wird der und die") == ()
    assert suchen(stand, "wie wird der und die") == ()
    assert suchen(stand, "") == ()


def test_fts_syntaxzeichen_werfen_nicht(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    treffer = suchen(stand, '"newsletter" -freigegeben: * NEAR AND (pfad:x) OR ""')
    assert DOKU in [t.id for t in treffer]


def test_anfrage_tokens():
    assert anfrage_tokens("kaufeLaden und newsletter_freigeben") == ("kaufe", "laden", "newsletter", "freigeben")
    assert len(anfrage_tokens(" ".join(f"wort{i:02d}" for i in range(30)))) == 12
    assert anfrage_tokens("ab to the Newsletter newsletter") == ("newsletter",)


def test_dicht_ohne_modul_ist_wirkungslos(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    assert suchen(stand, "newsletter", dicht=True) == suchen(stand, "newsletter")


def test_nachbarn(git_repo, graph_stand):
    stand = _stand(git_repo, graph_stand)
    assert nachbarn(stand, FREIGEBEN) == ((AUFRUFER, FREIGEBEN, True),)
    assert nachbarn(stand, FREIGEBEN, "aus") == ()
    assert nachbarn(stand, AUFRUFER, "aus") == ((AUFRUFER, FREIGEBEN, True),)
    assert nachbarn(stand, FREIGEBEN, "ein", ("importiert",)) == ()
    assert nachbarn(stand, DATEI, "beide", ("importiert",)) == ((DATEI, "datei:repo:app/cron.py", False),)
