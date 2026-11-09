"""Task 9: der Abholer am PC setzt Labels, die der Freigabe-Eingang aufträgt."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from selbstbild import freigabe_abholer
from selbstbild.freigabe_abholer import abholen
from selbstbild.funde.github import GitHubFehler
from selbstbild.funde.sync import Ziel

REPO = "Flissel/Vibemind_V1"


class FakeFreigabe:
    def __init__(self, auftraege: list[dict[str, Any]] | None) -> None:
        self._auftraege = auftraege
        self.ergebnisse: list[tuple[str, bool, str | None]] = []

    def auftraege(self) -> list[dict[str, Any]] | None:
        return self._auftraege

    def ergebnis(self, auftrag_id: str, ok: bool, fehler: str | None = None) -> bool:
        self.ergebnisse.append((auftrag_id, ok, fehler))
        return True


class FakeGitHub:
    def __init__(self, fehler: Exception | None = None, setzen_fehler: Exception | None = None) -> None:
        self.aufrufe: list[tuple[Any, ...]] = []
        self.fehler = fehler
        self.setzen_fehler = setzen_fehler

    def label_entfernen(self, repo: str, nummer: int, name: str) -> None:
        self.aufrufe.append(("entfernen", repo, nummer, name))
        if self.fehler:
            raise self.fehler

    def labels_setzen(self, repo: str, nummer: int, labels: list[str]) -> None:
        self.aufrufe.append(("setzen", repo, nummer, list(labels)))
        if self.setzen_fehler:
            raise self.setzen_fehler


def _a(label: str = "antwort:bleibt", repo: str = REPO, nummer: object = 7, art: str = "selbstbild_label",
       id_: str = "a1") -> dict[str, Any]:
    return {"id": id_, "art": art, "nutzlast": {"repo": repo, "nummer": nummer, "label": label}}


def test_abholer_setzt_label_und_entfernt_altes() -> None:
    fg, gh = FakeFreigabe([_a("antwort:bleibt")]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 1  # type: ignore[arg-type]
    assert ("entfernen", REPO, 7, "antwort:weg") in gh.aufrufe
    assert ("entfernen", REPO, 7, "antwort:benutzt") in gh.aufrufe
    assert ("entfernen", REPO, 7, "antwort:bleibt") not in gh.aufrufe
    assert gh.aufrufe[-1] == ("setzen", REPO, 7, ["antwort:bleibt"])  # erst entfernen, dann setzen
    assert fg.ergebnisse == [("a1", True, None)]


def test_abholer_lehnt_fremdes_label_ab() -> None:
    fg, gh = FakeFreigabe([_a("bug")]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == [("a1", False, "label_unzulaessig")]


def test_abholer_lehnt_fremdes_repo_ab() -> None:
    fg, gh = FakeFreigabe([_a(repo="fremd/repo")]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == [("a1", False, "repo_unzulaessig")]


@pytest.mark.parametrize("nummer", [True, "7", 0, -1, 7.0, None])
def test_abholer_lehnt_ungueltige_nummer_ab(nummer: object) -> None:
    fg, gh = FakeFreigabe([_a(nummer=nummer)]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == [("a1", False, "nummer_unzulaessig")]


def test_abholer_unbekannte_art() -> None:
    fg, gh = FakeFreigabe([_a(art="kurs_freigabe_zeigen")]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == [("a1", False, "unbekannte_art")]


def test_abholer_ohne_netz_beendet_sich_ruhig() -> None:
    fg, gh = FakeFreigabe(None), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == []


def test_abholer_fehler_meldet_nur_den_typ() -> None:
    fg = FakeFreigabe([_a(), _a(id_="a2", nummer=8)])
    gh = FakeGitHub(RuntimeError("token ghp_geheim"))
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert fg.ergebnisse == [("a1", False, "RuntimeError"), ("a2", False, "RuntimeError")]


def test_abholer_ueberspringt_kaputte_form() -> None:
    fg, gh = FakeFreigabe([{"art": "selbstbild_label"}, {"id": "x", "art": "selbstbild_label", "nutzlast": 5}]), FakeGitHub()
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert gh.aufrufe == [] and fg.ergebnisse == [("x", False, "nutzlast_unzulaessig")]


def test_cli_ohne_umgebung_ist_aus(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
                                   tmp_path: Path) -> None:
    monkeypatch.delenv("FREIGABE_URL", raising=False)
    monkeypatch.delenv("FREIGABE_SCHLUESSEL_DATEI", raising=False)
    with caplog.at_level("INFO", logger="selbstbild.freigabe_abholer"):
        assert freigabe_abholer.main([str(tmp_path)]) == 0
    assert caplog.text == ""  # ganz aus: alle 2 Minuten nichts ins Log schreiben
    monkeypatch.setenv("FREIGABE_URL", "http://x")  # halb gesetzt: Fehler des Betreibers, sichtbar
    with caplog.at_level("INFO", logger="selbstbild.freigabe_abholer"):
        assert freigabe_abholer.main([str(tmp_path)]) == 0
    assert "Abholer aus" in caplog.text


def test_umgebung_braucht_url_und_vorhandene_datei(tmp_path: Path) -> None:
    schluessel = tmp_path / "k"
    assert freigabe_abholer.freigabe_aus_umgebung({}) is None
    assert freigabe_abholer.freigabe_aus_umgebung({"FREIGABE_URL": "http://x", "FREIGABE_SCHLUESSEL_DATEI": str(schluessel)}) is None
    schluessel.write_text("s", encoding="utf-8")
    assert freigabe_abholer.freigabe_aus_umgebung({"FREIGABE_SCHLUESSEL_DATEI": str(schluessel)}) is None
    assert freigabe_abholer.freigabe_aus_umgebung(
        {"FREIGABE_URL": "http://x", "FREIGABE_SCHLUESSEL_DATEI": str(schluessel)}) is not None


def test_github_fehler_meldet_die_schluesselfreie_art() -> None:
    fg = FakeFreigabe([_a()])
    gh = FakeGitHub(GitHubFehler("http_404"))
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]
    assert fg.ergebnisse == [("a1", False, "http_404")]


def test_labels_setzen_scheitert_nach_dem_entfernen() -> None:
    fg = FakeFreigabe([_a(), _a(id_="a2", nummer=8)])
    gh = FakeGitHub(setzen_fehler=GitHubFehler("http_502"))
    assert abholen(fg, gh, REPO) == 0  # type: ignore[arg-type]  # wirft nicht
    assert ("entfernen", REPO, 7, "antwort:weg") in gh.aufrufe
    assert fg.ergebnisse == [("a1", False, "http_502"), ("a2", False, "http_502")]


def test_umgebung_oserror_beim_pruefen_ergibt_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def kaputt(self: Path) -> bool:
        raise PermissionError("kein Zugriff")
    monkeypatch.setattr(Path, "is_file", kaputt)
    assert freigabe_abholer.freigabe_aus_umgebung(
        {"FREIGABE_URL": "http://x", "FREIGABE_SCHLUESSEL_DATEI": str(tmp_path / "k")}) is None


class _Abholer:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, ziel: Ziel | None, n: int = 0) -> None:
        self.ziel_aufrufe = 0
        self.abgeholt: list[str] = []
        self.gh_konten: list[str] = []
        outer = self

        def ziel_fn(wurzel: Path, aussen: str) -> Ziel | None:
            outer.ziel_aufrufe += 1
            return ziel

        def holen(f: object, gh: object, repo: str) -> int:
            outer.abgeholt.append(repo)
            return n

        class GH:
            def __init__(self, konto: str) -> None:
                outer.gh_konten.append(konto)

        monkeypatch.setattr(freigabe_abholer, "freigabe_aus_umgebung", lambda *a, **k: object())
        monkeypatch.setattr(freigabe_abholer, "rueckfrage_ziel", ziel_fn)
        monkeypatch.setattr(freigabe_abholer, "_aussen_name", lambda w: "aussen")
        monkeypatch.setattr(freigabe_abholer, "abholen", holen)
        monkeypatch.setattr(freigabe_abholer, "GitHub", GH)


def test_cli_nimmt_das_erste_ziel_und_verlangt_erst_dann_ein_konto(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    a = _Abholer(monkeypatch, Ziel(repo=REPO, konto=None, oeffentlich=False))
    assert freigabe_abholer.main([str(tmp_path)]) == 0
    assert a.ziel_aufrufe == 1 and a.abgeholt == [] and a.gh_konten == []


def test_cli_nutzt_konto_und_repo_des_ziels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                            caplog: pytest.LogCaptureFixture) -> None:
    a = _Abholer(monkeypatch, Ziel(repo=REPO, konto="Flissel", oeffentlich=False))
    with caplog.at_level("INFO", logger="selbstbild.freigabe_abholer"):
        assert freigabe_abholer.main([str(tmp_path)]) == 0
    assert a.abgeholt == [REPO] and a.gh_konten == ["Flissel"]
    assert caplog.text == ""  # nichts zu tun: kein Logeintrag (das Log wird nie rotiert)


def test_cli_loggt_nur_bei_ausgefuehrten_auftraegen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    _Abholer(monkeypatch, Ziel(repo=REPO, konto="Flissel", oeffentlich=False), n=2)
    with caplog.at_level("INFO", logger="selbstbild.freigabe_abholer"):
        freigabe_abholer.main([str(tmp_path)])
    assert "2 Auftrag" in caplog.text
