"""Gemeinsamer Client für den Freigabe-Eingang (Task 8)."""
from __future__ import annotations

import io
import json
import logging
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from selbstbild.freigabe_eingang import FreigabeEingang

SCHLUESSEL = "geheim-schluessel-xyz"


class _Antwort:
    def __init__(self, daten: Any, status: int = 200) -> None:
        self._roh = json.dumps(daten).encode()
        self.status = status

    def __enter__(self) -> "_Antwort":
        return self

    def __exit__(self, *a: object) -> None:
        return None

    def read(self) -> bytes:
        return self._roh


class _Oeffnen:
    def __init__(self, antworten: list[Any]) -> None:
        self.antworten = antworten
        self.aufrufe: list[Any] = []

    def __call__(self, req: Any, timeout: float | None = None) -> _Antwort:
        self.aufrufe.append((req, timeout))
        a = self.antworten.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a


@pytest.fixture
def datei(tmp_path: Path) -> Path:
    d = tmp_path / "schluessel"
    d.write_bytes((SCHLUESSEL + "\r\n").encode())  # VM-Dateien können CRLF haben
    return d


def _client(datei: Path, antworten: list[Any], url: str = "https://host:1/") -> tuple[FreigabeEingang, _Oeffnen]:
    o = _Oeffnen(antworten)
    return FreigabeEingang(url, datei, oeffnen=o), o


def test_eintragen_sendet_bearer_und_json(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"kennung": "fg-abcdefghij", "neu": True}, 201)])
    assert c.eintragen("debugger_pr", "o/r#1", "Titel", "https://x/y", {"tests": "ok"}) == "fg-abcdefghij"
    (req, timeout), = o.aufrufe
    assert req.full_url == "https://host:1/api/freigaben/eintrag"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {SCHLUESSEL}"
    assert req.get_header("Content-type") == "application/json"
    assert json.loads(req.data) == {"art": "debugger_pr", "quelle_ref": "o/r#1", "titel": "Titel",
                                    "link": "https://x/y", "details": {"tests": "ok"}}
    assert timeout == 10


def test_eintragen_ohne_link_und_details_laesst_felder_weg(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"kennung": "fg-abcdefghij", "neu": False})])
    c.eintragen("debugger_pr", "o/r#1", "T")
    assert json.loads(o.aufrufe[0][0].data) == {"art": "debugger_pr", "quelle_ref": "o/r#1", "titel": "T"}


def test_netzfehler_gibt_none(datei: Path, caplog: pytest.LogCaptureFixture) -> None:
    c, _ = _client(datei, [urllib.error.URLError("weg"), OSError("x"), TimeoutError()])
    with caplog.at_level(logging.DEBUG):
        assert c.eintragen("debugger_pr", "o/r#1", "T") is None
        assert c.offene() is None
        assert c.ergebnis("1", True) is False
    assert SCHLUESSEL not in caplog.text


def test_http_fehler_gibt_none_ohne_schluessel_im_log(datei: Path, caplog: pytest.LogCaptureFixture) -> None:
    fehler = urllib.error.HTTPError("u", 403, "x", {}, io.BytesIO(b'{"fehler":"art_nicht_erlaubt"}'))  # type: ignore[arg-type]
    c, _ = _client(datei, [fehler])
    with caplog.at_level(logging.DEBUG):
        assert c.eintragen("debugger_pr", "o/r#1", "T") is None
    assert "403" in caplog.text and SCHLUESSEL not in caplog.text


def test_kaputtes_json_gibt_none(datei: Path) -> None:
    class Kaputt(_Antwort):
        def read(self) -> bytes:
            return b"<html>"

    c, _ = _client(datei, [Kaputt({})])
    assert c.eintragen("debugger_pr", "o/r#1", "T") is None


def test_fehlende_schluesseldatei_gibt_none(tmp_path: Path) -> None:
    o = _Oeffnen([])
    c = FreigabeEingang("https://h", tmp_path / "fehlt", oeffnen=o)
    assert c.eintragen("debugger_pr", "o/r#1", "T") is None and o.aufrufe == []


def test_schluesselmuster_wird_nicht_gesendet(datei: Path, caplog: pytest.LogCaptureFixture) -> None:
    geheim = "ghp_" + "a" * 36
    c, o = _client(datei, [])
    with caplog.at_level(logging.DEBUG):
        assert c.eintragen("debugger_pr", "o/r#1", f"Titel {geheim}") is None
        assert c.eintragen("debugger_pr", "o/r#1", "T", details={"tests": f"ok {geheim}"}) is None
    assert o.aufrufe == [] and geheim not in caplog.text


def test_schliessen(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"geschlossen": 1})])
    assert c.schliessen("o/r#1") == 1
    req = o.aufrufe[0][0]
    assert req.full_url == "https://host:1/api/freigaben/schliessen"
    assert json.loads(req.data) == {"quelle_ref": "o/r#1", "status": "erledigt"}


def test_offene_und_auftraege(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"eintraege": [{"quelle_ref": "o/r#1", "art": "debugger_pr"}]}),
                           _Antwort({"auftraege": [{"id": "7", "art": "selbstbild_label", "nutzlast": {}}]})])
    assert c.offene() == [{"quelle_ref": "o/r#1", "art": "debugger_pr"}]
    assert c.auftraege() == [{"id": "7", "art": "selbstbild_label", "nutzlast": {}}]
    assert [a[0].full_url for a in o.aufrufe] == ["https://host:1/api/freigaben/offen",
                                                  "https://host:1/api/freigaben/auftraege"]
    assert all(a[0].get_method() == "GET" for a in o.aufrufe)


def test_ergebnis(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"ok": True}), _Antwort({"ok": True})])
    assert c.ergebnis("7", True) is True
    assert c.ergebnis("8", False, "label_fehlt") is True
    assert o.aufrufe[0][0].full_url == "https://host:1/api/freigaben/auftraege/7"
    assert json.loads(o.aufrufe[0][0].data) == {"ok": True}
    assert json.loads(o.aufrufe[1][0].data) == {"ok": False, "fehler": "label_fehlt"}


def test_nicht_serialisierbare_details_ohne_ausnahme(datei: Path) -> None:
    from datetime import datetime

    c, o = _client(datei, [])
    assert c.eintragen("debugger_pr", "o/r#1", "T", details={"t": datetime.now()}) is None
    zyklus: dict[str, Any] = {}
    zyklus["selbst"] = zyklus
    assert c.eintragen("debugger_pr", "o/r#1", "T", details=zyklus) is None
    assert o.aufrufe == []


def test_auftrag_id_wird_quotiert(datei: Path) -> None:
    c, o = _client(datei, [_Antwort({"ok": True})])
    c.ergebnis("a/../b?x", True)
    assert o.aufrufe[0][0].full_url == "https://host:1/api/freigaben/auftraege/a%2F..%2Fb%3Fx"


def test_http_fehler_wird_geschlossen(datei: Path) -> None:
    geschlossen: list[bool] = []

    class Fehler(urllib.error.HTTPError):
        def close(self) -> None:
            geschlossen.append(True)

    c, _ = _client(datei, [Fehler("u", 401, "x", {}, io.BytesIO(b"{}"))])  # type: ignore[arg-type]
    assert c.eintragen("debugger_pr", "o/r#1", "T") is None
    assert geschlossen


def test_standard_oeffner_vertraut_auch_certifi(datei: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows-Python 3.11/OpenSSL 1.1.1 kennt ISRG Root X2 nur als Zwischenzertifikat; certifi ergänzt die Wurzel."""
    import ssl
    import urllib.request

    pytest.importorskip("certifi")
    gesehen: dict[str, Any] = {}

    def falsch_oeffnen(req: Any, timeout: float | None = None, context: Any = None) -> _Antwort:
        gesehen["context"] = context
        return _Antwort({"eintraege": []})

    monkeypatch.setattr(urllib.request, "urlopen", falsch_oeffnen)
    assert FreigabeEingang("https://host:1/", datei).offene() == []
    kontext = gesehen["context"]
    assert isinstance(kontext, ssl.SSLContext)
    assert kontext.verify_mode == ssl.CERT_REQUIRED and kontext.check_hostname
    namen = {wert for c in kontext.get_ca_certs() for teil in c["subject"] for (k, wert) in teil if k == "commonName"}
    assert "ISRG Root X2" in namen
