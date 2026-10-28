from pathlib import Path

import pytest
from pydantic import ValidationError

from selbstbild.regeln import REGELDATEI, STANDARD_AUSGENOMMEN, regeln_laden

BEISPIEL = Path(__file__).resolve().parents[2] / "docs" / "selbstbild" / "selbstbild.beispiel.yaml"


def test_vorgaben_ohne_datei():
    r = regeln_laden(None)
    assert r.doku_pfade == ["**/*.md"]
    assert r.struktur == []
    assert r.issue_ziel.privat is True


def test_unbekannter_schluessel_bricht():
    with pytest.raises(ValidationError):
        regeln_laden("foo: 1")


def test_unbekannter_verschachtelter_schluessel_bricht():
    with pytest.raises(ValidationError):
        regeln_laden("issue_ziel:\n  tippfehler: 1\n")


def test_beispieldatei_valide():
    r = regeln_laden(BEISPIEL.read_text(encoding="utf-8"))
    space = [s for s in r.struktur if s.art == "space"]
    assert len(space) == 1
    assert space[0].ort == "vibemind-os/spaces/*/"
    assert "capabilities.yaml" in space[0].pflicht
    assert space[0].verboten


def test_leere_datei():
    assert regeln_laden("") == regeln_laden(None)


def test_konstanten():
    assert REGELDATEI == "selbstbild.yaml"
    assert ".git/**" in STANDARD_AUSGENOMMEN
    assert len(STANDARD_AUSGENOMMEN) == 12
