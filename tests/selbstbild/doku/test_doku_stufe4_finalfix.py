"""Stufe 4, Final-Fix-Welle (Ruling FINAL-a): C1, C2b, I1-I5, M1, M3, M4.

Alle Schluessel in diesen Tests sind synthetisch (Wiederholungen von Buchstaben/Ziffern), nie echte Werte. Laeufe
gehen gegen das `git init`-Fixture aus `test_doku_tick.py` unter tmp_path; Judge, Budget und Torwaechter sind Fakes.
"""

import re

import pytest

from selbstbild.doku import schluessel as sl

# synthetische Fake-Werte (keine echten Schluessel)
GHP = "ghp_" + "A1b2C3d4E5" * 3 + "F6g7H8"  # 36 Zeichen nach dem Praefix
SK_ANT = "sk-ant-api03-" + "Ab1_Cd2-" * 4
SK_PROJ = "sk-proj-" + "Zz9_Yy8-" * 4
TELEGRAM = "123456789:" + "AAbbCCddEEffGGhhIIjjKKllMMnnOOppQQr"  # 9 Ziffern, 35 Zeichen
PEM = "\n".join([
    "-----BEGIN RSA PRIVATE KEY-----",
    "MIIEowIBAAKCAQEAfakefakefakefakefakefakefakefakefakefakefakefake",
    "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2",
    "-----END RSA PRIVATE KEY-----",
])


# ---------- I1: PEM ganz schwaerzen ----------

def test_i1_pem_koerper_und_end_zeile_verschwinden():
    text = f"vorher\n{PEM}\nnachher"
    aus = sl.schwaerzen(text)
    assert "vorher" in aus and "nachher" in aus
    assert "MIIEowIBAAKCAQEA" not in aus and "QUJDREVGR0hJSktM" not in aus
    assert "END RSA PRIVATE KEY" not in aus and "BEGIN" not in aus
    assert "[Schlüssel-Verdacht: privater Schlüssel (PEM)]" in aus
    assert sl.schluessel_muster(text) == "privater Schlüssel (PEM)"


def test_i1_pem_ohne_end_bis_textende():
    text = "Kopf\n" + PEM.rsplit("\n", 1)[0] + "\nund noch eine Zeile"
    aus = sl.schwaerzen(text)
    assert aus.startswith("Kopf\n") and "MIIE" not in aus and "QUJD" not in aus and "noch eine Zeile" not in aus


# ---------- I2: sk-ant/sk-proj, Telegram ----------

@pytest.mark.parametrize("wert", [SK_ANT, SK_PROJ])
def test_i2_sk_mit_bindestrich_und_unterstrich(wert):
    text = f"Schluessel {wert} im Text"
    assert sl.schluessel_muster(text) == "OpenAI-artiger API-Schlüssel"
    aus = sl.schwaerzen(text)
    assert wert[3:12] not in aus and "sk-" not in aus and aus.endswith(" im Text")
    assert sl.schluessel_muster(f"KEY={wert}") is not None and sl.schluessel_muster(f'"{wert}"') is not None


def test_i2_sk_kein_fehltreffer_im_fliesstext():
    for harmlos in ("risk-assessment-report-v2-final", "task-runner-for-the-whole-space", "desk-sk-"):
        assert sl.schluessel_muster(harmlos) is None, harmlos


def test_i2_telegram_bot_token():
    text = f"Bot {TELEGRAM} fertig"
    assert sl.schluessel_muster(text) == "Telegram-Bot-Token"
    aus = sl.schwaerzen(text)
    assert "AAbbCC" not in aus and "123456789" not in aus and aus == "Bot [Schlüssel-Verdacht: Telegram-Bot-Token] fertig"
    # keine Fehltreffer: Uhrzeit, Port, kurze Kennung
    for harmlos in ("um 12:30 Uhr", "localhost:8130", "1234:abc", "12345678:" + "x" * 20):
        assert sl.schluessel_muster(harmlos) is None, harmlos


def test_i2_namen_ohne_wertteile():
    for name, _ in sl.SCHLUESSEL_MUSTER:
        assert not re.search(r"sk-|ghp_|xox|AKIA|eyJ|\d{8}:", name)
