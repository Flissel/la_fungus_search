from __future__ import annotations

import pytest

from selbstbild.funde.text import BEGIN, END, block_ersetzen, einzel_text, pruefen, sammel_text
from selbstbild.funde.typen import Fund, fund_schluessel


def _fund(i: int = 0, schwere: str = "mittel", repo: str = "vibemind-os", art: str = "doppelung", anker: str | None = None) -> Fund:
    return Fund(art=art, schwere=schwere, code_repo=repo, titel=f"Fund {i}", text="t",  # type: ignore[arg-type]
                belege=(f"src/mod{i}.py",), anker=anker if anker is not None else f"src/mod{i}.py")


def test_einzel_text_titel_marke_hinweis() -> None:
    f = _fund(1, "hoch", art="doku_widerspruch")
    titel, text = einzel_text(f)
    assert titel == "[selbstbild] Doku-Widerspruch: Fund 1 (vibemind-os)"
    assert f"<!-- selbstbild:schluessel={fund_schluessel(f)} -->" in text
    assert "- `src/mod1.py`" in text and "hoch" in text
    assert "automatisch von Selbstbild; mit Label wontfix schließen, um zu unterdrücken" in text


def test_marke_im_text() -> None:
    titel, text = sammel_text("doppelung", "vibemind-os", [_fund(1)])
    assert titel == "[selbstbild] Sammlung Doppelung – vibemind-os"
    assert "<!-- selbstbild:sammel=doppelung:vibemind-os -->" in text
    assert text.count(BEGIN) == 1 and text.count(END) == 1


def test_sammel_sortierung_und_escape() -> None:
    fs = [_fund(1, "niedrig"), Fund("doppelung", "hoch", "vibemind-os", "a|b", "t", ("x",), "z"), _fund(2, "mittel")]
    _, text = sammel_text("doppelung", "vibemind-os", fs)
    assert text.index("a" + chr(92) + "|b") < text.index("Fund 2") < text.index("Fund 1")
    assert sammel_text("doppelung", "vibemind-os", fs) == sammel_text("doppelung", "vibemind-os", list(reversed(fs)))


def test_sammel_kappe() -> None:
    fs = [_fund(i, anker=f"src/sehr/langer/pfad/zu/modul_{i:05d}.py") for i in range(5000)]
    _, text = sammel_text("doppelung", "vibemind-os", fs)
    assert len(text) < 65536 and len(text) <= 60000
    assert "… und " in text and " weitere, siehe lokaler Bericht" in text
    k = int(text.split("… und ")[1].split(" weitere")[0])
    assert 0 < k < 5000 and text.count("\n| mittel |") == 5000 - k


def test_sammel_ohne_kappe_bei_wenig() -> None:
    _, text = sammel_text("tot", "r", [_fund(1)], max_zeichen=60000)
    assert "weitere, siehe" not in text


def test_block_ersetzen_erhaelt_handtext() -> None:
    alt = f"Mein Text oben\n{BEGIN}\nalt\n{END}\nMein Text unten\n"
    neu = block_ersetzen(alt, f"{BEGIN}\nneu\n{END}")
    assert neu == f"Mein Text oben\n{BEGIN}\nneu\n{END}\nMein Text unten\n"


def test_block_ersetzen_crlf() -> None:
    alt = f"oben\r\n{BEGIN}\r\nalt\r\n{END}\r\nunten\r\n"
    neu = block_ersetzen(alt, f"{BEGIN}\nzeile1\nzeile2\n{END}")
    assert neu == f"oben\r\n{BEGIN}\r\nzeile1\r\nzeile2\r\n{END}\r\nunten\r\n"


def test_block_ersetzen_fehlt_haengt_an() -> None:
    neu = block_ersetzen("Hand\n", f"{BEGIN}\nx\n{END}")
    assert neu.startswith("Hand\n") and neu.endswith(f"{BEGIN}\nx\n{END}")


def test_block_ersetzen_doppelt_oder_kaputt() -> None:
    with pytest.raises(ValueError):
        block_ersetzen(f"{BEGIN}\na\n{END}\n{BEGIN}\nb\n{END}", f"{BEGIN}\nx\n{END}")
    with pytest.raises(ValueError):
        block_ersetzen(f"{BEGIN}\na\n", f"{BEGIN}\nx\n{END}")
    with pytest.raises(ValueError):
        block_ersetzen(f"{END}\na\n{BEGIN}\n", f"{BEGIN}\nx\n{END}")


def test_schluesselverdacht_im_pfad() -> None:
    assert pruefen("pfad/ghp_" + "a" * 36 + ".py") == "GitHub-Token (klassisch)"
    assert pruefen("src/normal.py") is None
