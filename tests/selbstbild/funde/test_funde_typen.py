from __future__ import annotations

from selbstbild.funde.typen import Fund, fund_schluessel


def _fund(anker: str) -> Fund:
    return Fund(art="tot", schwere="mittel", code_repo="r", titel="t", text="x", belege=("a.py",), anker=anker)


def test_schluessel_stabil() -> None:
    assert fund_schluessel(_fund("a")) == fund_schluessel(_fund("a"))
    assert fund_schluessel(_fund("a")) != fund_schluessel(_fund("b"))
    art, h = fund_schluessel(_fund("a")).split(":")
    assert art == "tot" and len(h) == 16
