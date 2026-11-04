"""Issue-Texte, Sammel-Bloecke, Kappe und Schluesselfilter (Stufe 5, Task 3)."""
from __future__ import annotations

from selbstbild.doku.schluessel import schluessel_muster
from selbstbild.funde.typen import Fund, fund_schluessel

BEGIN = "<!-- selbstbild:begin -->"
END = "<!-- selbstbild:end -->"
_ART_NAME = {
    "doku_widerspruch": "Doku-Widerspruch",
    "doppelung": "Doppelung",
    "tot": "Toter Code",
    "konfig": "Konfig",
}
_RANG = {"hoch": 0, "mittel": 1, "niedrig": 2}
_HINWEIS = "Issue automatisch von Selbstbild; mit Label wontfix schließen, um zu unterdrücken."


def _art_name(art: str) -> str:
    return _ART_NAME.get(art, art)


def _zelle(s: str) -> str:
    return " ".join(s.split()).replace("|", chr(92) + "|")


def einzel_text(f: Fund) -> tuple[str, str]:
    titel = f"[selbstbild] {_art_name(f.art)}: {f.titel} ({f.code_repo})"
    belege = "\n".join(f"- `{b}`" for b in f.belege) or "- (keine)"
    text = (
        f"<!-- selbstbild:schluessel={fund_schluessel(f)} -->\n\n"
        f"{f.text}\n\n"
        f"**Schwere:** {f.schwere}\n\n"
        f"**Belege:**\n{belege}\n\n"
        f"_{_HINWEIS}_\n"
    )
    return titel, text


def _sortiert(funde: list[Fund]) -> list[Fund]:
    return sorted(funde, key=lambda f: (_RANG.get(f.schwere, 9), f.code_repo, f.anker, f.titel))


def _zeile(f: Fund) -> str:
    return f"| {f.schwere} | `{_zelle(f.anker)}` | {_zelle(f.titel)} |"


def kappen_zeile(k: int) -> str:
    """Der Kappensatz fuer `k` weggelassene Zeilen (ohne fuehrende Leerzeilen)."""
    return f"… und {k} weitere, siehe lokaler Bericht"


# Platz fuer die Kappenzeile, die immer reserviert wird (K hat hoechstens 7 Stellen)
KAPPEN_RESERVE = len("\n\n" + kappen_zeile(9999999))


def kappen(zeilen: list[str], budget: int) -> tuple[list[str], int]:
    """Die ersten Zeilen, die (je mit einem Zeilenumbruch) ins Budget passen, und die Zahl der weggelassenen."""
    n, used = 0, 0
    for z in zeilen:
        if used + len(z) + 1 > budget:
            break
        used += len(z) + 1
        n += 1
    return zeilen[:n], len(zeilen) - n


def sammel_text(art: str, code_repo: str, funde: list[Fund], *, max_zeichen: int = 60000) -> tuple[str, str]:
    titel = f"[selbstbild] Sammlung {_art_name(art)} – {code_repo}"
    kopf = f"<!-- selbstbild:sammel={art}:{code_repo} -->\n\n"
    fuss = f"\n\n_{_HINWEIS}_\n"
    sortiert = _sortiert(funde)
    zeilen = [_zeile(f) for f in sortiert]
    tabellenkopf = f"{BEGIN}\n**{len(sortiert)} Funde**\n\n| Schwere | Fund | Beschreibung |\n|---|---|---|"
    reserve = KAPPEN_RESERVE + len(f"\n{END}")
    budget = max_zeichen - len(kopf) - len(fuss) - len(tabellenkopf) - reserve
    behalten, k = kappen(zeilen, budget)
    tab = tabellenkopf + "".join("\n" + z for z in behalten)
    if k:
        tab += "\n\n" + kappen_zeile(k)
    block = f"{tab}\n{END}"
    return titel, f"{kopf}{block}{fuss}"


def block_ersetzen(alter_text: str, neuer_block: str) -> str:
    """Ersetzt nur den markierten Block; Rest bleibt byte-gleich. Fehlt er, wird er angehaengt."""
    crlf = "\r\n" in alter_text
    block = neuer_block.replace("\r\n", "\n")
    if crlf:
        block = block.replace("\n", "\r\n")
    nb, ne = alter_text.count(BEGIN), alter_text.count(END)
    if nb > 1 or ne > 1:
        raise ValueError("mehrere Selbstbild-Blockmarken im Text")
    if nb != ne:
        raise ValueError("Selbstbild-Block unvollständig (Anfang oder Ende fehlt)")
    if nb == 0:
        zl = "\r\n" if crlf else "\n"
        sep = "" if (not alter_text or alter_text.endswith("\n")) else zl
        return f"{alter_text}{sep}{zl if alter_text else ''}{block}"
    a = alter_text.index(BEGIN)
    e = alter_text.index(END)
    if e < a:
        raise ValueError("Selbstbild-Block verdreht (Ende vor Anfang)")
    return alter_text[:a] + block + alter_text[e + len(END):]


def pruefen(text: str) -> str | None:
    """Mustername bei Schluessel-Verdacht, sonst None."""
    return schluessel_muster(text)
