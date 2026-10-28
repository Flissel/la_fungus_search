"""Python-Parser: Symbole, Importe, Aufrufe, Env-Lesungen, Einstiege, Tabellenzugriffe.

Arbeitet per Knotenlauf ueber den tree-sitter-Baum. Gespeichert werden nur Namen,
nie String-Inhalte (ausser dem Env-Namen selbst und erkannten Tabellennamen).
"""
from __future__ import annotations

import ast
import hashlib
import posixpath
import re

from tree_sitter import Node

from selbstbild.sprachen import parser_fuer
from selbstbild.typen import (
    Aufruf, DateiErgebnis, Einstieg, EnvLesung, Import, Symbol,
)

_ENV_AUFRUFE = {"os.getenv", "os.environ.get", "environ.get", "getenv"}
_ENV_TABELLEN = {"os.environ", "environ"}
# Tabellenzugriff nur in Strings, die mit einem SQL-Verb beginnen (R9). UPDATE zaehlt nur
# als "UPDATE <name> SET"; sonst waere "update the docs please" ein Treffer.
_SQL_ANFANG_RE = re.compile(
    r"\s*(?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP|MERGE|UPSERT)\b", re.IGNORECASE)
_STRING_KOPF_RE = re.compile(r"^[A-Za-z]*(?:\"\"\"|'''|\"|')")
_NAME = r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)"
_TABELLE_RE = re.compile(
    rf"\b(?:(?:FROM|JOIN|INTO|TABLE)\s+{_NAME}|UPDATE\s+{_NAME}\s+SET\b)", re.IGNORECASE)


def _text(n: Node) -> str:
    return n.text.decode("utf-8", errors="replace") if n.text is not None else ""


def _digest(n: Node) -> str:
    text = _text(n).replace("\r\n", "\n").replace("\r", "\n")
    zeilen = text.split("\n")
    while zeilen and not zeilen[0].strip():
        zeilen.pop(0)
    while zeilen and not zeilen[-1].strip():
        zeilen.pop()
    return hashlib.sha256("\n".join(zeilen).encode("utf-8")).hexdigest()


def _literal(n: Node | None) -> str | None:
    """Wert eines einfachen String-Literals (ohne f-String), sonst None."""
    if n is None or n.type != "string":
        return None
    try:
        wert = ast.literal_eval(_text(n))
    except (ValueError, SyntaxError):
        return None
    return wert if isinstance(wert, str) else None


def _punktname(n: Node) -> str | None:
    """Dotted name aus identifier/attribute-Ketten; None bei komplexeren Ausdruecken."""
    if n.type == "identifier":
        return _text(n)
    if n.type == "attribute":
        obj = n.child_by_field_name("object")
        attr = n.child_by_field_name("attribute")
        if obj is None or attr is None:
            return None
        kopf = _punktname(obj)
        return None if kopf is None else f"{kopf}.{_text(attr)}"
    return None


def _ist_testdatei(pfad: str) -> bool:
    name = posixpath.basename(pfad.replace("\\", "/"))
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


class _Sammler:
    def __init__(self, pfad: str) -> None:
        self.testdatei = _ist_testdatei(pfad)
        self.symbole: list[Symbol] = []
        self.importe: list[Import] = []
        self.aufrufe: list[Aufruf] = []
        self.env: list[EnvLesung] = []
        self.einstiege: list[Einstieg] = []
        self.tabellen: dict[tuple[str, int, str | None], None] = {}  # dict: dedupliziert, Reihenfolge stabil
        self.alias: dict[str, str] = {}

    # --- Importe ---------------------------------------------------------
    def _import(self, n: Node) -> None:
        zeile = n.start_point[0] + 1
        if n.type == "import_statement":
            for k in n.children_by_field_name("name"):
                if k.type == "aliased_import":
                    name = k.child_by_field_name("name")
                    alias = k.child_by_field_name("alias")
                    if name is not None:
                        self.importe.append(Import(_text(name), (), zeile, False))
                        if alias is not None:
                            self.alias[_text(alias)] = _text(name)
                else:
                    self.importe.append(Import(_text(k), (), zeile, False))
            return
        mod = n.child_by_field_name("module_name")
        relativ = mod is not None and mod.type == "relative_import"
        modul = _text(mod).lstrip(".") if mod is not None else ""
        namen: list[str] = []
        for k in n.children_by_field_name("name"):
            if k.type == "aliased_import":
                name = k.child_by_field_name("name")
                alias = k.child_by_field_name("alias")
                if name is not None:
                    namen.append(_text(name))
                    if alias is not None:
                        self.alias[_text(alias)] = _text(name)
            else:
                namen.append(_text(k))
        if any(c.type == "wildcard_import" for c in n.children):
            namen.append("*")
        self.importe.append(Import(modul, tuple(namen), zeile, relativ))

    # --- Aufrufe, Env ----------------------------------------------------
    def _aufloesen(self, name: str) -> str:
        kopf, punkt, rest = name.partition(".")
        return f"{self.alias.get(kopf, kopf)}{punkt}{rest}"

    def _aufruf(self, n: Node, scope: str | None) -> None:
        fn = n.child_by_field_name("function")
        if fn is None:
            return
        zeile = n.start_point[0] + 1
        name = _punktname(fn)
        if name is None and fn.type == "attribute":
            attr = fn.child_by_field_name("attribute")
            name = _text(attr) if attr is not None else None
        if name is None:
            return
        name = self._aufloesen(name)
        self.aufrufe.append(Aufruf(scope, name, zeile))
        if name in _ENV_AUFRUFE:
            args = n.child_by_field_name("arguments")
            erstes = args.named_children[0] if args is not None and args.named_children else None
            wert = _literal(erstes)
            if wert is not None:
                self.env.append(EnvLesung(wert, zeile, scope))

    def _subscript(self, n: Node, scope: str | None) -> None:
        wert = n.child_by_field_name("value")
        if wert is None or _punktname(wert) not in _ENV_TABELLEN:
            return
        name = _literal(n.child_by_field_name("subscript"))
        if name is not None:
            self.env.append(EnvLesung(name, n.start_point[0] + 1, scope))

    def _string(self, n: Node, scope: str | None) -> None:
        eltern = n.parent
        if eltern is not None:
            if eltern.type == "expression_statement":
                return  # loser String
            if eltern.type in ("block", "module"):
                erstes = next((k for k in eltern.named_children if k.type != "comment"), None)
                if erstes is not None and erstes.id == n.id:
                    return  # Docstring
        text = _text(n)
        inhalt = _STRING_KOPF_RE.sub("", text, count=1)
        if not _SQL_ANFANG_RE.match(inhalt):
            return
        for m in _TABELLE_RE.finditer(text):
            zeile = n.start_point[0] + 1 + text.count("\n", 0, m.start())
            self.tabellen[(m.group(1) or m.group(2), zeile, scope)] = None

    # --- Lauf ------------------------------------------------------------
    def besuche(self, n: Node, scope: str | None, in_klasse: bool) -> None:
        t = n.type
        if t in ("function_definition", "class_definition"):
            name_n = n.child_by_field_name("name")
            if name_n is not None:
                kurz = _text(name_n)
                qual = kurz if scope is None else f"{scope}.{kurz}"
                if t == "class_definition":
                    art = "klasse"
                else:
                    art = "methode" if in_klasse else "funktion"
                self.symbole.append(Symbol(
                    qual, art, n.start_point[0] + 1, n.end_point[0] + 1, _digest(n)))
                self._test_einstieg(kurz, art, scope, n)
                for k in n.children:
                    self.besuche(k, qual, t == "class_definition")
                return
        elif t in ("import_statement", "import_from_statement"):
            self._import(n)
        elif t == "call":
            self._aufruf(n, scope)
        elif t == "subscript":
            self._subscript(n, scope)
        elif t == "string":
            self._string(n, scope)
        elif t == "if_statement" and scope is None and self._ist_main_if(n):
            self.einstiege.append(Einstieg("main", None, n.start_point[0] + 1))
        # in_klasse gilt nur fuer die direkten Klassenkoerper-Kinder (block)
        for k in n.children:
            self.besuche(k, scope, in_klasse and t in ("block", "decorated_definition"))

    def _test_einstieg(self, kurz: str, art: str, scope: str | None, n: Node) -> None:
        if not self.testdatei or not kurz.startswith("test"):
            return
        if art == "funktion" and scope is None:
            qual = kurz
        elif art == "methode" and scope is not None and "." not in scope and scope.startswith("Test"):
            qual = f"{scope}.{kurz}"
        else:
            return
        self.einstiege.append(Einstieg("test", qual, n.start_point[0] + 1))

    @staticmethod
    def _ist_main_if(n: Node) -> bool:
        b = n.child_by_field_name("condition")
        if b is None or b.type != "comparison_operator":
            return False
        teile = _text(b).replace(" ", "").replace("'", '"')
        return teile in ('__name__=="__main__"', '"__main__"==__name__')


def parse_python(pfad: str, quelle: bytes) -> DateiErgebnis:
    baum = parser_fuer("python").parse(quelle)
    s = _Sammler(pfad)
    s.besuche(baum.root_node, None, False)
    return DateiErgebnis(
        symbole=tuple(s.symbole),
        importe=tuple(s.importe),
        aufrufe=tuple(s.aufrufe),
        env=tuple(s.env),
        einstiege=tuple(s.einstiege),
        tabellen_zugriffe=tuple(s.tabellen),
        fehlerhaft=baum.root_node.has_error,
    )
