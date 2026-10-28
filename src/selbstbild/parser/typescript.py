"""TypeScript-/JavaScript-Parser: Symbole, Importe, Aufrufe, Env-Lesungen, Einstiege.

Arbeitet per Knotenlauf ueber den tree-sitter-Baum (Sprachen typescript, tsx,
javascript). Gespeichert werden nur Namen, nie String-Inhalte (ausser dem Env-Namen
selbst und Import-Modulnamen).
"""
from __future__ import annotations

import posixpath
import re
from typing import Literal

from tree_sitter import Node

from selbstbild.parser.python import _digest, _text
from selbstbild.sprachen import parser_fuer
from selbstbild.typen import (
    Aufruf, DateiErgebnis, Einstieg, EnvLesung, Import, Symbol,
)

_ENV_OBJEKTE = {"process.env", "import.meta.env"}
_HTTP_METHODEN = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_EXPRESS_VERBEN = {"get", "post", "put", "patch", "delete", "use"}
_TEST_FUNKTIONEN = {"it", "test", "describe"}
_NEXT_ROUTE_RE = re.compile(r"(?:^|/)app/(?:.*/)?route\.(?:ts|tsx|js|jsx|mjs|cjs|mts|cts)$")
_TESTDATEI_RE = re.compile(r"\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs|mts|cts)$")
_FUNKTIONSWERTE = {"arrow_function", "function_expression", "function", "generator_function"}
_FUNKTIONS_DEKL = {"function_declaration", "generator_function_declaration"}
_KLASSEN_DEKL = {"class_declaration", "abstract_class_declaration"}


def _literal(n: Node | None) -> str | None:
    """Inhalt eines einfachen String-Literals (ohne Interpolation), sonst None."""
    if n is None:
        return None
    if n.type == "template_string":
        if any(k.type == "template_substitution" for k in n.children):
            return None
    elif n.type != "string":
        return None
    return "".join(_text(k) for k in n.children if k.type in ("string_fragment", "escape_sequence"))


def _punktname(n: Node) -> str | None:
    """Dotted name aus identifier/this/member_expression-Ketten, sonst None."""
    if n.type in ("identifier", "this", "super"):
        return _text(n)
    if n.type == "meta_property":
        return "import.meta"
    if n.type == "member_expression":
        obj = n.child_by_field_name("object")
        prop = n.child_by_field_name("property")
        if obj is None or prop is None:
            return None
        kopf = _punktname(obj)
        return None if kopf is None else f"{kopf}.{_text(prop)}"
    return None


class _Sammler:
    def __init__(self, pfad: str) -> None:
        norm = pfad.replace("\\", "/")
        self.next_route = bool(_NEXT_ROUTE_RE.search(norm))
        self.testdatei = bool(_TESTDATEI_RE.search(posixpath.basename(norm)))
        self.symbole: list[Symbol] = []
        self.importe: list[Import] = []
        self.aufrufe: list[Aufruf] = []
        self.env: list[EnvLesung] = []
        self.einstiege: list[Einstieg] = []
        self.alias: dict[str, str] = {}

    # --- Importe ---------------------------------------------------------
    def _import(self, n: Node) -> None:
        modul = _literal(n.child_by_field_name("source"))
        if modul is None:
            return
        namen: list[str] = []
        for k in n.children:
            if k.type != "import_clause":
                continue
            for c in k.children:
                if c.type == "identifier":
                    namen.append(_text(c))
                elif c.type == "namespace_import":
                    ident = next((x for x in c.children if x.type == "identifier"), None)
                    namen.append("*")
                    if ident is not None:
                        self.alias[_text(ident)] = modul
                elif c.type == "named_imports":
                    for s in c.children:
                        if s.type != "import_specifier":
                            continue
                        name = s.child_by_field_name("name")
                        alias = s.child_by_field_name("alias")
                        if name is None:
                            continue
                        namen.append(_text(name))
                        if alias is not None:
                            self.alias[_text(alias)] = _text(name)
        self.importe.append(Import(modul, tuple(namen), n.start_point[0] + 1, modul.startswith(".")))

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
        if name is None and fn.type == "member_expression":
            prop = fn.child_by_field_name("property")
            name = _text(prop) if prop is not None else None
        if name is None:
            return
        self.aufrufe.append(Aufruf(scope, self._aufloesen(name), zeile))
        if self.testdatei and name in _TEST_FUNKTIONEN:
            self.einstiege.append(Einstieg("test", None, zeile))
        obj = fn.child_by_field_name("object") if fn.type == "member_expression" else None
        if obj is not None and obj.type == "identifier" \
                and name.rpartition(".")[2] in _EXPRESS_VERBEN:
            args = n.child_by_field_name("arguments")
            erstes = args.named_children[0] if args is not None and args.named_children else None
            pfad = _literal(erstes)
            if pfad is not None and pfad.startswith("/"):
                self.einstiege.append(Einstieg("route", scope, zeile))

    def _env_member(self, n: Node, scope: str | None) -> None:
        obj = n.child_by_field_name("object")
        prop = n.child_by_field_name("property")
        if obj is None or prop is None or prop.type != "property_identifier":
            return
        if _punktname(obj) in _ENV_OBJEKTE:
            self.env.append(EnvLesung(_text(prop), n.start_point[0] + 1, scope))

    def _env_index(self, n: Node, scope: str | None) -> None:
        obj = n.child_by_field_name("object")
        if obj is None or _punktname(obj) not in _ENV_OBJEKTE:
            return
        name = _literal(n.child_by_field_name("index"))
        if name is not None:
            self.env.append(EnvLesung(name, n.start_point[0] + 1, scope))

    # --- Symbole ---------------------------------------------------------
    def _symbol(self, n: Node, kurz: str, scope: str | None,
                art: Literal["funktion", "klasse", "methode"]) -> str:
        qual = kurz if scope is None else f"{scope}.{kurz}"
        self.symbole.append(Symbol(qual, art, n.start_point[0] + 1, n.end_point[0] + 1, _digest(n)))
        return qual

    def _route_einstieg(self, kurz: str, scope: str | None, n: Node) -> None:
        if self.next_route and scope is None and kurz in _HTTP_METHODEN:
            self.einstiege.append(Einstieg("route", kurz, n.start_point[0] + 1))

    def _rekursiv(self, n: Node, scope: str | None) -> None:
        for k in n.children:
            self.besuche(k, scope)

    def besuche(self, n: Node, scope: str | None) -> None:
        t = n.type
        name_n = n.child_by_field_name("name")
        if t in _FUNKTIONS_DEKL or t in _KLASSEN_DEKL:
            if name_n is not None:
                art: Literal["funktion", "klasse"] = "klasse" if t in _KLASSEN_DEKL else "funktion"
                qual = self._symbol(n, _text(name_n), scope, art)
                self._route_einstieg(_text(name_n), scope, n)
                self._rekursiv(n, qual)
                return
        elif t == "method_definition":
            if name_n is not None and name_n.type != "computed_property_name":
                qual = self._symbol(n, _text(name_n), scope, "methode")
                self._rekursiv(n, qual)
                return
        elif t in ("variable_declarator", "public_field_definition"):
            wert = n.child_by_field_name("value")
            if (name_n is not None and wert is not None and wert.type in _FUNKTIONSWERTE
                    and name_n.type in ("identifier", "property_identifier")):
                art2: Literal["funktion", "methode"] = (
                    "methode" if t == "public_field_definition" else "funktion")
                qual = self._symbol(n, _text(name_n), scope, art2)
                if t == "variable_declarator":
                    self._route_einstieg(_text(name_n), scope, n)
                self._rekursiv(n, qual)
                return
        elif t == "import_statement":
            self._import(n)
        elif t == "call_expression":
            self._aufruf(n, scope)
        elif t == "member_expression":
            self._env_member(n, scope)
        elif t == "subscript_expression":
            self._env_index(n, scope)
        self._rekursiv(n, scope)


def parse_ts(pfad: str, quelle: bytes,
             sprache: Literal["typescript", "tsx", "javascript"] = "typescript") -> DateiErgebnis:
    baum = parser_fuer(sprache).parse(quelle)
    s = _Sammler(pfad)
    s.besuche(baum.root_node, None)
    return DateiErgebnis(
        symbole=tuple(s.symbole),
        importe=tuple(s.importe),
        aufrufe=tuple(s.aufrufe),
        env=tuple(s.env),
        einstiege=tuple(s.einstiege),
        fehlerhaft=baum.root_node.has_error,
    )
