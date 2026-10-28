"""Rust- und Go-Parser: Symbole, Importe, Aufrufe, Env-Lesungen, Einstiege.

Knotenlauf ueber den tree-sitter-Baum. Gespeichert werden nur Namen, nie
String-Inhalte (ausser dem Env-Namen selbst und Import-Modulnamen).
"""
from __future__ import annotations

import posixpath
from typing import Literal

from tree_sitter import Node

from selbstbild.parser.python import _digest, _text
from selbstbild.sprachen import parser_fuer
from selbstbild.typen import (
    Aufruf, DateiErgebnis, Einstieg, EnvLesung, Import, Symbol,
)

_RUST_ENV_FUNKTIONEN = {"std.env.var", "env.var", "std.env.var_os", "env.var_os"}
_RUST_ENV_MAKROS = {"env", "option_env"}
_RUST_TYPEN = {"struct_item", "enum_item", "trait_item", "union_item"}
_RUST_TESTATTRIBUTE = {"test", "tokio::test"}
_GO_ENV_FUNKTIONEN = {"os.Getenv", "os.LookupEnv"}
_GO_TEST_PRAEFIXE = ("Test", "Benchmark", "Fuzz")


def _rust_literal(n: Node | None) -> str | None:
    """Inhalt eines einfachen String-Literals, sonst None."""
    if n is None or n.type != "string_literal":
        return None
    return "".join(_text(k) for k in n.children if k.type == "string_content")


def _go_literal(n: Node | None) -> str | None:
    if n is None or n.type not in ("interpreted_string_literal", "raw_string_literal"):
        return None
    return "".join(
        _text(k) for k in n.children
        if k.type in ("interpreted_string_literal_content", "raw_string_literal_content"))


def _ohne_leerraum(n: Node) -> str:
    return "".join(_text(n).split())


class _RustSammler:
    def __init__(self) -> None:
        self.symbole: list[Symbol] = []
        self.importe: list[Import] = []
        self.aufrufe: list[Aufruf] = []
        self.env: list[EnvLesung] = []
        self.einstiege: list[Einstieg] = []
        self.alias: dict[str, str] = {}

    # --- Importe ---------------------------------------------------------
    def _use_eintrag(self, importe: dict[str, list[str]], modul: str, name: str,
                     alias: str | None = None) -> None:
        importe.setdefault(modul, [])
        if name:
            importe[modul].append(name)
        if alias is not None and name:
            self.alias[alias] = name

    def _use_pfad(self, importe: dict[str, list[str]], prefix: str, pfad: str,
                  alias: str | None) -> None:
        voll = f"{prefix}::{pfad}" if prefix else pfad
        modul, trenner, name = voll.rpartition("::")
        if not trenner:
            self._use_eintrag(importe, voll, "", None)
            if alias is not None:
                self.alias[alias] = voll
            return
        self._use_eintrag(importe, modul, name, alias)

    def _use_knoten(self, importe: dict[str, list[str]], n: Node, prefix: str) -> None:
        t = n.type
        if t in ("identifier", "scoped_identifier"):
            self._use_pfad(importe, prefix, _ohne_leerraum(n), None)
        elif t == "use_as_clause":
            pfad = n.child_by_field_name("path")
            alias = n.child_by_field_name("alias")
            if pfad is not None:
                self._use_pfad(importe, prefix, _ohne_leerraum(pfad),
                               _text(alias) if alias is not None else None)
        elif t == "scoped_use_list":
            pfad = n.child_by_field_name("path")
            liste = n.child_by_field_name("list")
            neu = prefix
            if pfad is not None:
                neu = f"{prefix}::{_ohne_leerraum(pfad)}" if prefix else _ohne_leerraum(pfad)
            if liste is not None:
                self._use_liste(importe, liste, neu)
        elif t == "use_list":
            self._use_liste(importe, n, prefix)
        elif t == "use_wildcard":
            rest = "::".join(p for p in _ohne_leerraum(n).split("::") if p != "*")
            modul = f"{prefix}::{rest}" if prefix and rest else (prefix or rest)
            self._use_eintrag(importe, modul, "*")

    def _use_liste(self, importe: dict[str, list[str]], liste: Node, prefix: str) -> None:
        for k in liste.named_children:
            if k.type == "self":
                self._use_eintrag(importe, prefix, "")
            else:
                self._use_knoten(importe, k, prefix)

    def _use(self, n: Node) -> None:
        arg = n.child_by_field_name("argument")
        if arg is None:
            return
        importe: dict[str, list[str]] = {}
        self._use_knoten(importe, arg, "")
        zeile = n.start_point[0] + 1
        for modul, namen in importe.items():
            erstes = modul.partition("::")[0]
            self.importe.append(Import(modul, tuple(namen), zeile, erstes in ("self", "super")))

    # --- Aufrufe, Env ----------------------------------------------------
    def _aufloesen(self, name: str) -> str:
        kopf, punkt, rest = name.partition(".")
        return f"{self.alias.get(kopf, kopf)}{punkt}{rest}"

    def _name(self, fn: Node) -> str | None:
        if fn.type == "identifier":
            return _text(fn)
        if fn.type == "scoped_identifier":
            return _ohne_leerraum(fn).replace("::", ".")
        if fn.type == "field_expression":
            wert = fn.child_by_field_name("value")
            feld = fn.child_by_field_name("field")
            if feld is None:
                return None
            if wert is not None and wert.type in ("self", "identifier"):
                return f"{_text(wert)}.{_text(feld)}"
            return _text(feld)
        if fn.type == "generic_function":
            inner = fn.child_by_field_name("function")
            return self._name(inner) if inner is not None else None
        return None

    def _aufruf(self, n: Node, scope: str | None) -> None:
        fn = n.child_by_field_name("function")
        if fn is None:
            return
        name = self._name(fn)
        if name is None:
            return
        zeile = n.start_point[0] + 1
        self.aufrufe.append(Aufruf(scope, self._aufloesen(name), zeile))
        if name in _RUST_ENV_FUNKTIONEN:
            args = n.child_by_field_name("arguments")
            erstes = args.named_children[0] if args is not None and args.named_children else None
            lit = _rust_literal(erstes)
            if lit is not None:
                self.env.append(EnvLesung(lit, zeile, scope))

    def _makro(self, n: Node, scope: str | None) -> None:
        name = n.child_by_field_name("macro")
        if name is None or _text(name) not in _RUST_ENV_MAKROS:
            return
        baum = next((k for k in n.children if k.type == "token_tree"), None)
        erstes = baum.named_children[0] if baum is not None and baum.named_children else None
        lit = _rust_literal(erstes)
        if lit is not None:
            self.env.append(EnvLesung(lit, n.start_point[0] + 1, scope))

    # --- Symbole ---------------------------------------------------------
    def _symbol(self, n: Node, kurz: str, scope: str | None,
                art: Literal["funktion", "klasse", "methode"]) -> str:
        qual = kurz if scope is None else f"{scope}.{kurz}"
        self.symbole.append(Symbol(qual, art, n.start_point[0] + 1, n.end_point[0] + 1, _digest(n)))
        return qual

    @staticmethod
    def _ist_test(n: Node) -> bool:
        vor = n.prev_named_sibling
        while vor is not None and vor.type in ("attribute_item", "line_comment", "block_comment"):
            if vor.type == "attribute_item":
                attr = next((k for k in vor.named_children if k.type == "attribute"), None)
                if attr is not None and _ohne_leerraum(attr) in _RUST_TESTATTRIBUTE:
                    return True
            vor = vor.prev_named_sibling
        return False

    @staticmethod
    def _impl_typ(n: Node) -> str | None:
        typ = n.child_by_field_name("type")
        while typ is not None and typ.type in ("generic_type", "reference_type"):
            typ = typ.child_by_field_name("type")
        return _text(typ) if typ is not None else None

    def _rekursiv(self, n: Node, scope: str | None) -> None:
        for k in n.children:
            self.besuche(k, scope)

    def besuche(self, n: Node, scope: str | None) -> None:
        t = n.type
        name_n = n.child_by_field_name("name")
        if t == "function_item" and name_n is not None:
            kurz = _text(name_n)
            art: Literal["funktion", "methode"] = "methode" if scope is not None else "funktion"
            qual = self._symbol(n, kurz, scope, art)
            if kurz == "main" and scope is None and n.parent is not None \
                    and n.parent.type == "source_file":
                self.einstiege.append(Einstieg("main", qual, n.start_point[0] + 1))
            if self._ist_test(n):
                self.einstiege.append(Einstieg("test", qual, n.start_point[0] + 1))
            self._rekursiv(n, qual)
            return
        if t in _RUST_TYPEN and name_n is not None:
            qual = self._symbol(n, _text(name_n), scope, "klasse")
            self._rekursiv(n, qual if t == "trait_item" else scope)
            return
        if t == "impl_item":
            typ = self._impl_typ(n)
            self._rekursiv(n, typ if typ is not None else scope)
            return
        if t == "use_declaration":
            self._use(n)
            return
        if t == "call_expression":
            self._aufruf(n, scope)
        elif t == "macro_invocation":
            self._makro(n, scope)
        self._rekursiv(n, scope)


def parse_rust(pfad: str, quelle: bytes) -> DateiErgebnis:
    baum = parser_fuer("rust").parse(quelle)
    s = _RustSammler()
    s.besuche(baum.root_node, None)
    return DateiErgebnis(
        symbole=tuple(s.symbole),
        importe=tuple(s.importe),
        aufrufe=tuple(s.aufrufe),
        env=tuple(s.env),
        einstiege=tuple(s.einstiege),
        fehlerhaft=baum.root_node.has_error,
    )


class _GoSammler:
    def __init__(self, pfad: str) -> None:
        norm = pfad.replace("\\", "/")
        self.testdatei = posixpath.basename(norm).endswith("_test.go")
        self.paket: str | None = None
        self.symbole: list[Symbol] = []
        self.importe: list[Import] = []
        self.aufrufe: list[Aufruf] = []
        self.env: list[EnvLesung] = []
        self.einstiege: list[Einstieg] = []
        self.alias: dict[str, str] = {}

    def _import_spec(self, n: Node) -> None:
        modul = _go_literal(n.child_by_field_name("path"))
        if modul is None:
            return
        name = n.child_by_field_name("name")
        if name is not None and name.type == "package_identifier":
            self.alias[_text(name)] = modul
        self.importe.append(Import(modul, (), n.start_point[0] + 1, False))

    def _aufloesen(self, name: str) -> str:
        kopf, punkt, rest = name.partition(".")
        return f"{self.alias.get(kopf, kopf)}{punkt}{rest}"

    def _name(self, fn: Node) -> str | None:
        if fn.type == "identifier":
            return _text(fn)
        if fn.type == "selector_expression":
            operand = fn.child_by_field_name("operand")
            feld = fn.child_by_field_name("field")
            if feld is None:
                return None
            if operand is not None and operand.type == "identifier":
                return f"{_text(operand)}.{_text(feld)}"
            return _text(feld)
        return None

    def _aufruf(self, n: Node, scope: str | None) -> None:
        fn = n.child_by_field_name("function")
        if fn is None:
            return
        name = self._name(fn)
        if name is None:
            return
        zeile = n.start_point[0] + 1
        self.aufrufe.append(Aufruf(scope, self._aufloesen(name), zeile))
        if self._aufloesen(name) in _GO_ENV_FUNKTIONEN or name in _GO_ENV_FUNKTIONEN:
            args = n.child_by_field_name("arguments")
            erstes = args.named_children[0] if args is not None and args.named_children else None
            lit = _go_literal(erstes)
            if lit is not None:
                self.env.append(EnvLesung(lit, zeile, scope))

    def _symbol(self, n: Node, kurz: str, scope: str | None,
                art: Literal["funktion", "klasse", "methode"]) -> str:
        qual = kurz if scope is None else f"{scope}.{kurz}"
        self.symbole.append(Symbol(qual, art, n.start_point[0] + 1, n.end_point[0] + 1, _digest(n)))
        return qual

    @staticmethod
    def _empfaengertyp(n: Node) -> str | None:
        rec = n.child_by_field_name("receiver")
        if rec is None:
            return None
        for p in rec.named_children:
            typ = p.child_by_field_name("type")
            while typ is not None and typ.type in ("pointer_type", "generic_type", "parenthesized_type"):
                inner = typ.child_by_field_name("type") or next(iter(typ.named_children), None)
                typ = inner
            if typ is not None:
                return _text(typ)
        return None

    def _rekursiv(self, n: Node, scope: str | None) -> None:
        for k in n.children:
            self.besuche(k, scope)

    def besuche(self, n: Node, scope: str | None) -> None:
        t = n.type
        if t == "package_clause":
            ident = next((k for k in n.children if k.type == "package_identifier"), None)
            self.paket = _text(ident) if ident is not None else None
            return
        if t == "function_declaration":
            name_n = n.child_by_field_name("name")
            if name_n is not None:
                kurz = _text(name_n)
                qual = self._symbol(n, kurz, scope, "funktion")
                zeile = n.start_point[0] + 1
                if kurz == "main" and self.paket == "main":
                    self.einstiege.append(Einstieg("main", qual, zeile))
                if self.testdatei and kurz.startswith(_GO_TEST_PRAEFIXE):
                    self.einstiege.append(Einstieg("test", qual, zeile))
                self._rekursiv(n, qual)
                return
        elif t == "method_declaration":
            name_n = n.child_by_field_name("name")
            if name_n is not None:
                typ = self._empfaengertyp(n)
                oberscope = typ if typ is not None else scope
                qual = self._symbol(n, _text(name_n), oberscope, "methode")
                self._rekursiv(n, qual)
                return
        elif t == "type_spec":
            name_n = n.child_by_field_name("name")
            if name_n is not None:
                self._symbol(n, _text(name_n), scope, "klasse")
        elif t == "import_spec":
            self._import_spec(n)
            return
        elif t == "call_expression":
            self._aufruf(n, scope)
        self._rekursiv(n, scope)


def parse_go(pfad: str, quelle: bytes) -> DateiErgebnis:
    baum = parser_fuer("go").parse(quelle)
    s = _GoSammler(pfad)
    s.besuche(baum.root_node, None)
    return DateiErgebnis(
        symbole=tuple(s.symbole),
        importe=tuple(s.importe),
        aufrufe=tuple(s.aufrufe),
        env=tuple(s.env),
        einstiege=tuple(s.einstiege),
        fehlerhaft=baum.root_node.has_error,
    )
