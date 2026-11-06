"""Regeldatei `selbstbild.yaml`: Laden und Validieren."""
from __future__ import annotations

from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field

REGELDATEI = "selbstbild.yaml"

STANDARD_AUSGENOMMEN: tuple[str, ...] = (
    "node_modules/**",
    "**/node_modules/**",
    "dist/**",
    "build/**",
    "**/.venv*/**",
    "**/__pycache__/**",
    "**/*.min.js",
    "**/*.lock",
    "**/package-lock.json",
    "**/pnpm-lock.yaml",
    "**/uv.lock",
    ".git/**",
)


class Strukturregel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    art: str
    ort: str
    pflicht: list[str]
    verboten: list[str] = Field(default_factory=list)
    hinweis: str | None = None


class IssueZiel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str | None = None
    privat: bool = True


FundArt = Literal["doku_widerspruch", "doppelung", "tot", "konfig"]
# = selbstbild.funde.typen.ARTEN (gepinnt durch test_fund_arten_wie_typen)
FUND_ARTEN: tuple[str, ...] = get_args(FundArt)
_REPO_MUSTER = r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
_KONTO_MUSTER = r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$"


class IssueZielEintrag(BaseModel):
    """Ein Issue-Ziel (Stufe 5): GitHub-Repo, gh-Konto (nur je Aufruf, D2) und ob das Repo oeffentlich ist."""
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(pattern=_REPO_MUSTER)
    konto: str | None = Field(default=None, pattern=_KONTO_MUSTER)
    # Fix-Runde 1 (I2): D6 fail-closed -- ohne Angabe gilt ein Ziel als oeffentlich (kein konfig)
    oeffentlich: bool = True


class IssuesAbschnitt(BaseModel):
    """Abschnitt `issues:` (Stufe 5). Ersetzt `issue_ziel` fachlich (Ruling S5-1)."""
    model_config = ConfigDict(extra="forbid")

    ziel_standard: IssueZielEintrag | None = None
    ziele: dict[str, IssueZielEintrag] = Field(default_factory=dict)
    arten: list[FundArt] = Field(default_factory=lambda: list(FUND_ARTEN))
    max_einzel_je_tag: int = Field(default=5, ge=0, le=20)


class Regeln(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    doku_pfade: list[str] = Field(default_factory=lambda: ["**/*.md"])
    struktur: list[Strukturregel] = Field(default_factory=list)
    einstiegspunkte: list[str] = Field(default_factory=list)
    ausgenommen: list[str] = Field(default_factory=list)
    waechter_zuordnung: dict[str, list[str]] = Field(default_factory=dict)
    issue_ziel: IssueZiel = Field(default_factory=IssueZiel)
    # Stufe 5: fehlt der Abschnitt, gilt `issue_ziel.repo` als Ziel ohne Konto (nur Bericht, Ruling S5-1)
    issues: IssuesAbschnitt | None = None
    dienste: str | None = None
    max_dateigroesse_kb: int = 512
    # Globs relativ zur äußeren Wurzel. Gilt für alle darunter vermessenen Repos, wie `doku.ausschluss_pfade`.
    fremdcode: list[str] = Field(default_factory=list)
    # Doku-Loop (Stufe 4): Schluessel und Werte prueft erst `selbstbild.doku.konfig.aus_regeln`
    doku: dict[str, Any] | None = None


def regeln_laden(text: str | None) -> Regeln:
    """Parst YAML-Text; `None`, leer oder nur Kommentare ergibt die Vorgaben."""
    if text is None:
        return Regeln()
    daten = yaml.safe_load(text)
    if daten is None:
        return Regeln()
    return Regeln.model_validate(daten)
