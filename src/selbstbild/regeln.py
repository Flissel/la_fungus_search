"""Regeldatei `selbstbild.yaml`: Laden und Validieren."""
from __future__ import annotations

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


class Regeln(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    doku_pfade: list[str] = Field(default_factory=lambda: ["**/*.md"])
    struktur: list[Strukturregel] = Field(default_factory=list)
    einstiegspunkte: list[str] = Field(default_factory=list)
    ausgenommen: list[str] = Field(default_factory=list)
    waechter_zuordnung: dict[str, list[str]] = Field(default_factory=dict)
    issue_ziel: IssueZiel = Field(default_factory=IssueZiel)
    dienste: str | None = None
    max_dateigroesse_kb: int = 512


def regeln_laden(text: str | None) -> Regeln:
    """Parst YAML-Text; `None`, leer oder nur Kommentare ergibt die Vorgaben."""
    if text is None:
        return Regeln()
    daten = yaml.safe_load(text)
    if daten is None:
        return Regeln()
    return Regeln.model_validate(daten)
