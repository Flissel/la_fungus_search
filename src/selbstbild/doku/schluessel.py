"""Schluessel-Filter fuer alle Ausgaben (Stufe 4, Task 9; Fix-Runde 1: zentral fuer Notizen, Seiten und Pruefliste).

Genau die Muster des Task-9-Briefs. Ein Treffer wird nur mit dem NAMEN des Musters gemeldet -- der Name enthaelt nie
einen Teil des Werts (auch kein Praefix wie `ghp_` oder `sk-`)."""
from __future__ import annotations

import re

SCHLUESSEL_MUSTER: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Final-Fix I2: auch `sk-ant-api03-…` und `sk-proj-…` (Bindestrich/Unterstrich im Wert); links kein Buchstabe und
    # keine Ziffer, sonst traefe schon Fliesstext wie `risk-assessment-report-v2`
    ("OpenAI-artiger API-Schlüssel", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("GitHub-Token (klassisch)", re.compile(r"ghp_[A-Za-z0-9]{30,}")),
    ("GitHub-Token (feingranular)", re.compile(r"github_pat_")),
    ("AWS-Zugriffsschlüssel", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("Slack-Token", re.compile(r"xox[bp]-")),
    ("JSON Web Token", re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.")),
    # Final-Fix I1: der ganze Schluessel bis zur END-Zeile (oder bis zum Textende), nicht nur die BEGIN-Zeile
    ("privater Schlüssel (PEM)", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.DOTALL)),
    ("Telegram-Bot-Token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),  # Final-Fix I2
)


def schluessel_muster(text: str) -> str | None:
    """Name des ersten Musters, das im Text vorkommt, sonst None."""
    for name, muster in SCHLUESSEL_MUSTER:
        if muster.search(text):
            return name
    return None


def sieht_aus_wie_schluessel(text: str) -> bool:
    """Trifft eines der Muster `sk-…`, `ghp_…`, `github_pat_`, `AKIA…`, `xox[bp]-`, JWT, PEM-Privatschluessel oder
    Telegram-Bot-Token?"""
    return schluessel_muster(text) is not None


def schwaerzen(text: str) -> str:
    """Jeden Treffer durch `[Schlüssel-Verdacht: <Muster>]` ersetzen (fuer Texte, die nie einfrieren duerfen:
    Pruefliste, Index, Pruefpunkt-Texte). Der Wert verschwindet ganz, auch ein Praefix wie `xoxb-`."""
    for name, muster in SCHLUESSEL_MUSTER:
        text = muster.sub(f"[Schlüssel-Verdacht: {name}]", text)
    # die kurzen Muster (`github_pat_`, `xox[bp]-`) nehmen den Rest des Werts nicht mit: bis zum Wortende entfernen
    return re.sub(r"(\[Schlüssel-Verdacht: [^\]]+\])[A-Za-z0-9_\-.]*", r"\1", text)
