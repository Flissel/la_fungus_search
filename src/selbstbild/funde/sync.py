"""Issue-Abgleich mit Bremsen (Stufe 5, Task 4).

Wahrheit fuer Dubletten ist GitHub, nicht der lokale Stand (Review-Fokus 1-2):
- Jeder Lauf liest den Bestand neu (`GitHub.issues`: Label `selbstbild`, Zustand `all`) und erkennt Issues an der
  Marke `<!-- selbstbild:schluessel=<art>:<hash16> -->` (Einzel) bzw. `<!-- selbstbild:sammel=<art>:<code_repo> -->`
  (Sammel), die `selbstbild.funde.text` in den Text schreibt.
- Der lokale Stand `issues-<hash>.json` ist nur Beschleuniger: Tageszaehler (zaehlt nur erfolgreiches Anlegen) und
  Marken-Cache. Fehlt er oder ist er kaputt, entstehen keine Dubletten; die Tagesgrenze nimmt zusaetzlich die heute
  angelegten Einzel-Issues aus dem Bestand (`created_at`), damit auch ohne Stand nicht mehr als die Grenze entsteht.
- Der erste gescheiterte Schreibaufruf bricht alle weiteren Schreibaufrufe ab (kein halber Zustand ueber den Fehler
  hinaus); der naechste Lauf findet bereits Angelegtes ueber die Marke.

Bremsen (Spec 6.9): nur `hoch` wird Einzel-Issue, hoechstens `max_einzel_je_tag` neue je Ziel und Tag, der Rest
landet im Sammel-Issue je Art und Code-Repo. `wontfix` (Label) oder `state_reason == not_planned` heisst: nie wieder
oeffnen. Oeffentliche Ziele bekommen nie `konfig` (D6); oeffentlich ist ein Ziel, wenn die Konfiguration es sagt
(Vorgabe) ODER GitHub es nicht als privat meldet (Fix-Runde 1 I2). Geschlossen wird nur fuer Code-Repos, die der Lauf
wirklich geprueft hat (`geprueft`), und nur ueber die Repo-Marke, nie ueber den Titel (Fix-Runde 1 I1). Fuer Arten mit unlesbarer Quelle (Ruling S5-7) wird nichts
geschlossen und das Sammel-Issue nicht angefasst. `senden=False` macht keinen einzigen schreibenden gh-Aufruf und
schreibt auch den lokalen Stand nicht.

`konfig` (Ruling S5-11) ist immer genau ein Einzel-Issue je Fund (das Inventar liefert einen je Code-Repo), nie eine
Sammel-Zeile: Text = `einzel_text` des Funds mit Repo-Marke, jeder Lauf ersetzt den ganzen Text, wenn er sich
geaendert hat (sonst kein Schreibaufruf). Es zaehlt nicht gegen `max_einzel_je_tag`. Schliessen wie jedes
Einzel-Issue (nur gepruefte Code-Repos, nicht bei unlesbarer Quelle); D6 und Schluesselfilter gelten unveraendert.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..doku.stand import repo_schluessel
from ..regeln import Regeln
from ..speicher import stand_basis
from .github import GitHub, GitHubFehler
from .sammeln import UNLESBAR_PRAEFIX
from .text import BEGIN, END, block_ersetzen, einzel_text, pruefen, sammel_text
from .typen import ARTEN, Fund, fund_schluessel

STAND_VERSION = 1
# Fix-Runde 1 (Minor 4): Leerraum um die Teile der Marken wird toleriert
_MARKE = re.compile(r"<!--\s*selbstbild:schluessel\s*=\s*([a-z_]+:[0-9a-f]{16})\s*-->")
_SAMMEL = re.compile(r"<!--\s*selbstbild:sammel\s*=\s*([a-z_]+):([^\s>][^>]*?)\s*-->")
_REPO_MARKE = re.compile(r"<!--\s*selbstbild:repo\s*=\s*([^\s>][^>]*?)\s*-->")
_BELEGE = re.compile(r"(\*\*Belege:\*\*\r?\n)(.*?)(?=\r?\n\r?\n|\Z)", re.DOTALL)
_COMMIT = re.compile(r"[0-9a-fA-F]{7,64}")
_RANG = {"hoch": 0, "mittel": 1, "niedrig": 2}
# Quelle -> Arten, die ihr Ergebnis aus dieser Quelle haben (S5-7)
_QUELLEN_ARTEN: dict[str, frozenset[str]] = {
    "doku": frozenset({"doku_widerspruch"}),
    "analyse": frozenset({"doppelung", "tot"}),
    # Ruling S5-12: Quellen des Konfig-Inventars (Task 5)
    "graph": frozenset({"konfig"}),
    "dienste": frozenset({"konfig"}),
    "submodul": frozenset({"konfig"}),
    "dateien": frozenset({"konfig"}),  # git ls-files des Inventars gescheitert
}
_TOR_GESPERRT = "toter Code: gesperrt"


# ---------- Ziele ----------

@dataclass(frozen=True)
class Ziel:
    """Ein GitHub-Repo, in das Funde gehen. `konto=None` heisst: nur Bericht, kein Senden (S5-1)."""
    repo: str
    konto: str | None
    oeffentlich: bool
    arten: tuple[str, ...] = ARTEN
    max_einzel_je_tag: int = 5
    nur: frozenset[str] | None = None  # None = alle Code-Repos ausser `ausser`
    ausser: frozenset[str] = frozenset()

    def zustaendig(self, code_repo: str) -> bool:
        return (self.nur is None or code_repo in self.nur) and code_repo not in self.ausser


def ziele_aus_regeln(regeln: Regeln) -> list[Ziel]:
    """Ziele aus `issues:` (Standard zuerst, dann nach Repo). Ohne `issues:` gilt `issue_ziel.repo` ohne Konto."""
    i = regeln.issues
    if i is None:
        if not regeln.issue_ziel.repo:
            return []
        return [Ziel(repo=regeln.issue_ziel.repo, konto=None, oeffentlich=not regeln.issue_ziel.privat)]
    arten = tuple(dict.fromkeys(i.arten))
    standard = i.ziel_standard
    gruppen: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for code_repo, e in sorted(i.ziele.items()):
        gruppen[e.repo].append((code_repo, e))
    aus: list[Ziel] = []

    def bauen(repo: str, eintraege: list[Any], nur: frozenset[str] | None, ausser: frozenset[str]) -> Ziel:
        konten = {e.konto for e in eintraege}
        if len(konten) > 1:
            raise ValueError(f"issues: Ziel {repo} mit verschiedenen Konten")
        return Ziel(repo=repo, konto=konten.pop(), oeffentlich=any(e.oeffentlich for e in eintraege), arten=arten,
                    max_einzel_je_tag=i.max_einzel_je_tag, nur=nur, ausser=ausser)

    if standard is not None:
        fremd = frozenset(c for r, es in gruppen.items() if r != standard.repo for c, _ in es)
        mit = [standard, *(e for _, e in gruppen.get(standard.repo, []))]
        aus.append(bauen(standard.repo, mit, None, fremd))
    for repo, es in sorted(gruppen.items()):
        if standard is not None and repo == standard.repo:
            continue
        aus.append(bauen(repo, [e for _, e in es], frozenset(c for c, _ in es), frozenset()))
    return aus


def nach_ziel(funde: Iterable[Fund], ziele: Sequence[Ziel]) -> tuple[list[tuple[Ziel, list[Fund]]], list[Fund]]:
    """Funde je Ziel (Reihenfolge der Ziele) und die Funde ohne Ziel."""
    je: dict[int, list[Fund]] = {n: [] for n in range(len(ziele))}
    ohne: list[Fund] = []
    for f in funde:
        n = next((k for k, z in enumerate(ziele) if z.zustaendig(f.code_repo)), None)
        (ohne if n is None else je[n]).append(f)
    return [(z, je[n]) for n, z in enumerate(ziele)], ohne


def unlesbare_arten(notizen: Iterable[str]) -> frozenset[str]:
    """Arten, deren Quelle in diesem Lauf nicht lesbar war (S5-7); fuer sie wird nichts geschlossen.

    `quelle-unlesbar:doku...` -> doku_widerspruch, `quelle-unlesbar:analyse...` -> doppelung und tot,
    `quelle-unlesbar:graph|dienste|submodul|dateien...` -> konfig (S5-12), eine unbekannte Quelle -> alle Arten (fail-closed). Ein gesperrtes Tor fuer toten Code zaehlt wie unlesbar:
    es liefert keine Tot-Funde, ohne dass der Code sich geaendert hat.
    """
    aus: set[str] = set()
    for n in notizen:
        if n.startswith(UNLESBAR_PRAEFIX):
            rest = n[len(UNLESBAR_PRAEFIX):]
            quelle = next((q for q in _QUELLEN_ARTEN if rest.startswith(q)), None)
            aus |= _QUELLEN_ARTEN[quelle] if quelle else set(ARTEN)
        elif n.startswith(_TOR_GESPERRT):
            aus.add("tot")
    return frozenset(aus)


# ---------- lokaler Stand ----------

def stand_pfad(wurzel: Path, basis: Path | None = None) -> Path:
    """`<Selbstbild-Stand>/issues-<repo_schluessel(wurzel)>.json`; `basis` fuer Tests."""
    return Path(basis if basis is not None else stand_basis()) / f"issues-{repo_schluessel(Path(wurzel))}.json"


@dataclass
class IssueStand:
    """Tageszaehler je Ziel-Repo und Marken-Cache. Nur Beschleuniger, nie Grundlage fuer Dubletten-Entscheidungen."""
    pfad: Path
    zaehler: dict[str, dict[str, Any]] = field(default_factory=dict)
    marken: dict[str, dict[str, int]] = field(default_factory=dict)
    # Ziel-Repo -> Grund: Senden gesperrt, bis ein Mensch den Eintrag loescht (z. B. `label_fehlt`)
    sperren: dict[str, str] = field(default_factory=dict)
    unlesbar: bool = False

    @classmethod
    def laden(cls, pfad: Path) -> IssueStand:
        pfad = Path(pfad)
        try:
            roh = json.loads(pfad.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(pfad)
        except (OSError, ValueError):
            return cls(pfad, unlesbar=True)
        try:
            zaehler = {str(r): {"tag": str(z["tag"]), "neu": int(z["neu"])} for r, z in roh["zaehler"].items()}
            marken = {str(r): {str(k): int(n) for k, n in m.items()} for r, m in roh["marken"].items()}
            sperren = {str(r): str(g) for r, g in (roh.get("sperren") or {}).items()}
        except (KeyError, TypeError, ValueError, AttributeError):
            return cls(pfad, unlesbar=True)
        return cls(pfad, zaehler, marken, sperren)

    def neu_heute(self, repo: str, heute: date) -> int:
        z = self.zaehler.get(repo)
        return int(z["neu"]) if z and z.get("tag") == heute.isoformat() else 0

    def zaehlen(self, repo: str, heute: date) -> None:
        self.zaehler[repo] = {"tag": heute.isoformat(), "neu": self.neu_heute(repo, heute) + 1}

    def speichern(self) -> None:
        self.pfad.parent.mkdir(parents=True, exist_ok=True)
        daten = {"version": STAND_VERSION, "zaehler": self.zaehler, "marken": self.marken, "sperren": self.sperren}
        fd, tmp = tempfile.mkstemp(prefix=self.pfad.name + ".", suffix=".tmp", dir=self.pfad.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(daten, f, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, self.pfad)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.unlesbar = False


# ---------- Bericht ----------

@dataclass
class Aktion:
    """`art`: anlegen, bearbeiten, oeffnen, schliessen, sammel_anlegen, sammel_bearbeiten, sammel_oeffnen,
    sammel_schliessen. `status`: ausgefuehrt, trocken, fehlgeschlagen, abgebrochen."""
    art: str
    schluessel: str
    titel: str
    nummer: int | None
    status: str


@dataclass
class Bericht:
    ziel: str
    senden: bool
    aktionen: list[Aktion] = field(default_factory=list)
    neu: int = 0  # neue Einzel-Issues in diesem Lauf (angelegt bzw. trocken geplant)
    unveraendert: int = 0
    unterdrueckt: list[str] = field(default_factory=list)  # Schluessel mit wontfix / not planned
    ueber_grenze: list[str] = field(default_factory=list)  # hoch, aber Tagesgrenze erreicht -> Sammel-Issue
    gesperrt: list[str] = field(default_factory=list)  # Schluesselfilter: "<schluessel>: <Mustername>"
    ausgelassen: list[str] = field(default_factory=list)
    notizen: list[str] = field(default_factory=list)
    fehler: str | None = None  # GitHubFehler.art des ersten gescheiterten Schreibaufrufs

    def text(self) -> str:
        modus = "senden" if self.senden else "trocken (nichts gesendet)"
        z = [f"## Issues → {self.ziel} ({modus})", ""]
        if self.fehler:
            z += [f"**Abgebrochen:** GitHub-Fehler `{self.fehler}`; weitere Schreibaufrufe ausgelassen.", ""]
        z.append(f"- neue Einzel-Issues: {self.neu}; unverändert: {self.unveraendert}")
        if self.ueber_grenze:
            z.append(f"- über der Tagesgrenze (diesmal im Sammel-Issue): {len(self.ueber_grenze)}")
        if self.unterdrueckt:
            z.append(f"- unterdrückt (wontfix / not planned): {len(self.unterdrueckt)}")
        for a in self.aktionen:
            nr = f"#{a.nummer}" if a.nummer is not None else "neu"
            z.append(f"- {a.art} {nr} [{a.status}]: {a.titel}")
        for g in self.gesperrt:
            z.append(f"- nicht gesendet (Schlüssel-Verdacht): {g}")
        for x in self.ausgelassen:
            z.append(f"- ausgelassen: {x}")
        for n in self.notizen:
            z.append(f"- Hinweis: {n}")
        return "\n".join(z) + "\n"


# ---------- Abgleich ----------

def _labels(issue: dict[str, Any]) -> set[str]:
    aus: set[str] = set()
    for lab in issue.get("labels") or ():
        name = lab.get("name") if isinstance(lab, dict) else lab
        if isinstance(name, str):
            aus.add(name.casefold())
    return aus


def _wontfix(issue: dict[str, Any]) -> bool:
    return "wontfix" in _labels(issue) or issue.get("state_reason") == "not_planned"


def _offen(issue: dict[str, Any]) -> bool:
    return issue.get("state") == "open"


def _body(issue: dict[str, Any]) -> str:
    b = issue.get("body")
    return b if isinstance(b, str) else ""


def _code_repo(issue: dict[str, Any]) -> str | None:
    """Code-Repo nur aus der Repo-Marke; der Titel ist vom Menschen aenderbar und zaehlt nie (Fix-Runde 1 I1)."""
    m = _REPO_MARKE.search(_body(issue))
    return m.group(1) if m else None


def _heute_angelegt(issue: dict[str, Any], heute: date) -> bool:
    roh = issue.get("created_at")
    if not isinstance(roh, str):
        return False
    try:
        return datetime.fromisoformat(roh.replace("Z", "+00:00")).astimezone().date() == heute
    except ValueError:
        return False


def _einzel_mit_repo(f: Fund) -> tuple[str, str]:
    """Text aus `einzel_text` plus Repo-Marke direkt nach der Schluessel-Marke (fuer das Schliessen je Ziel)."""
    titel, text = einzel_text(f)
    kopf, sep, rest = text.partition("\n")
    return titel, f"{kopf}\n<!-- selbstbild:repo={f.code_repo} -->{sep}{rest}"


def _block(text: str) -> str:
    return text[text.index(BEGIN): text.index(END) + len(END)]


def _belege_ersetzen(alt: str, neu: str) -> str | None:
    """Ersetzt nur den Belege-Abschnitt im alten Text; None, wenn er dort nicht (mehr) zu finden ist."""
    m_alt, m_neu = _BELEGE.search(alt), _BELEGE.search(neu)
    if not m_alt or not m_neu:
        return None
    teil = m_neu.group(2)
    if "\r\n" in alt:
        teil = teil.replace("\r\n", "\n").replace("\n", "\r\n")
    return alt[: m_alt.start(2)] + teil + alt[m_alt.end(2):]


class _Schreiber:
    """Fuehrt Schreibaufrufe aus (senden) oder plant sie nur (trocken); nach dem ersten Fehler: abgebrochen."""

    def __init__(self, gh: GitHub | None, repo: str, senden: bool, bericht: Bericht) -> None:
        self.gh, self.repo, self.senden, self.bericht = gh, repo, senden, bericht
        self._label = False

    def __call__(self, art: str, schluessel: str, titel: str, nummer: int | None,
                 tun: Callable[[GitHub], int | None]) -> tuple[str, int | None]:
        if not self.senden:
            status = "trocken"
        elif self.bericht.fehler is not None:
            status = "abgebrochen"
        else:
            assert self.gh is not None
            try:
                if not self._label:
                    self.gh.label_sicherstellen(self.repo)
                    self._label = True
                ergebnis = tun(self.gh)
                if ergebnis is not None:
                    nummer = ergebnis
                status = "ausgefuehrt"
            except GitHubFehler as f:
                self.bericht.fehler = f.art
                status = "fehlgeschlagen"
        self.bericht.aktionen.append(Aktion(art, schluessel, titel, nummer, status))
        return status, nummer


def abgleichen(funde: Sequence[Fund], ziel: Ziel, gh: GitHub | None, stand: IssueStand, *, heute: date,
               senden: bool, commit: str, geprueft: frozenset[str],
               unlesbar: frozenset[str] = frozenset()) -> Bericht:
    """Gleicht die Funde fuer ein Ziel mit dem GitHub-Bestand ab (siehe Moduldoku).

    `commit`: SHA des vermessenen Stands (fuer den Schliess-Kommentar, nur die ersten 7 Zeichen).
    `geprueft`: Code-Repos, die dieser Lauf wirklich geprueft hat (`sammeln.geprueft_repos`). Nur deren Issues
    werden geschlossen; ein Issue ohne Repo-Marke oder fuer ein anderes Code-Repo bleibt offen (Fix-Runde 1 I1).
    `unlesbar`: Arten, deren Quelle in diesem Lauf unlesbar war (`unlesbare_arten(notizen)`).
    Ein Fehler beim Lesen des Bestands oder der Sichtbarkeit bricht mit `GitHubFehler` ab, bevor geschrieben wird.
    """
    if senden and gh is None:
        raise ValueError("senden braucht ein GitHub-Objekt (Konto)")
    if not _COMMIT.fullmatch(commit or ""):
        raise ValueError("commit muss ein Git-SHA sein")
    sha7 = commit[:7].lower()
    repo = ziel.repo
    b = Bericht(ziel=repo, senden=senden)
    if stand.unlesbar:
        b.notizen.append("lokaler Stand unlesbar: neu aufgebaut (Dubletten-Schutz über die Marken in GitHub)")
    sperre = stand.sperren.get(repo)
    if sperre:
        b.notizen.append(f"Ziel gesperrt ({sperre}): nichts gesendet; nach Prüfung den Eintrag unter `sperren` "
                         f"in {stand.pfad.name} löschen")
        if senden:
            b.fehler = "ziel_gesperrt"  # alle Schreibaufrufe werden `abgebrochen`

    # 1. Sichtbarkeit (D6 fail-closed, Fix-Runde 1 I2) und Bestand -- jeder Lauf neu, nur lesend
    oeffentlich = ziel.oeffentlich
    if gh is not None:
        if not gh.ist_privat(repo):
            if not oeffentlich:
                b.notizen.append("Ziel laut GitHub nicht privat: behandelt als öffentlich (D6)")
            oeffentlich = True
        bestand = gh.issues(repo)
    else:
        bestand = []
        b.notizen.append("kein gh: Bestand als leer angenommen (nur Bericht)")
    einzel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sammel: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for issue in sorted(bestand, key=lambda i: int(i.get("number") or 0)):
        text = _body(issue)
        m = _MARKE.search(text)
        if m:
            einzel[m.group(1)].append(issue)
            continue
        s = _SAMMEL.search(text)
        if s:
            sammel[(s.group(1), s.group(2))].append(issue)

    # 2. Funde filtern
    verwaltet = set(ziel.arten) - ({"konfig"} if oeffentlich else set())
    for art in sorted(unlesbar & verwaltet):
        if art == "konfig":
            b.notizen.append("konfig: Quelle unlesbar: nichts geschlossen")
        else:
            b.notizen.append(f"{art}: Quelle unlesbar: nichts geschlossen, Sammel-Issue unverändert")
    alle_schluessel: set[str] = set()
    aktiv: list[tuple[str, Fund]] = []
    konfig_oeffentlich = 0
    for f in funde:
        k = fund_schluessel(f)
        if k in alle_schluessel:
            continue
        alle_schluessel.add(k)
        if f.art == "konfig" and oeffentlich:
            konfig_oeffentlich += 1
            b.ausgelassen.append(f"{k} ({f.titel}): konfig nie in ein öffentliches Repo (D6)")
        elif f.art not in ziel.arten:
            b.ausgelassen.append(f"{k}: Art {f.art} ist für dieses Ziel nicht eingeschaltet")
        elif not ziel.zustaendig(f.code_repo):
            b.ausgelassen.append(f"{k}: Code-Repo {f.code_repo} hat ein anderes Ziel")
        else:
            aktiv.append((k, f))
    if konfig_oeffentlich:
        b.notizen.append(f"konfig: {konfig_oeffentlich} Funde nicht gesendet, Ziel ist öffentlich (D6)")
    aktiv.sort(key=lambda kf: (_RANG.get(kf[1].schwere, 9), kf[1].art, kf[1].code_repo, kf[1].anker))

    schreiben = _Schreiber(gh, repo, senden, b)

    def sichern() -> None:
        try:
            stand.speichern()
        except OSError as e:
            b.notizen.append(f"lokaler Stand nicht gespeichert ({type(e).__name__})")

    def label_verloren(status: str) -> bool:
        """GitHub hat angelegt, das Label aber verworfen: das Issue ist im Bestand unsichtbar, jeder weitere Lauf
        wuerde es neu anlegen. Darum Ziel lokal sperren (Bremse; die Dubletten-Erkennung bleibt die Marke)."""
        if status != "fehlgeschlagen" or b.fehler != "label_fehlt":
            return False
        stand.sperren[repo] = "label_fehlt"
        b.notizen.append("GitHub hat das Label `selbstbild` verworfen (fehlende Rechte?): Ziel gesperrt, "
                         "das angelegte Issue von Hand prüfen")
        return True

    def konfig_abgleichen(k: str, f: Fund, treffer: list[dict[str, Any]], offen: list[dict[str, Any]]) -> None:
        """S5-11: ein Einzel-Issue, ganzer Text ersetzt bei Aenderung, ohne Tagesgrenze."""
        titel, neu_text = _einzel_mit_repo(f)
        muster = pruefen(f"{titel}\n{neu_text}")
        if muster:
            b.gesperrt.append(f"{k}: {muster}")
            return
        if not offen and any(_wontfix(i) for i in treffer):
            b.unterdrueckt.append(k)
            return
        if not treffer:
            status, nummer = schreiben("anlegen", k, titel, None, lambda g: g.anlegen(repo, titel, neu_text))
            if status == "ausgefuehrt":
                assert nummer is not None
                einzel[k].append({"number": nummer, "title": titel, "body": neu_text, "state": "open"})
            elif label_verloren(status):
                sichern()
            return
        i = offen[0] if offen else treffer[-1]
        n = int(i["number"])
        if not offen:  # geschlossen ohne wontfix, Inventar wieder da: oeffnen
            status, _ = schreiben("oeffnen", k, str(i.get("title") or titel), n, lambda g: g.oeffnen(repo, n))
            if status == "ausgefuehrt":
                i["state"] = "open"
        if _body(i).replace("\r\n", "\n") == neu_text:
            b.unveraendert += 1
            return
        status, _ = schreiben("bearbeiten", k, str(i.get("title") or titel), n,
                              lambda g: g.bearbeiten(repo, n, neu_text))
        if status == "ausgefuehrt":
            i["body"] = neu_text

    neu_heute = max(stand.neu_heute(repo, heute),
                    sum(1 for issues in einzel.values() for i in issues if _heute_angelegt(i, heute)))

    # 3. Einzel-Issues
    sammel_funde: dict[tuple[str, str], list[Fund]] = defaultdict(list)
    for k, f in aktiv:
        treffer = einzel.get(k, [])
        offen = [i for i in treffer if _offen(i)]
        if f.art == "konfig":
            konfig_abgleichen(k, f, treffer, offen)
            continue
        if offen:
            i = offen[0]
            titel, neu_text = _einzel_mit_repo(f)
            ersetzt = _belege_ersetzen(_body(i), neu_text)
            if ersetzt is None or ersetzt == _body(i):
                b.unveraendert += 1
                continue
            muster = pruefen(neu_text)
            if muster:
                b.gesperrt.append(f"{k}: {muster}")
                continue
            status, _ = schreiben("bearbeiten", k, str(i.get("title") or titel), int(i["number"]),
                                  lambda g, n=int(i["number"]), t=ersetzt: g.bearbeiten(repo, n, t))
            if status == "ausgefuehrt":
                i["body"] = ersetzt
            continue
        if any(_wontfix(i) for i in treffer):
            b.unterdrueckt.append(k)
            continue
        if treffer:  # geschlossen ohne wontfix, Fund wieder da: oeffnen (zaehlt nicht als neu)
            i = treffer[-1]
            status, _ = schreiben("oeffnen", k, str(i.get("title") or f.titel), int(i["number"]),
                                  lambda g, n=int(i["number"]): g.oeffnen(repo, n))
            if status == "ausgefuehrt":
                i["state"] = "open"
            continue
        if f.schwere == "hoch":
            titel, text = _einzel_mit_repo(f)
            muster = pruefen(f"{titel}\n{text}")
            if muster:
                b.gesperrt.append(f"{k}: {muster}")
                continue
            if neu_heute < ziel.max_einzel_je_tag:
                status, nummer = schreiben("anlegen", k, titel, None,
                                           lambda g, t=titel, x=text: g.anlegen(repo, t, x))
                if status in ("ausgefuehrt", "trocken"):
                    neu_heute += 1
                    b.neu += 1
                if status == "ausgefuehrt":
                    assert nummer is not None
                    einzel[k].append({"number": nummer, "title": titel, "body": text, "state": "open"})
                    stand.zaehlen(repo, heute)
                    sichern()  # sofort: ein Abbruch spaeter laesst den Zaehler stimmen
                elif label_verloren(status):
                    stand.zaehlen(repo, heute)  # das Issue existiert, nur ohne Label
                    sichern()
                continue
            b.ueber_grenze.append(k)
        sammel_funde[(f.art, f.code_repo)].append(f)

    # 4. Sammel-Issues je Art und Code-Repo
    paare = set(sammel_funde) | {p for p, issues in sammel.items() if any(_offen(i) for i in issues)}
    nicht_geprueft: set[str] = set()
    for art, code_repo in sorted(paare):
        if art not in verwaltet or not ziel.zustaendig(code_repo) or art in unlesbar:
            continue
        sk = f"sammel:{art}:{code_repo}"
        issues = sammel.get((art, code_repo), [])
        offen = [i for i in issues if _offen(i)]
        # Fix-Runde 1 (Minor 6): ein Fund mit Schluessel-Verdacht faellt nur aus seiner eigenen Zeile
        fs: list[Fund] = []
        verdacht = 0
        for f in sammel_funde.get((art, code_repo), []):
            m = pruefen(f"{f.titel}\n{f.anker}\n{f.schwere}")
            if m:
                verdacht += 1
                b.gesperrt.append(f"{fund_schluessel(f)}: {m}")
            else:
                fs.append(f)
        if not fs and verdacht:
            continue  # die Funde sind da, nur nicht sendbar: nichts schliessen, Block unveraendert
        if not fs:
            if offen and code_repo not in geprueft:
                nicht_geprueft.add(code_repo)
                continue
            for i in offen:
                n = int(i["number"])
                status, _ = schreiben("sammel_schliessen", sk, str(i.get("title") or sk), n,
                                      lambda g, n=n: g.schliessen(
                                          repo, n, f"Sammlung leer: alle Funde im Code verschwunden (Commit `{sha7}`)"))
                if status == "ausgefuehrt":
                    i["state"] = "closed"
            continue
        titel, text = sammel_text(art, code_repo, fs)
        muster = pruefen(f"{titel}\n{text}")
        if muster:
            b.gesperrt.append(f"{sk}: {muster}")
            continue
        if not issues:
            status, _ = schreiben("sammel_anlegen", sk, titel, None, lambda g, t=titel, x=text: g.anlegen(repo, t, x))
            if label_verloren(status):
                sichern()
            continue
        if not offen and any(_wontfix(i) for i in issues):
            b.unterdrueckt.append(sk)
            continue
        i = offen[0] if offen else issues[-1]
        n = int(i["number"])
        try:
            neu_body = block_ersetzen(_body(i), _block(text))
        except ValueError:
            b.notizen.append(f"Sammel-Issue #{n} ({art}, {code_repo}): Block beschädigt, nicht bearbeitet")
            continue
        if not offen:
            status, _ = schreiben("sammel_oeffnen", sk, str(i.get("title") or titel), n,
                                  lambda g, n=n: g.oeffnen(repo, n))
            if status == "ausgefuehrt":
                i["state"] = "open"
        if neu_body == _body(i):
            b.unveraendert += 1
            continue
        status, _ = schreiben("sammel_bearbeiten", sk, str(i.get("title") or titel), n,
                              lambda g, n=n, t=neu_body: g.bearbeiten(repo, n, t))
        if status == "ausgefuehrt":
            i["body"] = neu_body

    # 5. Einzel-Issues, deren Fund verschwunden ist
    kommentar = f"Fund im Code verschwunden (Commit `{sha7}`)"
    nicht_zuordenbar = 0
    for k in sorted(einzel):
        art = k.partition(":")[0]
        if k in alle_schluessel or art not in verwaltet or art in unlesbar:
            continue
        for i in einzel[k]:
            if not _offen(i):
                continue
            code_repo = _code_repo(i)
            if code_repo is None:
                nicht_zuordenbar += 1
                continue
            if not ziel.zustaendig(code_repo):
                continue
            if code_repo not in geprueft:
                nicht_geprueft.add(code_repo)
                continue
            n = int(i["number"])
            status, _ = schreiben("schliessen", k, str(i.get("title") or k), n,
                                  lambda g, n=n: g.schliessen(repo, n, kommentar))
            if status == "ausgefuehrt":
                i["state"] = "closed"
    if nicht_zuordenbar:
        b.notizen.append(f"{nicht_zuordenbar} offene Einzel-Issues ohne Repo-Marke: nicht geschlossen")
    if nicht_geprueft:
        b.notizen.append("nicht geprüfte Code-Repos, ihre Issues bleiben offen: " + ", ".join(sorted(nicht_geprueft)))

    # 6. lokalen Stand nachziehen (nur beim Senden)
    if senden:
        stand.marken[repo] = {k: int((next((i for i in v if _offen(i)), v[-1]))["number"])
                              for k, v in sorted(einzel.items()) if v}
        try:
            stand.speichern()
        except OSError as e:
            b.notizen.append(f"lokaler Stand nicht gespeichert ({type(e).__name__})")
    return b
