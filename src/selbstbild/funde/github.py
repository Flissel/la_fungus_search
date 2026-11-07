"""GitHub-Adapter fuer die Fund-Issues (Stufe 5, Task 4).

Nur `gh api` mit explizitem `--method` und Zeitlimit; Texte gehen ueber `--input -` (stdin), nie als Argument.
Das Konto gilt je Aufruf (D2): der Token kommt aus `gh auth token --user <konto>`, wird je Instanz gecacht und nur
als `GH_TOKEN` in das Env des Kindprozesses gelegt. Nie `gh auth switch`, nie das globale Konto, nie loggen.
Jeder Fehler wird zu `GitHubFehler(art)` -- ohne Token, ohne Antworttext.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

LABEL = "selbstbild"
LABEL_FARBE = "5319e7"
LABEL_TEXT = "Automatisch von Selbstbild (VibeMind) erzeugt"
SEITE = 100
MAX_SEITEN = 100  # Bremse: hoechstens 10.000 Issues mit Label `selbstbild`
TIMEOUT_S = 60.0
HOST = "github.com"
AUTO_MARKE = "<!-- selbstbild:auto -->"  # an jedem Kommentar der Tagesautomatik
DEBUGGER_MARKE = "<!-- selbstbild:debugger -->"  # an jedem Text des Debugger-Helfers
_GRUENDE = ("completed", "not_planned")
# Variablen, die gh als Token nehmen wuerde: fuer `auth token` entfernen, fuer `api` nur GH_TOKEN setzen
_TOKEN_VARS = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN")
_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_KONTO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
_HTTP = re.compile(rb"HTTP (\d{3})")


class GitHubFehler(Exception):
    """Fehler beim gh-Aufruf. Traegt nur die Art (`http_404`, `zeitlimit`, ...), nie Token oder Antwort."""

    def __init__(self, art: str) -> None:
        super().__init__(art)
        self.art = art

    def __str__(self) -> str:
        return f"GitHub: {self.art}"

    def __repr__(self) -> str:
        return f"GitHubFehler({self.art!r})"


class _Geheim:
    """Haelt den Token so, dass weder `repr` noch `str` ihn zeigen."""

    __slots__ = ("_wert",)

    def __init__(self, wert: str) -> None:
        self._wert = wert

    def wert(self) -> str:
        return self._wert

    def __repr__(self) -> str:
        return "<verborgen>"

    __str__ = __repr__


def _repo(repo: str) -> str:
    if not isinstance(repo, str) or not _REPO.fullmatch(repo) or ".." in repo:
        raise GitHubFehler("repo_ungueltig")
    return repo


def _label(name: str) -> str:
    """Label-Name: 1-50 Zeichen, ohne Steuerzeichen (GitHub-Grenze)."""
    if not isinstance(name, str) or not 0 < len(name) <= 50 or not name.isprintable() or name != name.strip():
        raise GitHubFehler("label_ungueltig")
    return name


_BRANCH = re.compile(r"debugger/[A-Za-z0-9][A-Za-z0-9._/-]*")


def _debugger_branch(name: str) -> str:
    """Nur Branches `debugger/...` (kein `..`, keine Sonderzeichen); sonst `ValueError`."""
    if not isinstance(name, str) or not _BRANCH.fullmatch(name) or ".." in name or name.endswith("/"):
        raise ValueError("Branch muss debugger/... heißen")
    return name


def _nummer(nummer: int) -> int:
    if isinstance(nummer, bool) or not isinstance(nummer, int) or nummer <= 0:
        raise GitHubFehler("nummer_ungueltig")
    return nummer


class GitHub:
    """`gh` mit einem festen Konto; `laufzeit` ist fuer Tests injizierbar (Signatur wie `subprocess.run`)."""

    def __init__(self, konto: str, *, laufzeit: Callable[..., Any] = subprocess.run, gh: str = "gh",
                 timeout_s: float = TIMEOUT_S, host: str = HOST, marke: str | None = AUTO_MARKE,
                 token_datei: Path | None = None) -> None:
        if not isinstance(konto, str) or not _KONTO.fullmatch(konto):
            raise ValueError("gh-Konto ungültig")
        self.konto = konto
        self.laufzeit = laufzeit
        self.gh = gh
        self.timeout_s = timeout_s
        self.host = host
        self.marke = marke
        self.token_datei = token_datei
        self._token: _Geheim | None = None
        self._label_da: set[tuple[str, str]] = set()

    def __repr__(self) -> str:
        return f"GitHub(konto={self.konto!r})"

    # ---------- Prozess ----------

    def _basis_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _TOKEN_VARS}
        env["GH_PROMPT_DISABLED"] = "1"
        env["GH_NO_UPDATE_NOTIFIER"] = "1"
        env["NO_COLOR"] = "1"
        return env

    @staticmethod
    def _art(returncode: int, stderr: object) -> str:
        fehler = stderr if isinstance(stderr, bytes) else str(stderr or "").encode()
        m = _HTTP.search(fehler)
        return f"http_{m.group(1).decode()}" if m else f"exit_{returncode}"

    def _starten(self, argv: list[str], env: dict[str, str], eingabe: bytes | None) -> bytes:
        stdin = {"input": eingabe} if eingabe is not None else {"stdin": subprocess.DEVNULL}
        art: str | None = None
        lauf: Any = None
        # Fix-Runde 1 (Minor 2): der GitHubFehler entsteht AUSSERHALB des except-Blocks, sonst haenge die
        # Ausnahme (mit Teil-Ausgabe, Argumenten, stderr) als __context__ daran
        try:
            lauf = self.laufzeit(argv, capture_output=True, env=env, timeout=self.timeout_s,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **stdin)
        except subprocess.TimeoutExpired:
            art = "zeitlimit"
        except subprocess.CalledProcessError as e:
            art = self._art(e.returncode, e.stderr)
        except subprocess.SubprocessError:
            art = "gh_fehler"
        except OSError:
            art = "gh_fehlt"
        if art is None and lauf.returncode != 0:
            art = self._art(lauf.returncode, lauf.stderr)
        if art is not None:
            raise GitHubFehler(art) from None
        aus = lauf.stdout
        return aus if isinstance(aus, bytes) else str(aus or "").encode("utf-8")

    def _token_holen(self) -> str:
        if self._token is None:
            if self.token_datei is not None:
                try:
                    wert = Path(self.token_datei).read_text(encoding="utf-8").strip()
                except (OSError, ValueError):
                    wert = ""  # Datei fehlt/unlesbar: gleiche Art wie leer, ohne Pfad oder Inhalt
            else:
                aus = self._starten([self.gh, "auth", "token", "--hostname", self.host, "--user", self.konto],
                                    self._basis_env(), None)
                wert = aus.decode("utf-8", errors="replace").strip()
            if not wert or any(z.isspace() for z in wert):
                raise GitHubFehler("kein_token") from None
            self._token = _Geheim(wert)
        return self._token.wert()

    def _api(self, methode: str, pfad: str, daten: dict[str, Any] | None = None) -> Any:
        env = self._basis_env()
        env["GH_TOKEN"] = self._token_holen()
        argv = [self.gh, "api", "--method", methode, "-H", "Accept: application/vnd.github+json",
                "-H", "X-GitHub-Api-Version: 2022-11-28", pfad]
        eingabe = None
        if daten is not None:
            argv += ["--input", "-"]
            eingabe = json.dumps(daten, ensure_ascii=False).encode("utf-8")
        aus = self._starten(argv, env, eingabe)
        if not aus.strip():
            return None
        try:
            return json.loads(aus.decode("utf-8"))
        except ValueError:
            raise GitHubFehler("antwort_unlesbar") from None

    # ---------- Lesen ----------

    def issues(self, repo: str) -> list[dict[str, Any]]:
        """Alle Issues mit Label `selbstbild`, offen und geschlossen, ohne Pull Requests."""
        r = _repo(repo)
        alle: list[dict[str, Any]] = []
        for seite in range(1, MAX_SEITEN + 1):
            teil = self._api("GET", f"/repos/{r}/issues?labels={LABEL}&state=all&per_page={SEITE}&page={seite}")
            if not isinstance(teil, list):
                raise GitHubFehler("antwort_unlesbar")
            alle += [i for i in teil if isinstance(i, dict) and "pull_request" not in i]
            if len(teil) < SEITE:
                return alle
        raise GitHubFehler("zu_viele_seiten")

    def ist_privat(self, repo: str) -> bool:
        """True nur, wenn `GET /repos/{repo}` `private: true` meldet (D6 fail-closed, Fix-Runde 1 I2)."""
        antwort = self._api("GET", f"/repos/{_repo(repo)}")
        return isinstance(antwort, dict) and antwort.get("private") is True

    # ---------- Schreiben ----------

    def label_sicherstellen(self, repo: str, name: str = LABEL, farbe: str = LABEL_FARBE,
                            text: str = LABEL_TEXT) -> None:
        """Legt das Label `name` an, falls es fehlt (je `(repo, name)` einmal je Instanz gecacht)."""
        r, n = _repo(repo), _label(name)
        if (r, n) in self._label_da:
            return
        fehlt = False
        try:
            self._api("GET", f"/repos/{r}/labels/{quote(n, safe='')}")
        except GitHubFehler as f:
            if f.art != "http_404":
                raise
            fehlt = True
        if fehlt:  # ausserhalb des except-Blocks: kein __context__
            self._api("POST", f"/repos/{r}/labels", {"name": n, "color": farbe, "description": text})
        self._label_da.add((r, n))

    def anlegen(self, repo: str, titel: str, text: str, labels: Sequence[str] = ()) -> int:
        """Legt ein Issue mit Label `selbstbild` (plus `labels`) an. Verwirft GitHub das Label `selbstbild` still
        (ohne Schreibrecht auf Labels), wirft `GitHubFehler("label_fehlt")`: das Issue existiert dann, ist aber im
        Bestand unsichtbar. Fuer die Zusatz-Labels gilt das nicht."""
        namen_neu = list(dict.fromkeys([LABEL, *(_label(x) for x in labels)]))
        antwort = self._api("POST", f"/repos/{_repo(repo)}/issues", {"title": titel, "body": text, "labels": namen_neu})
        nummer = antwort.get("number") if isinstance(antwort, dict) else None
        if isinstance(nummer, bool) or not isinstance(nummer, int):
            raise GitHubFehler("antwort_unlesbar")
        labels = antwort.get("labels") if isinstance(antwort, dict) else None
        namen = {(x.get("name") if isinstance(x, dict) else x) for x in labels} if isinstance(labels, list) else set()
        if LABEL not in namen:
            raise GitHubFehler("label_fehlt")
        return nummer

    def bearbeiten(self, repo: str, nummer: int, text: str) -> None:
        self._api("PATCH", f"/repos/{_repo(repo)}/issues/{_nummer(nummer)}", {"body": text})

    def oeffnen(self, repo: str, nummer: int) -> None:
        self._api("PATCH", f"/repos/{_repo(repo)}/issues/{_nummer(nummer)}", {"state": "open"})

    def _mit_marke(self, text: str) -> str:
        """Haengt `self.marke` an (hinter eine Leerzeile), falls gesetzt und noch nicht im Text."""
        if self.marke is None or self.marke in text:
            return text
        return text + "\n\n" + self.marke

    def kommentieren(self, repo: str, nummer: int, text: str) -> None:
        self._api("POST", f"/repos/{_repo(repo)}/issues/{_nummer(nummer)}/comments", {"body": self._mit_marke(text)})

    def schliessen(self, repo: str, nummer: int, kommentar: str, grund: str = "completed") -> None:
        """Erst der Kommentar, dann zu (`state_reason`: `completed` oder `not_planned`)."""
        if grund not in _GRUENDE:
            raise ValueError("Schließgrund ungültig")
        r, n = _repo(repo), _nummer(nummer)
        self.kommentieren(r, n, kommentar)
        self._api("PATCH", f"/repos/{r}/issues/{n}", {"state": "closed", "state_reason": grund})

    # ---------- Debugger-Waechter (Lesen und Schreiben) ----------

    def _seiten(self, pfad: str) -> list[dict[str, Any]]:
        """Blaettert `pfad` (ohne Query) mit `per_page=100`, bis eine Seite weniger als 100 Eintraege hat."""
        alle: list[dict[str, Any]] = []
        for seite in range(1, MAX_SEITEN + 1):
            teil = self._api("GET", pfad.format(seite=seite))
            if not isinstance(teil, list):
                raise GitHubFehler("antwort_unlesbar")
            alle += [x for x in teil if isinstance(x, dict)]
            if len(teil) < SEITE:
                return alle
        raise GitHubFehler("zu_viele_seiten")

    def kommentare_seit(self, repo: str, seit: str) -> list[dict[str, Any]]:
        """Alle Issue-/PR-Kommentare des Repos seit `seit` (ISO-Zeit)."""
        s = quote(seit, safe=":")
        return self._seiten(f"/repos/{_repo(repo)}/issues/comments?since={s}&per_page={SEITE}&page={{seite}}")

    def review_kommentare_seit(self, repo: str, seit: str) -> list[dict[str, Any]]:
        """Alle Code-Review-Kommentare an PRs des Repos seit `seit` (ISO-Zeit)."""
        s = quote(seit, safe=":")
        return self._seiten(f"/repos/{_repo(repo)}/pulls/comments?since={s}&per_page={SEITE}&page={{seite}}")

    def reviews(self, repo: str, pr: int) -> list[dict[str, Any]]:
        """Reviews eines PR, nur die mit Text."""
        alle = self._seiten(f"/repos/{_repo(repo)}/pulls/{_nummer(pr)}/reviews?per_page={SEITE}&page={{seite}}")
        return [x for x in alle if isinstance(x.get("body"), str) and x["body"].strip()]

    def reagieren(self, repo: str, kommentar_id: int, *, review: bool = False) -> None:
        """Setzt die Reaktion `eyes` auf einen Kommentar (Review-Kommentar mit `review=True`)."""
        art = "pulls" if review else "issues"
        self._api("POST", f"/repos/{_repo(repo)}/{art}/comments/{_nummer(kommentar_id)}/reactions",
                  {"content": "eyes"})

    def labels_setzen(self, repo: str, nummer: int, labels: Sequence[str]) -> None:
        """Haengt Labels an ein Issue/PR (bestehende bleiben)."""
        self._api("POST", f"/repos/{_repo(repo)}/issues/{_nummer(nummer)}/labels",
                  {"labels": [_label(x) for x in labels]})

    def pr_anlegen(self, repo: str, kopf: str, basis: str, titel: str, text: str) -> int:
        """Legt einen PR an. Nur fuer Branches `debugger/...`; der Text traegt `self.marke`."""
        r, k = _repo(repo), _debugger_branch(kopf)
        antwort = self._api("POST", f"/repos/{r}/pulls",
                            {"title": titel, "head": k, "base": basis, "body": self._mit_marke(text)})
        nummer = antwort.get("number") if isinstance(antwort, dict) else None
        if isinstance(nummer, bool) or not isinstance(nummer, int):
            raise GitHubFehler("antwort_unlesbar")
        return nummer

    def pr_lesen(self, repo: str, nummer: int) -> dict[str, Any]:
        antwort = self._api("GET", f"/repos/{_repo(repo)}/pulls/{_nummer(nummer)}")
        if not isinstance(antwort, dict):
            raise GitHubFehler("antwort_unlesbar")
        return antwort

    def prs_zum_kopf(self, repo: str, kopf: str) -> list[dict[str, Any]]:
        """Alle PRs (offen und zu) mit Kopf-Branch `kopf` im selben Repo (Eigentuemer = Repo-Eigentuemer)."""
        r = _repo(repo)
        antwort = self._api("GET", f"/repos/{r}/pulls?head={quote(r.split('/')[0] + ':' + _debugger_branch(kopf), safe=':/')}"
                                   "&state=all")
        if not isinstance(antwort, list):
            raise GitHubFehler("antwort_unlesbar")
        return [x for x in antwort if isinstance(x, dict)]

    def branch_loeschen(self, repo: str, branch: str) -> None:
        """Loescht einen Branch `debugger/...`; ein 404 (schon weg) zaehlt als erledigt."""
        r, b = _repo(repo), _debugger_branch(branch)
        gefehlt = False
        try:
            self._api("DELETE", f"/repos/{r}/git/refs/heads/{b}")
        except GitHubFehler as f:
            if f.art != "http_404":
                raise
            gefehlt = True
        if gefehlt:
            return
