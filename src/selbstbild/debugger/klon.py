"""Git-Arbeit des Debugger-Claw auf dem VM-Klon: nachziehen, Worktrees, Push-Inhalt, aufraeumen.

Alle Aufrufe laufen ohne Shell und ohne Token. Klon und Worktrees gehoeren dem Helfer-Nutzer
(`claw`); jeder Git-Aufruf darin laeuft mit dem modulweiten Praefix (`git_praefix_setzen`,
im Container `setpriv --reuid=1000 ...`), damit Hooks, Filter oder Konfiguration, die der
Helfer ins geteilte `.git` schreibt, nie mit Rechten des Waechters laufen. Zusaetzlich laeuft
jeder dieser Aufrufe ohne Hooks und ohne fsmonitor (`KLON_HAERTUNG`, R39): Hooks, die der Helfer
anlegt, laufen hoechstens in seinen eigenen Prozessen. Mit GitHub spricht nur `spiegel.py`.
Git-Fehler tragen nur Unterbefehl und Rueckgabecode, nie stderr.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO, Sequence

PRAEFIX = "debugger/"
TITEL_PRAEFIX = "[selbstbild] "
MAX_KURZ = 30
GIT_TIMEOUT_S = 300
TIMEOUT_CODE = -1
GROESSE_CODE = -2
LESE_CODE = -3
MAX_PUSH_INHALT = 5_000_000
_BLOCK = 1 << 20
_NACH_KILL_S = 5.0
ZU_GROSS = "zu_gross"

KLON_HAERTUNG: tuple[str, ...] = ("-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false")

_SHA = re.compile(r"[0-9a-f]{40}")
_ZWEIG = re.compile(r"debugger/[A-Za-z0-9][A-Za-z0-9._/-]*")
_ENV_GESPERRT = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GIT_ASKPASS")
_praefix: tuple[str, ...] = ()


class KonfliktFehler(Exception):
    """Lokaler und entfernter Zweig sind auseinandergelaufen."""


class GitFehler(Exception):
    """Ein Git-Aufruf ist gescheitert; nennt nur Unterbefehl und Code."""

    def __init__(self, unterbefehl: str, code: int) -> None:
        zusatz = {TIMEOUT_CODE: " (Zeitueberschreitung)", GROESSE_CODE: " (Ausgabe zu gross)",
                  LESE_CODE: " (Ausgabe nicht vollstaendig gelesen)"}.get(
            code, f" (Code {code})")
        super().__init__(f"git {unterbefehl} fehlgeschlagen{zusatz}")
        self.unterbefehl = unterbefehl
        self.code = code


class WorktreeFehler(ValueError):
    """`wt` ist kein registrierter Worktree des Klons auf dem erwarteten Zweig."""


class PushZuGross(Exception):
    """Was ein Push hochlaedt, ist fuer die Schluessel-Pruefung zu gross (fail-closed, R23)."""

    def __init__(self) -> None:
        super().__init__(ZU_GROSS)


def git_praefix_setzen(praefix: Sequence[str]) -> None:
    """Setzt das argv-Praefix fuer alle Git-Aufrufe in Klon und Worktrees (einmal beim Start)."""
    global _praefix
    if isinstance(praefix, str) or not all(isinstance(x, str) for x in praefix):
        raise TypeError("praefix muss eine Folge von Strings sein")
    _praefix = tuple(praefix)


def git_praefix() -> tuple[str, ...]:
    """Das aktuelle argv-Praefix (Standard leer)."""
    return _praefix


def _env() -> dict[str, str]:
    """Umgebung fuer Klon-Git: ohne Zugangsdaten, ohne Prompt, ohne Replace-Objekte, ohne Nachladen."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("DEBUGGER_") and k not in _ENV_GESPERRT}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"  # der Waechter sieht immer die echten Objekte
    env["GIT_NO_LAZY_FETCH"] = "1"  # kein Nachladen ueber eine vom Helfer eingetragene Promisor-Quelle
    return env


def gruppe_toeten(proc: subprocess.Popen[bytes]) -> None:
    """Beendet den Prozess samt allen Nachfahren seiner Sitzung (POSIX); unter Windows nur ihn (Tests)."""
    if os.name == "nt":
        try:
            proc.kill()
        except OSError:
            pass
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:  # Gruppe schon weg oder nicht erlaubt: wenigstens den Prozess selbst
        try:
            proc.kill()
        except OSError:
            pass


def _begrenzt(
    argv: list[str], unter: str, *, eingabe: bytes | None = None, ziel: BinaryIO | None = None,
    grenze: int | None = None,
) -> tuple[int, bytes]:
    """Einziger Laeufer fuer Prozesse als `claw`: eigene Sitzung (POSIX), Lesen mit Restfrist (R31).

    stdout liest ein Thread; der Waechter wartet hoechstens `GIT_TIMEOUT_S`, auch wenn ein Enkel
    (Hook, Filter, pytest des Helfers) die Pipe offen haelt. Bei Zeit- oder Groessenueberschreitung
    wird die ganze Gruppe beendet und `GitFehler(unter, TIMEOUT_CODE | GROESSE_CODE)` geworfen;
    scheitert Lesen oder Schreiben nach `ziel`, ebenso mit `LESE_CODE` (fail-closed, nie Erfolg).
    stdout geht nach `ziel` oder kommt als Bytes zurueck; stderr wird verworfen. Rueckgabe: Code, Bytes.
    """
    frist = GIT_TIMEOUT_S
    ende = time.monotonic() + frist
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL if eingabe is None else subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=_env(), start_new_session=os.name != "nt",
    )
    teile: list[bytes] = []
    halt, zu_gross, kaputt = threading.Event(), threading.Event(), threading.Event()

    def lesen() -> None:
        aus, gesamt = proc.stdout, 0
        try:
            while aus is not None:
                block = aus.read1(_BLOCK)
                if not block or halt.is_set():
                    return
                gesamt += len(block)
                if grenze is not None and gesamt > grenze:
                    zu_gross.set()
                    return
                if ziel is not None:
                    ziel.write(block)
                else:
                    teile.append(block)
        except (OSError, ValueError):  # Lese- oder Schreibfehler (Platte voll, ziel zu): nie Erfolg
            kaputt.set()
            return
        finally:  # der Thread besitzt die Pipe; er kann den Aufruf überleben (Windows-Enkel)
            if aus is not None:
                aus.close()

    def schreiben() -> None:
        ein = proc.stdin
        try:
            if ein is not None and eingabe is not None:
                ein.write(eingabe)
        except (OSError, ValueError):
            pass
        finally:
            if ein is not None:
                try:
                    ein.close()
                except OSError:
                    pass

    leser = threading.Thread(target=lesen, daemon=True)
    leser.start()
    if eingabe is not None:
        threading.Thread(target=schreiben, daemon=True).start()
    leser.join(frist)
    code: int | None = None
    if not leser.is_alive() and not zu_gross.is_set() and not kaputt.is_set():
        try:
            code = proc.wait(timeout=max(0.0, ende - time.monotonic()))
        except subprocess.TimeoutExpired:
            code = None
    if code is None:
        halt.set()
        gruppe_toeten(proc)
        leser.join(_NACH_KILL_S)  # POSIX: nach dem Gruppen-Kill kommt EOF; sonst bleibt der Thread liegen
        try:
            proc.wait(timeout=_NACH_KILL_S)
        except subprocess.TimeoutExpired:
            pass
        art = GROESSE_CODE if zu_gross.is_set() else LESE_CODE if kaputt.is_set() else TIMEOUT_CODE
        raise GitFehler(unter, art)
    return code, b"".join(teile)


def _argv(ort: Path, args: tuple[str, ...]) -> list[str]:
    return [*_praefix, "git", *KLON_HAERTUNG, "-C", str(ort), *args]


def git_lauf(
    ort: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Fuehrt `<praefix> git -C <ort> ...` aus, ohne Eingabe und ohne Passwort-Abfrage.

    Ausgabe als UTF-8 (Ersatzzeichen bei Fehlern). Zeitueberschreitung wird zu
    GitFehler(code=TIMEOUT_CODE); bei check=True gibt es nur einen schlanken Fehler (nie stderr).
    """
    argv = _argv(ort, args)
    code, aus = _begrenzt(argv, args[0] if args else "?")
    if check and code != 0:
        raise GitFehler(args[0] if args else "?", code)
    return subprocess.CompletedProcess(argv, code, aus.decode("utf-8", errors="replace"), "")


def git_roh(ort: Path, *args: str, eingabe: bytes) -> bytes:
    """Wie `git_lauf`, aber mit Bytes auf stdin und rohen Bytes als Ergebnis (immer check)."""
    code, aus = _begrenzt(_argv(ort, args), args[0] if args else "?", eingabe=eingabe)
    if code != 0:
        raise GitFehler(args[0] if args else "?", code)
    return aus


def git_in_datei(ort: Path, *args: str, ziel: BinaryIO, grenze: int) -> None:
    """Wie `git_lauf` (mit Praefix), aber stdout geht blockweise nach `ziel` (Datei des Waechters).

    Mehr als `grenze` Bytes: die Gruppe wird beendet, `GitFehler(<unterbefehl>, GROESSE_CODE)`
    (fail-closed). Zeitgrenze wie bei `git_lauf`; stderr wird verworfen.
    """
    code, _ = _begrenzt(_argv(ort, args), args[0] if args else "?", ziel=ziel, grenze=grenze)
    if code != 0:
        raise GitFehler(args[0] if args else "?", code)


def ist_vorfahr(ort: Path, alt: str, neu: str) -> bool:
    """`merge-base --is-ancestor`: 0 ja, 1 nein, alles andere ist ein Fehler."""
    code = git_lauf(ort, "merge-base", "--is-ancestor", alt, neu, check=False).returncode
    if code not in (0, 1):
        raise GitFehler("merge-base", code)
    return code == 0


def _norm(pfad: str | Path) -> str:
    return os.path.normcase(str(Path(pfad).resolve()))


def _worktree_pruefen(repo_klon: Path, wt: Path, branch: str) -> None:
    """Wirft WorktreeFehler, wenn `wt` kein zusaetzlicher Worktree auf `branch` ist."""
    liste = git_lauf(repo_klon, "worktree", "list", "--porcelain").stdout
    pfade = [z[len("worktree "):] for z in liste.splitlines() if z.startswith("worktree ")]
    # Der erste Eintrag ist der Haupt-Checkout selbst und nie ein Helfer-Worktree.
    if _norm(wt) not in {_norm(p) for p in pfade[1:]}:
        raise WorktreeFehler("kein_worktree_des_klons")
    aktuell = git_lauf(wt, "symbolic-ref", "--short", "HEAD", check=False)
    if aktuell.returncode != 0 or aktuell.stdout.strip() != branch:
        raise WorktreeFehler("falscher_zweig_im_worktree")


def _ref_da(ort: Path, ref: str) -> bool:
    return git_lauf(ort, "rev-parse", "--verify", "-q", ref, check=False).returncode == 0


def debugger_zweig_pruefen(branch: str) -> None:
    """Nur `debugger/...` ohne `..`, Leer- oder Sonderzeichen (wie github.py); sonst `ValueError`."""
    if not isinstance(branch, str) or not _ZWEIG.fullmatch(branch) or ".." in branch or branch.endswith("/"):
        raise ValueError("kein_debugger_zweig")


def nachziehen(klon: Path, basis: str, submodule: Sequence[str]) -> str | None:
    """Zieht den Klon per Fast-Forward auf `origin/<basis>`.

    Rueckgabe None bei Erfolg, sonst "falscher_zweig", "nicht_sauber",
    "kein_fast_forward" oder "submodul_fehler". Bei den ersten beiden Werten
    bleibt der Arbeitsbaum unberuehrt.

    R40: Der Sauberkeitstest sieht die Submodule nicht (`--ignore-submodules=all`); jedes
    verwaltete Submodul wird getrennt geprueft. Eigene Aenderungen darin heissen "nicht_sauber",
    ein nur abweichender Checkout (Gitlink weiter als das Submodul) ist in Ordnung, denn das
    Submodul-Update laeuft in jedem Durchgang nach dem ff-Merge erneut.
    """
    git_lauf(klon, "fetch", "--prune", "--no-recurse-submodules")
    aktuell = git_lauf(klon, "symbolic-ref", "--short", "HEAD", check=False)
    if aktuell.returncode != 0 or aktuell.stdout.strip() != basis:
        return "falscher_zweig"
    if git_lauf(klon, "status", "--porcelain", "--ignore-submodules=all").stdout.strip():
        return "nicht_sauber"
    for pfad in submodule:
        ort = klon / pfad
        # nicht initialisiert: kein eigenes .git, git -C liefe sonst im aeusseren Repo
        if (ort / ".git").exists() and git_lauf(ort, "status", "--porcelain").stdout.strip():
            return "nicht_sauber"
    if git_lauf(klon, "merge", "--ff-only", f"origin/{basis}", check=False).returncode != 0:
        return "kein_fast_forward"
    for pfad in submodule:
        erg = git_lauf(klon, "submodule", "update", "--init", "--", pfad, check=False)
        if erg.returncode != 0:
            return "submodul_fehler"
    return None


def branch_name(issue: int, titel: str, runde: int = 1) -> str:
    """`debugger/<issue>-<kurz>`; kurz hoechstens 30 Zeichen aus [a-z0-9-].

    Ab Runde 2 endet kurz auf `-<runde>` (R37). Das Suffix zaehlt zur Grenze: gekuerzt
    wird nur der Titelteil, nie das Suffix.
    """
    if isinstance(runde, bool) or not isinstance(runde, int) or runde < 1:
        raise ValueError("runde_ungueltig")
    suffix = f"-{runde}" if runde > 1 else ""
    text = titel.removeprefix(TITEL_PRAEFIX).lower()
    # Zeichenweise ersetzen, kuerzen, dann zusammenfassen (so verlangt es
    # der verbindliche Testfall aus dem Brief).
    roh = re.sub(r"[^a-z0-9]", "-", text)[:MAX_KURZ - len(suffix)]
    kurz = re.sub(r"-{2,}", "-", roh).strip("-") or "fund"
    return f"{PRAEFIX}{issue}-{kurz}{suffix}"


def worktree_bereit(
    repo_klon: Path, wt_ordner: Path, branch: str, basis: str, *, start: str | None = None
) -> Path:
    """Stellt einen Worktree fuer `branch` bereit.

    Fehlt er, wird er aus `origin/<branch>` oder sonst aus `start` (Standard `origin/<basis>`)
    angelegt (ein lokaler Alt-Zweig ohne Worktree wird wiederverwendet, nie geloescht).
    Existiert er und gibt es `origin/<branch>`, wird per ff-only nachgezogen;
    laufen beide Seiten auseinander, folgt KonfliktFehler und nichts aendert sich.
    """
    debugger_zweig_pruefen(branch)
    if start is not None and not _SHA.fullmatch(start):
        raise ValueError("start_ungueltig")
    if wt_ordner.exists():
        _worktree_pruefen(repo_klon, wt_ordner, branch)
    git_lauf(repo_klon, "fetch", "--prune", "origin")
    remote = f"origin/{branch}"
    remote_da = _ref_da(repo_klon, f"refs/remotes/{remote}")
    if not wt_ordner.exists():
        wt_ordner.parent.mkdir(parents=True, exist_ok=True)
        git_lauf(repo_klon, "worktree", "prune")
        if _ref_da(repo_klon, f"refs/heads/{branch}"):
            git_lauf(repo_klon, "worktree", "add", str(wt_ordner), branch)
        else:
            von = remote if remote_da else (start or f"origin/{basis}")
            git_lauf(repo_klon, "worktree", "add", "--no-track", "-b", branch, str(wt_ordner), von)
            return wt_ordner
    if not remote_da:
        return wt_ordner
    erg = git_lauf(wt_ordner, "merge", "--ff-only", remote, check=False)
    if erg.returncode != 0:
        if not ist_vorfahr(wt_ordner, "HEAD", remote) and not ist_vorfahr(
            wt_ordner, remote, "HEAD"
        ):
            raise KonfliktFehler(branch)
        raise GitFehler("merge", erg.returncode)
    return wt_ordner


def voraus(wt: Path, branch: str, *, basis: str = "master") -> int:
    """Zahl der Commits auf HEAD, die nicht auf `origin/<branch>` liegen.

    Gibt es `origin/<branch>` nicht, gilt `origin/<basis>` als Gegenseite.
    """
    ref = f"origin/{branch}"
    if not _ref_da(wt, f"refs/remotes/{ref}"):
        ref = f"origin/{basis}"
    erg = git_lauf(wt, "rev-list", "--count", "HEAD", f"^{ref}")
    return int(erg.stdout.strip())


def aufraeumen(repo_klon: Path, wt: Path, branch: str) -> None:
    """Entfernt Worktree und lokalen Zweig des Helfers (nur `debugger/*`)."""
    debugger_zweig_pruefen(branch)
    if wt.exists():
        _worktree_pruefen(repo_klon, wt, branch)
        git_lauf(repo_klon, "worktree", "remove", "--force", str(wt))
    git_lauf(repo_klon, "worktree", "prune")
    if _ref_da(repo_klon, f"refs/heads/{branch}"):
        git_lauf(repo_klon, "branch", "-D", branch)


def betreffzeilen(wt: Path, gegen: str) -> list[str]:
    """Betreffzeilen von `<gegen>..HEAD`, aelteste zuerst."""
    aus = git_lauf(wt, "log", "--format=%s", f"{gegen}..HEAD").stdout
    return [z for z in reversed(aus.splitlines()) if z.strip()]


def push_inhalt(wt: Path, branch: str, basis: str) -> str:
    """Prueftext fuer alles, was ein Push nach `branch` hochlaedt (R21, R23).

    Volle Commit-Texte (`%B`, auch Merge-Commits) von `<gegen>..HEAD` plus die rohen Inhalte
    aller neuen Blobs (`rev-list --objects` + `cat-file --batch`, also ohne Diff, Textconv oder
    Attribute), als UTF-8 mit Ersatzzeichen. Gegenseite ist `origin/<branch>`, sonst
    `origin/<basis>`. Mehr als `MAX_PUSH_INHALT` Bytes: `PushZuGross` (fail-closed).
    """
    da = git_lauf(wt, "rev-parse", "--verify", "-q", f"refs/remotes/origin/{branch}", check=False)
    gegen = f"origin/{branch}" if da.returncode == 0 else f"origin/{basis}"
    texte = git_lauf(wt, "log", "--format=%B", f"{gegen}..HEAD").stdout
    gesamt = len(texte.encode("utf-8"))
    if gesamt > MAX_PUSH_INHALT:
        raise PushZuGross()
    objekte = git_lauf(wt, "rev-list", "--objects", "--no-object-names", "HEAD", f"^{gegen}").stdout
    ids = [z.strip() for z in objekte.splitlines() if z.strip()]
    if not ids:
        return texte
    blobs: list[str] = []
    pruefung = git_roh(wt, "cat-file", "--batch-check", eingabe="".join(f"{i}\n" for i in ids).encode())
    for zeile in pruefung.splitlines():
        teile = zeile.decode("ascii", errors="replace").split()
        if len(teile) != 3 or not teile[2].isdigit():
            raise GitFehler("cat-file", 1)
        if teile[1] == "blob":
            blobs.append(teile[0])
            gesamt += int(teile[2])
    if gesamt > MAX_PUSH_INHALT:
        raise PushZuGross()
    if not blobs:
        return texte
    roh = git_roh(wt, "cat-file", "--batch", eingabe="".join(f"{b}\n" for b in blobs).encode())
    inhalte: list[str] = []
    pos = 0
    for _ in blobs:
        ende = roh.find(b"\n", pos)
        kopf = roh[pos:ende].split() if ende >= 0 else []
        if len(kopf) != 3 or kopf[1] != b"blob" or not kopf[2].isdigit():
            raise GitFehler("cat-file", 1)
        start = ende + 1
        stopp = start + int(kopf[2])
        if stopp > len(roh):
            raise GitFehler("cat-file", 1)
        inhalte.append(roh[start:stopp].decode("utf-8", errors="replace"))
        pos = stopp + 1
    return "\n".join([texte, *inhalte])
