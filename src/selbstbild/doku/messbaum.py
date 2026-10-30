"""Messbaum (Ruling 50, Task 15c): der Loop misst nur committeten Code.

Alles, was der Loop misst (Manifeste, Extraktoren, SQL-Definitionen, Compose, Dienst-Register, Beleg-Digests,
`belege_auffrischen`, Aussage-Verankerung), liest er aus einem Messbaum statt aus dem Arbeitsbaum. Nur die Doku, die
er schreibt (KETTEN.md, docs/graph/*), liest und schreibt er im echten Arbeitsbaum.

Der Messbaum liegt in `%LOCALAPPDATA%/VibeMind/doku_loop/messbaum-<repo-hash>/` und besteht aus detached Git-Worktrees,
die den Objektspeicher der echten Checkouts teilen (nie klonen; `vibemind-os/.git` hat 7 GB):

- die Wurzel: das aeussere Repo bei `HEAD`,
- je initialisiertem Submodul (rekursiv, auch verschachtelt) ein Worktree an seinem Pfad, bei dessen echtem `HEAD`.
  Entscheidung (Vorgabe des Briefs): `HEAD` des Checkouts, nicht der Gitlink-Pin -- der Loop dokumentiert den
  Code, der wirklich laeuft; ein `+` in `git submodule status` (HEAD != Pin) ist im echten Repo der Normalfall.

Submodule werden ueber die Gitlinks des committeten HEAD-Baums (`git ls-tree -r HEAD`, Modus 160000) gefunden, nicht
ueber `git submodule status --recursive`: das bricht im echten Repo mitten ab (`learnhouse` in `spaces/learning` hat
keinen Eintrag in `.gitmodules`). Initialisiert heisst: `<pfad>/.git` existiert und `git rev-parse --show-toplevel` ist
genau dieser Pfad (sonst liefe git ins Eltern-Repo hoch). Nicht initialisierte Submodule bleiben leer; Belege darauf
sind `nicht_messbar` (kein Signal, Pruefliste `beleg_nicht_messbar`).

Sicherheitsregeln:
- Jeder git-Aufruf laeuft ohne Git-LFS (`filter.lfs.*` leer, `filter.lfs.required=false`, `GIT_LFS_SKIP_SMUDGE=1`:
  kein Smudge, kein Netz) und -- bei allem, was der Loop im Messbaum ausfuehrt -- mit `core.hooksPath` auf einen
  leeren Ordner im lokalen Ordner des Loops (Fix-Runde 1: `worktree add`/`checkout` liefen sonst die Hooks des
  echten Repos, z. B. LFS-Stubs in `.githooks`, und schrieben dort Dateien).
- In einem Messbaum-Worktree immer mit explizitem `--git-dir`/`--work-tree` (`core.worktree` +
  `extensions.worktreeConfig`, wie bei `openclaw`, zeigte sonst auf den Modul-Ordner selbst).
- Jeder angelegte Worktree bekommt eine Markerdatei in seinem eigenen Verwaltungsordner (dort raeumt `git clean`
  nie). Der Loop fasst nur Pfade innerhalb seines Messbaum-Ordners an (absolut, nie im echten Repo), entfernt nur
  markierte eigene Worktrees (Submodul nicht mehr initialisiert; gescheiterte Entfernungen bleiben im Zustand und
  erscheinen in `status`), ueberschreibt nur eigene Registrierungen und ruft nie `git worktree prune` auf. Lokale
  Aenderungen im Messbaum (die es nie geben sollte) -> `checkout --force` + `clean -fdx`, nur nach Pfadpruefung.
  Nie Commit, nie Stash.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .konfig import Konfig
from .stand import _atomar, lokaler_ordner, repo_schluessel

MARKER = "doku-loop-messbaum"
GIT_TIMEOUT_S = 600  # das erste Anlegen von vibemind-os dauert
GITLINK = "160000 commit "  # `git ls-tree`: <modus> <typ> <objekt>\t<pfad>
HOOKS_LEER = "messbaum-hooks-leer"
# Git-LFS aus: kein Smudge/Clean-Filter, kein Prozess-Filter, fehlender Filter ist kein Fehler
OHNE_LFS = ("-c", "filter.lfs.smudge=", "-c", "filter.lfs.clean=", "-c", "filter.lfs.process=",
            "-c", "filter.lfs.required=false")


class MessbaumFehler(RuntimeError):
    """Messbaum laesst sich nicht bereitstellen; `grund` ist kurz und ohne Dateiinhalte."""

    def __init__(self, grund: str) -> None:
        super().__init__(grund)
        self.grund = grund


@dataclass(frozen=True)
class Messbaum:
    konfig: Konfig  # repo -> Messbaum-Wurzel, Waechter-Zustandsdatei absolut im echten Arbeitsbaum
    wurzel: Path
    commits: dict[str, str]  # repo-relativer Pfad ("" = aussen) -> ausgecheckter Commit
    nicht_messbar: tuple[str, ...]  # repo-relative Pfade nicht initialisierter Submodule


def messbaum_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"messbaum-{repo_schluessel(konfig.repo)}"


def _zustand_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / f"messbaum-{repo_schluessel(konfig.repo)}.json"


def hooks_pfad(konfig: Konfig) -> Path:
    return lokaler_ordner(konfig) / HOOKS_LEER


# ---------- git ----------

def _umgebung() -> dict[str, str]:
    return {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}


def _git(wo: Path, *args: str, hooks: Path | None = None,
         timeout: int = GIT_TIMEOUT_S) -> subprocess.CompletedProcess[bytes]:
    """git ohne LFS; mit `hooks` zusaetzlich `core.hooksPath` auf diesen (leeren) Ordner."""
    vorne = ["git", "-c", "core.quotepath=false", *OHNE_LFS]
    if hooks is not None:
        vorne += ["-c", f"core.hooksPath={hooks}"]
    try:
        return subprocess.run([*vorne, "-C", str(wo), *args], capture_output=True, timeout=timeout,
                              env=_umgebung())
    except subprocess.TimeoutExpired as fehler:
        raise MessbaumFehler(f"timeout:git {args[0]}") from fehler
    except OSError as fehler:
        raise MessbaumFehler(f"git:{type(fehler).__name__}") from fehler


def _git_text(wo: Path, *args: str, hooks: Path | None = None, timeout: int = GIT_TIMEOUT_S) -> str | None:
    lauf = _git(wo, *args, hooks=hooks, timeout=timeout)
    return lauf.stdout.decode("utf-8", errors="replace").strip() if lauf.returncode == 0 else None


def _kurz(stderr: bytes) -> str:
    """Erste Fehlerzeile von git, eine Zeile, kurz (z. B. `Filename too long` bei MAX_PATH)."""
    zeilen = [z.strip() for z in stderr.decode("utf-8", errors="replace").splitlines() if z.strip()]
    wichtig = next((z for z in zeilen if z.lower().startswith(("fatal", "error"))), zeilen[0] if zeilen else "")
    return " ".join(wichtig.split())[:160]


def _gleich(a: Path, b: Path) -> bool:
    return str(Path(a).resolve()).casefold() == str(Path(b).resolve()).casefold()


def _initialisiert(pfad: Path, hooks: Path | None = None, timeout: int = GIT_TIMEOUT_S) -> str | None:
    """HEAD eines initialisierten Checkouts an genau diesem Pfad, sonst None."""
    if not (pfad / ".git").exists():
        return None
    oben = _git_text(pfad, "rev-parse", "--show-toplevel", hooks=hooks, timeout=timeout)
    if oben is None or not _gleich(Path(oben), pfad):
        return None
    return _git_text(pfad, "rev-parse", "--verify", "-q", "HEAD^{commit}", hooks=hooks, timeout=timeout) or None


def _gitlinks(repo: Path, hooks: Path | None = None, timeout: int = GIT_TIMEOUT_S) -> list[str]:
    """Gitlinks des committeten HEAD-Baums (nicht des Index: ein nur gestagtes Submodul gehoert nicht in den
    Messbaum)."""
    lauf = _git(repo, "ls-tree", "-r", "-z", "HEAD", hooks=hooks, timeout=timeout)
    if lauf.returncode != 0:
        raise MessbaumFehler("git:ls-tree")
    return sorted(e.split("\t", 1)[1] for e in lauf.stdout.decode("utf-8", errors="replace").split("\0")
                  if e.startswith(GITLINK) and "\t" in e)


def submodule_finden(repo: Path, hooks: Path | None = None, timeout: int = GIT_TIMEOUT_S
                     ) -> tuple[dict[str, tuple[Path, str]], tuple[str, ...]]:
    """(initialisiert: repo-relativer Pfad -> (echter Checkout, HEAD), nicht initialisiert), rekursiv. Nur lesend
    (`ls-tree`, `rev-parse`: keine Hooks, keine Filter)."""
    da: dict[str, tuple[Path, str]] = {}
    leer: list[str] = []

    def gehen(checkout: Path, praefix: str) -> None:
        for p in _gitlinks(checkout, hooks, timeout):
            rel = praefix + p
            kopf = _initialisiert(checkout / p, hooks, timeout)
            if kopf is None:
                leer.append(rel)
                continue
            da[rel] = (checkout / p, kopf)
            gehen(checkout / p, rel + "/")

    gehen(Path(repo), "")
    return da, tuple(sorted(leer))


def submodul_koepfe(repo: Path, timeout: int = GIT_TIMEOUT_S) -> dict[str, str]:
    """HEAD je initialisiertem Submodul (repo-relativ, rekursiv) -- fuer die Aenderungsermittlung des Ticks."""
    return {rel: k for rel, (_, k) in submodule_finden(repo, timeout=timeout)[0].items()}


# ---------- Worktrees ----------

def _gitdir(ziel: Path) -> Path | None:
    """Verwaltungsordner eines Worktrees aus seiner `.git`-Datei (`gitdir: <pfad>`); kein Worktree -> None.

    Bewusst ohne git: in einem Worktree eines Submoduls mit `core.worktree` in der gemeinsamen Konfiguration und
    `extensions.worktreeConfig=true` (im echten Repo: `openclaw`) loest git die Arbeitskopie falsch auf -- auf den
    Modul-Ordner `.git/modules/<name>` selbst. Darum laeuft jeder git-Aufruf im Messbaum mit explizitem
    `--git-dir`/`--work-tree` (`_git_wt`)."""
    datei = ziel / ".git"
    if not datei.is_file():  # ein Worktree hat eine .git-Datei, ein Haupt-Checkout einen Ordner
        return None
    try:
        zeile = datei.read_bytes().decode("utf-8").strip()
    except OSError:
        return None
    except UnicodeDecodeError as fehler:
        raise MessbaumFehler(f"gitdir_unlesbar:{ziel.name}") from fehler
    if not zeile.startswith("gitdir:"):
        return None
    pfad = Path(zeile[len("gitdir:"):].strip())
    return pfad if pfad.is_absolute() else (ziel / pfad).resolve()


def _git_wt(ziel: Path, *args: str, hooks: Path) -> subprocess.CompletedProcess[bytes]:
    """git in einem Messbaum-Worktree, Arbeitskopie und Verwaltungsordner explizit (nie aus der Konfiguration)."""
    gitdir = _gitdir(ziel)
    if gitdir is None:
        raise MessbaumFehler(f"kein_worktree:{ziel.name}")
    return _git(ziel, "--git-dir", str(gitdir), "--work-tree", str(ziel), *args, hooks=hooks)


def _zeigt_auf(verwaltung: Path, ziel: Path) -> bool:
    """Verwaltungsordner `<common>/worktrees/<name>` gehoert zu `ziel` (Rueckzeiger `gitdir` == `<ziel>/.git`)."""
    try:
        zurueck = Path((verwaltung / "gitdir").read_bytes().decode("utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return False
    return _gleich(zurueck, ziel / ".git")


def _eigener_worktree(ziel: Path) -> bool:
    """`ziel` ist ein Worktree, sein Verwaltungsordner traegt den Marker des Loops und zeigt auf genau `ziel`."""
    gitdir = _gitdir(ziel)
    return gitdir is not None and (gitdir / MARKER).is_file() and _zeigt_auf(gitdir, ziel)


def _innen(ziel: Path, wurzel: Path) -> None:
    z, w = Path(ziel).resolve(), Path(wurzel).resolve()
    if not (z == w or w in z.parents):
        raise MessbaumFehler(f"pfad_ausserhalb:{ziel.name}")


def _leer(ziel: Path) -> bool:
    try:
        return not any(ziel.iterdir())
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _sauber_machen(ziel: Path, wurzel: Path, hooks: Path) -> None:
    _innen(ziel, wurzel)
    lauf = _git_wt(ziel, "status", "--porcelain", "--ignore-submodules=all", "--untracked-files=all", hooks=hooks)
    if lauf.returncode != 0:
        raise MessbaumFehler("git:status")
    if not lauf.stdout.strip():
        return
    # zusaetzliche Plausibilitaetspruefung; die eigentliche Absicherung sind --git-dir/--work-tree und `_innen`
    oben = _git_wt(ziel, "rev-parse", "--show-toplevel", hooks=hooks)
    if oben.returncode != 0 or not _gleich(Path(oben.stdout.decode("utf-8", errors="replace").strip()), ziel):
        raise MessbaumFehler(f"git:toplevel:{ziel.name}")
    for args in (("checkout", "--force", "HEAD"), ("clean", "-fdx")):
        if _git_wt(ziel, *args, hooks=hooks).returncode != 0:
            raise MessbaumFehler(f"git:{args[0]}")


def _registrierung(quelle: Path, ziel: Path, hooks: Path) -> Path | None:
    """Verwaltungsordner, unter dem `ziel` in `quelle` als Worktree registriert ist (auch wenn der Ordner fehlt)."""
    gemeinsam = _git_text(quelle, "rev-parse", "--path-format=absolute", "--git-common-dir", hooks=hooks)
    if not gemeinsam:
        return None
    try:
        kandidaten = list((Path(gemeinsam) / "worktrees").iterdir())
    except OSError:
        return None
    return next((v for v in kandidaten if v.is_dir() and _zeigt_auf(v, ziel)), None)


def _anlegen(quelle: Path, ziel: Path, commit: str, wurzel: Path, rel: str, hooks: Path) -> None:
    _innen(ziel, wurzel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    lauf = _git(quelle, "worktree", "add", "--detach", str(ziel), commit, hooks=hooks)
    if lauf.returncode != 0 and b"already registered" in lauf.stderr:
        # Ordner weg, Eintrag noch da: nur ueberschreiben, wenn der Eintrag nachweislich unserer ist (kein prune)
        alt = _registrierung(quelle, ziel, hooks)
        if alt is None or not (alt / MARKER).is_file():
            raise MessbaumFehler(f"fremde_registrierung:{rel or '.'}")
        lauf = _git(quelle, "worktree", "add", "--force", "--detach", str(ziel), commit, hooks=hooks)
    if lauf.returncode != 0:
        raise MessbaumFehler(f"anlegen:{rel or '.'}:{_kurz(lauf.stderr)}")
    gitdir = _gitdir(ziel)
    if gitdir is None or not gitdir.is_dir():
        raise MessbaumFehler(f"anlegen:{rel or '.'}:kein_gitdir")
    (gitdir / MARKER).write_text(str(ziel), encoding="utf-8")


def _auschecken(ziel: Path, commit: str, wurzel: Path, rel: str, hooks: Path) -> None:
    _sauber_machen(ziel, wurzel, hooks)
    kopf = _git_wt(ziel, "rev-parse", "HEAD", hooks=hooks)
    if kopf.returncode == 0 and kopf.stdout.decode("utf-8", errors="replace").strip() == commit:
        return
    lauf = _git_wt(ziel, "checkout", "--detach", "--force", commit, hooks=hooks)
    if lauf.returncode != 0:
        raise MessbaumFehler(f"auschecken:{rel or '.'}:{_kurz(lauf.stderr)}")


def _entfernen(ziel: Path, wurzel: Path, hooks: Path) -> bool:
    """Nur eigene, markierte Worktrees innerhalb des Messbaums; nie prune. True = weg (oder nie da gewesen)."""
    if not ziel.exists():
        return True
    try:
        _innen(ziel, wurzel)
    except MessbaumFehler:
        return False
    gitdir = _gitdir(ziel)
    if gitdir is None or not _eigener_worktree(ziel):
        return False
    try:  # Verwaltungsordner <common>/worktrees/<name>, `commondir` relativ dazu
        gemeinsam = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
    except (OSError, UnicodeDecodeError):
        return False
    try:
        lauf = subprocess.run(["git", *OHNE_LFS, "-c", f"core.hooksPath={hooks}", "--git-dir", str(gemeinsam),
                               "worktree", "remove", "--force", str(ziel)], capture_output=True,
                              timeout=GIT_TIMEOUT_S, env=_umgebung())
    except (OSError, subprocess.TimeoutExpired):
        return False
    return lauf.returncode == 0


def _bereit(quelle: Path, ziel: Path, commit: str, wurzel: Path, rel: str, hooks: Path) -> None:
    if (ziel / ".git").exists():
        if not _eigener_worktree(ziel):
            raise MessbaumFehler(f"fremder_worktree:{rel or '.'}")
        _auschecken(ziel, commit, wurzel, rel, hooks)
    elif _leer(ziel):
        _anlegen(quelle, ziel, commit, wurzel, rel, hooks)
    else:
        raise MessbaumFehler(f"fremder_pfad:{rel or '.'}")


def _wurzel_pruefen(wurzel: Path, repo: Path) -> None:
    """Der Messbaum-Ordner ist absolut (auch ein nicht expandiertes `%LOCALAPPDATA%` waere relativ) und liegt weder
    im echten Repo noch umgekehrt."""
    roh = str(wurzel)
    if not wurzel.is_absolute() or "%" in roh or "$" in roh:
        raise MessbaumFehler("pfad_nicht_absolut")
    w, r = wurzel.resolve(), Path(repo).resolve()
    if w == r or r in w.parents or w in r.parents:
        raise MessbaumFehler("pfad_im_repo")


def _zustand_lesen(konfig: Konfig) -> dict[str, object]:
    try:
        daten = json.loads(_zustand_pfad(konfig).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return daten if isinstance(daten, dict) else {}


def _liste(daten: dict[str, object], schluessel: str) -> list[str]:
    wert = daten.get(schluessel)
    return [str(x) for x in wert] if isinstance(wert, list) else []


def bereitstellen(konfig: Konfig) -> Messbaum:
    """Messbaum auf den Stand der echten Checkouts bringen (aussen `HEAD`, je initialisiertem Submodul dessen
    `HEAD`). Wirft `MessbaumFehler`; aendert nie etwas im echten Arbeitsbaum oder an Branches."""
    repo = Path(konfig.repo)
    wurzel = messbaum_pfad(konfig)
    _wurzel_pruefen(wurzel, repo)
    hooks = hooks_pfad(konfig)
    hooks.mkdir(parents=True, exist_ok=True)
    if any(hooks.iterdir()):
        raise MessbaumFehler("hooks_ordner_nicht_leer")
    kopf = _initialisiert(repo, hooks)
    if kopf is None:
        raise MessbaumFehler("kein_head")
    subs, leer = submodule_finden(repo, hooks)
    _bereit(repo, wurzel, kopf, wurzel, "", hooks)
    for rel in sorted(subs, key=lambda r: (r.count("/"), r)):  # Eltern vor Kindern
        quelle, k = subs[rel]
        _bereit(quelle, wurzel / rel, k, wurzel, rel, hooks)
    # eigene Worktrees nicht mehr initialisierter Submodule entfernen (sonst misst der Loop alten Code); was nicht
    # entfernt werden kann, bleibt im Zustand, wird beim naechsten Mal erneut versucht und in `status` gezeigt
    daten = _zustand_lesen(konfig)
    weg = {*_liste(daten, "submodule"), *_liste(daten, "entfernen_offen")} - set(subs)
    offen = [rel for rel in sorted(weg, key=lambda r: -r.count("/")) if not _entfernen(wurzel / rel, wurzel, hooks)]
    commits = {"": kopf, **{rel: k for rel, (_, k) in subs.items()}}
    _atomar(_zustand_pfad(konfig), json.dumps(
        {"submodule": sorted(subs), "commits": commits, "nicht_messbar": list(leer), "entfernen_offen": sorted(offen)},
        ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    mess = dataclasses.replace(konfig, repo=wurzel,
                               waechter_zustandsdatei=str(repo / konfig.waechter_zustandsdatei))
    return Messbaum(mess, wurzel, commits, leer)


def messbaum_bereitstellen(konfig: Konfig) -> Konfig:
    """Wie `bereitstellen`, liefert nur die Messbaum-Konfig (`repo` -> Messbaum, sonst gleich; die Waechter-
    Zustandsdatei ist Laufzeitzustand und wird absolut im echten Arbeitsbaum gelesen)."""
    return bereitstellen(konfig).konfig


def messbaum_status(konfig: Konfig) -> dict[str, object]:
    """Fuer `status`: Ordner, Zahl der Worktrees, Commit aussen, nicht messbare Submodule, offene Entfernungen."""
    wurzel = messbaum_pfad(konfig)
    daten = _zustand_lesen(konfig)
    commits = daten.get("commits")
    commits = commits if isinstance(commits, dict) else {}
    return {"pfad": str(wurzel), "vorhanden": (wurzel / ".git").is_file(), "worktrees": len(commits),
            "commit_aussen": commits.get(""), "nicht_messbar": _liste(daten, "nicht_messbar"),
            "entfernen_offen": _liste(daten, "entfernen_offen")}


def vorhanden_lesend(konfig: Konfig, timeout: int = 60) -> Messbaum | None:
    """Der vorhandene Messbaum, rein lesend (Trockenlauf, Ruling T3-a); `None`, wenn er nicht genau dem aktuellen
    aeusseren HEAD und den Submodul-HEADs entspricht. Kein `worktree add`, kein Checkout, kein Schreiben: er gehoert
    dem laufenden alten Loop. Geprueft wird der aufgezeichnete Zustand (`messbaum-<hash>.json`) und, dass jeder
    Worktree wirklich auf seinem Commit steht (`HEAD` im Verwaltungsordner)."""
    repo = Path(konfig.repo)
    wurzel = messbaum_pfad(konfig)
    try:
        _wurzel_pruefen(wurzel, repo)
        daten = _zustand_lesen(konfig)
        commits = daten.get("commits")
        if not isinstance(commits, dict) or not commits:
            return None
        aufgezeichnet = {str(k): str(v) for k, v in commits.items()}
        kopf = _initialisiert(repo, timeout=timeout)
        if kopf is None:
            return None
        jetzt = {"": kopf, **submodul_koepfe(repo, timeout)}
        if aufgezeichnet != jetzt:
            return None
        for rel, commit in jetzt.items():
            gitdir = _gitdir(wurzel / rel) if rel else _gitdir(wurzel)
            if gitdir is None or (gitdir / "HEAD").read_bytes().decode("utf-8").strip() != commit:
                return None
    except (MessbaumFehler, OSError, UnicodeDecodeError):
        return None
    mess = dataclasses.replace(konfig, repo=wurzel,
                               waechter_zustandsdatei=str(repo / konfig.waechter_zustandsdatei))
    return Messbaum(mess, wurzel, jetzt, tuple(_liste(daten, "nicht_messbar")))
