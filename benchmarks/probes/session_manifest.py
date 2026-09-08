"""Session-Archiv als Gate-2-Manifest: der dritte Korpus-Adapter.

`vault_manifest.py` konvertiert einen Wikilink-Vault, `benchmarks/gate2/manifest.py`
einen Python-Baum. Dieser hier nimmt ein Session-Archiv (brain_claude
`10-Sessions/`): pro Datei eine Claude-Code-Session mit Frontmatter
(`session_id`, `workspace`, `captured`, `title`) und einem Prompt-Auszug.

**Warum ein eigener Adapter noetig war.** Der Wikilink-Parser findet hier 13
Kanten auf 5 062 Dateien — die Expansion, der gemessene Kernwert des Stacks
(Report Sektionen 27/28), liefe praktisch leer. Ein Session-Archiv traegt seine
Struktur aber woanders: **zeitlich benachbarte Sessions desselben Workspace**
gehoeren zusammen. Diese Kante ist mechanisch gegeben wie eine Aufruf-Kante,
nicht authoriert und nicht geraten — genau die Sorte, die §27 als tragfaehig
gemessen hat.

Bewusst *eng* gewaehlt: nur Vorgaenger und Nachfolger, nicht der ganze Workspace.
Ein Workspace haelt hier bis zu vierstellig viele Sessions; "alle im selben
Workspace" waere keine Kante, sondern eine Partition, und wuerde die
Kandidatenmenge sprengen statt sie zu schaerfen.

`symbol` wird lesbar gemacht: viele Sessions heissen "(untitled session)", und ein
GUID-Dateiname hilft niemandem. Ersatz ist dann `<workspace-kurz> <datum>`.

**Privat.** Session-Auszuege enthalten echte Prompts und Pfade. Ausgabe geht nach
`benchmarks/results/vault/` — gitignoriert, Remote ist public.

Aufruf::

    PYTHONPATH=src python -m benchmarks.probes.session_manifest \
        --sessions "<pfad-zum-session-archiv>/10-Sessions" \
        --out-dir benchmarks/results/vault --manifest-id sessions-v1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
UNTITLED = "(untitled session)"


def _frontmatter(text: str) -> dict[str, str]:
    treffer = FRONTMATTER.match(text)
    if not treffer:
        return {}
    felder: dict[str, str] = {}
    for zeile in treffer.group(1).splitlines():
        if ":" not in zeile:
            continue
        schluessel, _, wert = zeile.partition(":")
        felder[schluessel.strip()] = wert.strip()
    return felder


def _kurz(workspace: str) -> str:
    """Lesbarer Kurzname fuer den Workspace.

    Die Werte sind heterogen: pfadkodiert (`c--Users-<name>-Desktop-<projekt>`),
    aber auch `subagents` oder `wf_<id>`. Nur die pfadkodierten werden gekuerzt —
    bei den anderen IST der Wert bereits der Name, und ihn zu zerlegen erzeugt
    Unfug wie `f94` oder `d9b`, an dem niemand Relevanz erkennt.
    """
    if not workspace.lower().startswith(("c--", "d--", "e--")):
        return workspace
    teile = [t for t in workspace.split("-") if t]
    for anker in ("Desktop", "Documents"):
        if anker in teile:
            rest = teile[teile.index(anker) + 1 :]
            if len(rest) > 2:
                return "-".join(rest[:2] + rest[-1:])
            if rest:
                return "-".join(rest)
    return "-".join(teile[-2:]) if len(teile) >= 2 else workspace


def build(sessions_root: Path, out_dir: Path, manifest_id: str) -> None:
    dokumente: list[dict[str, object]] = []
    for pfad in sorted(sessions_root.rglob("*.md")):
        text = pfad.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        felder = _frontmatter(text)
        workspace = felder.get("workspace", "unknown")
        captured = felder.get("captured", "")[:10]
        titel = felder.get("title", "").strip()
        if not titel or titel == UNTITLED:
            titel = f"{_kurz(workspace)} {captured}".strip() or pfad.stem[:8]
        dokumente.append(
            {
                "document_id": pfad.relative_to(sessions_root).as_posix(),
                "file": pfad.relative_to(sessions_root).as_posix(),
                "start_line": 1,
                "end_line": max(1, text.count("\n") + 1),
                "symbol": titel,
                "source": text,
                "_workspace": workspace,
                "_captured": felder.get("captured", ""),
            }
        )

    # Kanten: zeitliche Nachbarn je Workspace. Ungeordnete captured-Werte sortieren
    # stabil ans Ende, damit eine fehlende Zeitangabe keine falsche Nachbarschaft
    # erfindet.
    nach_workspace: dict[str, list[dict[str, object]]] = {}
    for dokument in dokumente:
        nach_workspace.setdefault(str(dokument["_workspace"]), []).append(dokument)

    callees: dict[str, set[str]] = {str(d["document_id"]): set() for d in dokumente}
    for gruppe in nach_workspace.values():
        gruppe.sort(key=lambda d: (str(d["_captured"]) or "9999", str(d["document_id"])))
        for davor, danach in zip(gruppe, gruppe[1:]):
            callees[str(davor["document_id"])].add(str(danach["document_id"]))

    callers: dict[str, set[str]] = {str(d["document_id"]): set() for d in dokumente}
    for origin, ziele in callees.items():
        for ziel in ziele:
            callers[ziel].add(origin)

    for dokument in dokumente:
        dokument.pop("_workspace", None)
        dokument.pop("_captured", None)

    payload = {
        "manifest_id": manifest_id,
        "corpus_root": str(sessions_root),
        "commit_sha": "session-archive",
        "documents": dokumente,
        "callees_by_document": {k: sorted(v) for k, v in sorted(callees.items())},
        "callers_by_document": {k: sorted(v) for k, v in sorted(callers.items())},
        "discarded_names": [],
        "unresolved_names": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_pfad = out_dir / f"{manifest_id}.json"
    manifest_pfad.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    quellen_pfad = out_dir / f"{manifest_id}-sources.json"
    quellen_pfad.write_text(
        json.dumps([d["source"] for d in dokumente]), encoding="utf-8"
    )

    kanten = sum(len(v) for v in callees.values())
    print(f"sessions   : {len(dokumente)}")
    print(f"workspaces : {len(nach_workspace)}")
    print(f"kanten     : {kanten} (zeitliche Nachbarn je Workspace)")
    print(f"manifest   : {manifest_pfad}")
    print(f"sources    : {quellen_pfad}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Session-Archiv -> Gate-2-Manifest")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-id", default="sessions-v1")
    arguments = parser.parse_args()
    build(arguments.sessions, arguments.out_dir, arguments.manifest_id)


if __name__ == "__main__":
    main()
