# Selbstbild Stufe 1: Abnahme

Stand 2026-10-28. Paket `selbstbild` (`src/selbstbild/`), Tests `tests/selbstbild/` (155 grün mit
`-W error`). Alle Läufe lesen nur committeten Code, schreiben nur in den lokalen Stand
(`%LOCALAPPDATA%\selbstbild\<name>-<hash>\graph.sqlite`) und nie in die vermessenen Arbeitsbäume.

## Bedienung

```
python -m selbstbild vermessen <wurzel> [--commit C] [--voll]   # Bericht als JSON
python -m selbstbild export <wurzel> <ziel.json>                # deterministischer Export
python -m selbstbild status <wurzel>                            # letzter Commit, Zähler, Alter
python -m selbstbild -v vermessen ...                           # Fortschritt je Repo nach stderr
```

`export` und `status` finden den Stand über den Namen aus der `selbstbild.yaml` von `HEAD`
(sonst den Ordnernamen).

## Abnahme-Läufe

Host: Windows 11, 32 GB RAM. Python 3.11, tree-sitter 0.26.0, Language-Pack 1.22.1.

| Lauf | Commit | Dateien | geparst | aus Cache | übersprungen (Grund: Anzahl) | Knoten | Kanten | unsicher | fehlerhafte Dateien | fehlende Submodule | Dauer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a) `Vibemind_V1 --voll`, 20 Repos (äußeres Repo + 19 Submodule) | `6f8e4e0f` | 19 607 | 19 607 | 0 | keine_sprache 2004, binaer 332, zu_gross 92, ausgenommen 41, submodul_fehlt 7, lfs 4, symlink 2 | 234 874 | 483 389 | 25 800 (5,3 %) | 87 | 7 (s. u.) | **290 s** voll |
| b) zweiter Lauf, gleicher Commit, ohne `--voll` | `6f8e4e0f` | 19 607 | **0** | 19 607 | wie a) | 234 874 | 483 389 | 25 800 | 87 | 7 | **46 s** inkrementell |
| c) `vibemind-os/spaces/sales-claw --voll` (eigenes Git-Repo, keine Regeldatei, also Vorgaben) | `c8679622` | 599 | 599 | 0 | keine_sprache 49, binaer 25, ausgenommen 2, zu_gross 1 | 9 068 | 21 119 | 2 288 (10,8 %) | 1 | 0 | 7,3 s |

Die Dauer ist `Bericht.dauer_s`, also nur `vermessen()` ohne Prozessstart und Export.
Lauf c) endete ohne Änderung am Kern. Lauf c) misst den `HEAD` des sales-claw-Checkouts, nicht
den Pin aus vibemind-os, deshalb 599 statt 228 Dateien wie in a).

**Determinismus:** Der Export von a) wurde zweimal erzeugt (zwei getrennte volle Läufe) und
einmal nach b). Alle drei waren byte-gleich (366 MB, sha256 `3132d1f5…0b4915`).

**Speicher (Peak Working Set, `K32GetProcessMemoryInfo`):**

| Prozess | Peak |
|---|---|
| a) `vermessen` voll + `export_json` | 2,83 GB |
| b) `vermessen` inkrementell + `export_json` | 2,83 GB |
| b') `vermessen` inkrementell, ohne Export | 1,12 GB |
| c) sales-claw voll + Export | 0,15 GB |

Den Peak treibt `export_json`, nicht die Vermessung (siehe Grenzen). `graph.sqlite` für a) ist
814 MB groß (Parse-Cache, Graph und Symbolverlauf).

**Fehlende Submodule in a)** (nicht initialisiert, als `submodul_fehlt` gemeldet, der Rest wurde
vermessen): `vibemind-os/spaces/desktop/Automation_ui`, `vibemind-os/spaces/sales`,
`vibemind-os/spaces/learning/learnhouse`, `vibemind-os/spaces/learning/penecho`,
`vibemind-os/voice/external/Autogen_AgentFarm`, `vibemind-os/voice/external/openclaw`,
`vibemind-os/voice/external/zeroclaw`.

**Graph a) nach Art:** 205 038 Symbole, 19 607 Dateien, 3 999 Doku-Seiten, 3 406 Env-Namen,
1 379 Routen, 670 MCP-Werkzeuge, 446 Tabellen, 158 SQL-Funktionen, 151 Dienste, 20 Repos.

| Kantenart | Anzahl | davon unsicher |
|---|---|---|
| enthaelt | 224 645 | 0 |
| ruft_auf | 220 598 | 23 006 (10,4 %) |
| importiert | 24 723 | 1 009 (4,1 %) |
| liest_env | 7 889 | 0 |
| liest_schreibt | 2 743 | 1 776 (64,7 %) |
| registriert | 673 | 0 |
| definiert_route | 1 512 | 0 |
| nutzt_dienst | 457 | 0 |
| verbindet | 149 | 9 |

## d) Secret-Probe (startet der User selbst)

`scripts/secret_probe.py` liest die Werte einer `.env` nur im Speicher und sucht jeden Wert
(roh und JSON-escaped, als UTF-8-Bytes) in `graph.sqlite`, einer evtl. vorhandenen
`graph.sqlite-wal` und im Export. Ausgegeben werden nur Variablennamen mit Treffer und Zähler.
Exit 1 bei mindestens einem Treffer. Werte unter 6 Zeichen werden ignoriert, weil sie als
gewöhnliche Teilstrings überall vorkommen.

Abweichung von Spec 12a: Statt sha256-Vergleich mit Kandidaten-Strings wird direkt nach
Teilstrings gesucht (Ruling R16). Das ist strenger, weil es auch Werte findet, die in
längere Strings eingebettet sind.

Aufruf (nach Lauf a, mit Export):

```
python -m selbstbild export C:\Users\User\Desktop\Vibemind_V1 %TEMP%\vibemind-export.json
python scripts\secret_probe.py --env C:\Users\User\Desktop\Vibemind_V1\.env ^
  --graph %LOCALAPPDATA%\selbstbild\vibemind-8ac5c0f3db0b\graph.sqlite ^
  --export %TEMP%\vibemind-export.json
```

Der Implementierer hat das Skript nur mit einer Fake-`.env` in den Unit-Tests ausgeführt
(`tests/selbstbild/test_secret_probe.py`), nie mit echten Werten. In den Tests von `vermessen`
prüft `test_keine_secret_werte` dieselbe Eigenschaft für `.env`, `.env.example`, Compose,
PowerShell und Python.

## Bekannte Grenzen

1. **Aufrufkanten über Paket- und Repo-Grenzen.** Es gibt null `ruft_auf`-Kanten zwischen zwei
   Repos, denn die Auflösung läuft je Repo. Brücken über Repos entstehen nur über geteilte
   Ressourcen (Env-Namen, Dienste, Tabellen, Routen). Von den 60 016 Aufrufkanten zwischen
   verschiedenen Top-Level-Ordnern sind 8 462 (14 %) unsicher. TS-Namensimporte bleiben
   unsicher (R15), relative Python-Importe über mehr als eine Ebene ebenfalls (R13).
2. **Tabellenzugriffe meist unsicher** (65 % in a, 99,8 % in sales-claw). Die Tabellen sind
   nicht im selben Repo per SQL definiert (Supabase-Schema liegt woanders), und die Regex
   erkennt nur SQL mit führendem Verb (R9).
3. **`seit` hängt von den gemessenen Läufen ab.** Der Symbolverlauf wird je Lauf
   fortgeschrieben. Ein frischer Stand kennt nur den aktuellen Commit, dort ist `seit` die
   Commit-Zeit dieses Laufs. „Inkrementell == voll, byte-gleich“ gilt deshalb für alles außer
   `seit`, sobald der inkrementelle Stand ältere Commits gesehen hat (der Test
   `test_inkrementell_gleich_voll` nutzt bewusst gleiche Commit-Zeiten).
   Submodul-Symbole tragen die Commit-Zeit ihres Submoduls.
4. **Export braucht viel RAM.** `export_json` baut das ganze Dokument im Speicher (Objekte,
   JSON-Text und UTF-8-Bytes). Bei 235 k Knoten und 483 k Kanten sind das etwa 1,7 GB
   zusätzlich zur Vermessung. Für Stufe 2 sollte der Export streamen.
5. **Regeln gelten je Repo.** Die `selbstbild.yaml` im äußeren Repo wirkt nur auf dessen
   Pfade. Die Globs `vibemind-os/spaces/**` (doku_pfade, Strukturregel, Einstiegspunkte)
   erreichen das Submodul nicht, dort gelten dessen eigene Regeln bzw. die Vorgaben
   (`**/*.md` als Doku). Das Dienst-Register (`config/dienste.yaml`) wird dagegen über alle
   Repos gemischt, das äußere gewinnt bei Doppelungen. Strukturregeln und `issue_ziel` werden
   in Stufe 1 nur geladen, nicht ausgewertet.
6. **Fehlerhafte Dateien:** 87 in a), davon 51 YAML (Vorlagen und Mehrfach-Dokumente), 35 TS/TSX/JSX
   und 1 Python. Was tree-sitter erkennt, wird übernommen. Eine Parser-Ausnahme bricht den Lauf
   nicht ab, die Datei wird nur als fehlerhaft markiert (im Lauf a trat keine auf).
7. **Jeder Lauf liest alle Blobs einmal**, auch der inkrementelle. Das liegt an der Binär- und
   LFS-Prüfung in `baum_lesen`. Zusammen mit `git log` je Repo für die Zeitachse ergibt das die
   46 s für einen Lauf ohne Änderung.
8. **Parse-Cache** ist pfadabhängig (R11). Umbenennungen werden deshalb neu geparst. Der Cache
   wird nie aufgeräumt und wächst mit jedem neuen Blob.
9. Gleiche Blobs am gleichen Pfad in zwei Repos teilen einen Cache-Eintrag. Das ist gewollt,
   denn Parser-Ergebnisse sind repo-frei (R10). In a) sind es 19 605 Einträge für 19 607 Dateien.
