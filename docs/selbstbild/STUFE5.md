# Selbstbild Stufe 5 – Funde als Issues und Konfig-Inventar: Abnahme

Plan: `Vibemind_V1/docs/superpowers/plans/2026-11-04-selbstbild-stufe5-funde.md` (User-OK 04.11.2026).
Code: `src/selbstbild/funde/` (730aa6f … d412b0a), Befehl `python -m selbstbild funde <wurzel> [--senden | --offline] [--nur ART...] [--goldsatz PFAD]`.
Ziel-Konfiguration: äußeres `selbstbild.yaml`, Abschnitt `issues:` (Vibemind_V1 845789f7).

## Ergebnis

Am 04.11.2026 gab es den ersten Versand mit User-OK nach `Flissel/Vibemind_V1` (privat, Konto `Flissel` je Aufruf, Label `selbstbild`).

| Lauf | angelegt | geändert | geschlossen | unverändert | Exit |
|---|---|---|---|---|---|
| Trockenlauf (GitHub nur gelesen) | 27 geplant | 0 | 0 | 0 | 0 |
| 1. `--senden` | 27 | 0 | 0 | 0 | 0 |
| 2. `--senden` direkt danach | 0 | 0 | 0 | 27 | 0 |

Rücklesen bei GitHub: 27 Issues, alle offen, jede Schlüssel-Marke kommt genau einmal vor. Die Schlüssel-Suche (`text.pruefen`) über alle gesendeten Texte ergab 0 Treffer. Der größte Text hat 63.089 Zeichen, das GitHub-Limit liegt bei 65.536.

### Funde (Analyse-Stand 2de425b, HEAD 845789f7)

| Art | hoch | mittel | niedrig |
|---|---|---|---|
| doku_widerspruch | 14 | 19 | 0 |
| doppelung | 0 | 278 | 1381 |
| konfig | 0 | 1 | 0 |
| tot | 0 | 0 | 0 |

- **Toter Code: gesperrt (0/0 bestätigt).** Der Goldsatz (Seite `KCbvPztEFunb8sr6UajstK`) hat noch keine Urteile. Es geht kein Issue raus, und bestehende `tot`-Issues würden nicht geschlossen.
- **Tagesgrenze 5:** 5 der 14 hoch-Funde wurden Einzel-Issues. Die anderen 9 stehen vorerst im Sammel-Issue und werden an den Folgetagen befördert.
- **Konfig-Inventar** (nur Namen): env 3139, key 304, port 54. Befunde: doppelt 193, tot 271, ungesetzt 2450, widersprüchlich 6. LLM-Clients: anthropic 15, openai 93, vibemind_shared 16.
  - 5 Dateien wurden nach der Geheimnis-Regel nicht geöffnet.
  - 11 Submodule sind nicht ausgecheckt. Deshalb gilt `konfig` als unvollständig und wird nicht geschlossen.
- **Stichprobe Inventar:** 5 Einträge wurden von Hand gegen die Quelle geprüft (`ACCEPT_THRESHOLD`, `EMBEDDING_HTTP_MAX_RETRIES`, `CAPTAIN_PORTAL_IMAGE_REFERENCE`, `MCP_UI_PORT`, Port 11434). Es stehen nur Namen drin, die Vorgabewerte der Quellen (z. B. `"0.65"`) fehlen.

### Angelegte Issues

| Nr. | Titel |
|---|---|
| #282 | Doku-Widerspruch: `schedule.snooze` hat je Quelle ein anderes Werkzeug |
| #283 | Doku-Widerspruch: `video.demo_analyze` hat je Quelle ein anderes Werkzeug |
| #284 | Doku-Widerspruch: `video.demo_build` hat je Quelle ein anderes Werkzeug |
| #285 | Doku-Widerspruch: `video.lipsync` hat je Quelle ein anderes Werkzeug |
| #286 | Doku-Widerspruch: `video.team_run` hat je Quelle ein anderes Werkzeug |
| #287 | Konfig-Inventar (vibemind) |
| #288–#289 | Sammlung Doku-Widerspruch: vibemind, vibemind/vibemind-os |
| #290–#308 | Sammlung Doppelung, eine je Code-Repo (19) |

## Abnahme gegen Spec 9 / Review Focus

| Kriterium | Ergebnis |
|---|---|
| Keine Dublette | 2. Lauf: 0 neu. Dubletten werden nur über die Marke im GitHub-Text erkannt, das ist auch ohne lokalen Stand getestet. |
| ≤ 5 Einzel-Issues pro Tag | 5 angelegt, 9 `ueber_grenze`. Konfig zählt nicht mit (S5-11). |
| Secrets | 0 Treffer in allen gesendeten Texten. Inventar nur mit Namen; echte `.env`- und Secrets-Dateien werden nie geöffnet (Deny by default). |
| Konfig nie öffentlich (D6) | Vorgabe `oeffentlich: true`, dazu `GET /repos` → `private` vor jedem Schreiben. |
| Konto je Aufruf (D2) | `GH_TOKEN` nur im Kind-Prozess, nie `gh auth switch`. |
| Nie falsch schließen | Geschlossen wird nur, was zu `geprueft` gehört. Nicht geschlossen werden unlesbare Quellen, nicht gewählte Arten (`--nur`), Arten bei geschlossenem Tor und Analysen, die kein Vorfahre von HEAD sind. `wontfix`/`not_planned` werden nie wieder geöffnet. |
| Sammel-Issue unter Limit | max 59.810 Zeichen. Allgemeine Sperre bei über 65.000 (`zu_lang:`). |
| Hash-Probe des Users | offen, führt der User selbst aus (Befehl in STUFE4.md). |

Tests: `tests/selbstbild/funde` 178 passed, `tests/selbstbild` 1796 passed / 36 skipped (`-W error`).
Reviews: je Task Spec- und Quality-Review mit Fix-Runden. Das finale Gesamt-Review (Simulation von Tag 1 bis 4 mit gefälschtem gh) ergab „Ready after fixes“, die Fixes sind in d412b0a.

## Bekannte offene Punkte

1. **Goldsatz-Urteile fehlen.** Erst ab mindestens 30 beurteilten Kandidaten mit mindestens 90 % „ja“ geht toter Code raus.
2. **Kein automatischer Lauf (D7).** Das Auslösen kommt in Stufe 6, dazu eine Dateisperre gegen parallele `--senden`-Läufe.
3. **`sperren` (label_fehlt) lebt nur im lokalen Stand.** Geht der verloren, ist ein unsichtbares Issue mehr möglich.
4. **`ungesetzt` rauscht (2450).** Ursache sind vendorte Agenten ohne `vendor/`-Pfad. Abhilfe ist eine Ausnahmeliste in `selbstbild.yaml`.
5. **Stufe 1 öffnet getrackte `.env.*`-Dateien beim Extrahieren.** Sie speichert dabei nur Namen. Das liegt außerhalb von Stufe 5.
6. **Abstand gilt je Writer-Aufruf.** Kommentar und Schließen kommen deshalb direkt hintereinander.
7. **D1 bündelt 19 Code-Repos in ein Ziel.** Das ergibt 21 Sammel-Issues; beim Umstellen je Repo neu bewerten.
8. **Die Analyse ist älter als HEAD (2de425b).** Ein frischer Analyse-Lauf aktualisiert die Doppelungen beim nächsten Versand.
