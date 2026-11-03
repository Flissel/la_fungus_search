# Selbstbild Stufe 4: Doku-Loop in `selbstbild.doku` (Abnahme)

Stand: 03.11.2026, **Teil A** von Task 11 (Ruling T11-c): nur lesend messen und diesen Entwurf schreiben. Code aus
Task 1 bis 10, la_fungus `main` @ `4de58f4`. Für diese Abnahme wurde **kein Code geändert**. Gemessen wurde VibeMind
(`C:\Users\User\Desktop\Vibemind_V1`, Zweig `codex/vibemind-ops-baseline`) am Commit `23bd42ba`, Messbaum `4048bc83`
(nur loop-eigene Dateien dahinter, T3-c).

**Nachtrag Final-Fix-Welle (03.11.2026):** Nach dem Final-Review kam eine Fix-Welle; der Abschnitt
„Final-Fix-Welle“ unten listet sie. Schritt 2 beschreibt jetzt den Ablauf von Teil B.

**Teil B ist live (03.11.2026, ab 21:27).** Die Windows-Aufgabe ist umgeschaltet. Erster erzwungener Lauf, Ruhe über
10 Läufe und die deterministischen Rowboat-Seiten sind erledigt; der alte Einstieg verweigert `tick`. Im Betrieb
kamen zwei Befunde, beide mit Hotfix (Abschnitt „Befunde im Betrieb“). Offen bleiben die erste Nacht der
Erstbefüllung und die Punkte unter „Bekannte Grenzen und offene Punkte“.

Stufe 4 verpflanzt den Doku-Loop aus `scripts/doku_loop/` als Paket `selbstbild.doku` (CLI
`python -m selbstbild doku …`). Neu sind die Doku-Ziele `docs/spaces/0*.md` und `docs/modules/0*.md`, die nur
gelesenen Space-READMEs im Submodul, Events-Blöcke je Space, generierte Doku für Spaces ohne Doku, die nächtliche
Erstbefüllung, erzählte und deterministische Rowboat-Seiten und das Umschalten mit Rückweg.

**Kurzfassung:**

| Ziel | Ergebnis | Status |
|---|---|---|
| Trockenvergleich: gleicher `digest_stabil` wie das committete `graph.json` (T11-a: ohne `gemessen`) | Altbestand **byte-gleich im Digest** (`b232ebd2…`); dazu nur neuer Inhalt: 1 294 Aussagen, 72 Modul-Knoten, 144 Events, 2 071 `belegt_durch`, 144 `ereignis_von`. KETTEN-Teil (108 Hops, 13 Aussagen, 18 Belege) identisch | **erreicht** |
| Signale nur aus neuen Zielen, nie aus KETTEN.md; LLM 0 | 0 Signale, 0 LLM-Aufrufe, 0 Commits | erreicht |
| Belege mit Anker ≥ 95 % exakt (Spec 9) | KETTEN.md: **119 von 127 (93,7 %)**; alle 8 Abweichungen liegen in Dateien mit fremden, nicht committeten Änderungen (ohne sie 119/119). Neue Ziele: keine gerenderten Belege mit Zeile (s. Schritt 4) | **knapp verfehlt** (Ursache außerhalb des Loops, s. Schritt 4) |
| Events: Blöcke für die Registry-Spaces, Video-Fall | 144 Events in 14 Spaces, 13 Blöcke in 12 Dokus; Video 9 (Registry) gegen 11 (`EVENT_TO_TOOL`) als 10 × `ereignis_quellen_abweichend`; `video.reel` u. a. als 5 × `ereignis_nur_doku` | erreicht |
| Neue Spaces | geplant nur `docs/spaces/021_sales.md` + eine Zeile im README-Block (Ruling T6-a); Fixture-Test 19/19 grün | erreicht (Inhalt weicht bewusst vom Brief ab, T6-a) |
| Rowboat-Seiten | Live geschrieben: `Index.md`, `Prüfliste.md` (gruppiert), `Zustand.md`, `Mitbauen/…`. Die 40 erzählten Seiten (1 + 21 + 18) warten auf die Nacht (P3); 124 vorhandene Notizen, 0 von Hand bearbeitet | deterministisch **erreicht**; erzählt **offen (erste Nacht)** |
| Secrets: 0 Treffer der Schlüsselmuster | 0 Treffer in 183 Dateien bzw. Ausgaben (7 Muster); nach der Fix-Welle 9 Muster, 0 Treffer in 40 Doku-Dateien | erreicht; Hash-Probe mit echten Werten **startet der User** |
| Tests | `tests/selbstbild` 1 602 passed, 36 skipped (`8f5f240`); `tests/selbstbild/doku` nach T4-c3 957 passed, 35 skipped (`f0e5060`); 0 failed | erreicht |
| Umschaltung (Teil B) | 21:27:17 `aufgabe --umschalten`, Sicherung `doku_loop-sicherung-20261103-212717` (6 Dateien, ohne `messbaum-*`), `aufgabe-alt.json` geschrieben, neue Aktion zurückgelesen = Ziel, `--pruefen`: `Loop: neu` | **erreicht** |
| Erster erzwungener Lauf (Teil B) | 21:27:32, 86 s: Commit `be80fb3` (15 Dateien: 13 Events-Blöcke in 12 Dokus, `021_sales.md`, README-Block, `graph.json` 0,35 → 2,9 MB), 1 294 Aussagen neu gemerkt, 0 Signale, 0 LLM, Torwächter ohne neue Fehler | **erreicht** |
| Ruhe (Spec 9): 10 Läufe ohne Änderung | 22:53–23:38, 10 Läufe (Lauf-Nummern #322–#323; Läufe ohne Anstoß zählen keine Nummer): 0 LLM, 0 Commits, 0 Fehler | **erreicht** (nach T4-c2/T4-c3) |
| Betrieb ohne Lärm (Anti-Ziel „Lauter Hintergrund“) | zwei Befunde im ersten Betrieb, beide mit Hotfix (`fdd99ac`, `f0e5060`) | erreicht nach Hotfix |
| Erste Nacht der Erstbefüllung | — | **offen** (ab 00:00) |

## Befehle (PowerShell, im la_fungus-Arbeitsbaum)

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH="src"
.\.venv-selbstbild\Scripts\python -m selbstbild doku aufgabe C:\Users\User\Desktop\Vibemind_V1 --pruefen          # nur lesend
.\.venv-selbstbild\Scripts\python -m selbstbild doku tick C:\Users\User\Desktop\Vibemind_V1 --trocken             # schreibt nichts
.\.venv-selbstbild\Scripts\python -m selbstbild doku aufgabe C:\Users\User\Desktop\Vibemind_V1 --umschalten --trocken
.\.venv-selbstbild\Scripts\python -m selbstbild doku aufgabe C:\Users\User\Desktop\Vibemind_V1 --zurueck --trocken
```

Der alte Loop wurde einmal von Hand angestoßen (Ruling T11-d), aus `C:\Users\User\Desktop\Vibemind_V1`:
`cmd /c "scripts\doku-loop.cmd tick --erzwingen >> logs\doku-loop.log 2>&1"`, also mit derselben Umleitung wie die
Aufgabe.

`doku messen` lief **nicht** über die CLI: Sie nimmt `tick.lock` im lokalen Ordner des alten Loops und stellt den
Messbaum bereit (`messbaum_bereitstellen`), beides gehört bis zur Umschaltung dem alten Loop (Ruling T3-a). Stattdessen
rief ein Skript dieselbe Funktion `messen.belege_messen` gegen den **vorhandenen** Messbaum auf
(`messbaum.vorhanden_lesend`, wie der Trockenlauf), je Doku-Datei.

## Läufe und Speicher

Host: Windows 11, 32 GB RAM, parallel andere Sessions. Vor jedem Lauf ≥ 3 GB freier Commit
(`FreeVirtualMemory`). Kein Lauf wurde abgebrochen, kein fremder Prozess beendet.

| Lauf | Zeit | Dauer | freier Commit vorher |
|---|---|---|---|
| `aufgabe --pruefen` (Sperre frei, Loop alt) | 19:28 | < 5 s | 5,46 GB |
| alter Loop `tick --erzwingen` (T11-d) | 19:29:08–19:30:10 | 62 s | 5,46 GB |
| `doku tick --trocken` (CLI) | 19:30:17 | 165 s | 5,41 GB |
| Trockenlauf mit Mitschnitt (gleicher Code, Graph und Texte festgehalten) | 19:33 | 155 s | 5,92 GB |
| Belege messen (lesend, vorhandener Messbaum) | ca. 19:37 | 51 s | 5,67 GB |
| `test_doku_neue_spaces.py` | ca. 19:39 | 92 s | 6,02 GB |
| volle Suite `tests/selbstbild` | 19:41:43–19:55:04 | 800 s | 5,90 GB |

Nachweis „schreibt nichts“: Schnappschuss (Name, Größe, mtime) des Ordners `%LOCALAPPDATA%\VibeMind\doku_loop\`
(ohne `messbaum-*`), seiner Geschwister `doku_loop-*`, von `knowledge\Doku\**`, des äußeren HEAD und des Hashes von
`docs/graph/graph.json` vor beiden Trockenläufen und danach: **135 Einträge, 0 Unterschiede.** Der geplante Tick des
alten Loops um 19:33:57 lief dazwischen (`kein_anstoss`).

## Schritt 1: Trockenvergleich vor der Umschaltung

**Alter Loop, einmal erzwungen (T11-d).** Vorher `aufgabe --pruefen`: Loop alt, Sperre frei. Direkt nach dem geplanten
Tick 19:28:57 (`kein_anstoss`) gestartet. Log (`logs/doku-loop.log`):

```
Anstoß: erzwungen; Marker gelöscht
Urteile: 0 gelesen, 0 neu, 0 unlesbar, Prüfliste +0
Änderungen: 0 Dateien außen, 0 im Submodul -> Spaces: brain, marketing, sales-claw; Beleg-Dateien außerhalb: 0; offene Signale: 0
Fremde Änderungen: keine
Messbaum: 20 Worktree(s) auf HEAD, 7 Submodul(e) nicht messbar
Graph: 3 Space(s) gebaut, +0/-0 Kanten
Belege: 0 Signal(e) (Hops 0, Aussagen 0); Judge 0 Aufruf(e), 0 umgeschrieben
Render: 19 Block/Blöcke, 3 neu, 0 neue Hand-Edits
Torwächter: keine neuen Fehler
Commit: 23bd42b (2 Datei(en))
Stand: Lauf 305
Notizen: 2 geschrieben, 0 entfernt, 0 von Hand, 0 umgeleitet (nicht angefasst)
Ergebnis: Lauf 2026-11-03T19:29:08#305; Spaces brain, marketing, sales-claw; Signale 0; LLM 0; umgeschrieben 0; Prüfliste +0; Commit 23bd42b
```

Commit `23bd42ba` (`docs(loop): brain, marketing, sales-claw – 0 Aussagen nachgeführt, 0 in Prüfliste`):
`docs/graph/graph.json` und `docs/spaces/KETTEN.md`, +46/−46. Inhalt: Er übernimmt die neun seit dem 30.10. ausstehenden
Lesungen `gemessen: bricht` für K6/1, 3, 4, 5 und K7/1, 2, 4, 5, 6 (Hysterese, s. Task-3-Bericht). Kein Code, keine
Aussage.

**Trockenlauf direkt danach** (`doku tick … --trocken`, exit 0, 165 s):

```
Messung: 2 Hop(s) mit geänderter Lesung – gilt erst beim nächsten gleichen Wert
Graph: 3 Space(s) gebaut, +0/-0 Kanten
Erstbefüllung: 532 Aussage(n) offen; nicht fällig, hier nicht gefragt
Aussagen: 29 Doku-Datei(en), 9 nur lesend, 0 ausgelassen; 2947 Aussagen, 1307 belegt, 1294 neu gemerkt
Belege: 0 Signal(e) (Hops 0, Aussagen 0); Judge 0 Aufruf(e), 0 umgeschrieben
Neue Spaces: 19 Space-Ordner, ohne Doku sales; angelegt docs/spaces/021_sales.md; generierte Dokus 1
Ereignisse: 144 in 14 Space(s), 13 mit Doku; Prüfliste ereignis_nur_doku 15, ereignis_quellen_abweichend 17, space_ohne_doku 1
Render: 36 Block/Blöcke, 17 neu, 0 neue Hand-Edits
Seiten: 40 erzählte Seite(n), 40 fällig, 0 erzählt (0 Aufruf(e) unter -), 40 noch nie geschrieben
Notizen: 3 geschrieben, 0 entfernt, 0 von Hand, 0 umgeleitet (nicht angefasst)
```

Ergebnis-JSON (Auszug): `uebersprungen: null`, `hinweis: messbaum_hinter_head_nur_eigene_dateien`, `messbaum_commit:
4048bc83`, `signale 0`, `llm_aufrufe 0`, `commit null`, `graph_digest e2740853…84d4`. Der zweite Trockenlauf (mit
Mitschnitt) lieferte denselben Digest.

**Vergleich der schlanken Graphen** (committetes `graph.json` an `23bd42ba` gegen den Trockenlauf):

| | committet | Trockenlauf | Unterschied |
|---|---|---|---|
| Knoten `aussage` | 13 | 1 307 | +1 294 (neue Ziele) |
| Knoten `ereignis` | 0 | 144 | +144 (Task 5) |
| Knoten `modul` | 3 | 75 | +72 (Endpunkte neuer `belegt_durch`, je eine Code-Datei) |
| Knoten `dienst` / `kette` / `komponente` / `space` / `mcp_werkzeug` / `mensch` / `tabelle` | 101 / 18 / 155 / 3 / 2 / 2 / 1 | gleich | 0 |
| Kanten `hop` | 108 | 108 | 0 |
| Kanten `verbindet` | 78 | 78 | 0 |
| Kanten `belegt_durch` | 18 | 2 089 | +2 071 |
| Kanten `ereignis_von` | 0 | 144 | +144 |

- **Kein vorhandener Knoten und keine vorhandene Kante ist geändert oder weg.** Ohne die neuen Knoten und Kanten hat der
  Trockenlauf-Graph genau den `digest_stabil` des committeten Graphen, `b232ebd246f0…a8da`, **sogar mit
  `gemessen`**. Die Abweichung der vollen Digests (`b232ebd2…` gegen `e2740853…`) kommt also nur aus neuem Inhalt.
- Struktur-Digest nach T11-a (alle Schlüssel `gemessen` und `zeit` entfernt, `Beleg.commit` entfernt): committet
  `e4d91ac9…`, Trockenlauf `c0dd2c17…`; der Unterschied ist derselbe neue Inhalt. `gemessen`-Abweichungen: **0**
  (der erzwungene Tick hatte die ausstehenden Lesungen gerade übernommen; zwei neue Lesungen stehen wieder aus).
- **KETTEN-Teil identisch:** 108 Hops, 13 KETTEN-Aussagen und ihre 18 `belegt_durch`-Kanten gleich.
- Neue Aussagen je Datei: Module 27–89, Space-Dokus 13–70, nur-lesend-READMEs 1–27 (`sales-claw` 27). Neue
  `belegt_durch`-Belege nach Art: `datei` 1 413, `symbol` 657, `yaml_key` 87, `sql` 6.
- Kein `messbaum_veraltet`: Seit dem Messbaum-Commit `4048bc83` kam nur der eigene Commit `23bd42ba` dazu.

**Geplante Doku-Schreibungen** (nur Diff, nicht geschrieben):

| Datei | Block | Zeilen | Zeilenenden |
|---|---|---|---|
| `001_agentfarm.md` | `ereignisse agentfarm` | +9 | LF |
| `003_coding.md` | `ereignisse coding` | +15 | LF |
| `004_desktop.md` | `ereignisse desktop` | +35 | CRLF bleibt (351 → 386 CRLF) |
| `005_flowzen.md` | `ereignisse flowzen` | +10 | LF |
| `006_ideas.md` | `ereignisse bubbles`, `ereignisse ideas` | +51 | LF |
| `008_marketing.md` | `ereignisse marketing` | +20 | LF |
| `009_minibook.md` | `ereignisse minibook` | +11 | LF |
| `010_mirofish.md` | `ereignisse mirofish` | +14 | LF |
| `013_research.md` | `ereignisse research` | +13 | LF |
| `014_rowboat.md` | `ereignisse rowboat` | +15 | LF |
| `016_schedule.md` | `ereignisse schedule` | +14 | LF |
| `018_video.md` | `ereignisse video` | +18 | CRLF bleibt (237 → 255 CRLF) |
| `021_sales.md` (neu) | `stand`, `steckbrief`, `belege` | +23 | LF |
| `README.md` | `neue-spaces` | +8 | LF |

Alle Diffs fügen nur ein (0 entfernte Zeilen): eine Überschrift `## Events (generiert)` und den markierten Block. KETTEN.md
ändert sich nicht. Keine BOM.

**Erstbefüllung und Seitenplan:** 532 Aussagen offen (nicht fällig um 19:30, Fenster 00:00–06:00), dazu 40 erzählte
Seiten im Zustand `neu`. Schätzung: 532 × 2 + 40 = 1 104 Aufrufe, bei 150/Tag **mindestens 8 Nächte**. Task 7 hatte
451 geschätzt; die Näherung damals kannte nicht alle Space-Manifeste.

## Schritt 2: Umschaltung – Teil B (erledigt)

**Ergebnis (03.11.2026, live, Angaben aus dem Ledger des Controllers):**

| Schritt | Zeit | Ergebnis | Status |
|---|---|---|---|
| 1. `aufgabe --umschalten` | 21:27:17 | Sicherung `doku_loop-sicherung-20261103-212717` (6 Dateien, kein `messbaum-*`), `aufgabe-alt.json` geschrieben, neue Aktion zurückgelesen = Ziel | erledigt |
| 2. erster Lauf `tick --erzwingen` | 21:27:32 | 86 s; 0 Signale, 0 LLM-Aufrufe; 1 294 Aussagen neu gemerkt | erledigt |
| 3. den einen Commit prüfen | – | äußerer Commit `be80fb3`, 15 Dateien: 13 Events-Blöcke in 12 Dokus, `docs/spaces/021_sales.md`, README-Block `neue-spaces`, `docs/graph/graph.json` (0,35 → 2,9 MB, M5 beobachten); Torwächter: keine neuen Fehler | erledigt |
| 4. Rowboat prüfen | – | geschrieben: `Index.md`, `Prüfliste.md` (gruppiert, I5), `Zustand.md`, `Mitbauen/Neuen Space erstellen mit Claude Code.md`; die 40 erzählten Seiten warten auf die Nacht (Plan-Ruling P3) | erledigt |
| 5. `aufgabe --pruefen` | – | `Loop: neu` | erledigt |
| 6. Aufgabe laufen lassen | ab 21:27 | live; zwei Befunde, s. „Befunde im Betrieb“ | erledigt |

Der Ablauf, wie er vorher geplant war:

**Der erste Inhalt kommt am Tag, erzwungen und ohne LLM** (Final-Review C2, Teil a): Ohne
`--erzwingen` täte der erste ungezwungene Tick nichts, weil sich seit dem letzten Lauf des alten Loops kein Code
geändert hat. Der erste Inhalt käme dann erst nachts mit der Erstbefüllung. Ablauf nach dem OK, in dieser Reihenfolge:

1. **Am Tag** (außerhalb 00:00–06:00) und mit **≥ 3 GB freiem Commit** (`FreeVirtualMemory`): Der Controller setzt den
   WORKBOARD-Eintrag für `cc-doku-loop` („abgelöst durch selbstbild.doku“, nur die eigene Zeile gestaget) und zeigt
   `aufgabe --umschalten --trocken`. Dann `aufgabe --umschalten`. Es sichert `%LOCALAPPDATA%\VibeMind\doku_loop\`
   (ohne `messbaum-*`) nach `doku_loop-sicherung-<zeit>`, merkt die alte Aktion in `aufgabe-alt.json`, setzt die neue
   Aktion (`conhost.exe --headless cmd /c ""…\.venv-selbstbild\Scripts\python.exe" -m selbstbild doku tick
   "C:\Users\User\Desktop\Vibemind_V1" >> "…\logs\doku-loop.log" 2>&1"`) und liest sie zurück. Trigger, Limits und
   Principal bleiben. Ab jetzt verweigert `scripts\doku-loop.cmd tick` mit Exit 2 (Final-Fix I3).
2. **Sofort** den ersten Lauf von Hand, erzwungen:
   `python -m selbstbild doku tick C:\Users\User\Desktop\Vibemind_V1 --erzwingen`. Am Tag ist die Erstbefüllung nicht
   fällig, ohne Signal fragt der Lauf kein LLM. Erwartet: `llm_aufrufe 0`, genau ein Commit.
3. **Diesen einen Commit prüfen** (`git show --stat HEAD`):
   - 13 Events-Blöcke in 12 Dokus (Tabelle in Schritt 1), nur eingefügt, CRLF bleibt in `004_desktop.md` und
     `018_video.md`;
   - `docs/spaces/021_sales.md` neu, dazu eine Zeile im README-Block `neue-spaces`;
   - `docs/graph/graph.json`: Größe notieren (Trockenlauf: +1 294 Aussagen, +2 071 `belegt_durch`, +144 Events;
     M5: nur beobachten);
   - keine weitere Datei, `KETTEN.md` unverändert.
4. **Rowboat prüfen** (`C:\Users\User\.rowboat\knowledge\Doku\`): `Index.md` neu, `Prüfliste.md` (Struktur-Hinweise
   nach Art und Datei gruppiert, I5), `Mitbauen/Neuen Space erstellen mit Claude Code.md`, `Zustand.md`. Im Log
   `Seiten: … 0 Aufruf(e)` – keine erzählte Seite am Tag, kein LLM.
5. `aufgabe --pruefen` zeigt Aktion `neu`, die Sicherung, `aufgabe-alt.json` vorhanden, Sperre frei.
6. Erst dann die Aufgabe laufen lassen (alle 5 min; Anstoß bei Marker, bewegtem HEAD oder nach 30 min).

Weicht Schritt 3 oder 4 ab: `aufgabe --zurueck --trocken`, dann `aufgabe --zurueck` (Schritt 9).

**Was der User danach erwarten soll:**
- Nachts (00:00–06:00), solange die Erstbefüllung läuft, Commits `docs(loop): Erstbefüllung – N Aussagen nachgeführt,
  M in Prüfliste` in `docs/spaces/` und `docs/modules/`. Ein Commit entsteht nur bei einem Urteil `veraltet` mit
  gültigem Ersatz oder bei geändertem stabilen Digest. Jeder Lauf, den nur die Erstbefüllung auslöst, baut alle Spaces
  (C2b), höchstens alle 30 min und nur, solange das Tagesbudget von `selbstbild-erstbefuellung` reicht.
- Etwa 3 erzählte Rowboat-Seiten je Lauf (`rowboat_seiten_je_lauf: 3`), nachts nach den Aussagen; eine Seite wird nur
  gefragt, wenn bis zur Deadline noch ein ganzer Aufruf passt (M3).
- Eine lange erste `Prüfliste.md` (rund 500 Struktur-Hinweise), gruppiert nach Art und Datei mit Summen und höchstens
  drei Beispielen je Datei (I5).
- **Mindestens 8 Nächte**, bis die Erstbefüllung fertig ist (532 Aussagen × 2 + 40 Seiten ≈ 1 104 Aufrufe bei 150/Tag).
  Danach gilt wieder `doku-loop` mit 30/Tag.

## Schritt 3: Ruhe – erreicht (Teil B)

Ziel (Spec 9): 10 Läufe ohne Codeänderung außerhalb des Erstbefüllungs-Fensters, 0 LLM-Aufrufe, 0 Commits.

Ergebnis: 10 Läufe von 22:53 bis 23:38, nach den Hotfixes T4-c2 und T4-c3, ohne Änderung im Repo. **0 LLM-Aufrufe,
0 Commits, 0 Fehler. Erreicht.** Die Lauf-Nummern gehen dabei nur von #322 bis #323: Ein Lauf ohne Anstoß
(`kein_anstoss`) speichert keinen Stand und zählt keine Nummer.

## Befunde im Betrieb

Beide Befunde traten in den ersten Stunden nach der Umschaltung auf. Beide wurden live mit einem Hotfix behoben. Die
Windows-Aufgabe läuft direkt aus dem la_fungus-Arbeitsbaum, der Fix wirkt also ab dem nächsten Lauf.

**Befund 1 – Budget-Verbrauch durch `WORKBOARD.md` (21:33 und 22:03, Läufe #316/#317).**
- 18 Aussagen in Space- und Modul-Dokus belegen `WORKBOARD.md` als Ganzdatei-Anker ohne Ankernamen. Jede Session
  committet dort ihren Claim, die Datei ändert sich also mehrmals am Tag. Jeder WORKBOARD-Commit ergab 18
  Judge-Signale.
- Lauf #317 verbrauchte 16 Aufrufe unter `doku-loop` (30/Tag). Die Urteile waren nur `unklar`, `uneinig` oder
  `noch_wahr`, umgeschrieben wurde nichts. Für einen Anker ohne Namen kann der Judge nie gültig `veraltet` sagen
  (`erlaubte_anker` filtert leere Namen). Commit `e0aec1d` (nur `graph.json`).
- **Hotfix Ruling T4-c2, `fdd99ac`:** In Doku-Zielen und nur-lesend-Dateien (nicht KETTEN.md) trägt ein
  `datei`-Anker ohne Ankernamen kein Judge-Signal mehr, gleich welche Endung. Er wird still nachgeführt und je Datei
  als `beleg_datei_geaendert` gemeldet. Offene Signale dieser Art erledigt der nächste Lauf ohne Aufruf (#319: 20
  erledigt, 0 LLM).
- Wirkung auf den Bestand: 275 Aussagen haben jetzt nur noch gesammelte Prüfpunkte, 63 verlieren einen Teil ihrer
  signaltragenden Anker. Die Erstbefüllung ändert sich nicht, weil T7-c solche Aussagen schon übersprang.
- Weil der Extraktor jeden `datei`-Anker ohne Namen anlegt, tragen in Doku-Zielen nur noch `symbol`-, `yaml_key`- und
  `sql`-Anker ein Signal.

**Befund 2 – Commit-Lärm durch die Nachführung (22:33, Lauf #319).**
- Die Nachführung der Digests nicht signaltragender Anker änderte `docs/graph/graph.json`, und der Loop committete die
  Datei (`1444a7d`, 2,9 MB). Jede Änderung an `WORKBOARD.md` oder `vibemind.config.json` hätte einen solchen Commit
  `docs(loop): Belege – …` ohne Doku-Änderung erzeugt. Das verletzt das Anti-Ziel „Lauter Hintergrund“.
- **Hotfix Ruling T4-c3, `f0e5060`:** Die Commit-Entscheidung (`digest_stabil` des schlanken Graphen) ignoriert den
  Digest jedes nicht signaltragenden Belegs in Doku-Zielen und nur-lesend-Dateien. Der Trockenlauf-`graph_digest` nutzt
  denselben Maßstab. Der nachgeführte Digest steht im lokalen Vollgraph und fährt mit dem nächsten fälligen Commit mit.
  `beleg_datei_geaendert` bleibt, der Commit-Zeiger wird fortgeschrieben. KETTEN.md und signaltragende Anker zählen
  wie bisher.
- Folge: Bis zum nächsten fälligen Commit führt jeder Lauf, der die Dokus liest, diese Anker erneut nach. Das kostet
  kein LLM und keinen Commit, der Prüfpunkt ist dedupliziert. Der nächste echte Commit trägt dann alle nachgeführten
  Digests mit, sein `graph.json`-Diff ist also größer, als sein Betreff vermuten lässt.

## Schritt 4: Belege

Gemessen mit `messen.belege_messen` über den vorhandenen Messbaum (`4048bc83`, vibemind-os am Checkout-HEAD
`ba4bd8e5`), 51 s, nur lesend.

**KETTEN.md:**

| Klasse | gerendert mit Anker (C.9.1) | gerendert ohne Anker | roh |
|---|---|---|---|
| exakt | **119** | 49 | 17 |
| verschoben | 7 | 23 | 4 |
| token_fehlt | 0 | 6 | 3 |
| datei_fehlt | 1 | 0 | 0 |
| nicht_aufloesbar | 0 | 2 | 0 |

C.9.1: **119 von 127 = 93,7 %**, Ziel ≥ 95 %. Die 8 Abweichungen:

| Zeile | Kette | Stelle | Klasse | Ursache |
|---|---|---|---|---|
| 57 | K1 | `claw/werkzeuge.py:313` `versand_beauftragen` | verschoben → 297 | Datei mit fremden, nicht committeten Änderungen; 313 ist die Zeile im Arbeitsbaum |
| 58 | K1 | `api/server.py:521` `versandauftrag_anlegen_route` | verschoben → 516 | dito |
| 59 | K1 | `db/093_versandauftraege_youtube.sql:25` | datei_fehlt | Datei ist im Submodul **nicht versioniert** (`??`) |
| 85 | K2 | `api/pult.py:86` `_schluessel` | verschoben → 71 | fremde Änderungen |
| 85 | K2 | `api/freigabe.py:192` `freigeben` | verschoben → 143 | fremde Änderungen |
| 86 | K2 | `api/freigabe.py:203` `export_nachholen` | verschoben → 154 | fremde Änderungen |
| 196 | K7 | `claw/werkzeuge.py:183` `wissensquellen` | verschoben → 169 | fremde Änderungen (183 = Arbeitsbaum) |
| 196 | K7 | `claw/werkzeuge.py:774` `wissen_fragen` | verschoben → 557 | fremde Änderungen (Arbeitsbaum heute 808) |

Alle Pfade unter `vibemind-os/spaces/marketing/`. Dort hat eine andere Session 184 nicht committete Änderungen im
Submodul, die vier Python-Dateien sind `M`, die SQL-Datei ist `??`. Gemessen wird nur committeter Code (Leitplanke),
die angezeigten Zeilen stammen aber aus dem Arbeitsbaum. Der Graph trägt dieselben Zeilen (`zeile` 313, 192, 774 …,
`commit ba4bd8e5`), und der Loop vergleicht Belege über den Digest: Ist der Rumpf im Commit gleich, entsteht kein
Signal, und die Zeile wird nicht nachgezogen. Das ist Verhalten des alten Loops, 1:1 verpflanzt (D1). Ohne die 8
Stellen in fremd geänderten Dateien: 119/119. **Offen für das Final-Review:** ob der Loop eine nur verschobene Zeile
bei gleichem Digest nachführen soll.

**Neue Ziele (28 Dokus, 9 nur lesend):** Sie haben heute **keine gerenderten Belege mit Zeile**. Die Events-Blöcke
zitieren bewusst Datei plus Anker ohne Zeilennummer (Task 5, Fix I1), deshalb zählt C.9.1 sie nicht. Die Prosa-Belege
(roh, Token-Methode) zum Vergleich:

| Klasse | Zahl | Anteil |
|---|---|---|
| exakt | 1 834 | 64 % |
| verschoben | 425 | 15 % |
| datei_fehlt | 360 | 13 % |
| token_fehlt | 199 | 7 % |
| nicht_aufloesbar | 45 | 2 % |

Am meisten `datei_fehlt`: `012_plugin_setup.md` 66, `004_desktop.md` 56, `011_musik.md` 33, `013_research.md` 33
(Kurzpfade relativ zum Space, die der Auflöser nicht findet, und Dateien in nicht messbaren Submodulen). Die
nur-lesend-READMEs nennen keine `pfad:zeile`-Stellen (0 Belege). Die 2 163 Anker der neuen Aussagen (`symbol` 657,
`yaml_key` 87, `sql` 6, `datei` 1 413) entstehen im ersten Lauf aus dem Messbaum und sind damit per Konstruktion
aktuell; ihre Drift misst erst der Betrieb nach der Umschaltung.

## Schritt 5: Events

Echter, lesender Abgleich im Trockenlauf (Quellen aus dem Messbaum, T5-a): **144 Events in 14 Spaces.**

| Space | Events | Quellen |
|---|---|---|
| desktop | 28 | Registry 28 |
| ideas | 25 | Registry 25 |
| bubbles | 14 | Registry 14 |
| marketing | 13 | `EVENT_TO_TOOL` 13 |
| video | 11 | Registry **9**, `EVENT_TO_TOOL` **11** |
| coding, n8n, rowboat | je 8 | Registry |
| mirofish, schedule | je 7 | Registry (schedule zusätzlich `EVENT_TO_TOOL` 6) |
| research | 6 | Registry |
| minibook | 4 | Registry |
| flowzen | 3 | Registry |
| agentfarm | 2 | Registry |

Blöcke: 13 in 12 Dokus (006_ideas trägt `bubbles` und `ideas`). Prüfliste aus dem Abgleich:

| Art | Zahl | Verteilung |
|---|---|---|
| `ereignis_quellen_abweichend` | 17 | video 10, schedule 7 |
| `ereignis_nur_doku` | 15 | 001_agentfarm 6, 009_minibook 4, 018_video 5 |
| `space_ohne_doku` | 1 | n8n |

**Video-Fall:** 8 Events mit anderem Werkzeug je Quelle, `video.lipsync_analyze` und `video.publish` nur in
`EVENT_TO_TOOL`; `video.status` stimmt überein. Auszug aus der Prüfliste, wie er erscheinen würde (nur Namen):

```
ereignis_quellen_abweichend  video: `video.team_status` hat je Quelle ein anderes Werkzeug: `db_query` (space_agent_registry.yml), `team_pipeline_status` (EVENT_TO_TOOL)
ereignis_quellen_abweichend  video: `video.voice_tts` hat je Quelle ein anderes Werkzeug: `db_video_project_create` (space_agent_registry.yml), `voice_tts` (EVENT_TO_TOOL)
ereignis_quellen_abweichend  video: `video.publish` steht nur in EVENT_TO_TOOL, nicht in space_agent_registry.yml
ereignis_nur_doku            docs/spaces/018_video.md: `video.reel` steht in der Doku, aber in keiner Ereignis-Quelle (Präfix `video.` ist bekannt)
ereignis_nur_doku            docs/spaces/018_video.md: `video.import` steht in der Doku, aber in keiner Ereignis-Quelle (Präfix `video.` ist bekannt)
ereignis_nur_doku            docs/spaces/018_video.md: `video.job_status` …, `video.export_get` …, `video.generate_clips` …
space_ohne_doku              n8n: Events aus space_agent_registry.yml, aber keine Space-Doku (docs/spaces/<NNN>_n8n.md oder doku.space_dokumente)
```

Der Block in `018_video.md` hat 11 Zeilen, 8 mit `⚠` (z. B. `video.team_status`: `db_query` (Registry) /
`team_pipeline_status` (`EVENT_TO_TOOL`)), Beleg je Zeile `vibemind-os/config/space_agent_registry.yml`
`spaces.video.events["video.team_status"]` und `vibemind-os/spaces/video/agents/video_agent.py`
`VideoBackendAgent.EVENT_TO_TOOL`.

## Schritt 6: Neue Spaces

- 19 Space-Ordner sichtbar, ohne Doku nur `sales` (`vibemind-os/spaces/sales`, uninitialisierter Gitlink). Geplant:
  `docs/spaces/021_sales.md` in Kurzform (Stand-Block mit den Gitlink-Pins `vibemind-os@4672d02`,
  `vibemind-os/spaces/sales@94a9ec8`, Steckbrief „Unter-Repo, im Messbaum nicht vermessen“, Beleg B1 auf den Ordner)
  und die Zeile `| 021 | [sales](021_sales.md) | … |` im Block `neue-spaces` von `docs/spaces/README.md`.
- Abweichung vom Brief („auf dem echten Repo wird nichts erzeugt“) nach **Ruling T6-a**: Ordner, die Gitlink sind
  oder enthalten, zählen als Spaces (Spec §3 Schritt 4, §6.7). Abschaltbar über `doku.space_dokumente`. Der User sieht
  die Datei im Diff des ersten echten Ticks.
- Fixture-Test `tests/selbstbild/doku/test_doku_neue_spaces.py`: **19 passed** (92 s).
- Der Durchstich mit einem echten neuen Test-Space ist Stufe 7.

## Schritt 7: Rowboat – Plan; Schreiben offen (Teil B)

**Seitenplan** (Trockenlauf, ohne LLM):

| Art | Zahl | Zustand |
|---|---|---|
| erzählt `Was ist VibeMind.md` | 1 | neu |
| erzählt `Spaces/<name>.md` | 21 | neu |
| erzählt `Abläufe/K1.md` … `K18.md` | 18 | neu |
| deterministisch `Mitbauen/Neuen Space erstellen mit Claude Code.md` | 1 | würde geschrieben |
| deterministisch `Zustand.md` | 1 | würde geschrieben |
| `Index.md` | 1 | ändert sich |

Geschätzt 40 Abo-Aufrufe, höchstens 3 je Lauf (`rowboat_seiten_je_lauf`), nachts unter `selbstbild-erstbefuellung`
nach den Aussagen.

**Vorhandene Notizen in `C:\Users\User\.rowboat\knowledge\Doku\`** (nur Manifest `notizen-2264cd6a4e2a.json` und
Hashes gelesen, nichts geschrieben): 124 Dateien, alle im Manifest, **alle unbearbeitet** (Hash gleich), 0 von Hand.

| Ordner | Dateien | was passiert nach der Umschaltung |
|---|---|---|
| `Dienste/` | 101 | bleibt beim deterministischen Schreiber |
| `Ketten/` | 18 | bleibt (Daten); `Abläufe/K<n>.md` kommen daneben (T9-a) |
| `Spaces/` | 3 (`brain`, `marketing`, `sales-claw`) | werden von der erzählten Seite überschrieben, sobald sie erzählt ist (unbearbeitet, T9-a); bis dahin bleiben sie stehen |
| `Index.md` | 1 | wird neu geschrieben |
| `Prüfliste.md` | 1 | bleibt beim deterministischen Schreiber (geschwärzt bei Schlüssel-Treffer, T9-c) |

Offen (Teil B): `Mitbauen/…` und `Zustand.md` liegen im Ordner; Stand der erzählten Seiten nach der ersten Nacht.

## Schritt 8: Secrets

`selbstbild.doku.schluessel` (zentraler Filter, 7 Muster) lief nur lesend über alle Ausgaben. Gezählt wurden Treffer
je Muster, nie Werte. JSON zusätzlich entschlüsselt (`\u`-Escapes).

| Quelle | Dateien | Treffer |
|---|---|---|
| Doku-Ziele + KETTEN.md (Arbeitsbaum) | 29 | 0 |
| `docs/spaces/README.md`, Space-READMEs, `CONTEXT.md` | 10 | 0 |
| geplante Doku-Diffs des Trockenlaufs | 14 | 0 |
| `docs/graph/graph.json` (HEAD) | 1 | 0 |
| schlanker Graph des Trockenlaufs | 1 | 0 |
| `knowledge/Doku/**` | 124 | 0 |
| Prüfliste des alten Loops | 1 | 0 |
| Prüfpunkte, Seitenplan und Log des Trockenlaufs | 3 | 0 |
| **gesamt** | **183** | **0** (alle 7 Muster: `sk-…`, `ghp_…`, `github_pat_`, `AKIA…`, `xox[bp]-`, JWT, PEM) |

Nachtrag Final-Fix I1/I2: Seit der Fix-Welle gibt es 9 Muster (PEM bis zur END-Zeile, `sk-` mit `-`/`_`, neu
Telegram-Bot-Token). Ein neuer lesender Lauf über 40 Doku-Dateien von VibeMind (`docs/spaces`, `docs/modules`,
`graph.json`, Space-READMEs) ergab 0 Treffer.

**Hash-Probe mit echten Werten (startet der User selbst).** Sie nutzt `scripts/secret_probe.py` aus Stufe 1 (Werte
nur im Speicher, roh und JSON-escaped, nur Namen und Zähler, Exit 0 / 1 Treffer / 2 nicht prüfbar), hier über die
Ausgaben der Stufe 4 statt über `graph.sqlite`:

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH = "scripts"
@'
import os, sys
from pathlib import Path
import secret_probe as s
v = Path(r"C:\Users\User\Desktop\Vibemind_V1")
env = Path(os.environ.get("SONDE_ENV", v / ".env"))
d = [*v.glob("docs/spaces/*.md"), *v.glob("docs/modules/0*.md"), v / "docs/graph/graph.json",
     *[p for p in (Path.home() / ".rowboat/knowledge/Doku").rglob("*") if p.is_file()],
     *Path(os.environ["LOCALAPPDATA"], "VibeMind", "doku_loop").glob("pruefliste-*.json")]
try:
    s._lesbar(env, "--env")
    werte = s.env_werte(env.read_text(encoding="utf-8", errors="replace"))
    t = s.pruefen(werte, d, None)
except s.ProbeFehler as f:
    print(f"FEHLER: {f}. Pruefung nicht moeglich, Exit 2"); sys.exit(2)
for n, a in t.items():
    print(f"TREFFER {n}: {a}")
print(f"{len(d)} Dateien, {sum(len(w) >= s.MIN_LAENGE for w in werte.values())} Werte geprueft, Treffer: {len(t)} Variablen")
sys.exit(1 if t else 0)
'@ | .\.venv-selbstbild\Scripts\python -
"Exit $LASTEXITCODE"
```

Selbsttest nur mit einer Fake-`.env` (`$env:SONDE_ENV`): ein Wert aus KETTEN.md → `TREFFER FAKE_A: 14`, Exit 1;
fehlende Datei → Exit 2. Mit echten Werten lief der Befehl nicht. Nach Teil B noch einmal ausführen, dann mit den
geschriebenen Seiten.

## Schritt 9: Rückweg und Frist

Rückweg (nur nach der Umschaltung sinnvoll; vorher endet `--zurueck` mit Exit 2, weil `aufgabe-alt.json` fehlt):

```powershell
cd C:\Users\User\ClaudeWork\wt-fungus-gate1
$env:PYTHONPATH="src"
.\.venv-selbstbild\Scripts\python -m selbstbild doku aufgabe C:\Users\User\Desktop\Vibemind_V1 --zurueck --trocken   # jeden Schritt zeigen
.\.venv-selbstbild\Scripts\python -m selbstbild doku aufgabe C:\Users\User\Desktop\Vibemind_V1 --zurueck
```

Was `--zurueck` tut (hält `tick.lock` die ganze Zeit; frische Sperre → Exit 2, nichts geändert), Reihenfolge nach
**Ruling T10-e**:
1. **Rowboat-Archiv:** Unbearbeitete eigene Seiten (`Mitbauen/…`, `Zustand.md`, `Was ist VibeMind.md`, `Spaces/*`,
   `Abläufe/*` mit `erzeugt_von: selbstbild`) wandern nach `<stand>\erzaehlt-archiv\`, ihre Manifest-Einträge fallen
   weg. Bearbeitete Seiten bleiben unberührt (T10-a, T10-b).
2. **Stand:** Die Sicherung wird zurückgespielt. Abweichende Dateien werden ersetzt, neue nach
   `doku_loop-nachher-<zeit>` verschoben, nie gelöscht. Unbearbeitete Notizen des neuen Loops (Index, Ketten, Dienste)
   behalten ihren aktuellen Hash im Manifest (**T10-c**), damit der alte Loop sie nicht als von Hand meldet.
3. **Graph:** Knoten- und Kantenarten, die der alte Loop nicht kennt (`ereignis`, `ereignis_von`), werden aus
   `docs/graph/graph.json` entfernt und in einem eigenen Commit committet (**T10-d**, Betreff
   `docs(loop): Rueckweg – Knotenarten des neuen Loops entfernt`). Fremde Änderung an `graph.json` → Exit 2 vorher.
   Seit Final-Fix M1 entfernt dieser Schritt auch jede Aussage, deren Datei nicht im Doku-Satz des alten Loops liegt
   (`ketten_md` + `weitere_doku` aus `config/doku_loop.yaml`, sonst nur KETTEN.md), samt Kanten und dadurch
   verwaisten Modulen. `--pruefen` und `--trocken` nennen die Zahlen.
4. **Aufgabe zuletzt:** alte Aktion setzen und zurücklesen; `aufgabe-alt.json` wandert nach `nachher`. Erst dann
   läuft `scripts\doku-loop.cmd tick` wieder. Scheitert genau dieses Verschieben, endet `--zurueck` mit Exit 1 und
   nennt die Datei, die von Hand verschoben werden muss.

Jeder Schritt ist wiederholbar. Ein erneutes `--umschalten` legt die archivierten Seiten zurück (0 Aufrufe). Kosten
des Rückwegs: Der Fortschritt der Erstbefüllung geht verloren (der Urteils-Cache begrenzt die Kosten). Zwischen einem
gescheiterten Rückweg und seiner Wiederholung kann der neue Loop noch ticken (Task-10-Bericht, Concern 2).

**Frist:** Der alte Code `scripts/doku_loop/`, `scripts/doku-loop.cmd` und `config/doku_loop.yaml` bleiben **14 Tage**
nach der Umschaltung liegen. Ihre Entfernung ist ein eigener Schritt mit User-OK.

## Rulings, die Nutzer betreffen

| Ruling | Wirkung |
|---|---|
| T4-b | Fremd geänderte Doku-Ziele blockieren den Tick nicht; die Datei wird übersprungen (`doku_ziel_fremd`), KETTEN.md und `docs/graph/*` warten weiter. Heute 0 Ziele fremd geändert. |
| T4-c | Reine `datei`-Anker auf Code-Dateien lösen bei den neuen Zielen kein Signal aus, sondern einen gesammelten Prüfpunkt `beleg_datei_geaendert` je Datei. Grob belegte Aussagen werden nicht automatisch nachgeführt. |
| T5-a, T5-b | Event-Quellen kommen aus dem Messbaum (committeter Stand); Space-Namen werden normalisiert (`sales-claw` = `sales_claw`). |
| T6-a | Gitlink-Spaces bekommen Doku: `021_sales.md` entsteht im ersten echten Tick. |
| T7-c | Die Erstbefüllung überspringt Aussagen ohne Ankernamen; sie werden nie einmalig bestätigt, bleiben aber signalfähig. |
| T9-a | Erzählte Space-Seiten besitzen `Spaces/<name>.md`; die drei alten Space-Notizen werden beim ersten Erzählen überschrieben (unbearbeitet). |
| T9-c | Prüfliste und Index werden bei Schlüssel-Verdacht geschwärzt statt übersprungen; andere Seiten werden übersprungen. |
| T10-c | Nach dem Rückweg behalten unbearbeitete Notizen des neuen Loops ihren Hash; das Manifest ist dann nicht byte-gleich mit der Sicherung. |
| T10-d | Der Rückweg macht einen eigenen Commit an `docs/graph/graph.json`. |
| T10-e | Rückweg-Reihenfolge: Archiv → Stand → Graph → Aufgabe zuletzt. |
| T11-a | Der Trockenvergleich ignoriert `gemessen`; eine echte `gemessen`-Regression fiele hier nicht auf. |

## Tests

Volle Suite (`-W error`, `--basetemp C:/Users/User/AppData/Local/Temp/selbstbild-pt`, ohne `SELBSTBILD_VIBEMIND`):
**1 577 passed, 34 skipped, 0 failed** in 800 s (19:41:43–19:55:04, freier Commit vorher und nachher 5,90 GB). Das ist
die Basis nach Task 10 (1 577/34). Dazu `test_doku_neue_spaces.py` einzeln: 19 passed.

**Nach der Final-Fix-Welle** (la_fungus `8f5f240`, gleicher Aufruf, ohne `SELBSTBILD_VIBEMIND`): **1 602 passed,
36 skipped, 0 failed** in 529 s (20:48:52–20:57:41, freier Commit vorher 5,1 GB). Die zwei zusätzlichen Skips sind
die `.cmd`-Tests (I3); sie brauchen das echte äußere Repo und liefen mit `SELBSTBILD_VIBEMIND` grün.

**Trockenlauf nach der Fix-Welle** (21:00, 100 s, freier Commit 6,6 GB): Mein äußerer Commit `8ca7c497` (nur
`scripts/doku-loop.cmd`) liegt hinter dem Messbaum `4048bc83`. Diese Datei gehört nicht dem Loop, deshalb meldet der
einfache Trockenlauf `messbaum_veraltet`, bis der alte Loop wieder baut; nach der Umschaltung stellt der erste echte
Lauf seinen Messbaum selbst bereit. Gelaufen ist deshalb derselbe Trockenlauf mit einer Ausnahme nur im Speicher: Die
T3-c-Prüfung lässt zusätzlich genau diese eine Datei zu. Sie liegt unter keiner Space-Wurzel und ist kein Beleg.
Ergebnis wie in Teil A: `graph_digest e2740853…84d4`, 0 Signale, 0 LLM-Aufrufe, kein Commit, 532 Aussagen offen,
40 Seiten fällig, 144 Events, `021_sales.md` geplant. Notizen: 4 statt 3 würden geschrieben, denn `Prüfliste.md`
erscheint jetzt gruppiert (I5). Lokaler Ordner und `knowledge\Doku` blieben unverändert.

## Final-Fix-Welle (Ruling FINAL-a)

Nach dem Final-Review („With fixes“) kam eine Fix-Welle, jede mit Test (zuerst rot). Die Tests stehen in
`tests/selbstbild/doku/test_doku_stufe4_finalfix.py`.

| Punkt | Änderung |
|---|---|
| C1 | `judge.antwort_parsen` verwirft einen `neuer_text` mit Schlüssel-Verdacht (Ergebnis `ungueltig`, gebucht als `fehler`). `tick._ersetzen` ist die zweite Linie, auch für Antworten aus dem Urteils-Cache. Sein Prüfpunkt `ersetzen_abgelehnt` nennt nur das Muster. Gilt für Signal und Erstbefüllung. |
| C2a | Ablauf Teil B mit erzwungenem ersten Lauf am Tag (Schritt 2). |
| C2b | Ein Lauf, den nur die Erstbefüllung auslöst, baut alle Spaces. Sonst würden neue Aussagen ohne Space-Manifeste verankert, also ohne Symbol-Anker, und nie erstgeprüft. Betreff `docs(loop): Erstbefüllung – …`. |
| I1 | PEM-Muster bis zur END-Zeile oder zum Textende; erkannt und geschwärzt wird der ganze Schlüssel. |
| I2 | `sk-` auch mit `-`/`_` im Wert (`sk-ant-api03-…`, `sk-proj-…`), links mit Wortgrenze gegen Treffer in Fließtext; neues Muster Telegram-Bot-Token. 9 Muster, 0 Treffer in 40 Doku-Dateien von VibeMind. |
| I3 | `scripts\doku-loop.cmd` (äußeres Repo) verweigert `tick` mit Exit 2, solange `aufgabe-alt.json` liegt. `--trotzdem` lässt den Lauf zu, andere Unterbefehle bleiben erlaubt. `--zurueck` endet mit Exit 1, wenn es die Datei nicht verschieben kann. |
| I4 | `doku_ziel_fremd` erscheint einmal je Datei, nicht einmal je fremdem Inhalt. |
| I5 | `Prüfliste.md` gruppiert die Struktur-Hinweise nach Art und Datei; 500 Hinweise ergeben rund 50 Zeilen. Beispieltexte werden vor dem Kürzen geschwärzt. Das Zustandsformat bleibt. |
| M1 | Der Rückweg entfernt Aussagen außerhalb des alten Doku-Satzes (Schritt 9). |
| M3 | Das Notizen-Manifest wird nach jeder geschriebenen Notiz gespeichert, außer bei der Übernahme ohne Manifest. Eine erzählte Seite wird nur gefragt, wenn noch `judge.TIMEOUT_S` + 30 s bis zur Deadline bleiben. |
| M4 | Die Budget-Vorprüfung der Erstbefüllung läuft erst nach dem Anstoß. Ruhige 5-Minuten-Aufrufe laden das Budget-Modul nicht. |

Geparkt: M2 (Zeilen in unsauberen Dateien nachführen), M5 (Größe von `graph.json` beobachten).

## Bekannte Grenzen und offene Punkte

- **Universell nur auf Konfig-Ebene (M6, offen).** Der Loop ist über `selbstbild.yaml` (`doku:`) für andere Repos
  konfigurierbar. Ein Repo ohne `KETTEN.md` und ohne `docs/graph/graph.json` endet heute aber mit `kein_seed`: Es
  gibt keinen Weg zum ersten Graph ohne den Ketten-Seed. Das ist ein Punkt für eine spätere Stufe.

- **Offen nach Teil B:**
  - erste Nacht der Erstbefüllung (ab 00:00): Commits `docs(loop): Erstbefüllung – …` und die ersten erzählten Seiten
    prüfen;
  - Hash-Probe mit echten Werten (startet der User; Schritt 8), dann mit den geschriebenen Seiten;
  - Push von la_fungus (User-OK);
  - Frist 14 Tage für den alten Code (`scripts/doku_loop/`, `scripts/doku-loop.cmd`, `config/doku_loop.yaml`), danach
    Entfernung mit User-OK;
  - C.9.1 neu messen, sobald die fremden Marketing-Dateien committet sind;
  - Größe von `graph.json` (M5: 0,35 → 2,9 MB) als Folgepunkt;
  - die 30-Tage-Abnahme der Doku beginnt erst nach dem Ende der Erstbefüllung (Spec 12a).
- **C.9.1 bei 93,7 %.** Alle Abweichungen in fremd geänderten Marketing-Dateien. Ob der Loop eine bei gleichem Digest
  nur verschobene Zeile nachziehen soll, ist eine Frage für das Final-Review. Die neuen Ziele haben noch keinen
  Belegbestand für C.9.1.
- **Prüfliste wächst stark.** Der Trockenlauf hat 500 `struktur`-Punkte statt heute 15: `beleg_unsicher` 369,
  `anker_zu_grob` 78, `ereignis_*`/`space_ohne_doku` 33, `env_unbekannt` 12, `aussage_id_doppelt` 8. Dazu 17 offene
  (`beleg_unsauber` 15, `beleg_untracked` 2 – fremde Arbeit im Submodul). Das wird in `Prüfliste.md` und `Zustand.md`
  sichtbar. Die Struktur-Hinweise zeigt `Prüfliste.md` seit I5 gruppiert.
- **Erstbefüllung länger als geschätzt:** 532 offene Aussagen statt 451, mit den Seiten etwa 1 104 Aufrufe, mindestens
  8 Nächte.
- **Trockenlauf dauert jetzt 155–165 s** (Task 3: 52 s), wegen 29 Doku-Dateien, 9 nur-lesend-Dateien und der Events.
  Der echte Tick liest die Doku nur bei Anstoß. Im Erstbefüllungs-Fenster liest jeder angestoßene Lauf alle Dokus und
  baut seit C2b alle Spaces. Ist das Tagesbudget schon verbraucht (lesende Vorprüfung, nach dem Anstoß, M4), entfällt
  beides.
- `021_sales.md` nennt als Datum ein Commit-Datum des Stands (2026-08-17), nicht das Laufdatum.
