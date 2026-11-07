# Selbstbild Stufe 6a – täglicher Lauf, Rückfragen, Telegram: Abnahme

- **Plan:** `Vibemind_V1/docs/superpowers/plans/2026-11-06-selbstbild-stufe6a-automatik.md`
- **Spec:** `Vibemind_V1/docs/superpowers/specs/2026-11-06-selbstbild-stufe6-automatik-design.md`, User-OK 06.11.2026
- **Code:** cb0d24b..cb8fde0, 11 Commits
- **Befehle:**
  - `python -m selbstbild auto <wurzel> [--trocken]`
  - `python -m selbstbild auto aufgabe <wurzel> [--pruefen | --zurueck] [--python PFAD]`
- **Windows-Aufgabe:** `VibeMind Selbstbild-Funde`, LIVE seit 07.11.2026 mit User-OK.
  - Läuft täglich um 07:30.
  - Startet nach, wenn der PC zur Startzeit aus war.
  - Startet keine zweite Kopie neben einer laufenden (IgnoreNew).
  - Wird nach 2 Stunden beendet (PT2H).
  - Läuft ohne Fenster über `conhost --headless` und schreibt nach `auto-aufgabe.log`.

## Ablauf

1. Sperre `auto.lock`.
2. Zweig prüfen.
3. Änderungs-Tor: HEAD, Hash des Goldsatzes, Antwort-Labels der offenen Rückfragen.
4. Speicher prüfen: mindestens 3 GB freier Commit-Speicher.
5. Kindprozesse `vermessen` und `analysieren`.
6. Rückfragen abgleichen.
7. `funde --senden`.
8. Höchstens eine Telegram-Nachricht je Kalendertag.

Die Gesamtfrist je Lauf ist 100 min. Jeder Kindprozess bekommt die Restzeit.

**Abweichung durch die Abnahme (cb8fde0):** Laut Plan kam der Speicher vor dem Tor. Der zweite Lauf am 07.11. ist bei 2,85 GB frei mit `übersprungen: Speicher` ausgestiegen, obwohl er nichts zu tun hatte. Die Zeile wäre am Folgetag als Meldung verschickt worden. Das Tor liest nur Git und GitHub, deshalb steht es jetzt vor der Speicherprüfung.

## Läufe am 07.11.2026 (Desktop-Checkout, HEAD 84966013, Zweig `codex/vibemind-ops-baseline`)

| Lauf | Dauer | Ergebnis | GitHub | Telegram |
|---|---|---|---|---|
| Trockenlauf 06.11. | 7:20 | Exit 0, kein Stand geschrieben | nur gelesen: 30 unverändert, 4 über Tagesgrenze | keine |
| 1. echter Lauf | 7:22 | Exit 0, `ok` | #314–#317 neu, 1 Sammel-Issue aktualisiert, 29 unverändert, 0 geschlossen | 1 Nachricht |
| 2. Lauf direkt danach (alte Reihenfolge) | 0,5 s | `übersprungen:speicher` (frei 2,85 GB) | nichts | keine, Zeile wartete auf den Folgetag |
| 2. Lauf nach cb8fde0 | 1,8 s | `unveraendert`, kein Kindprozess | nichts | keine |
| Lauf bei belegter Sperre | <1 s | Exit 3, Stand byte-gleich | nichts | keine |

Die Telegram-Nachricht ist nur laut `letzte_nachricht_tag` gesendet. Ob sie beim User angekommen ist, habe ich nicht geprüft.

Gemessene Dauern des echten Laufs:

| Schritt | Dauer |
|---|---|
| vermessen | ~2 min |
| analysieren | ~3,5 min |
| funde | ~1,5 min |

Der Speicher erreichte in der Spitze 1,18 GB. Bis zur Frist von 100 min bleibt viel Luft.

## Abnahme gegen Spec 6

| Kriterium | Stand |
|---|---|
| Zwei Läufe ohne Code-Änderung ergeben 0 Nachrichten | ✅ Nach cb8fde0 `unveraendert` ohne Nachricht und ohne Kindprozess. Die Tests decken auch den Folgetag ab. |
| Kette unklar → Issue → `antwort:bleibt` → Goldsatz `ja` + `bleibt: true`, Issue zu | ⏳ **Offen.** Der Goldsatz hat noch kein `unklar`. Der Knopf ist auf der Seite `KCbvPztEFunb8sr6UajstK` (Version 3, Sammlung `urteile2`). Nach dem Übernehmen legt der nächste Lauf die Rückfrage an. Die Tests decken die Kette ab (`test_funde_rueckfrage.py`, `test_auto.py`). |
| Ein zweiter, gleichzeitiger Lauf endet ohne Wirkung | ✅ Live mit einer Sperre einer lebenden PID geprüft: Exit 3, Stand unverändert. `funde --senden` hat dazu `senden.lock`. |
| Unter 3 GB frei: kein Lauf und genau eine Nachricht | ✅ Live: kein Kindprozess bei 2,85 GB. Die Nachricht wartete, weil heute schon eine verschickt war. Derselbe Grund wird nur einmal gemeldet (Tests). |
| 0 Schlüssel-Treffer in Issue- und Telegram-Texten | ✅ `schluessel_treffer: []` im echten Lauf. Der Telegram-Text läuft durch dieselbe Prüfung, sonst wird `ERSATZ_TEXT` gesendet. |

**Tests:** Die volle Suite ist bei cfc90cd grün (1986 passed, 36 skipped). Bei cb8fde0 ist die volle Suite grün (1987 passed, 36 skipped, 9:44).

## Bekannte offene Punkte

- **Zweig-Regel (R16):** Steht der Desktop-Checkout auf einem anderen Zweig als `doku.zweig`, pausiert der Lauf. Er meldet das einmal und dann jeden Montag, solange es über 7 Tage anhält. Spec 4.4.1 wollte, dass ein Zweigwechsel den Lauf nicht beeinflusst. Dafür müsste `funde` umgebaut werden. Der User hat am 07.11. entschieden, es so zu lassen.
- **Gelöschter Log-Ordner:** Löscht jemand den Ordner, scheitert die cmd-Umleitung still (conhost Exit 0). `auto aufgabe` legt den Ordner an, und er wird nie automatisch gelöscht.
- **Rückfragen außerhalb der Frist:** Der Rückfragen-Schritt läuft im eigenen Prozess außerhalb von `FRIST_S`. Er dauert GitHub-Aufrufe zu je höchstens 60 s plus den Commit-Hook. Theoretisch kann der Lauf so über PT2H kommen. Gemessen: ohne `unklar` dauert der Schritt Sekunden.
- **Sonderzeichen im Pfad:** `--zurueck` scheitert bei einer Wurzel mit cmd-Sonderzeichen. Die echte Wurzel hat keine.
- **Rückweg:** `python -m selbstbild auto aufgabe <wurzel> --zurueck` entfernt die Aufgabe. Der Stand in `%LOCALAPPDATA%\selbstbild\<repo>\auto-stand.json` bleibt erhalten.
