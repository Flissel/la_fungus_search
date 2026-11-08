# Debugger-Claw Plan 1: Rulings bei der Umsetzung

Diese Entscheidungen hat der Controller getroffen, wo Spec und Plan nichts festlegten oder sich widersprachen
(Sitzung 07.–08.11.2026). Jede Zeile nennt, was es kostet, wenn die Entscheidung falsch war. **Fett** sind die,
die du zuerst prüfen solltest.

| R | Entscheidung | Kosten, falls falsch |
|---|---|---|
| 1 | Zielrepo aus Repo-Marke, sonst Sammel-Marke, sonst Titel-Endung. Das erste Segment fällt weg, der Rest muss genau ein Repo-Pfad sein, sonst nur einordnen | Fund nur eingeordnet statt gefixt (an #306 geprüft: richtig, da Submodul) |
| 2 | `pr_anlegen` hängt die Debugger-Marke an den PR-Text | eine unsichtbare Marke zu viel |
| 3 | Test `test_vor_seit_ignoriert` prüft gegen `stand.start` | irreführender Testname |
| 4 | Lesemarke `seit` hält nur für zugeordnete, unverarbeitete Kommentare zurück | ein Kommentar wird erneut gelesen |
| 5 | `voraus(…, basis="master")` als Parameter | ein Parameter mehr |
| 6 | Worktrees außerhalb des Klons unter `stand/wt/` | nur ein anderer Pfad |
| 7 | Push über Askpass-Skript aus `DEBUGGER_ASKPASS`, fehlt es → Fehler | Start scheitert sichtbar |
| 8 | `worktree remove --force` nur nach Merge oder Close des eigenen PR | Reste in einem erledigten Worktree gehen verloren |
| **9** | Pin-Hebung per `update-index --cacheinfo` erst nach vier Prüfungen (Commit, Vorfahr von origin/basis, auf dem Remote) | ein falscher Pin wäre ein falscher Submodul-Stand in Vibemind_V1 |
| 10 | Tasks 1 und 2 gebündelt | größeres Review-Paket |
| 11 | Branch-Name ≤ 30 Zeichen, das Plan-Beispiel gilt | anderer Branch-Name |
| 12 | Git mit Timeout und ohne Prompts | etwas mehr Code |
| **13** | Helfer bekommt eine bereinigte Umgebung ohne Tokens oder Askpass | keine (nur weniger Umgebung) |
| 14 | Prompt ≤ 120 000 Byte, Material wird gekürzt | Helfer sieht weniger Material |
| 15 | `is_error` → Helfer-Fehler (teilweise ersetzt durch 42 und 49) | ein gezählter Versuch zu viel |
| **16** | Wächter als root im Container, Helfer als claw (uid 1000) über `setpriv` | die ganze Token-Grenze hängt daran |
| **17** | Antworten zu PRs im öffentlichen vibemind-os gehen ins private Issue | Gespräch steht im Issue statt im PR |
| 18 | Feste Wächter-Hinweise stehen außerhalb von AUFTRAG und MATERIAL | zwei Sätze mehr im Prompt |
| **19** | Öffentlicher PR-Titel = erste Commit-Zeile, nicht der Issue-Titel | anderer Titel |
| **20** | Nur root-eigene bare Spiegel sprechen mit GitHub, Token nur im Env des Push- oder Fetch-Prozesses | Kern der Token-Grenze |
| 21 | Vor jedem Push `log -p` durch die Schlüsselprüfung | ein Push mehr wird zurückgehalten |
| 22 | Im Material nur Kommentare von Flissel und Vibemind-LAB | weniger Kontext |
| 23 | Prüftext = Commit-Texte und alle neuen Blobs, über 5 MB fail-closed | große Pushes werden zurückgehalten |
| 24 | `nachlauf(…, ordner)` | ein Parameter mehr |
| 25 | Pin-PR ohne Merge geschlossen → erledigt | Issue bleibt bis zum täglichen Lauf offen |
| 26 | Je Pin-Fehlschlag ein Kommentar | ein Kommentar mehr oder weniger |
| 27 | Konfiguration und Werkzeuge in jedem Durchgang neu | ein Neustart weniger nötig |
| **28/29** | Push als Bundle: claw erzeugt es, root prüft es im Spiegel und pusht den geprüften SHA ohne `+` | Kern gegen TOCTOU und Force-Push |
| 30 | Sicherheits-Minors (`--git-dir`, fester PATH, nur lokale URL) mit in die Fix-Runde | mehr Code |
| 31 | Läufer für claw: eigene Prozessgruppe, killpg bei Timeout, Lesen mit Frist | — |
| 32 | `init: true`, `pids_limit`, nach jedem Durchgang alle Prozesse von uid 1000 beenden | ein Helfer-Hintergrundprozess stirbt mit |
| 33 | Linux-Test im Image (am 08.11.: 298 bestanden) | — |
| 34 | flock für den ganzen Durchgang | Windows-Sonderweg nur im Test |
| **35** | Der Einstieg bricht ab, wenn claw das Secret lesen kann | Start scheitert, bis die Datei richtig liegt (gewollt) |
| 36 | Unlesbarer Stand → Exit 1, nichts gespeichert | Wächter steht, bis die Rechte stimmen |
| 37 | Neuer FundStand nach erledigt, Branch-Suffix `-<runde>` | neuer Branch statt des alten |
| 38 | Öffentlicher Branch heißt `debugger/<n>-fix` | anderer Branch-Name |
| **39** | `core.hooksPath=/dev/null` für Git als claw, Pin-Prüfung im root-Spiegel, `nur_gitlink` | Kern gegen Hook-Ausbruch |
| 40 | Sauberkeitsprüfung ohne Submodule, Update in jedem Durchgang | — |
| **41** | Budget vorab gebucht (`laeuft`), fail-closed | eine Runde mehr gezählt |
| 42 | Limit wartet und zählt nicht, Pause 10 min × Versuche | spätere Wiederholung |
| 43 | Worktree ohne PR und ohne Vorlauf entfernen | — |
| 44 | claw-Kill direkt nach dem Helfer | — |
| 45 | Pin-Branches ohne `nur_gitlink` werden abgewiesen, auf einem Pin-Branch nur einordnen | — |
| 46 | (ersetzt durch 49) | — |
| 47/47b | Branch-Name nie im Pin-Muster | — |
| 48 | Runde auf einem PR, der zu ist, wartet auf den Nachlauf | Runde 90 s später |
| 49 | Limit global 30 min, dreimal in Folge → gezählt, Sitzung verworfen | eine Sitzung geht verloren |
| 50 | Keine Test-Push-Probe auf master (Ruleset nur lesend geprüft) | Ruleset-Wirkung nicht live belegt |
| 51 | Rootful Docker auf der VM → Token root:root 0400 per sudo | — |
| 52 | gh aus dem Image entfernt (**falsch**, ersetzt durch 53) | erster Durchgang `gh_fehlt` |
| 53 | gh als Release-`.deb` mit fester Prüfsumme, Drift-Test `test_image.py` | Update von Hand |
| 55 | `STOP` verhindert neue Runden, ein laufender Helfer endet erst durch den Timeout | Helfer läuft nach STOP höchstens bis zum Timeout weiter |
| A1 | Kein Präfix-Treffer für Sammel-Issues (Pfade sind verschachtelte Submodule) | Sammel-Issues in nicht verschachtelten Unterordnern werden nur eingeordnet |
| A2 | Testzeile des Helfers (≤ 300 Zeichen, geprüft) in den PR-Text | eine Zeile aus Helfer-Text im öffentlichen PR |
| A3 | Rückfrage-Label auch bei `aktion: antwort` | — |
