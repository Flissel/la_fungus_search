# Freigabe-Eingang — Betrieb und Abnahme

Alles, was auf den Betreiber wartet, steht als Vorschlag in Rowboat. Annehmen in Rowboat führt **nichts** aus
(Stufe 1). Es markiert den Eintrag nur in der Sales-UI `/freigaben`. Ausgeführt wird nur dort.

Spec: `Vibemind_V1/docs/superpowers/specs/2026-11-08-rowboat-freigabe-eingang-design.md`
Plan: `Vibemind_V1/docs/superpowers/plans/2026-11-08-rowboat-freigabe-eingang.md`

## Was am Selbstbild-Teil läuft

| Teil | Wo | Wie |
|---|---|---|
| Debugger-Claw meldet PRs, schließt sie bei Merge/Close | VM, `~/debugger-claw` | Schlüssel `~/debugger-claw/freigabe/schluessel` (root, 0400) |
| Rückfragen melden | PC, täglicher Lauf `auto` | gleicher Client wie der Abholer |
| Label-Aufträge abholen | PC, Aufgabe „Selbstbild Freigabe-Abholer“ (alle 2 min) | `deploy/selbstbild-abholer.ps1`, Log `%LOCALAPPDATA%\selbstbild\freigabe-abholer.log` |

Umgebung am PC (User-Ebene, nur Namen; Werte nie in Dateien des Repos):

- `SELBSTBILD_FREIGABE_URL`: Basis der Sales-UI
- `SELBSTBILD_FREIGABE_SCHLUESSEL_DATEI`: `%LOCALAPPDATA%\VibeMind\selbstbild\freigabe.schluessel` (ACL nur der Benutzer)

Nicht die unpräfixierten `FREIGABE_*` auf User-Ebene setzen. Sonst nimmt der Learning-Claw am selben PC den
Selbstbild-Schlüssel.

Ausschalten: `deploy/selbstbild-abholer.ps1 -Zurueck`. Die Debugger-Claw meldet nichts mehr, wenn
`FREIGABE_URL` in ihrer `.env` leer ist.

### TLS am PC

Windows-Python 3.11.0 (OpenSSL 1.1.1q) hat `ISRG Root X2` nur im Zwischenspeicher (CA), nicht unter ROOT. Für die
Let's-Encrypt-Kette der Sales-UI (YE1 → Root YE → X2) wählt es deshalb die seit 09/2025 abgelaufene
X2-von-X1-Querzertifizierung und meldet `CERTIFICATE_VERIFY_FAILED` („certificate has expired“). Der Client lädt
darum zusätzlich `certifi`, wenn es installiert ist (`_tls_kontext`, Commit 54bc239).

## Abnahme 09.11.2026

Uhrzeiten in UTC. Testobjekte: PR Flissel/vibemind-os#157 (leer, geschlossen ohne Merge),
Issue Flissel/Vibemind_V1#320 (ohne Rückfrage-Marke, geschlossen).

| # | Prüfung | Beleg | Ergebnis |
|---|---|---|---|
| 1 | Einstellen → Vorschlag in Rowboat ≤ 2 min | `fg-scbrakvvoi` per Debugger-Client eingetragen 08:17:49, `rowboat_stand=vorgeschlagen` 08:18:32; Vorschlagstext im Format `[Debugger-Claw] … · [PR #157](…) · [Sales](…#fg-scbrakvvoi)` | ✅ 43 s |
| 2 | Annehmen in Rowboat → nur Markierung | `todo:acceptSuggestion` 08:18:48 → `angenommen` 08:19:40; Aufträge 0 → 0, PR-Ereignisse 0 → 0, Kommentare 0 → 0 | ✅ |
| 3 | Rückfrage ablehnen → nichts ausgeführt | `fg-cq6cgnn3ul` (#320) `todo:declineSuggestion` → `abgelehnt` 08:32:35; Aufträge 0, Issue-Ereignisse nur `labeled` (Anlage) | ✅ |
| 4 | PR schließen → `erledigt`, in Rowboat abgehakt | #157 geschlossen 08:19:52 → `erledigt` von `debugger-claw` 08:20:06; Bote: Zeile `checked: true` (Durchgang 08:20:32), aufgeräumt 08:21:32 | ✅ |
| 5 | Bote mehrfach → eine Zeile | über 80 Bote-Durchgänge seit 07:47: jede der 7 Kennungen genau einmal in `todo:get` | ✅ |
| 6 | Von Hand geschriebene Zeilen → kein `angenommen` | Zeile mit `#fg-aaaaaaaaaa` und Zeile mit echter offener `#fg-o5y745qoto` per `todo:addItem` 08:35:25; nach 4 Durchgängen weiter `vorgeschlagen`, „0 Staende“; Zeilen danach entfernt | ✅ |
| 7 | je Art ein echter Durchlauf | Debugger-PR: siehe 1/2/4. Selbstbild: Auftrag `antwort:bleibt` für #320 08:33:18, PC-Abholer setzt Label 08:34:36, Auftrag `erfolg`, Eintrag `ausgefuehrt` von `selbstbild` (*) | ✅ Debugger, ✅ Selbstbild; Entwurf und Kursfreigabe offen (Klick des Betreibers) |
| 8 | Stufe 2 gesperrt | live: `stufe2_lesen()` alle `False`, 4 Kästchen `disabled`, keins `checked`; POST → 409 durch `test_freigabe_eingang_ui` (sales_test) | ✅ |

(*) Den Auftrag hat `fe.auftrag_anlegen` im sales-ui-Container angelegt (Akteur `abnahme-claude`), denselben Aufruf
wie hinter dem Knopf „soll bleiben“. Den Klick selbst gab es nicht, weil der Browser einer anderen Session gehörte.
Die HTTP-, CSRF- und Rollen-Schicht prüft `test_freigabe_eingang_ui`.

Gefunden beim Ausrollen:

- `deploy/update.sh` lief zweimal in den Rückbau, weil die Smoke-Liste `sales-freigabe-bote` nicht kannte
  (behoben in sales-claw 4830dc7).
- TLS am PC (siehe oben): behoben in la_fungus 54bc239 und Learning b8cc069.
