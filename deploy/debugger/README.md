# Debugger-Claw im Container (VM)

Der Wächter läuft als root im Container und liest alle `intervall_s` Sekunden (Standard 90) die
Kommentare von `Flissel` in Selbstbild-Issues und PRs. Je Auftrag startet er eine Runde. Claude Code
arbeitet darin als Nutzer `claw` (uid 1000) in einem Worktree. root pusht über einen Spiegel
(bare Repo je Repo unter `/data/spiegel`) als Vibemind-LAB. `claw` kommt nie an den Token.

| Ort im Container | Art | Inhalt |
|---|---|---|
| `/data/klon` | Volume `klon` (claw) | Klon `Vibemind_V1` mit Submodul `vibemind-os`, `origin` = Spiegel |
| `/data/stand` | Volume `stand` (root) | `debugger-stand.json`, Budget, `STOP`, Sperre; `wt/` gehört claw |
| `/data/spiegel` | Volume `spiegel` (root, für claw lesbar) | `Flissel__Vibemind_V1.git`, `Flissel__vibemind-os.git` |
| `/home/claw/.claude` | Volume `claude` (claw) | Claude-Login (`CLAUDE_CONFIG_DIR`) |
| `/data/konfig` | Bind-Mount, nur lesen | `selbstbild.yaml` mit Abschnitt `debugger:` |
| `/run/secrets/lab_token` | Secret-Datei | klassischer Token von Vibemind-LAB |

## Einrichten (Schritte des Users)

Alle `docker compose`-Befehle laufen in `~/debugger-claw/la_fungus/deploy/debugger`.

1. **Ordner auf der VM anlegen.** `mkdir -p ~/debugger-claw/secrets ~/debugger-claw/konfig`. Dann la_fungus
   nach `~/debugger-claw/la_fungus` klonen. Das ist der Build-Kontext: das Image kopiert `pyproject.toml`,
   `README.md`, `src/` und die Skripte aus `deploy/debugger/` und installiert la_fungus per `pip install`.
2. **Vibemind-LAB einladen.** Vibemind-LAB in `Flissel/Vibemind_V1` und `Flissel/vibemind-os` mit
   Write einladen (nie Admin). Die Einladung als Vibemind-LAB annehmen.
3. **Token ablegen.** Als Vibemind-LAB einen klassischen Token mit Scope `repo` erzeugen und auf der VM als
   `~/debugger-claw/secrets/lab_token` ablegen, dann `chmod 400 ~/debugger-claw/secrets/lab_token`.
4. **Konfiguration.** `~/debugger-claw/konfig/selbstbild.yaml` anlegen. Sie liegt bewusst nicht im Klon,
   denn `origin/master` hat kein `selbstbild.yaml`:

   ```yaml
   debugger:
     issue_repo: Flissel/Vibemind_V1
     repos:
       Flissel/Vibemind_V1: "."
       Flissel/vibemind-os: vibemind-os
   ```

5. **`.env` und Image.** Neben `compose.yml` eine Datei `.env` anlegen (sie ist in `.gitignore`):

   ```
   DEBUGGER_HEIM=/home/<nutzer>/debugger-claw
   DEBUGGER_GIT_EMAIL=<id>+Vibemind-LAB@users.noreply.github.com
   ```

   Dann `docker compose build`.
6. **Spiegel und Klon einmalig einrichten.**
   1. Zweimal einen Durchgang starten:
      `docker compose run --rm debugger python -m selbstbild debugger durchgang /data/konfig --klon /data/klon --ordner /data/stand --spiegel /data/spiegel`.
      Der erste Durchgang setzt nur die Lesemarke (erster Start, ältere Kommentare gelten als verarbeitet).
      Der zweite legt die Spiegel an und holt sie von GitHub. Danach endet er mit Exit 1, weil der Klon
      noch fehlt. Das ist hier erwartet.
   2. Den Klon als claw aus dem Spiegel ziehen (`<basis>` ist `basis` aus der Konfiguration, Standard `master`):
      `docker compose run --rm debugger setpriv --reuid=1000 --regid=1000 --init-groups -- env HOME=/home/claw git clone -b <basis> /data/spiegel/Flissel__Vibemind_V1.git /data/klon`
   3. Das Submodul auf den Spiegel umstellen und holen, ebenfalls als claw (Präfix wie oben):
      `... git -C /data/klon config submodule.vibemind-os.url /data/spiegel/Flissel__vibemind-os.git`
      und danach `... git -C /data/klon submodule update --init vibemind-os`.
7. **Claude-Anmeldung als claw.**
   `docker compose run --rm debugger setpriv --reuid=1000 --regid=1000 --init-groups -- env HOME=/home/claw claude`
   starten, mit dem Abo anmelden, dann beenden. Der Login liegt im Volume `claude`.
8. **Starten.** `docker compose up -d`. Logs zeigt `docker compose logs -f debugger`.
9. **Rückweg.** `docker compose down`. Die Volumes bleiben, auch Stand und Login.

## Betrieb

- **Not-Aus:** `docker compose exec debugger touch /data/stand/STOP`. Keine neue Runde startet, bis die
  Datei wieder weg ist (`rm /data/stand/STOP`).
- **Token drehen:** die Datei `secrets/lab_token` ersetzen. Der Wächter baut seine Werkzeuge in jedem
  Durchgang neu, ein Neustart ist nicht nötig.
- **Klon nicht sauber oder auf einem anderen Branch:** keine Runde. Der Grund steht einmal im Log und als
  `klon:<grund>` im Stand. Der Wächter verwirft nichts. Erst wenn der Klon wieder sauber auf `<basis>` steht,
  läuft es weiter.
- **Kaputter Stand:** die Datei wird als `debugger-stand.json.kaputt-<zeit>` beiseitegelegt. Danach gilt der
  erste Start.
- Nach jedem Durchgang beendet der Wächter alle Prozesse von uid 1000 (`pkill -KILL -U 1000`, R32).
  `init: true` räumt verwaiste Kindprozesse ab, `pids_limit` begrenzt ihre Zahl.

## Bekannte Grenzen

- Das Netz im Container ist offen. Der Helfer (claw) kann Repo-Inhalte lesen und ins Netz schicken. An den
  Token kommt er nicht: die Datei gehört root (0400), und der Token steht nur in der Umgebung der
  Push-/Fetch-Prozesse von root.
- Gelesen werden Issue-Kommentare und Review-Zeilenkommentare. Review-Zusammenfassungen (der Text beim
  Absenden eines Reviews) liest der Wächter in Plan 1 nicht.
- Die Container-Dateien sind nicht automatisch getestet (am PC gibt es kein Docker). Geprüft ist nur die
  Syntax der Skripte.
