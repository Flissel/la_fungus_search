<#
.SYNOPSIS
  Legt die geplante Aufgabe "Selbstbild Freigabe-Abholer" an (alle 2 Minuten) oder entfernt sie (-Zurueck).

.DESCRIPTION
  Die Aufgabe ruft `python -m selbstbild.freigabe_abholer <wurzel>` auf: ein Durchgang holt Label-Auftraege aus dem
  Freigabe-Eingang von sales-claw und setzt das Label am GitHub-Issue (Konto aus den Regeln, nie `gh auth switch`).

  Aktion und Arbeitsordner spiegeln die bestehende taegliche Aufgabe "VibeMind Selbstbild-Funde":
  `conhost.exe --headless cmd /c ""<python>" -m ... >> "<log>" 2>&1"`, Arbeitsordner = Wurzel des selbstbild-Repos,
  Principal = angemeldeter Benutzer (InteractiveToken, LeastPrivilege).

  UMGEBUNG: Die taegliche Aufgabe setzt keine Umgebungsvariablen. Der Abholer liest deshalb wie sie die
  USER-Umgebung des angemeldeten Benutzers. Der Betreiber setzt dort selbst (Werte stehen NICHT in diesem Skript):
      SELBSTBILD_FREIGABE_URL                Basis der Sales-UI
      SELBSTBILD_FREIGABE_SCHLUESSEL_DATEI   Pfad der Schluesseldatei (nur fuer den Benutzer lesbar)
  zum Beispiel mit [Environment]::SetEnvironmentVariable('SELBSTBILD_FREIGABE_URL', '<wert>', 'User').
  Diese praefixierten Namen auf User-Ebene setzen, NICHT FREIGABE_URL / FREIGABE_SCHLUESSEL_DATEI: der Learning-Claw
  am selben PC (gleicher Benutzer) nutzt die unpraefixierten Namen, und die Prozess-Umgebung schlaegt seine .env,
  er wuerde sonst den Selbstbild-Schluessel verwenden. Der Abholer faellt je Name auf die unpraefixierte Variable
  zurueck, wenn die praefixierte fehlt oder leer ist. Fehlt eine der beiden Variablen oder die Datei, loggt der Abholer "aus" und endet mit Exit 0.

  Dieses Skript ist ein Deploy-Schritt: es braucht das ausdrueckliche OK des Betreibers und wird nicht automatisch
  ausgefuehrt. Es ist idempotent (-Force ersetzt eine vorhandene Aufgabe).

.PARAMETER Zurueck
  Entfernt die Aufgabe wieder (spiegelt `auto aufgabe --zurueck`). Fehlt sie schon, ist das kein Fehler.

.PARAMETER Wurzel
  Das vermessene Repo (Regeln und Issue-Ziel); Vorgabe wie bei der taeglichen Aufgabe.

.PARAMETER Python
  Python mit allen selbstbild-Abhaengigkeiten; Vorgabe: .venv-selbstbild im Arbeitsordner.

.PARAMETER Arbeitsordner
  Wurzel des selbstbild-Repos (hier liegt `src/selbstbild`); Vorgabe: Elternordner dieses Skripts.
#>
[CmdletBinding()]
param(
    [switch]$Zurueck,
    [string]$Wurzel = 'C:\Users\User\Desktop\Vibemind_V1',
    [string]$Arbeitsordner = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = (Join-Path (Split-Path -Parent $PSScriptRoot) '.venv-selbstbild\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$Name = 'Selbstbild Freigabe-Abholer'

if ($Zurueck) {
    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Output "${Name}: entfernt"
    } else {
        Write-Output "${Name}: war nicht eingerichtet"
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python nicht gefunden: $Python" }
if (-not (Test-Path -LiteralPath $Arbeitsordner)) { throw "Arbeitsordner nicht gefunden: $Arbeitsordner" }
if ($Wurzel -match '["%^&|<>]' -or $Python -match '["%^&|<>]') { throw 'Zeichen, die cmd.exe umdeutet, im Pfad' }

$LogOrdner = Join-Path $env:LOCALAPPDATA 'selbstbild'
New-Item -ItemType Directory -Force -Path $LogOrdner | Out-Null
$Log = Join-Path $LogOrdner 'freigabe-abholer.log'

$Argumente = "--headless cmd /c `"`"$Python`" -m selbstbild.freigabe_abholer `"$Wurzel`" >> `"$Log`" 2>&1`""
$Aktion = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $Argumente -WorkingDirectory $Arbeitsordner

# Start in einer Minute, danach alle 2 Minuten ohne Ende.
$Ausloeser = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 2)
$Einstellungen = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$Benutzer = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Prinzipal = New-ScheduledTaskPrincipal -UserId $Benutzer -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $Name -Action $Aktion -Trigger $Ausloeser -Settings $Einstellungen `
    -Principal $Prinzipal -Description 'Selbstbild: Label-Auftraege aus dem Freigabe-Eingang abholen (alle 2 min)' `
    -Force | Out-Null
Write-Output "${Name}: eingerichtet (alle 2 Minuten, Log: $Log)"
