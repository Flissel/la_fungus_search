param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python venv not found: $python"
    exit 1
}

# Default to running the project app if no args were provided
if (-not $Rest -or $Rest.Length -eq 0) {
    $Rest = @('run', 'streamlit_fungus_backup.py')
}

& $python -m streamlit @Rest


