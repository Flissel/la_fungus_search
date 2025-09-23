param(
    [int]$Port = 8011
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python venv not found: $python"
    exit 1
}

& $python -m pip install -q fastapi "uvicorn[standard]" websockets wsproto plotly | Out-Null

# Ensure uvicorn can import from src/
& $python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --port $Port --reload


