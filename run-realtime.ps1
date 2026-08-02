param(
    [int]$Port = 8011
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:VIBEMIND_CONFIG_DIR)) {
    throw "VIBEMIND_CONFIG_DIR must point to the canonical VibeMind configuration directory."
}

$configPath = Join-Path $env:VIBEMIND_CONFIG_DIR "llm_config.yml"
if (-not (Test-Path $configPath)) {
    throw "Canonical VibeMind configuration not found: $configPath"
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python venv not found: $python"
    exit 1
}

& $python -m pip install -q fastapi "uvicorn[standard]" websockets wsproto plotly python-dotenv | Out-Null

# Ensure uvicorn can import from src/
& $python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --port $Port --reload


