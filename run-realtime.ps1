param(
    [int]$Port = 8011
)

$ErrorActionPreference = "Stop"

# Try to use venv Python, fall back to system Python
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$useVenv = $false

if (Test-Path $python) {
    # Check if venv has pip (suppress error output)
    $ErrorActionPreference = "SilentlyContinue"
    $pipCheck = & $python -m pip --version 2>&1 | Out-String
    $pipWorks = $LASTEXITCODE -eq 0
    $ErrorActionPreference = "Stop"

    if ($pipWorks) {
        $useVenv = $true
        Write-Host "Using virtual environment Python" -ForegroundColor Green
    } else {
        Write-Warning "Virtual environment exists but pip is not installed"
        Write-Host "Recreating virtual environment..." -ForegroundColor Yellow

        # Remove broken venv
        Write-Host "  Removing old venv..." -ForegroundColor DarkGray
        Remove-Item -Path (Join-Path $PSScriptRoot ".venv") -Recurse -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500  # Give Windows time to release file handles

        # Create new venv with system Python
        Write-Host "  Creating new venv..." -ForegroundColor DarkGray
        $venvOutput = python -m venv .venv 2>&1

        if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"))) {
            $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
            $useVenv = $true
            Write-Host "[OK] Virtual environment recreated successfully" -ForegroundColor Green
        } else {
            Write-Warning "Failed to create venv, falling back to system Python"
            Write-Host "Venv error: $venvOutput" -ForegroundColor DarkGray
            $python = "python"
        }
    }
}

if (-not $useVenv) {
    Write-Host "Using system Python..." -ForegroundColor Yellow
    $python = "python"

    # Test if system Python is available
    try {
        & $python --version 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
    } catch {
        Write-Error "No Python found. Please install Python or fix your PATH."
        exit 1
    }
}

Write-Host "`n=== Installing Dependencies ===" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor Gray
Write-Host ""

# Upgrade pip first
Write-Host "[1/2] Upgrading pip..." -ForegroundColor Gray
$ErrorActionPreference = "SilentlyContinue"
$pipUpgrade = & $python -m pip install --upgrade pip -q 2>&1 | Out-String
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] Failed to upgrade pip, but continuing..." -ForegroundColor Yellow
} else {
    Write-Host "  [OK] pip upgraded" -ForegroundColor Green
}

# Install all dependencies
Write-Host "[2/2] Installing packages from requirements.txt..." -ForegroundColor Gray
Write-Host "      (FastAPI, LangChain, Supermemory, etc.)" -ForegroundColor DarkGray
$ErrorActionPreference = "SilentlyContinue"
$installResult = & $python -m pip install -q -r requirements.txt 2>&1 | Out-String
$installSuccess = $LASTEXITCODE -eq 0
$ErrorActionPreference = "Stop"

if (-not $installSuccess) {
    Write-Host "  [WARN] Some dependencies may have failed" -ForegroundColor Yellow
    if ($installResult -match "error|failed|ERROR|FAILED") {
        Write-Host "  Error preview: $($installResult.Substring(0, [Math]::Min(200, $installResult.Length)))" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  [OK] All dependencies installed" -ForegroundColor Green
}

Write-Host "`n=== Starting Server ===" -ForegroundColor Cyan
Write-Host "Port: $Port" -ForegroundColor Gray
Write-Host "URL: http://localhost:$Port" -ForegroundColor Gray
Write-Host "`nPress Ctrl+C to stop the server`n" -ForegroundColor Yellow

# Ensure uvicorn can import from src/
& $python -m uvicorn --app-dir src embeddinggemma.realtime.server:app --port $Port --reload


