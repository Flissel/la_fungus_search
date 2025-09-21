@echo off
setlocal enabledelayedexpansion
set PYTHON="%~dp0\.venv\Scripts\python.exe"
if not exist %PYTHON% (
  echo Python venv not found: %PYTHON%
  exit /b 1
)
if "%~1"=="" (
  %PYTHON% -m streamlit run streamlit_fungus_backup.py
) else (
  %PYTHON% -m streamlit %*
)
