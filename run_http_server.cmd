@echo off
cd /d C:\Users\User\Desktop\Vibemind_V1\vibemind-os\la-fungus-search
if "%VIBEMIND_CONFIG_DIR%"=="" (
  echo VIBEMIND_CONFIG_DIR must point to the canonical VibeMind configuration directory.
  exit /b 1
)
if not exist "%VIBEMIND_CONFIG_DIR%\llm_config.yml" (
  echo Canonical VibeMind configuration not found: %VIBEMIND_CONFIG_DIR%\llm_config.yml
  exit /b 1
)
set FUNGUS_CODEBASE=C:/Users/User/Desktop/Vibemind_V1/vibemind-os
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
C:\Users\User\Desktop\Vibemind_V1\.venv\Scripts\python.exe -u mcp_server.py --http 8412 >> .fungus_cache\mcp_http.log 2>&1
