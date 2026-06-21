@echo off
cd /d C:\Users\User\Desktop\Vibemind_V1\vibemind-os\la-fungus-search
set FUNGUS_DEVICE=cuda
set FUNGUS_CODEBASE=C:/Users/User/Desktop/Vibemind_V1/vibemind-os
set FUNGUS_EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
C:\Users\User\Desktop\Vibemind_V1\.venv\Scripts\python.exe -u mcp_server.py --http 8412 >> .fungus_cache\mcp_http.log 2>&1
