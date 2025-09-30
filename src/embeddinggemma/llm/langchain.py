from __future__ import annotations

import os


def build_langchain_chat_ollama():
    try:
        try:
            import langchain_ollama as _lco  # type: ignore
            LCChatOllama = _lco.ChatOllama  # type: ignore
        except Exception:
            from langchain_community.chat_models import ChatOllama as LCChatOllama  # type: ignore
        model = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        return LCChatOllama(model=model)
    except Exception:
        return None



