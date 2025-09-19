import json
from typing import Any, Dict, List


def build_langchain_agent_if_available(settings: Dict[str, Any]):
    try:
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool
        try:
            import langchain_ollama as _lco  # type: ignore
            LCChatOllama = _lco.ChatOllama  # type: ignore
        except Exception:
            from langchain_community.chat_models import ChatOllama as LCChatOllama  # type: ignore
        try:
            from langgraph.prebuilt import create_react_agent  # type: ignore
        except Exception:
            create_react_agent = None  # type: ignore
    except Exception:
        return None

    from .mcmp_runner import quick_search_with_mcmp

    def tool_search(q: str) -> str:
        res = quick_search_with_mcmp(settings, q, settings.get('top_k', 5))
        if res.get('error'):
            return res['error']
        items = res.get('results', [])
        lines = []
        for i, it in enumerate(items[: settings.get('top_k', 5)]):
            src = (it.get('metadata', {}) or {}).get('file_path', 'chunk')
            lines.append(f"#{i+1} {float(it.get('relevance_score', 0.0)):.3f} {src}")
        return "\n".join(lines) or "No results"

    def tool_get_settings(_: str = "") -> str:
        safe = {k: v for k, v in settings.items() if k not in {"docs"}}
        return json.dumps(safe)

    def tool_set_root_dir(new_dir: str) -> str:
        import os
        if os.path.isdir(new_dir):
            settings['use_repo'] = False
            settings['root_folder'] = new_dir
            return f"Root folder set to: {new_dir}"
        return f"Directory does not exist: {new_dir}"

    tools = [
        Tool(name="search_code", func=tool_search, description="Search codebase for a query and return top sources."),
        Tool(name="get_settings", func=tool_get_settings, description="Return current search settings as JSON."),
        Tool(name="set_root_dir", func=tool_set_root_dir, description="Set the root directory for searching. Provide an absolute path."),
    ]

    llm = LCChatOllama(model=os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')) if 'os' in globals() and LCChatOllama else None  # type: ignore
    if llm is None:
        return None

    if create_react_agent is not None:
        try:
            if hasattr(llm, "bind_tools"):
                graph = create_react_agent(llm, tools)

                class _GraphWrapper:
                    def __init__(self, g):
                        self._g = g

                    def invoke(self, payload: Dict[str, Any]):
                        msg = [{"role": "user", "content": str(payload.get("input", ""))}]
                        out = self._g.invoke({"messages": msg})
                        try:
                            messages = out.get("messages", [])
                            content = messages[-1].get("content", "") if messages else str(out)
                            return {"output": content}
                        except Exception:
                            return {"output": str(out)}

                return _GraphWrapper(graph)
        except Exception:
            pass

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
