"""Test LangChain imports"""

print("Testing LangChain imports...")

# Test 1: langchain.agents.AgentExecutor
try:
    from langchain.agents import AgentExecutor
    print("[OK] Found AgentExecutor in langchain.agents")
except ImportError as e:
    print(f"[FAIL] langchain.agents.AgentExecutor: {e}")

# Test 2: langchain_core.agents.AgentExecutor
try:
    from langchain_core.agents import AgentExecutor
    print("[OK] Found AgentExecutor in langchain_core.agents")
except ImportError as e:
    print(f"[FAIL] langchain_core.agents.AgentExecutor: {e}")

# Test 3: Check what's in langchain.agents
try:
    import langchain.agents as agents
    print(f"\n[INFO] Available in langchain.agents:")
    for attr in sorted(dir(agents)):
        if not attr.startswith('_'):
            print(f"  - {attr}")
except Exception as e:
    print(f"[FAIL] Error listing langchain.agents: {e}")

# Test 4: Try the old create_react_agent
try:
    from langchain.agents import create_react_agent
    print("\n[OK] Found create_react_agent in langchain.agents")
except ImportError as e:
    print(f"\n[FAIL] langchain.agents.create_react_agent: {e}")
