# Supermemory MCP Setup for Claude Code

## Quick Setup (5 minutes)

### Automated Installation

Run this command in your terminal:

```bash
npx -y install-mcp@latest https://api.supermemory.ai/mcp --client claude-code --oauth=yes
```

**What happens:**
1. Browser opens to https://mcp.supermemory.ai
2. You get a unique URL (your authentication token)
3. Configuration is automatically written
4. **You must restart Claude Code completely**

---

## Manual Setup

### Step 1: Get Supermemory URL

Visit: https://mcp.supermemory.ai

Copy your unique URL: `https://abc123.supermemory.ai`

**⚠️ IMPORTANT**: Save this URL securely! It's your access key.

### Step 2: Locate Config File

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

### Step 3: Edit Configuration

Open the config file and add:

```json
{
  "mcpServers": {
    "supermemory": {
      "command": "npx",
      "args": ["-y", "@supermemoryai/supermemory-mcp"],
      "env": {
        "SUPERMEMORY_URL": "https://your-unique-url.supermemory.ai"
      }
    }
  }
}
```

**Alternative: Using API Key**

If you have a `SUPERMEMORY_API_KEY`:

```json
{
  "mcpServers": {
    "supermemory": {
      "command": "npx",
      "args": ["-y", "@supermemoryai/supermemory-mcp"],
      "env": {
        "SUPERMEMORY_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

### Step 4: Restart Claude Code

- Quit Claude Code completely
- Reopen the application
- Look for MCP icon in input area

---

## Verification

### Check if MCP is Working

1. **Look for MCP indicator** in Claude Code UI (bottom-right or input box)

2. **Test with a query:**
   ```
   Use getProjects to list my Supermemory containers
   ```

3. **Expected response:**
   ```
   I can see your Supermemory projects: [project list]
   ```

### Troubleshooting

**Problem: No MCP icon appears**
- Solution: Restart Claude Code completely (quit, not just close window)
- Check config file syntax (valid JSON)

**Problem: Authentication error**
- Solution: Verify SUPERMEMORY_URL or API key is correct
- Get new URL from https://mcp.supermemory.ai

**Problem: Command not found**
- Solution: Ensure Node.js 18+ is installed
- Test: `node --version`

---

## Using MCP with Your Backend

### Shared Memory Containers

Your backend stores insights with `container_tag = run_id`:

```python
# Backend stores to:
container_tag = "la_fungus_search_20251113_140522"
```

To query these insights from Claude Code:

```
Search Supermemory project "la_fungus_search_20251113_140522" for entry points
```

### Available MCP Tools

Once configured, Claude Code can use:

1. **getProjects** - List all memory containers/projects
2. **addMemory** - Store new insights
3. **search** - Search stored memories
4. **whoAmI** - Check authentication

### Example Workflow

**1. Run simulation (backend stores insights):**
```bash
# Your backend stores:
# - Entry points discovered
# - Architectural patterns
# - Security findings
# All stored with container_tag = run_id
```

**2. Query from Claude Code:**
```
You: "Use getProjects to find the latest run"
Claude: [Shows: la_fungus_search_20251113_140522]

You: "Search that project for security vulnerabilities"
Claude: [Uses MCP search → retrieves insights]
Claude: "Found 3 security issues: ..."
```

---

## Integration with Backend

### Backend (Python SDK) - What You Have

```python
# src/embeddinggemma/memory/supermemory_client.py
await memory_manager.add_insight(
    content="FastAPI entry at server.py:50",
    insight_type="entry_point",
    container_tag=run_id,
    confidence=0.95,
    metadata={"file_path": "server.py", "phase": 0}
)
```

**Characteristics:**
- ✅ Autonomous automation
- ✅ Custom metadata and business logic
- ✅ Better performance
- ✅ Full programmatic control

### MCP (Interactive) - Optional Addition

```
You: "Search for FastAPI entry points"
Claude: [Uses MCP search tool]
Claude: "Found: FastAPI entry at server.py:50"
```

**Characteristics:**
- ✅ Conversational queries
- ✅ Cross-tool sharing
- ✅ Manual exploration
- ⚠️ Simpler tools (no custom metadata)

### Both Work Together!

- **Backend**: Stores insights automatically during simulation
- **MCP**: Lets you query those insights conversationally
- **Same data**: Both access the same Supermemory store

---

## Configuration Examples

### Minimal (Just Supermemory)

```json
{
  "mcpServers": {
    "supermemory": {
      "command": "npx",
      "args": ["-y", "@supermemoryai/supermemory-mcp"],
      "env": {
        "SUPERMEMORY_URL": "https://abc123.supermemory.ai"
      }
    }
  }
}
```

### With Multiple MCP Servers

```json
{
  "mcpServers": {
    "supermemory": {
      "command": "npx",
      "args": ["-y", "@supermemoryai/supermemory-mcp"],
      "env": {
        "SUPERMEMORY_URL": "https://abc123.supermemory.ai"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_token"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\bauma\\Desktop\\Felix_code"
      ]
    }
  }
}
```

---

## Security Notes

### Supermemory URL is Sensitive!

Your unique URL acts as your authentication token:

```
https://abc123.supermemory.ai
         ^^^^^^
         This is your secret key!
```

**⚠️ Security Best Practices:**
- Don't share your URL publicly
- Don't commit to version control
- If lost, get new URL from https://mcp.supermemory.ai
- Store in password manager

### Container Isolation

Use different container tags for different projects:

```python
# Production data
container_tag = "prod_analysis_20251113"

# Test data
container_tag = "test_run_20251113"

# Development
container_tag = "dev_exploration_20251113"
```

This prevents cross-contamination of insights.

---

## Next Steps

1. **Test Backend Integration First**
   - Run a simulation
   - Verify insights are stored
   - Check manifest.json for memory_stats

2. **Then Add MCP (Optional)**
   - Set up MCP configuration
   - Query insights from Claude Code
   - Enjoy conversational access!

3. **Advanced: Use Both Together**
   - Backend stores during autonomous runs
   - MCP queries for interactive debugging
   - Best of both worlds!

---

## Resources

- **MCP Docs**: https://modelcontextprotocol.io
- **Supermemory MCP**: https://supermemory.ai/docs/supermemory-mcp
- **Get MCP URL**: https://mcp.supermemory.ai
- **Backend Integration**: See [SUPERMEMORY_INTEGRATION.md](SUPERMEMORY_INTEGRATION.md)

---

## Troubleshooting

### Common Issues

**1. "npx: command not found"**
- Install Node.js 18+ from https://nodejs.org
- Restart terminal
- Verify: `node --version`

**2. "MCP server failed to start"**
- Check config file syntax (must be valid JSON)
- Verify SUPERMEMORY_URL is correct
- Try removing and re-adding the server

**3. "Authentication failed"**
- Get new URL from https://mcp.supermemory.ai
- Update SUPERMEMORY_URL in config
- Restart Claude Code

**4. "No projects found"**
- Run a simulation first (backend stores insights)
- Use `getProjects` to list available containers
- Ensure same Supermemory account in backend and MCP

---

## Summary

**MCP Setup**: Optional enhancement for interactive queries
**Backend Integration**: Already working (production-ready)
**Best Approach**: Test backend first, add MCP later if needed

The Python SDK integration you have is optimal for autonomous exploration. MCP is a nice-to-have for conversational debugging! 🎉
