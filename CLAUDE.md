# LangForge — Specifications (CLAUDE.md)

> **Save this file as both `SPECIFICATIONS.md` and `CLAUDE.md` at the repo root.**
> Claude Code reads `CLAUDE.md` automatically as its primary instruction file.

> **Powered by Temporal** | Durable LangGraph execution. Every node is a Temporal Activity. Every tool call is a Temporal Activity. State lives in Temporal's event history — not SQLite.

---

## 0. North Star

LangForge is a **durable execution runtime for LangGraph graphs**. It decomposes a LangGraph graph into Temporal primitives:

| LangGraph concept | Temporal primitive |
|---|---|
| Graph node execution | Activity |
| Tool call within a node | Activity (scheduled by Workflow, not by NodeActivity) |
| MCP server call | Activity (scheduled by Workflow, not by NodeActivity) |
| Graph state | Workflow state (persisted in Temporal event history) |
| Edge evaluation / routing logic | Workflow code (deterministic) |
| ReAct loop (agent → tools → agent) | Workflow code (drives the loop explicitly) |
| Checkpointer | **Replaced entirely by Temporal** |
| Human-in-the-loop | Temporal Signal |

> LangForge doesn't run LangGraph inside a workflow.
> LangForge **is** LangGraph, re-expressed as Temporal primitives.
