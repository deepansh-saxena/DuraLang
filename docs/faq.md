# FAQ

Common questions and troubleshooting for DuraLang.

---

## General

### What does `@dura` actually do?

It wraps your async function so that when called, it starts a Temporal Workflow. Inside the workflow, proxy objects intercept `llm.ainvoke()`, `tool.arun()`, and `session.call_tool()` — routing each call through a Temporal Activity. Outside of a `@dura` function, LangChain works exactly as normal.

### Do I need to change my LangChain code?

No. You add `@dura` to your function and `from duralang import dura`. Everything else stays the same.

### Does `@dura` work with any LLM provider?

Yes. Any LangChain-compatible `BaseChatModel` works: Anthropic, OpenAI, Google, Ollama. DuraLang extracts `LLMIdentity` from the instance to reconstruct it inside the Activity.

### What happens if my agent crashes mid-execution?

Temporal replays the workflow from its event history. Completed activities are not re-executed — only the currently running or failed activity is retried.

---

## Architecture

### How does interception work without changing my code?

DuraLang patches `BaseChatModel.__init__` and `BaseTool.__init__` at import time. Every instance gets proxy methods installed that check for `DuraContext`. If no context exists (outside `@dura`), the original methods are called. If context exists (inside `@dura`), calls route to Temporal Activities.

### What is `DuraContext`?

A `contextvars.ContextVar` that carries workflow state. Set by `DuraLangWorkflow` before calling your function, read by proxy objects to know how to route calls.

### How do multi-agent calls work?

If a `@dura`-decorated function calls another `@dura`-decorated function, the wrapper detects the existing `DuraContext` and routes to `workflow.execute_child_workflow()` instead of starting a new top-level workflow.

---

## Troubleshooting

### "Tool not in registry" error

This means the `dura__tool` Activity couldn't find the tool by name. Ensure:
1. Tools are created inside or before your `@dura` function
2. Tool names match what the LLM calls

### "Cannot determine LLM provider" error

DuraLang needs to identify your LLM provider to reconstruct it inside the Activity. Supported providers: `ChatAnthropic`, `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatOllama`.

### "@dura cannot wrap lambda functions"

`@dura` functions must be importable by the Temporal worker. Lambdas and closures cannot be resolved by import path. Define your function at module top level.

### My function works outside `@dura` but fails inside

Check that:
1. All arguments are serializable (primitives, lists, dicts, LangChain messages)
2. The function is defined at module level (not nested inside another function)
3. Temporal server is running and reachable
