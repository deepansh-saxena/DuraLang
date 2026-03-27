# Human-in-the-Loop

DuraLang will support pausing a running workflow to wait for human input using Temporal Signals. This is planned for v2.

---

## Concept

Temporal Signals allow external systems to send data into a running workflow. For DuraLang, this means an agent can pause mid-execution, wait for a human to approve or provide input, and resume from exactly where it left off.

---

## How It Will Work

```python
# Inside the workflow, the agent pauses for human input
# The workflow waits for a signal before continuing

# External code sends the signal:
handle = client.get_workflow_handle(workflow_id)
await handle.signal(DuraLangWorkflow.human_input, "Yes, proceed with the plan")
```

---

## Current Status

This feature is planned for v2. The signal handler is defined in `DuraLangWorkflow` but the full flow (pause triggers, resume logic, approval patterns) is not yet implemented.

For now, if you need human-in-the-loop behavior, you can split your workflow into two `@dura` functions — the first runs up to the approval point, and the second continues after your application collects the human input.
