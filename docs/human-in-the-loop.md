# Human-in-the-Loop

DuraLang supports pausing a running workflow to wait for human input using Temporal Signals. This is a v2 feature currently in development.

---

## Concept

The `DuraLangWorkflow` supports a `human_input` signal that can inject a `HumanMessage` into the message history mid-execution. This allows external systems to pause an agent, collect human feedback, and resume.

---

## How It Will Work

```python
# Inside the workflow, a pause can be triggered
# The workflow waits for a signal before continuing

# External code sends the signal:
handle = client.get_workflow_handle(workflow_id)
await handle.signal(DuraLangWorkflow.human_input, "Yes, proceed with the plan")
```

---

## Current Status

The signal handler is defined in `DuraLangWorkflow` but the full human-in-the-loop flow (pause triggers, resume logic) is planned for a future release.
