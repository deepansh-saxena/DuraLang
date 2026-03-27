# Human-in-the-Loop

DuraLang supports pausing a running agent to wait for human input using Temporal Signals. This feature leverages Temporal's native signal mechanism, which allows external systems to send data into a running workflow at any point.

---

## Concept

In production agent systems, many operations require human oversight before proceeding:

- **Approval gates** — "The agent wants to send an email. Should it proceed?"
- **Input collection** — "The agent needs a password or API key to continue."
- **Review points** — "The agent has a draft. Does the user want changes?"
- **Safety checks** — "The agent is about to execute a destructive action."

Temporal Signals are the ideal mechanism for this. A running workflow can pause indefinitely, wait for a signal containing human input, and resume from exactly where it left off — with the full durable execution guarantee.

---

## How It Works

When an agent needs human input, it pauses the workflow and waits for a Temporal Signal. An external system (a web UI, Slack bot, CLI, etc.) sends the signal with the human's response. The workflow resumes from the exact point of pause.

```python
# Inside the @dura function:
# The workflow pauses here, waiting for human input
human_response = await workflow.wait_for_signal("human_input")

# External code sends the signal:
handle = client.get_workflow_handle(workflow_id)
await handle.signal(DuraLangWorkflow.human_input, "Yes, proceed with the plan")
```

### Key Properties

| Property | Behavior |
|---|---|
| **Durability** | The pause is durable — if the worker process dies, the workflow resumes when a new worker picks it up |
| **No polling** | The workflow sleeps efficiently, not burning compute |
| **Exact resume** | All completed activities are preserved. The agent continues from the exact step where it paused |
| **Timeout support** | The pause can have a timeout — if no human responds, the agent can take a default action |

---

## Current Pattern (v1 Workaround)

Until the full signal-based flow is implemented in v2, you can achieve human-in-the-loop by splitting your workflow into two `@dura` functions:

```python
from langchain.agents import create_agent
from duralang import dura

@dura
async def phase_1_gather(task: str) -> str:
    """First phase — gathers information and produces a proposal."""
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=research_tools,
    )
    result = await agent.ainvoke({
        "messages": [HumanMessage(content=f"Research and propose a plan for: {task}")]
    })
    return result["messages"][-1].content

@dura
async def phase_2_execute(proposal: str, human_feedback: str) -> str:
    """Second phase — executes the proposal with human feedback incorporated."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")
    response = await llm.ainvoke([
        HumanMessage(content=(
            f"Original proposal:\n{proposal}\n\n"
            f"Human feedback:\n{human_feedback}\n\n"
            f"Proceed with the plan, incorporating the feedback."
        ))
    ])
    return response.content

# Your application orchestrates the two phases:
async def main():
    # Phase 1: Agent gathers info and proposes a plan
    proposal = await phase_1_gather("Migrate database to PostgreSQL")

    # Human review step (your application handles this)
    print(f"Proposal:\n{proposal}")
    human_feedback = input("Your feedback: ")

    # Phase 2: Agent executes with feedback
    result = await phase_2_execute(proposal, human_feedback)
    print(result)
```

Both phases are independently durable. If phase 1 crashes, it retries. If phase 2 crashes, phase 1's result is preserved.

---

## v2 Design

The v2 implementation will support inline pauses within a single `@dura` function using Temporal Signals:

```python
@dura
async def agent_with_approval(task: str) -> str:
    # Phase 1: Gather information
    proposal = await generate_proposal(task)

    # Pause for human approval — the workflow sleeps here
    approval = await dura.wait_for_human(
        prompt=f"Approve this plan?\n{proposal}",
        timeout=timedelta(hours=24),
    )

    if approval.approved:
        # Phase 2: Execute the plan
        result = await execute_plan(proposal, approval.feedback)
        return result
    else:
        return f"Plan rejected: {approval.feedback}"
```

The signal handler will be defined in `DuraLangWorkflow`. The external interface (web hook, API endpoint, CLI) will send signals to the workflow handle.
