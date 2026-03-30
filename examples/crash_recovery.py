"""Crash recovery demo — DuraLang agents that cannot fail.

Demonstrates two things:

1. AUTOMATIC RETRY: A flaky tool fails on first attempt. Temporal retries it
   automatically. Completed LLM calls are NOT re-executed.

2. PROCESS CRASH RECOVERY: The worker process is killed mid-execution.
   Re-run the script. Temporal replays all completed steps from event
   history (no LLM calls re-made, no money wasted) and continues from
   the exact point of failure.

Usage:
    # Mode 1: Automatic retry (tool fails once, Temporal retries)
    python examples/crash_recovery.py

    # Mode 2: Process crash (kills process mid-run, re-run to resume)
    python examples/crash_recovery.py --crash

    # Clean up state files
    python examples/crash_recovery.py --clean

Prerequisites:
    - Temporal server running: temporal server start-dev
    - ANTHROPIC_API_KEY set
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from duralang import dura, dura_agent

# ── State tracking ───────────────────────────────────────────────────────────
# File-based counter so tools can track attempts across retries and restarts.

STATE_FILE = Path("/tmp/duralang_crash_demo.json")
WORKFLOW_ID = "crash-demo-duralang-001"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tool_attempts": {}, "crash_triggered": False}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _increment_attempt(tool_name: str) -> int:
    """Increment and return the attempt count for a tool call."""
    state = _load_state()
    attempts = state.setdefault("tool_attempts", {})
    attempts[tool_name] = attempts.get(tool_name, 0) + 1
    _save_state(state)
    return attempts[tool_name]


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    attempt = _increment_attempt(f"get_stock_price:{ticker}")
    print(f"    [get_stock_price] ticker={ticker}, attempt={attempt}")

    # Simulate flaky API: first attempt always times out
    if attempt == 1:
        print(f"    [get_stock_price] FAILED - API timeout (Temporal will retry)")
        raise TimeoutError(f"Connection to stock API timed out for {ticker}")

    prices = {"AAPL": "$189.84", "GOOGL": "$176.32", "MSFT": "$420.21"}
    price = prices.get(ticker.upper(), "$100.00")
    print(f"    [get_stock_price] SUCCESS - {ticker}: {price}")
    return f"{ticker}: {price}"


@tool
def analyze_sentiment(topic: str) -> str:
    """Analyze market sentiment for a given topic."""
    attempt = _increment_attempt(f"analyze_sentiment:{topic[:30]}")
    print(f"    [analyze_sentiment] topic='{topic[:50]}', attempt={attempt}")

    # In --crash mode: kill the process on first attempt
    if "--crash" in sys.argv:
        state = _load_state()
        if not state.get("crash_triggered"):
            state["crash_triggered"] = True
            _save_state(state)
            print(f"    [analyze_sentiment] PROCESS KILLED (simulating crash)")
            print()
            print("=" * 64)
            print("  WORKER PROCESS CRASHED")
            print()
            print("  The Temporal server still has your workflow running.")
            print("  All completed LLM calls and tool calls are preserved.")
            print()
            print("  Run the script again to resume:")
            print("    python examples/crash_recovery.py --crash")
            print("=" * 64)
            os._exit(1)

    print(f"    [analyze_sentiment] SUCCESS - analysis complete")
    return f"Market sentiment for '{topic}' is bullish. Recent earnings beat expectations."


@tool
def get_analyst_rating(ticker: str) -> str:
    """Get the consensus analyst rating for a stock."""
    attempt = _increment_attempt(f"get_analyst_rating:{ticker}")
    print(f"    [get_analyst_rating] ticker={ticker}, attempt={attempt}")
    print(f"    [get_analyst_rating] SUCCESS")

    ratings = {"AAPL": "Strong Buy", "GOOGL": "Buy", "MSFT": "Strong Buy"}
    rating = ratings.get(ticker.upper(), "Hold")
    return f"{ticker} consensus rating: {rating} (14 analysts)"


# ── Agent ────────────────────────────────────────────────────────────────────


@dura
async def market_analyst(messages: list) -> list:
    """Market analysis agent with multiple tool calls."""
    agent = dura_agent(
        model="claude-sonnet-4-6",
        tools=[get_stock_price, analyze_sentiment, get_analyst_rating],
        system_prompt=(
            "You are a market analyst. Use the available tools to gather "
            "stock prices, analyst ratings, and sentiment analysis. "
            "Provide a comprehensive summary."
        ),
    )
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    if "--clean" in sys.argv:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("Cleaned up state files.")
        return

    crash_mode = "--crash" in sys.argv
    state = _load_state()
    is_resume = crash_mode and state.get("crash_triggered", False)

    print()
    print("=" * 64)
    print("  DuraLang Crash Recovery Demo")
    print("=" * 64)

    if is_resume:
        print()
        print("  RESUMING after crash.")
        print("  Temporal replays completed steps from event history.")
        print("  Completed LLM calls: NOT re-executed (replayed from history).")
        print("  Completed tool calls: NOT re-executed (replayed from history).")
        print("  Only the failed operation is retried.")
    elif crash_mode:
        print()
        print("  CRASH MODE: The worker will be killed mid-execution.")
        print("  Re-run the same command to see Temporal resume the workflow.")
    else:
        print()
        print("  RETRY MODE: get_stock_price will fail on first attempt.")
        print("  Temporal retries it automatically with backoff.")
        print("  The completed LLM call is NOT re-made.")

    print()
    print("-" * 64)

    prompt = (
        "Get the stock price and analyst rating for AAPL, "
        "then analyze market sentiment for Apple."
    )

    result = await market_analyst(
        [HumanMessage(content=prompt)],
        _workflow_id=WORKFLOW_ID if crash_mode else None,
    )

    print()
    print("-" * 64)
    print()
    print("FINAL ANSWER:")
    print(result[-1].content)
    print()

    # Clean up
    if STATE_FILE.exists():
        STATE_FILE.unlink()


if __name__ == "__main__":
    asyncio.run(main())
