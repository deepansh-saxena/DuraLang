"""DuraLang CLI — worker management."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="duralang",
        description="DuraLang — Durable LangChain agents powered by Temporal",
    )
    subparsers = parser.add_subparsers(dest="command")

    # duralang worker start
    worker_parser = subparsers.add_parser("worker", help="Worker management")
    worker_sub = worker_parser.add_subparsers(dest="worker_command")

    start_parser = worker_sub.add_parser("start", help="Start a DuraLang worker")
    start_parser.add_argument(
        "--host", default="localhost:7233", help="Temporal server address"
    )
    start_parser.add_argument(
        "--namespace", default="default", help="Temporal namespace"
    )
    start_parser.add_argument(
        "--task-queue", default="duralang", help="Worker task queue"
    )

    args = parser.parse_args()

    if args.command == "worker" and getattr(args, "worker_command", None) == "start":
        asyncio.run(_start_worker(args.host, args.namespace, args.task_queue))
    else:
        parser.print_help()
        sys.exit(1)


async def _start_worker(host: str, namespace: str, task_queue: str):
    """Start a standalone DuraLang worker."""
    from temporalio.client import Client
    from temporalio.worker import Worker

    from duralang.activities import llm_activity, mcp_activity, tool_activity
    from duralang.workflow import DuraLangWorkflow

    print(f"Connecting to Temporal at {host} (namespace: {namespace})")
    client = await Client.connect(host, namespace=namespace)

    print(f"Starting worker on task queue: {task_queue}")
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[DuraLangWorkflow],
        activities=[llm_activity, tool_activity, mcp_activity],
    )

    shutdown_event = asyncio.Event()

    def _request_shutdown():
        print("\nShutting down worker...")
        shutdown_event.set()

    # Install signal handlers for clean shutdown
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _request_shutdown)

    print("Worker running. Press Ctrl+C to stop.")
    try:
        async with worker:
            await shutdown_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # Clean exit on Ctrl+C


if __name__ == "__main__":
    main()
