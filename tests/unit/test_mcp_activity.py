"""Tests for forge__mcp activity with mocked MCP session."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langforge.activities.mcp import mcp_activity
from langforge.exceptions import MCPActivityError
from langforge.graph_def import MCPActivityPayload
from langforge.registry import MCPSessionRegistry


@pytest.fixture(autouse=True)
def clear_registry():
    MCPSessionRegistry.clear()
    yield
    MCPSessionRegistry.clear()


@pytest.fixture
def mock_heartbeat():
    with patch("langforge.activities.mcp.activity") as mock_activity:
        mock_activity.heartbeat = MagicMock()
        yield mock_activity


class TestMCPActivity:
    @pytest.mark.asyncio
    async def test_successful_mcp_call(self, mock_heartbeat):
        # Mock MCP session and result
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "file contents here"}

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        MCPSessionRegistry.register("filesystem", mock_session)

        payload = MCPActivityPayload(
            server_name="filesystem",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            tool_call_id="tc_mcp_1",
        )

        result = await mcp_activity(payload)

        assert result.tool_call_id == "tc_mcp_1"
        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        mock_session.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})

    @pytest.mark.asyncio
    async def test_mcp_server_not_registered(self, mock_heartbeat):
        payload = MCPActivityPayload(
            server_name="nonexistent",
            tool_name="some_tool",
            arguments={},
            tool_call_id="tc_1",
        )

        with pytest.raises(MCPActivityError, match="not registered"):
            await mcp_activity(payload)

    @pytest.mark.asyncio
    async def test_mcp_error_result(self, mock_heartbeat):
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "permission denied"}

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = True

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        MCPSessionRegistry.register("fs", mock_session)

        payload = MCPActivityPayload(
            server_name="fs",
            tool_name="write_file",
            arguments={"path": "/root/secret"},
            tool_call_id="tc_err",
        )

        result = await mcp_activity(payload)
        assert result.is_error is True
        assert result.tool_call_id == "tc_err"

    @pytest.mark.asyncio
    async def test_tool_call_id_preserved(self, mock_heartbeat):
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "ok"}

        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_result.isError = False

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        MCPSessionRegistry.register("test_server", mock_session)

        payload = MCPActivityPayload(
            server_name="test_server",
            tool_name="ping",
            arguments={},
            tool_call_id="preserve_this_id",
        )

        result = await mcp_activity(payload)
        assert result.tool_call_id == "preserve_this_id"
