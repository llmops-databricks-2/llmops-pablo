"""Expose Databricks managed MCP tools to the agent: the Genie space and the schema's UC functions.

Adapted from the course's `mcp.py` and notebook 3.2. `mcp_tools` lists a server's tools and wraps each as a
`ToolInfo` whose spec is the tool's own input schema. For the Genie space only the query tool is offered to the
model: when Genie has not finished, code polls that same message with the poll tool, every `POLL_SECONDS` for
at most `POLL_LIMIT_SECONDS`, and never resubmits the question. MCP calls run as the workspace client's
identity: the developer in dev, the serving endpoint's service principal once deployed.
"""

import json
import time
from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger

from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.agent_tools import ToolInfo

POLL_SECONDS = 5
POLL_LIMIT_SECONDS = 60
MAX_ROWS = 50
# Genie message statuses after which polling stops; any other status means Genie is still working.
GENIE_DONE = {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}


def genie_mcp_url(w: WorkspaceClient, space_id: str) -> str:
    """Return the managed MCP server URL of a Genie space.

    Args:
        w: The workspace client.
        space_id: The Genie space ID.

    Returns:
        `{host}/api/2.0/mcp/genie/{space_id}`.

    """
    return f"{w.config.host.rstrip('/')}/api/2.0/mcp/genie/{space_id}"


def functions_mcp_url(w: WorkspaceClient, cfg: ProjectConfig) -> str:
    """Return the managed MCP server URL of the schema's UC functions.

    Args:
        w: The workspace client.
        cfg: The project configuration for the target environment.

    Returns:
        `{host}/api/2.0/mcp/functions/{catalog}/{schema}`.

    """
    return f"{w.config.host.rstrip('/')}/api/2.0/mcp/functions/{cfg.catalog}/{cfg.db_schema}"


def _text(response: Any) -> str:  # noqa: ANN401 - an MCP CallToolResult
    """Join the text parts of an MCP tool result.

    Args:
        response: The result of `call_tool`.

    Returns:
        The concatenated text.

    """
    return "".join(getattr(part, "text", "") for part in response.content)


def _parse(text: str) -> Any:  # noqa: ANN401 - whatever JSON the server returned
    """Parse a tool result as JSON when it is JSON.

    Args:
        text: The tool's text result.

    Returns:
        The parsed value, or the text unchanged.

    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def condense_genie(result: dict[str, Any]) -> dict[str, Any]:
    """Keep what the model needs from a Genie message: status, IDs, text, SQL and the first rows.

    Args:
        result: The parsed result of the Genie query or poll tool.

    Returns:
        The status, `conversation_id`, `message_id`, any text answer, and for a query its SQL, description,
        column names and at most `MAX_ROWS` rows.

    """
    content = result.get("content") or {}
    condensed: dict[str, Any] = {
        "status": result.get("status"),
        "conversation_id": result.get("conversationId"),
        "message_id": result.get("messageId"),
    }
    # Text attachments come as plain strings, or as objects with a `content` field.
    texts = [t if isinstance(t, str) else t.get("content") for t in content.get("textAttachments") or []]
    texts = [t for t in texts if t]
    if texts:
        condensed["text"] = texts
    for attachment in content.get("queryAttachments") or []:
        statement = attachment.get("statement_response") or {}
        columns = (statement.get("manifest") or {}).get("schema", {}).get("columns") or []
        rows = (statement.get("result") or {}).get("data_array") or []
        condensed["sql"] = attachment.get("query")
        condensed["description"] = attachment.get("description")
        condensed["columns"] = [c.get("name") for c in columns]
        condensed["rows"] = rows[:MAX_ROWS]
        condensed["row_count"] = len(rows)
    return condensed


def make_genie_exec(
    call_tool: Callable[[str, dict[str, Any]], str],
    query_tool: str,
    poll_tool: str,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Callable[..., dict[str, Any]]:
    """Build the Genie tool's callable: ask once, then poll that message with a time bound.

    Args:
        call_tool: Calls an MCP tool by name with arguments and returns its text result.
        query_tool: The Genie query tool's name, `query_space_<id>`.
        poll_tool: The Genie poll tool's name, `poll_response_<id>`.
        sleep: Waits between polls; replaced in tests.
        clock: Returns a monotonic time in seconds; replaced in tests.

    Returns:
        A callable taking `query` and an optional `conversation_id`, returning the condensed Genie message.

    Raises:
        TimeoutError: Through the callable, when Genie is still working after `POLL_LIMIT_SECONDS`; the registry
            turns it into an error result.

    """

    def ask_genie(query: str, conversation_id: str | None = None) -> dict[str, Any]:
        """Ask Genie a question, continuing a conversation when its ID is given.

        Args:
            query: The question.
            conversation_id: The conversation to continue, for a follow-up.

        Returns:
            The condensed Genie message.

        Raises:
            TimeoutError: If Genie is still working after `POLL_LIMIT_SECONDS`.

        """
        args = {"query": query} | ({"conversation_id": conversation_id} if conversation_id else {})
        result = _parse(call_tool(query_tool, args))
        deadline = clock() + POLL_LIMIT_SECONDS
        while result.get("status") not in GENIE_DONE:
            if clock() >= deadline:
                raise TimeoutError(
                    f"Genie still pending after {POLL_LIMIT_SECONDS} s (status {result.get('status')}, "
                    f"conversation {result.get('conversationId')}, message {result.get('messageId')})"
                )
            logger.info(f"Genie status {result.get('status')}; polling message {result.get('messageId')}")
            sleep(POLL_SECONDS)
            ids = {"conversation_id": result["conversationId"], "message_id": result["messageId"]}
            result = _parse(call_tool(poll_tool, ids))
        return condense_genie(result)

    return ask_genie


def mcp_tools(w: WorkspaceClient, url: str, names: set[str] | None = None) -> list[ToolInfo]:
    """Wrap a managed MCP server's tools as agent tools.

    A Genie server's query tool gets the polling callable from `make_genie_exec`, and its poll tool is not
    offered to the model. Other tools return their parsed result.

    Args:
        w: The workspace client whose identity calls the tools.
        url: The MCP server URL.
        names: Keep only these tool names; all of them when omitted.

    Returns:
        One `ToolInfo` per kept tool.

    """
    client = DatabricksMCPClient(server_url=url, workspace_client=w)

    def call_tool(name: str, args: dict[str, Any]) -> str:
        """Call a tool on this server.

        Args:
            name: The tool name.
            args: Its arguments.

        Returns:
            The tool's text result.

        """
        try:
            return _text(client.call_tool(name, args))
        except ExceptionGroup as group:
            # The MCP client wraps failures in nested task groups; report the innermost error instead.
            error: BaseException = group
            while isinstance(error, BaseExceptionGroup):
                error = error.exceptions[0]
            raise error from group

    listed = client.list_tools()
    poll_tools = {t.name for t in listed if t.name.startswith("poll_response_")}
    tools = []
    for tool in listed:
        if tool.name in poll_tools or (names is not None and tool.name not in names):
            continue
        spec = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        }
        if tool.name.startswith("query_space_"):
            poll_tool = tool.name.replace("query_space_", "poll_response_", 1)
            exec_fn = make_genie_exec(call_tool, tool.name, poll_tool)
        else:

            def exec_fn(_name: str = tool.name, **kwargs: Any) -> Any:  # noqa: ANN401
                """Call the MCP tool and parse its result.

                Args:
                    _name: The tool name, bound at definition.
                    **kwargs: The tool's arguments.

                Returns:
                    The parsed result.

                """
                return _parse(call_tool(_name, kwargs))

        tools.append(ToolInfo(name=tool.name, spec=spec, exec_fn=exec_fn))
    return tools
