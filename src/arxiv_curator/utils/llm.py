"""Call the course's AI Gateway model services with the workspace's own authentication.

The gateway serves its model services (such as `course_ops.gateway.chat`) through an
OpenAI-compatible API under `<workspace>/ai-gateway/mlflow/v1`. The bearer token comes from a
`WorkspaceClient`, so the same code works with a CLI profile's OAuth login and with a job's
identity, without creating personal access tokens.
"""

from databricks.sdk import WorkspaceClient
from openai import OpenAI
from openai.types.chat import ChatCompletion


def gateway_client(w: WorkspaceClient) -> OpenAI:
    """Build an OpenAI client for the workspace's AI Gateway.

    Args:
        w: An authenticated workspace client.

    Returns:
        A client whose `model` argument takes a gateway model service name.

    """
    token = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(base_url=f"{w.config.host.rstrip('/')}/ai-gateway/mlflow/v1", api_key=token)


def response_text(response: ChatCompletion) -> str:
    """Return the answer text of a chat completion, without any reasoning summary.

    Reasoning models such as GPT-OSS return the message content as a list of parts: a reasoning
    part with a summary, then one or more text parts. Other models return a plain string.

    Args:
        response: The chat completion.

    Returns:
        The concatenated text parts.

    """
    content = response.choices[0].message.content
    if isinstance(content, str) or content is None:
        return content or ""
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")
