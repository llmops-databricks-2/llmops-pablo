# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Call the course chat model through the AI Gateway, authenticated with WorkspaceClient."""

# MAGIC %md
# MAGIC # Model service experiment
# MAGIC
# MAGIC Modelled on the course's `1.1_foundation_models_overview.py`, but through the AI Gateway:
# MAGIC `WorkspaceClient` authenticates, `arxiv_curator.utils.llm.gateway_client` builds an
# MAGIC OpenAI-compatible client for `<workspace>/ai-gateway/mlflow/v1`, and the model is the gateway
# MAGIC model service named in `project_config.yml` (`llm_endpoint`).
# MAGIC
# MAGIC The prompt is synthetic on purpose: real lecture text waits until it is settled what the
# MAGIC gateway path retains.

# COMMAND ----------

# MAGIC %md
# MAGIC Install the package wheel that `databricks bundle deploy` uploaded next to the bundle files.
# MAGIC Update the version in the path when `version.txt` changes. Locally, these magic cells are
# MAGIC comments and the package comes from the uv environment.

# COMMAND ----------

# MAGIC %pip install ../../artifacts/.internal/arxiv_curator-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.utils.llm import gateway_client, response_text

# The environment comes from the env widget; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
logger.info(f"Authenticated to {w.config.host} as {w.current_user.me().user_name}")

client = gateway_client(w)
logger.info(f"Model service: {cfg.llm_endpoint}")

# COMMAND ----------
# One synthetic test prompt.

response = client.chat.completions.create(
    model=cfg.llm_endpoint,
    messages=[
        {"role": "system", "content": "You answer questions about a university lecture, briefly."},
        {"role": "user", "content": "In one sentence, what is a Delta table?"},
    ],
    reasoning_effort="low",
    max_tokens=200,
)

logger.info(f"Served model: {response.model}")
logger.info(f"Answer: {response_text(response)}")
logger.info(
    f"Tokens: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion"
    f" = {response.usage.total_tokens} total"
)
