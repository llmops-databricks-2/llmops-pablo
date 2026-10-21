# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Run the lecture assistant's core LLM tasks through the gateway and record how the model does."""

# MAGIC %md
# MAGIC # Use-case prompts
# MAGIC
# MAGIC The facts path of the proposal runs an LLM over each lecture to extract questions, decide
# MAGIC whether they were answered, tag them to syllabus topics and spot confusion. This notebook
# MAGIC tries those tasks on `course_ops.gateway.chat` and records answers, token usage and latency.
# MAGIC
# MAGIC The lecture excerpt is **synthetic**: made-up names, in the format of the real Teams
# MAGIC transcripts and chats, with planted cases whose right answers are known (see `EXPECTED`).
# MAGIC Real lecture text waits until platform-side retention is confirmed.

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

import json
import time

from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.utils.llm import gateway_client, response_text

try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)
client = gateway_client(WorkspaceClient())

# COMMAND ----------
# The synthetic lecture: transcript turns (offset from the recording start) and chat messages
# (local clock time; the recording starts at 16:00).

TRANSCRIPT = """\
Ada Example   0:12
Good afternoon everyone. Today we continue with delta lick tables and then look at jobs.
Ada Example   2:05
A Delta table is Parquet files plus a transaction log, the _delta_log folder next to the data.
Bo Sample   4:40
Sorry, can you repeat what the transaction log stores?
Ada Example   4:52
Sure. Every commit is a JSON file in the log that says which data files were added or removed.
Ada Example   9:30
Time travel lets you query an older version of a table with VERSION AS OF.
Cy Test   12:10
Does time travel keep old versions forever, or is there a limit?
Ada Example   12:25
Good question. It depends on the retention settings, I'll come back to that another time.
Ada Example   20:00
Next topic: jobs. A job should be I dem potent, so running it twice gives the same result.
Dee Mock   23:45
How do we make a MERGE idempotent when the source has duplicate keys?
Ada Example   23:58
Let's park that, we're short on time. On to Unity Catalog permissions.
Ada Example   30:15
In Unity Catalog you grant privileges on catalogs, schemas and tables.
Eli Fake   33:40
Do I need USE SCHEMA if I already have SELECT on the table?
Ada Example   33:55
Yes. You need USE CATALOG and USE SCHEMA as well as SELECT to read a table.
Bo Sample   35:10
Can you repeat the part about USE CATALOG?
Ada Example   35:20
USE CATALOG on the catalog, USE SCHEMA on the schema, then SELECT on the table.
"""

CHAT = """\
Cy Test   16:05
lol my cat just walked over the keyboard
Dee Mock   16:13
where exactly is the _delta_log folder stored?
Eli Fake   16:33
do I need USE SCHEMA when I have SELECT?
Bo Sample   16:34
+1 same question
Cy Test   16:41
the audio cut out for me during the MERGE part
Dee Mock   16:42
brb coffee
"""

SYLLABUS = [
    "Delta tables and the transaction log",
    "Time travel and retention",
    "Idempotent jobs and MERGE",
    "Unity Catalog privileges",
    "Course logistics",
]

# The planted cases: what a good answer contains.
EXPECTED = {
    "questions": [
        "Q1 voice 4:40 Bo: what the transaction log stores — answered at 4:52 — Delta tables",
        "Q2 voice 12:10 Cy: time travel retention limit — partly answered at 12:25 — Time travel",
        "Q3 voice 23:45 Dee: idempotent MERGE with duplicate keys — not answered — Idempotent jobs",
        "Q4 chat 16:33 Eli (+1 Bo) and voice 33:40 Eli: USE SCHEMA with SELECT — the same question"
        " twice — answered at 33:55 — Unity Catalog",
        "Q5 chat 16:13 Dee: where _delta_log is stored — not answered (2:05 says 'next to the data'"
        " before it was asked) — Delta tables",
        "Q6 voice 35:10 Bo: repeat USE CATALOG — answered at 35:20 — Unity Catalog",
    ],
    "not_questions": ["chat 16:05 cat", "chat 16:42 coffee", "chat 16:41 audio (a confusion signal)"],
    "confusion": ["4:40 repeat request", "35:10 repeat request", "Q4 asked twice plus a +1", "16:41 audio cut"],
    "mangled_terms": ["delta lick = Delta Lake", "I dem potent = idempotent"],
}

LECTURE = f"TRANSCRIPT (speaker, offset from recording start 16:00):\n{TRANSCRIPT}\nCHAT (local time):\n{CHAT}"

# COMMAND ----------


def run(name: str, prompt: str, effort: str = "low", json_mode: bool = False, max_tokens: int = 2000) -> dict:
    """Send one prompt about the lecture and record the answer, usage and latency.

    Args:
        name: A label for the run.
        prompt: The task instruction; the lecture text is appended.
        effort: The model's reasoning effort.
        json_mode: Whether to request a JSON object response.
        max_tokens: The completion token limit.

    Returns:
        The run's label, answer text, token counts, latency in seconds and, in JSON mode,
        whether the answer parsed as JSON.

    """
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[
            {"role": "system", "content": "You analyse university lecture transcripts and chats for the lecturer."},
            {"role": "user", "content": f"{prompt}\n\n{LECTURE}"},
        ],
        reasoning_effort=effort,
        max_tokens=max_tokens,
        **extra,
    )
    result = {
        "name": name,
        "effort": effort,
        "text": response_text(response),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "seconds": round(time.perf_counter() - start, 1),
    }
    if json_mode:
        try:
            json.loads(result["text"])
            result["json_ok"] = True
        except json.JSONDecodeError:
            result["json_ok"] = False
    logger.info(f"{name} [{effort}]: {result['total_tokens']} tokens, {result['seconds']} s")
    return result


PROMPTS = {
    "1_extract": (
        "List every question students asked, in the transcript or the chat. For each, give: the "
        "question in one line, the source (voice or chat), the time, and who asked. Merge the same "
        "question asked twice into one item and say so. Do not list statements or off-topic chat."
    ),
    "2_answered": (
        "For each question students asked, decide whether the lecturer answered it: 'answered', "
        "'partly answered' or 'not answered'. Give the time of the answer, or say there is none. "
        "Be strict: a promise to come back later is not an answer."
    ),
    "3_topics": (
        "Tag each question students asked with exactly one topic from this syllabus list: "
        + "; ".join(SYLLABUS)
        + ". Answer as 'question → topic'."
    ),
    "4_confusion": (
        "List the moments that suggest students were confused or missed something: repeat "
        "requests, the same question asked more than once, technical problems. Give the time of each."
    ),
}

JSON_PROMPT = (
    "Return a JSON object with one key, 'questions': a list of objects with keys 'question' "
    "(string), 'source' ('voice' or 'chat'), 'time' (string), 'asked_by' (string), 'status' "
    "('answered', 'partly answered' or 'not answered'), 'answer_time' (string or null) and "
    "'topic' (one of: " + "; ".join(SYLLABUS) + "). Merge duplicates. Return only the JSON."
)

# COMMAND ----------
# The five use-case prompts.

results = [run(name, prompt) for name, prompt in PROMPTS.items()]
results.append(run("5_json", JSON_PROMPT, json_mode=True))

# COMMAND ----------
# Probes: repeatability of the answered labels, and a higher reasoning effort for extraction.

results += [run(f"2_answered_repeat{i}", PROMPTS["2_answered"]) for i in (2, 3)]
results.append(run("1_extract", PROMPTS["1_extract"], effort="medium"))

# COMMAND ----------
# Probe: a lecture-sized input (the real lecture-1 transcript plus chat is about 36k tokens).
# The synthetic excerpt is repeated with shifted labels, so only the first copy's questions count.

copies = [LECTURE.replace("Sample", f"Sample{i}") if i else LECTURE for i in range(60)]
big = "\n\n".join(copies)
start = time.perf_counter()
response = client.chat.completions.create(
    model=cfg.llm_endpoint,
    messages=[{"role": "user", "content": f"How many questions did 'Bo Sample' (exactly that name) ask?\n\n{big}"}],
    reasoning_effort="low",
    max_tokens=500,
)
results.append(
    {
        "name": "6_long_context",
        "effort": "low",
        "text": response_text(response),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "seconds": round(time.perf_counter() - start, 1),
    }
)

# COMMAND ----------
# Results: the answers, then a summary table.

for r in results:
    print(f"\n===== {r['name']} [{r['effort']}] =====\n{r['text']}")

print("\nrun | effort | prompt | completion | total | seconds | json")
for r in results:
    print(
        f"{r['name']} | {r['effort']} | {r['prompt_tokens']} | {r['completion_tokens']} | "
        f"{r['total_tokens']} | {r['seconds']} | {r.get('json_ok', '')}"
    )
