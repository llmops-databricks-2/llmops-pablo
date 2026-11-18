# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Compare semantic, filtered and hybrid search over the Lakebase lecture chunks on one query set."""

# MAGIC %md
# MAGIC # Search comparison
# MAGIC
# MAGIC Runs one fixed set of lecturer-style questions in three modes over `lecture_chunks_synced` in Lakebase,
# MAGIC all in pgvector SQL on the self-managed embeddings:
# MAGIC
# MAGIC | Mode | Ranking |
# MAGIC |---|---|
# MAGIC | semantic | cosine similarity to the question's embedding |
# MAGIC | filtered | the same, within a `lecture_id` and/or `source` |
# MAGIC | hybrid | 0.7 × cosine similarity + 0.3 × `ts_rank_cd` full-text rank |
# MAGIC
# MAGIC Each question is embedded once with `databricks-gte-large-en`, the model that embedded the chunks. The
# MAGIC questions are written for this comparison; no lecture text is sent. It writes nothing. Run the
# MAGIC `lecture_insights_assistant` job first, so the synced table is up to date.

# COMMAND ----------

# MAGIC %md
# MAGIC Install the package wheel that `databricks bundle deploy` uploaded next to the bundle files. Update the
# MAGIC version in the path when `version.txt` changes.

# COMMAND ----------

# MAGIC %pip install ../../artifacts/.internal/arxiv_curator-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
from databricks.sdk import WorkspaceClient
from loguru import logger

from arxiv_curator.config import get_env, load_config
from arxiv_curator.utils.lakebase import connect
from arxiv_curator.utils.lecture_search import embed_query, filtered_search, hybrid_search, semantic_search

# The environment comes from the env widget; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()
K = 5

# COMMAND ----------
# The query set: two exact terms, two conceptual questions, two constrained ones and one the lectures cannot
# answer. Each carries the filter that filtered mode applies.

QUERIES = [
    {"type": "exact term", "query": "ai_parse_document", "filter": {"lecture_id": "lecture-2"}},
    {"type": "exact term", "query": "provisioned throughput", "filter": {"lecture_id": "lecture-1"}},
    {
        "type": "conceptual",
        "query": "Why can adding more context make answers worse?",
        "filter": {"source": "transcript"},
    },
    {
        "type": "conceptual, other words",
        "query": "How do I log in to Databricks from my laptop?",
        "filter": {"lecture_id": "lecture-1"},
    },
    {
        "type": "constrained",
        "query": "What did we say about guardrails?",
        "filter": {"lecture_id": "lecture-1", "source": "transcript"},
    },
    {
        "type": "constrained",
        "query": "How should chunks be checked before indexing?",
        "filter": {"source": "slides"},
    },
    {
        "type": "not in the lectures",
        "query": "How do I deploy the model to Kubernetes with Helm?",
        "filter": {"source": "transcript"},
    },
]

# COMMAND ----------
# Run every question in the three modes on one connection.

results = []
with connect(cfg, w) as conn:
    for q in QUERIES:
        vector = embed_query(w, cfg.embedding_endpoint, q["query"])
        results.append(
            {
                **q,
                "semantic": semantic_search(conn, cfg, vector, K),
                "filtered": filtered_search(conn, cfg, vector, K, **q["filter"]),
                "hybrid": hybrid_search(conn, cfg, vector, q["query"], K),
            }
        )
modes = ["semantic", "filtered", "hybrid"]

# COMMAND ----------
# Top results of each mode side by side, with chunk ids and scores.

for r in results:
    rows = []
    for rank in range(K):
        row = {"rank": rank + 1}
        for mode in modes:
            hit = r[mode][rank] if rank < len(r[mode]) else None
            row[mode] = f"{hit['chunk_id']} ({hit['score']:.3f})" if hit else ""
        hybrid_hit = r["hybrid"][rank] if rank < len(r["hybrid"]) else None
        row["hybrid text score"] = round(float(hybrid_hit["text_score"]), 3) if hybrid_hit else None
        rows.append(row)
    logger.info(f"[{r['type']}] {r['query']!r}  filter={r['filter']}\n{pd.DataFrame(rows).to_string(index=False)}")
    for mode in modes:
        if r[mode]:
            logger.info(f"  top {mode}: {r[mode][0]['excerpt'][:160]!r}")

# COMMAND ----------
# Per question: the top score of each mode, how many of semantic search's top results each other mode keeps,
# and how often full-text matched in hybrid mode.

summary = []
for r in results:
    ids = {mode: [h["chunk_id"] for h in r[mode]] for mode in modes}
    summary.append(
        {
            "type": r["type"],
            "query": r["query"][:40],
            "top semantic": round(float(r["semantic"][0]["score"]), 3),
            "top filtered": round(float(r["filtered"][0]["score"]), 3) if r["filtered"] else None,
            "top hybrid": round(float(r["hybrid"][0]["score"]), 3),
            "filtered keeps": len(set(ids["filtered"]) & set(ids["semantic"])),
            "hybrid keeps": len(set(ids["hybrid"]) & set(ids["semantic"])),
            "hybrid top 1 changed": ids["hybrid"][0] != ids["semantic"][0],
            "hybrid rows with text match": sum(float(h["text_score"]) > 0 for h in r["hybrid"]),
            "max text score": round(max(float(h["text_score"]) for h in r["hybrid"]), 3),
        }
    )
logger.info(f"Summary over {len(results)} questions:\n{pd.DataFrame(summary).to_string(index=False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What changed between the modes
# MAGIC
# MAGIC Observed on 2026-10-28 over 272 chunks of kick-off, lecture-1 and lecture-2, top 5 per mode.
# MAGIC
# MAGIC | Question | Semantic top 1 | Filtered keeps | Hybrid keeps | Hybrid text match |
# MAGIC |---|---|---|---|---|
# MAGIC | ai_parse_document | 0.790 | 5 of 5 | 3 of 5, new top 1 | 5 rows, up to 0.70 |
# MAGIC | provisioned throughput | 0.754 | 5 of 5 | 3 of 5 | 5 rows, up to 0.31 |
# MAGIC | More context, worse answers | 0.782 | 2 of 5 | 5 of 5 | none |
# MAGIC | Log in from my laptop | 0.791 | 4 of 5 | 5 of 5 | none |
# MAGIC | Guardrails | 0.568 | 2 of 5 | 5 of 5 | none |
# MAGIC | Check chunks before indexing | 0.794 | 3 of 5 | 5 of 5 | 3 rows, up to 0.003 |
# MAGIC | Kubernetes with Helm (not taught) | 0.644 | 3 of 5 | 5 of 5 | none |
# MAGIC
# MAGIC - **Semantic search finds the right material, slides first.** Every answerable question's top hit is the
# MAGIC   slide or materials page on that topic, such as the CLI login slide for "log in from my laptop". Slides
# MAGIC   win because they are short and on one topic; transcript chunks mix several.
# MAGIC - **Filtered search changes the evidence only when the filter cuts.** A `lecture_id` filter for a
# MAGIC   lecture that semantic search already picked changes nothing (5 of 5 kept). A `source = 'transcript'`
# MAGIC   filter replaces slides with what was said, such as the spoken discussion of rate limits and fallback
# MAGIC   next to the guardrails slide. That is the evidence a "what did we say" question needs.
# MAGIC - **Hybrid search helps exact terms only.** For `ai_parse_document` it promotes the slide about the
# MAGIC   function itself over a table that lists it, and brings in transcript chunks that name it. For full
# MAGIC   questions the text score is 0: `plainto_tsquery` requires every word of the question in the chunk, so a
# MAGIC   sentence never matches and hybrid ranks exactly like semantic, scaled by 0.7.
# MAGIC - **No score separates "not in the lectures".** The Kubernetes question still returns 5 chunks at
# MAGIC   0.644, above the real guardrails question (0.568). A score threshold would drop real answers before it
# MAGIC   dropped this one.
# MAGIC - **Boilerplate leaks into the evidence.** Materials pages start with `CAUCHY · LLMOPS Lecture 1 · October
# MAGIC   2026`, and slides carry HTML table markup, as notebook 3.1 found.
# MAGIC
# MAGIC **For the agent:** use semantic search by default, add a `source` or `lecture_id` filter when the question
# MAGIC asks for it, and use hybrid search for short exact terms. Matching any of the question's words, for
# MAGIC example with `websearch_to_tsquery` and `OR`, would let hybrid search help full questions too; that is a
# MAGIC follow-up, not part of this comparison.
