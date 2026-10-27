# Databricks notebook source
# ruff: noqa: ERA001
# Databricks MAGIC comments are executable notebook cells.
"""Compare the lecture chunking strategy with a sentence-based alternative, in words and GTE tokens."""

# MAGIC %md
# MAGIC # Chunking comparison
# MAGIC
# MAGIC Builds the chunks of both strategies in memory from `lecture_transcript` and `lecture_slides`, and
# MAGIC compares their sizes in words and in tokens of the embedding model:
# MAGIC
# MAGIC | Strategy | Transcript pieces | Pages |
# MAGIC |---|---|---|
# MAGIC | `words` (the job's before CAU-2261) | whole turns, cut every 300 words when longer | one chunk each |
# MAGIC | `sentences` (the job's) | sentences, cut only when a sentence passes 300 words | one chunk each |
# MAGIC
# MAGIC Both pack pieces into windows of up to 300 words and carry a short last piece into the next window.
# MAGIC Token counts come from the tokenizer of `databricks-gte-large-en` (`Alibaba-NLP/gte-large-en-v1.5`),
# MAGIC whose input limit is 8,192 tokens. It also checks that figure descriptions reach `lecture_slides`.
# MAGIC
# MAGIC It writes nothing. Run the `lecture_insights_assistant` job first.

# COMMAND ----------

# MAGIC %md
# MAGIC Install the package wheel that `databricks bundle deploy` uploaded next to the bundle files, and the
# MAGIC tokenizer library. Update the version in the path when `version.txt` changes.

# COMMAND ----------

# MAGIC %pip install ../../artifacts/.internal/arxiv_curator-0.0.1-py3-none-any.whl tokenizers==0.23.3

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from collections import defaultdict
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import pandas as pd
from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.sql import functions as F
from tokenizers import Tokenizer

from arxiv_curator.config import get_env, load_config
from arxiv_curator.lecture_chunking import chunk_page, chunk_transcript, split_sentences, split_words

spark = DatabricksSession.builder.getOrCreate()

# The environment comes from the env widget; ENV or "dev" when run elsewhere.
try:
    from databricks.sdk.runtime import dbutils

    dbutils.widgets.text("env", "dev")
    env = dbutils.widgets.get("env")
except Exception:
    env = get_env()

cfg = load_config("../project_config.yml", env)
schema = cfg.full_schema_name

GTE_MAX_TOKENS = 8192
BGE_MAX_TOKENS = 512  # databricks-bge-large-en, shown for scale

# COMMAND ----------
# Build both strategies' chunks from the same tables. Pages are chunked the same way in both.

turns = defaultdict(list)
for row in spark.table(f"{schema}.lecture_transcript").collect():
    turns[row.lecture_id].append(row.asDict())
slides = spark.table(f"{schema}.lecture_slides").collect()
pages = [c for r in slides if (c := chunk_page(r.lecture_id, r.kind, r.page, r.text))]

strategies = {"words": split_words, "sentences": split_sentences}
chunks = {
    name: [c for lecture_id, rows in sorted(turns.items()) for c in chunk_transcript(lecture_id, rows, split=split)]
    + pages
    for name, split in strategies.items()
}

# COMMAND ----------
# Count tokens with GTE Large's own tokenizer. Its tokenizer.json pads and truncates to 512, which would cap
# every count, so both are switched off. Counts include the [CLS] and [SEP] tokens the model adds.

tokenizer = Tokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
tokenizer.no_padding()
tokenizer.no_truncation()

rows = []
for name, strategy_chunks in chunks.items():
    encodings = tokenizer.encode_batch([c["text"] for c in strategy_chunks])
    for c, enc in zip(strategy_chunks, encodings, strict=True):
        rows.append(
            {
                "strategy": name,
                "source": c["source"],
                "chunk_id": c["chunk_id"],
                "words": c["word_count"],
                "tokens": len(enc.ids),
                # A transcript chunk that does not end a sentence was cut mid-thought.
                "ends_sentence": c["text"].rstrip().endswith((".", "?", "!")),
                "text": c["text"],
            }
        )
df = pd.DataFrame(rows)

# COMMAND ----------
# Sizes per strategy and source, and how many chunks pass each model's input limit.

summary = (
    df.groupby(["strategy", "source"])
    .agg(
        chunks=("chunk_id", "count"),
        median_words=("words", "median"),
        max_words=("words", "max"),
        median_tokens=("tokens", "median"),
        max_tokens=("tokens", "max"),
        over_bge=("tokens", lambda t: int((t > BGE_MAX_TOKENS).sum())),
        over_gte=("tokens", lambda t: int((t > GTE_MAX_TOKENS).sum())),
        ends_sentence=("ends_sentence", "mean"),
    )
    .round(2)
)
logger.info(f"Chunks and sizes per strategy and source:\n{summary.to_string()}")

over_limit = df[df["tokens"] > GTE_MAX_TOKENS]
logger.info(f"Chunks over the GTE limit of {GTE_MAX_TOKENS} tokens: {len(over_limit)}")
assert over_limit.empty, over_limit[["strategy", "chunk_id", "tokens"]].to_string()

tokens_per_word = df.groupby("source").apply(lambda g: g["tokens"].sum() / g["words"].sum(), include_groups=False)
logger.info(f"Tokens per word:\n{tokens_per_word.round(2).to_string()}")

# COMMAND ----------
# Token distribution per source: the two strategies differ only on transcripts.

sources = sorted(df["source"].unique())
fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 3.5))
for ax, source in zip(axes, sources, strict=True):
    for name in strategies:
        ax.hist(df[(df["source"] == source) & (df["strategy"] == name)]["tokens"], bins=20, alpha=0.6, label=name)
    ax.set_title(source)
    ax.set_xlabel("GTE tokens per chunk")
axes[0].set_ylabel("chunks")
axes[0].legend()
fig.tight_layout()
plt.show()

# COMMAND ----------
# The same stretch of transcript in both strategies: the first chunk of lecture-2.

for name in strategies:
    first = df[(df["strategy"] == name) & (df["chunk_id"] == "lecture-2:transcript:0000")].iloc[0]
    logger.info(f"{name}: {first.words} words, {first.tokens} tokens, ends a sentence: {first.ends_sentence}")
    logger.info(f"  start: {first.text[:200]!r}")
    logger.info(f"  end:   {first.text[-200:]!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Figure descriptions
# MAGIC
# MAGIC `ai_parse_document` gives a figure without text an empty string as content. `lecture_slides` now falls
# MAGIC back to the figure's description, so every such description should appear in its page's text. This
# MAGIC cell parses lecture-2's slides again, which is billable.
# MAGIC
# MAGIC A model writes the descriptions, so a new parse can word them differently from the job's parse. Each
# MAGIC description is therefore matched to the most similar line of its page, not looked up exactly. Before
# MAGIC the fix that line was empty, which scores 0.

# COMMAND ----------

slides_path = next(r.source_path for r in slides if r.lecture_id == "lecture-2" and r.kind == "slides")
figures = (
    spark.read.format("binaryFile")
    .load(slides_path)
    .selectExpr("ai_parse_document(content, map('version', '2.0')) AS doc")
    .selectExpr("explode(variant_get(doc, '$.document.elements', 'ARRAY<VARIANT>')) AS el")
    .selectExpr(
        "variant_get(el, '$.bbox[0].page_id', 'INT') + 1 AS page",
        "variant_get(el, '$.type', 'STRING') AS type",
        "variant_get(el, '$.content', 'STRING') AS content",
        "variant_get(el, '$.description', 'STRING') AS description",
    )
    .where((F.col("type") == "figure") & (F.trim(F.col("content")) == "") & F.col("description").isNotNull())
    .collect()
)
page_lines = {r.page: r.text.split("\n") for r in slides if r.lecture_id == "lecture-2" and r.kind == "slides"}
MIN_SIMILARITY = 0.6

matches = []
for f in figures:
    score, line = max((SequenceMatcher(None, f.description, line).ratio(), line) for line in page_lines[f.page])
    matches.append(score >= MIN_SIMILARITY)
    logger.info(f"  page {f.page}: similarity {score:.2f}")
    logger.info(f"    parsed now: {f.description[:100]!r}")
    logger.info(f"    stored:     {line[:100]!r}")
logger.info(f"lecture-2 figures without text: {len(figures)}; descriptions in lecture_slides: {sum(matches)}")
assert all(matches), "Some figure descriptions are missing from lecture_slides"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Choice
# MAGIC
# MAGIC Observed on 2026-10-27 over kick-off, lecture-1 and lecture-2.
# MAGIC
# MAGIC | Transcript | `words` | `sentences` |
# MAGIC |---|---|---|
# MAGIC | Chunks | 177 | 158 |
# MAGIC | Median / max tokens | 369 / 421 | 371 / 419 |
# MAGIC | Chunks that end a sentence | 61% | 100% |
# MAGIC
# MAGIC **Keep `sentences`.** Both strategies give chunks of about 370 tokens, inside the 256 to 512 tokens the
# MAGIC course recommends for question answering. The difference is where they cut:
# MAGIC
# MAGIC - **`words` cuts mid-thought.** A long turn is cut every 300 words, so 39% of its chunks stop inside a
# MAGIC   sentence. It also leaves stubs: lecture-2's first chunk is 28 words of small talk, because the 2,217-word
# MAGIC   turn after it does not fit.
# MAGIC - **`sentences` costs a little overlap and accuracy.** It carries one sentence into the next window instead
# MAGIC   of a whole short turn, and its rule also breaks after abbreviations such as "e.g.".
# MAGIC
# MAGIC **Token limit.** No chunk of either strategy comes near GTE's 8,192 tokens: the largest is a slide page of
# MAGIC 870. Transcripts run at 1.29 tokens per word and pages at about 2.1, because of markup and symbols. Pages
# MAGIC are the same in both strategies, and 22 of 25 materials pages would pass BGE's 512-token limit, so the
# MAGIC model choice matters more for pages than for transcripts.
# MAGIC
# MAGIC **Figure descriptions.** All 5 lecture-2 figures without text now reach `lecture_slides`. Four are arrows,
# MAGIC so the fix adds some noise as well as the one diagram description.
# MAGIC
# MAGIC The share of page chunks that end a sentence is 0% because pages end in a page number; it only means
# MAGIC something for transcripts.
