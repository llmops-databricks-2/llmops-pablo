# Agent tools

The tools the lecture insights assistant could have, grouped by what each lets it do. The list is
kept short on purpose: a small, clearly named set makes it easier for the model to pick the right
tool, and easier to evaluate. See the [proposal](README.md) for the architecture these tools sit in.

## v1 tool set

The agent starts with three tools, chosen from the candidates below. Each one reads data that
already exists: the chunks in Lakebase and the Delta tables the prepare job writes. How the agent
calls them is in [ORCHESTRATION.md](ORCHESTRATION.md).

| Tool | Inputs | Outputs | Runs as | Kind, and why |
|---|---|---|---|---|
| `search_course_content` | `query`; optional `lecture_id` (such as `lecture-2`), `source` (`transcript`, `slides` or `materials`), `k` | Chunks with `lecture_id`, `source`, `page` or start and end offsets, an excerpt and a score | The agent's identity: the developer in dev, the serving endpoint's service principal once deployed. It needs a Postgres role that can read `lecture_chunks_synced` and permission to query `databricks-gte-large-en`. | **Custom function.** It embeds the query and runs pgvector SQL over a Postgres connection, which a UC function cannot do. It wraps the helpers in `arxiv_curator.utils.lecture_search`. |
| `get_transcript_segment` | `lecture_id`, `start_seconds`, `end_seconds` | Up to 100 transcript turns with `turn`, `offset_seconds` and `text`, without the speaker | The caller's UC grants: EXECUTE on the function, SELECT on `lecture_transcript`, and use of the course SQL warehouse. | **UC function through the managed MCP.** It is a plain SQL read of a Delta table, so UC grants govern it. It lets the agent read around a search hit before citing it. |
| Genie space | A question in plain language; a follow-up reuses the conversation ID | An answer, the SQL Genie ran, and the rows | The caller's identity: CAN RUN on the space, SELECT on its tables, and use of the course SQL warehouse. | **Managed MCP.** It answers open-ended counting and filtering questions over `lectures`, `lecture_chat` and `github_events`. |

**Left out of v1:**

- `list_questions`, `get_syllabus`, `get_coverage` and `get_pr_themes`: the tables they read
  (`questions`, `syllabus_topics`, `pr_activity`) do not exist yet.
- `get_slide`: search already returns the page number, and one fewer tool means one fewer wrong
  choice.
- The memory tools and the output tools: there is no app or digest for them to serve yet.
- Session history is not a tool. The agent loads it before the loop and saves it after.

**People stay out of Genie:** the space excludes `lecture_chat.sender`, `github_events.actor`,
`github_events.repo` (one repository per student) and `github_events.payload`, so Genie never sees a
name, and its instructions forbid listing or ranking people. `get_transcript_segment` leaves out the
speaker for the same reason.

## 1. Find things in course content (content chunks)

| Tool | What it does | Example use |
|---|---|---|
| `search_course_content(query, source?, lecture?, week?)` | Similarity search over slides, readings and transcripts in Lakebase pgvector. Returns chunks with citations: slide number, page, or lecture plus timestamp. | "When did I explain checkpointing?" |
| `get_transcript_segment(lecture_id, start, end)` | Returns the exact transcript between two timestamps, to read around a hit. | Checking whether a question was really answered at 00:34 |
| `get_slide(deck, slide_number)` | Returns a slide's full text and a link to the image. | Citing or quoting a specific slide in a recap |

The first one is the content-search tool in the diagram. The other two are cheap follow-ups
that let the agent look closer instead of searching again.

## 2. Ask questions about the LLM-derived data (course facts)

| Tool | What it does | Example use |
|---|---|---|
| **Genie space** (managed MCP) | Natural language to SQL over `lectures`, `questions`, `syllabus_topics`, `pr_activity` | "Which topics got the most unanswered questions?" |
| `list_questions(lecture_id?, topic?, status?)` | A fixed, parameterized query that returns extracted questions | Building the post-lecture digest |
| `get_syllabus(week?)` | Planned topics and learning objectives | "Did I cover week 3?" |
| `get_coverage(week)` | Planned topics compared with what was covered, asked about, and in PR reviews | The coverage map and the digest's coverage section |
| `get_pr_themes(assignment?, topic?)` | Review-comment themes, aggregated, never per student | "What kept coming up in assignment 2 reviews?" |

The fixed UC functions sit next to Genie because **the digest runs after every lecture and must be
reliable**. A fixed query gives the same correct answer every time. Genie is better for
open-ended questions the lecturer thinks of on the spot. Use Genie for exploring and functions for
the recurring workflows.

## 3. Remember the lecturer (memory)

| Tool | What it does |
|---|---|
| `get_lecturer_preferences()` / `save_lecturer_preference(key, value)` | Recap tone and length, topics to always highlight, how to show question authors |
| `mark_item_reviewed(item_id, action)` | Records that the lecturer dismissed, corrected or accepted a digest item. This is also the feedback signal used in Monitor. |

## 4. Produce outputs (actions)

| Tool | What it does | Guardrail |
|---|---|---|
| `save_draft_recap(lecture_id, text)` | Stores a recap draft for the lecturer to edit in the app | Draft only. The agent never posts to students. |
| `correct_question(question_id, field, value)` | Lets the lecturer fix an extracted fact, e.g. "this *was* answered" | Writes a correction; the original extraction is kept |

In v1, leave out anything that **sends** something: posting to Teams, emailing students,
commenting on PRs. The lecturer copies the recap out themselves. That avoids the agent acting on
the lecturer's behalf and keeps the privacy model simple.
