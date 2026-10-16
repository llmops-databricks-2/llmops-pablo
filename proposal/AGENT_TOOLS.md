# Agent tools

The tools the lecture insights assistant could have, grouped by what each lets it do. The list is
kept short on purpose: a small, clearly named set makes it easier for the model to pick the right
tool, and easier to evaluate. See the [proposal](README.md) for the architecture these tools sit in.

## 1. Find things in course content (content chunks)

| Tool | What it does | Example use |
|---|---|---|
| `search_course_content(query, source?, lecture?, week?)` | Similarity search over slides, readings and transcripts in Lakebase pgvector. Returns chunks with citations: slide number, page, or lecture plus timestamp. | "When did I explain checkpointing?" |
| `get_transcript_segment(lecture_id, start, end)` | Returns the exact transcript between two timestamps, to read around a hit. | Checking whether a question was really answered at 00:34 |
| `get_slide(deck, slide_number)` | Returns a slide's full text and a link to the image. | Citing or quoting a specific slide in a recap |

The first one is the content-search MCP tool in the diagram. The other two are cheap follow-ups
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
