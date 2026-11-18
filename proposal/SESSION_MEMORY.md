# Session memory

How the agent keeps a lecturer's conversation between turns, and who can read it. The steps around it are
in [ORCHESTRATION.md](ORCHESTRATION.md); the code is `arxiv_curator.utils.session_memory` and
`agent_loop.run_turn`.

## What is stored

Two tables in the project's Lakebase, in the Postgres schema named after the Unity Catalog schema:

| Table | Columns | Key |
|---|---|---|
| `sessions` | `session_id`, `owner`, `created_at` | `session_id` |
| `session_messages` | `session_id`, `seq`, `message` (JSONB), `created_at` | `(session_id, seq)` |

A turn stores its question, tool calls with their IDs, tool results and the answer, so a reloaded
conversation is valid input for the model.

- **Order:** `seq` is the message's position in the session, and loads order by it, not by timestamp.
- **No duplicates:** a turn writes from `seq = len(history)`, and a position already taken is skipped. A
  retried save writes nothing twice.
- **Index:** the primary key's index leads with `session_id`, so loading one session uses it.

## How session access is protected

1. **Connecting needs a role in the Lakebase project.** Roles exist only for the developer, their dev, acc
   and prd deploy service principals, and the course's onboarding principal. Other students have none and
   cannot connect. No password is stored: the SDK generates a short-lived credential for the caller's role.
2. **Postgres grants.** Every role above belongs to `databricks_superuser`, which has full rights on the
   tables. Lakebase's defaults also give SELECT to `databricks_reader_16406`, which has no members. So today
   the database keeps other students out, not the developer's own identities.
3. **The owner check keeps sessions apart.** One database role serves every lecturer, so Postgres cannot tell
   them apart; the code does. The first turn claims the session for its caller. A later turn from anyone else
   raises `PermissionError` before the history loads or the model runs. A session ID alone grants nothing.

The caller's name must come from the authenticated request, never from the request body, where anyone could
type someone else's name.

## Which identity the runtime uses

| Where | Identity | Postgres role |
|---|---|---|
| dev, the check script | The developer, through the CLI profile | The developer's own role, which owns the tables |
| Deployed endpoint (later) | `sp-deploy-pablo-dev`, the course-provisioned service principal | Its own role, a `databricks_superuser` member, so it can already read and write |

Its client ID and secret are UC secrets in the project schema.

## Limitations

- **Unbounded history:** history is not trimmed or summarised, and tool results are stored in full, so long
  sessions grow the prompt and its cost.
- **Concurrent turns:** two turns at once in one session keep only the first one's messages, and the save
  logs a warning.
- **Broad runtime rights:** the service principal can also update and delete history. A dedicated role with
  only USAGE, SELECT and INSERT would close that. The check runs as the developer, not the service principal.
- **No retention rule:** nothing deletes old sessions.
