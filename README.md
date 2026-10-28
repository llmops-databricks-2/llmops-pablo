# llmops-pablo

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Your schema in development, which you manage | `llmops_dev.pablo` |
| Your schema in acceptance, which you read | `llmops_acc.pablo` |
| Your schema in production, which you read | `llmops_prd.pablo` |
| Lecture files volume | `llmops_dev.pablo.lecture_files`, defined in the bundle for every target (acc and prd get theirs when those targets are deployed) |
| SQL warehouse id | `8d8d38f86ea6cc4a` |
| Usage policy id | `f91fbdfb-b1c9-39a9-9d93-d132a1e63daa` |
| MLflow experiments | Dev: `/llmops/pablo-dev`; acc: `/llmops/pablo-acc`; prd: `/llmops/pablo-prd` |
| Chat model service | `course_ops.gateway.chat` |
| Embedding model service | `course_ops.gateway.embed` |

Both settings name model services of the course’s AI Gateway, with a monthly budget per student.
Chat routes to GPT-OSS-120B; embeddings route to GTE Large. Call both through the gateway
with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(
    model="course_ops.gateway.chat",
    messages=[...],
    reasoning_effort="low",
    max_tokens=1024,
)
client.embeddings.create(model="course_ops.gateway.embed", input="A paper abstract")
```

The `databricks-openai` package is not included in this repo’s dependencies. The same APIs
are available through the installed `openai` client with the workspace base URL
`/ai-gateway/mlflow/v1` and Databricks authentication.

`embedding_endpoint` is the `databricks-gte-large-en` serving endpoint, not a gateway model service:
`ai_query` cannot call custom gateway model services yet. The chunking job embeds every chunk with it
(1,024 floats), and queries must be embedded with the same endpoint before they are searched with
pgvector in Lakebase.

## Lakebase

The `lakebase_project_id` value in `project_config.yml` identifies your Lakebase project.
All three environments share it; your Unity Catalog schemas remain separate.
Onboarding does not create an application database or install pgvector. The lecture job's
`lecture_sync` task enables the pgvector `vector` extension, syncs `lecture_chunks` into the
read-only synced table `lecture_chunks_synced` (Postgres schema `pablo`, database
`databricks_postgres`) with `embedding` as `vector(1024)`, and adds an HNSW cosine index and a GIN
full-text index. Search runs in Postgres on these self-managed embeddings; no Vector Search endpoint
is used. Only dev syncs for now: acc and prd share the project and need their own database first.

For a dedicated student project, you and your three deployment service principals have project
management access and OAuth Postgres roles with database administration privileges. The project
carries your usage policy and student cost tag. Its initial compute range is 0.5–2 CU, with
suspension after five idle minutes. Creating extra branches or computes can increase its cost.

## What is in the repo

- `src/arxiv_curator/`: starts with `config.py`, which loads `project_config.yml`.
  Add the other course modules as you work through the lectures.
- `lecture_1/`, `lecture_2/` and so on, when you add them: copy a lecture's folder from the course
  code as it is, to the root of this repo, where its notebooks find `project_config.yml` one level up.
  Commit it with the hooks installed (below): the first commit stops once, after the hooks have
  formatted the notebooks and fixed what ruff can, and goes through when you add their changes and
  commit again. Ruff rules are configured in `pyproject.toml`.
- `notebooks/`: this project's own notebooks, such as `1.3_lecture_data_ingestion.py`, which parses
  the raw lecture files in the volume into the `lectures`, `lecture_transcript`, `lecture_chat` and
  `lecture_slides` tables, and `3.1_lecture_chunks_inspection.py`, which checks those tables and
  `lecture_chunks` for counts, sizes, and empty or duplicate chunks, and compares a sample of chunks with
  the parsed PDF elements and transcript turns they came from, and `3.2_chunking_comparison.py`, which
  compares the job's chunking with a sentence-based alternative in words and GTE tokens. Like the lecture
  folders, they find `project_config.yml` one level up.
- `scripts/`: Python scripts that jobs run, such as `lecture_ingestion.py`
  (`python scripts/lecture_ingestion.py --env dev`), `lecture_chunking.py`, `lecture_sync.py` and
  `github_ingestion.py`. Scripts stay thin:
  the ingestion itself lives in `src/arxiv_curator/utils/`, with the table schemas in
  `src/arxiv_curator/schemas/`, so notebooks and jobs run the same code.
- `databricks.yml` and `resources/`: the bundle, with three targets, `dev` (your sandbox, in
  development mode, the default), `acc` and `prd`. It builds the package as a wheel with `uv build`
  and deploys the `hello` job, named after its target (`job-hello-pablo-dev`,
  `-acc`, `-prd`), one serverless notebook task that installs the wheel and writes one row to the
  `hello` table of your schema in the target's catalog, such as
  `llmops_dev.pablo.hello`. It defines the `lecture_files` volume in each target's schema, protected
  from `bundle destroy` with `prevent_destroy`. It also deploys the `lecture_insights_assistant` job
  (`job-lecture-insights-assistant-pablo-dev`, `-acc`, `-prd`), whose `lecture_ingestion` task runs
  `scripts/lecture_ingestion.py` on serverless to parse the lecture files in the target's
  volume into the lecture tables. Its `lecture_chunking` task then runs `scripts/lecture_chunking.py`,
  which packs the sentences of `lecture_transcript` into windows of up to 300 words and makes each page
  of `lecture_slides` one chunk, embeds each chunk with `embedding_endpoint`, and writes them to
  `lecture_chunks`, with change data feed on. Its `lecture_sync` task then runs
  `scripts/lecture_sync.py`, which syncs `lecture_chunks` into Lakebase (see Lakebase above).
  The job has no schedule, and only dev has lecture data so far. The
  `github_events` job (`job-github-events-pablo-dev`, `-acc`, `-prd`) runs
  `scripts/github_ingestion.py` at 08:00, 12:00 and 18:00 Amsterdam time, in dev too: its
  schedule is unpaused explicitly. It merges the new events of the GitHub organization's public
  repositories into the `github_events` table, using the read-only GitHub token in the secret scope
  that `project_config.yml` names (`pablo-github`, key `token`). The token expires: replace it with
  `databricks secrets put-secret pablo-github token`. The
  course's onboarding does not deploy these jobs yet. Deployment workflows are introduced in a later lecture.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/CODEOWNERS`: who reviews a pull request to `main`, which needs one approving review.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```

Each environment logs to its own MLflow experiment and UC trace tables in its matching catalog. You can manage the dev experiment and write dev traces; acc and prd are read-only for you. Each deploy service principal manages its own environment’s experiment and writes to that environment’s schema.
