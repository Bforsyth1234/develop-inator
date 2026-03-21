# Slack Bot Backend

Production-oriented FastAPI backend for a Slack bot that is evolving toward QUESTION and ACTION flows, with Slack, Supabase, LLM, and Git providers defined behind typed interfaces.

## Current repository status

- FastAPI app factory in `slack_bot_backend.main` with a health endpoint at `/api/healthz`
- QUESTION API route at `POST /api/question` backed by `slack_bot_backend.workflows.question.QuestionWorkflow`
- Checked-in Supabase persistence schema at `supabase/migrations/20260320164500_slack_bot_persistence.sql`
- Supabase persistence helpers for Slack thread history and pgvector document matching in `slack_bot_backend/services/supabase_persistence.py`
- Verification coverage for config, app startup, dependency wiring, QUESTION behavior, Supabase persistence helpers, and schema assertions in `tests/`

## Important implementation note

The repository now contains QUESTION workflow and Supabase persistence building blocks, but the default dependency container in `slack_bot_backend.dependencies` still wires stub Slack, Supabase, LLM, and Git services. That means the app boots safely out of the box, while concrete provider wiring can be layered in without changing the FastAPI entrypoint.

ACTION workflow orchestration, Slack `app_mention` ingestion, and automated PR execution should still be treated as in-progress unless you have separately wired and validated them in your environment.

## Requirements

- Python 3.11+

## Local setup

1. Create and activate a virtual environment.
2. Install the package and dev extras:
   `python3 -m pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and fill in the credentials you plan to use.
4. Start the API locally:
   `uvicorn slack_bot_backend.main:app --reload`
5. Verify the API is healthy:
   `curl http://127.0.0.1:8000/api/healthz`

## Environment variables

All runtime configuration is prefixed with `SLACK_BOT_`.

- App: `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL`, `API_PREFIX`
- Slack: `SLACK_ENABLED`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`
- Supabase: `SUPABASE_ENABLED`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- LLM: `LLM_PROVIDER` (`stub`, `openai`, `anthropic`), `LLM_API_KEY`
- GitHub: `GIT_PROVIDER` (`stub`, `github`), `GITHUB_TOKEN`, `GITHUB_REPOSITORY`

Practical guidance:

- Keep `SLACK_BOT_SLACK_ENABLED=false` until you have valid Slack credentials and a real Slack gateway wired in.
- Set `SLACK_BOT_SUPABASE_URL` and `SLACK_BOT_SUPABASE_SERVICE_ROLE_KEY` before using the checked-in Supabase persistence layer.
- Set `SLACK_BOT_LLM_PROVIDER` and `SLACK_BOT_LLM_API_KEY` when moving off the stub LLM.
- Set `SLACK_BOT_GIT_PROVIDER`, `SLACK_BOT_GITHUB_TOKEN`, and `SLACK_BOT_GITHUB_REPOSITORY` when enabling GitHub-backed PR actions.

## Supabase SQL setup

Before enabling real Supabase-backed persistence, apply the checked-in migration:

`supabase/migrations/20260320164500_slack_bot_persistence.sql`

This migration creates:

- the `vector` extension required for pgvector search
- `public.slack_thread_messages` for thread-history persistence
- `public.documentation_chunks` plus the `match_documentation_chunks` RPC for semantic retrieval

You can apply it using your normal Supabase migration workflow or by running the SQL directly in the Supabase SQL editor against the target project database.

## Running tests

Run the verification suite with:

`python3 -m unittest discover -s tests`

## How bootstrap and SQL fit together

The bootstrap layer owns application startup, settings loading, routing, and dependency boundaries. The checked-in Supabase migration and persistence helpers define the storage contract for thread history and documentation retrieval, while the QUESTION workflow consumes those interfaces. The default app wiring remains stub-backed today, so SQL setup and credentials are necessary prerequisites for real provider integration rather than something the scaffold automatically enables by itself.
