# Slack Bot Backend

FastAPI backend for a Slack bot that classifies user messages into **QUESTION**, **ACTION**, or **CONFIGURE** intents and orchestrates code changes across multiple GitHub repositories — all from Slack.

## Architecture overview

```
Slack @mention ──► FastAPI ──► Celery worker ──► IntentWorkflow (LLM classification)
                                                      │
                              ┌────────────────┬──────┴──────┬────────────────┐
                              ▼                ▼             ▼                ▼
                         QUESTION         ACTION         CONFIGURE       (fallback)
                        RAG answer     Pre-flight eval   Switch repo     Post message
                                            │
                              ┌─────────────┴─────────────┐
                              ▼                           ▼
                       simple → Aider            complex → OpenHands
                    (clone, branch, edit,          (REST API agent,
                     test, push, open PR)           polls for result)
```

### Entry points

| Route | Purpose |
|---|---|
| `GET /api/healthz` | Health check |
| `POST /api/question` | Direct question API |
| `POST /api/action` | Direct action API |
| `POST /api/slack/events` | Slack Event Subscriptions (app_mention) |
| `POST /api/slack/interactions` | Slack Block Kit interactions |
| `POST /api/github/webhook` | GitHub push / PR comment webhooks |

### Workflows

- **IntentWorkflow** — LLM-based classification of Slack mentions into QUESTION, ACTION, or CONFIGURE. Includes a fast-path for pending repo clarifications.
- **QuestionWorkflow** — Retrieval-augmented generation. Embeds the question, searches OpenViking for relevant code/doc chunks, loads Slack thread context from Supabase, and generates an answer.
- **ActionWorkflow** — Pre-flight evaluator determines actionability, complexity tier (simple/complex), and target repository. Simple tasks run **Aider** in an ephemeral clone. Complex tasks route to **OpenHands** via REST API. Both paths can open PRs.
- **ConfigureWorkflow** — Extracts a GitHub `owner/repo` from the user's message and persists it as the thread's target repository.

### Services

| Service | Implementation | Purpose |
|---|---|---|
| SlackGateway | `SlackWebAPIGateway` | Post messages to Slack |
| SupabaseRepository | `SupabasePersistenceRepository` | Thread history, thread context, PR mappings, pending requests |
| LanguageModel | `RoutingLanguageModel` | Routes prompts across Groq (light), Anthropic (standard), Anthropic (heavy) |
| GitService | `GitHubGitService` | Create PRs, resolve review threads |
| ContextSearch | `OpenVikingContextService` | Semantic search over indexed codebases (used by Aider + QuestionWorkflow) |
| Indexer | `OpenVikingIndexer` / `CodebaseIndexer` | Re-index repos on push to default branch |

### Background execution

Slack mentions, GitHub webhooks, and PR comments are dispatched to **Celery workers** (Redis broker) for durable background processing. PR comment tasks use a **Redis distributed lock** to serialize concurrent runs on the same branch.

## Requirements

- Python 3.11+
- Redis (for Celery broker and distributed locks)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [ngrok](https://ngrok.com/) (to expose local server for webhooks)

## Local setup

### 1. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
python3 -m pip install -e ".[dev]"
```

### 2. Configure environment

Copy `.env.example` to `.env` (or edit `.env` directly) and set the required values:

```bash
cp .env.example .env
```

See [Environment variables](#environment-variables) below for the full list.

### 3. Apply Supabase migrations

Run all migrations in order against your Supabase project (via the SQL editor or CLI):

1. `supabase/migrations/20260320164500_slack_bot_persistence.sql` — `vector` extension, `slack_thread_messages`, `documentation_chunks` + RPC
2. `supabase/migrations/20260322000000_active_pull_requests.sql` — `active_pull_requests` table for PR ↔ Slack thread mapping
3. `supabase/migrations/20260325000000_add_pending_request.sql` — `pending_request` column for repo-clarification replay

### 4. Run the stack

You need four processes running in separate terminals:

```bash
# 1. Redis (broker for Celery + distributed locks)
redis-server

# 2. FastAPI server
uv run uvicorn slack_bot_backend.main:app --reload

# 3. Celery worker (background task execution)
uv run celery -A slack_bot_backend.celery_app worker --loglevel=info

# 4. ngrok (expose local server for Slack & GitHub webhooks)
ngrok http 8000
```

Verify the API is healthy:

```bash
curl http://127.0.0.1:8000/api/healthz
```

### 5. Configure webhook URLs

Every time ngrok restarts you get a new public URL (e.g. `https://abc123.ngrok-free.app`). Update these places:

**Slack** ([api.slack.com/apps](https://api.slack.com/apps) → your app):

| Setting | URL |
|---|---|
| Event Subscriptions → Request URL | `https://<ngrok-url>/api/slack/events` |
| Interactivity & Shortcuts → Request URL | `https://<ngrok-url>/api/slack/interactions` |

**GitHub** (repo → Settings → Webhooks → your webhook):

| Setting | URL |
|---|---|
| Payload URL | `https://<ngrok-url>/api/github/webhook` |

Select events: `push`, `issue_comment`, `pull_request_review_comment`.

## Environment variables

All runtime configuration is prefixed with `SLACK_BOT_`.

| Category | Variable | Description |
|---|---|---|
| **App** | `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL`, `API_PREFIX` | Basic app config |
| **Slack** | `SLACK_ENABLED`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET` | Slack API credentials |
| **Supabase** | `SUPABASE_ENABLED`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Supabase persistence |
| **LLM** | `LLM_PROVIDER` (`stub`, `openai`, `anthropic`, `groq`, `router`), `LLM_API_KEY` | Primary LLM |
| **LLM Router** | `GROQ_API_KEY`, `OPENAI_API_KEY`, `ROUTER_LIGHT_MODEL`, `ROUTER_STANDARD_MODEL`, `ROUTER_HEAVY_MODEL` | Multi-tier model routing |
| **GitHub** | `GIT_PROVIDER` (`stub`, `github`), `GITHUB_TOKEN`, `GITHUB_WEBHOOK_SECRET`, `REPO_MAP` | GitHub integration |
| **OpenViking** | `OPENVIKING_ENABLED`, `OPENVIKING_URL`, `OPENVIKING_API_KEY` | Semantic code search (Aider + QuestionWorkflow) |
| **OpenHands** | `OPENHANDS_ENABLED`, `OPENHANDS_URL`, `OPENHANDS_MODEL`, `OPENHANDS_API_KEY` | Complex task agent |
| **Redis** | `REDIS_URL`, `CELERY_BROKER_URL` | Celery broker + distributed locks |

`REPO_MAP` is a JSON array of `"owner/repo"` strings, e.g. `["Bforsyth1234/rfpinator", "Bforsyth1234/talk-to-a-folder"]`.

## Running tests

```bash
uv run pytest
```

Or:

```bash
python3 -m pytest tests/
```
