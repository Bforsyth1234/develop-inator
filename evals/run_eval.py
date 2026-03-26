#!/usr/bin/env python3
"""
Eval harness for testing ActionWorkflow code-editing capabilities.

For each task in dataset.json the harness:
  1. Copies the dummy React app into a temporary git repo (with a bare origin).
  2. Injects ActionWorkflow with no-op Slack, no-op Git, and a bypass LLM stub.
  3. Calls ActionWorkflow.run() which invokes Aider on the repo.
  4. Scores the result:
        Check 1 — Aider exit code 0 (PR opened).
        Check 2 — `npm test` passes on the new branch.
        Check 3 — `npm run lint` passes on the new branch.
     Score = 1.0 if all three pass, else 0.0.

Usage:
    python evals/run_eval.py                      # run all 20 tasks
    python evals/run_eval.py --ids fix-divide-by-zero add-dark-mode-toggle
    python evals/run_eval.py --out results.json   # write detailed results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import the workflow.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env from the project root so API keys are available.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env")

# Map SLACK_BOT_GROQ_API_KEY → GROQ_API_KEY if the latter isn't already set,
# since _build_subprocess_env forwards GROQ_API_KEY to Aider.
if os.environ.get("SLACK_BOT_GROQ_API_KEY") and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ["SLACK_BOT_GROQ_API_KEY"]

from slack_bot_backend.models.action import ActionRequest, ActionRouteResult  # noqa: E402
from slack_bot_backend.services.interfaces import (  # noqa: E402
    EmbeddingResult,
    LLMResult,
    PullRequestDraft,
)
from slack_bot_backend.workflows.action import ActionWorkflow  # noqa: E402

logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _NoOpSlack:
    """Slack stub that silently discards all messages."""
    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        pass

    async def post_blocks(self, channel: str, blocks: list, text: str = "", thread_ts: str | None = None) -> None:
        pass


class _NoOpGit:
    """Git stub — records the PR draft but does not call GitHub."""
    def __init__(self) -> None:
        self.last_draft: PullRequestDraft | None = None

    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str:
        self.last_draft = draft
        return f"https://github.com/fake/repo/pull/0#{draft.branch_name}"


_CATEGORY_TO_TIER: dict[str, str] = {
    "bugfix": "simple",
    "feature": "simple",
    "refactor": "simple",
    "performance": "simple",
}


class _BypassLLM:
    """LLM stub that always returns is_actionable=True with the raw task as the prompt."""
    def __init__(self, task_prompt: str, category: str = "unknown") -> None:
        self._task_prompt = task_prompt
        self._tier = _CATEGORY_TO_TIER.get(category, "complex")

    async def generate(self, prompt: str) -> LLMResult:
        payload = json.dumps({
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": self._task_prompt,
            "complexity_tier": self._tier,
            "target_repository": "eval/dummy-react-app",
        })
        return LLMResult(content=payload, provider="bypass-eval")

    async def embed(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=(0.0,), provider="bypass-eval")


# ---------------------------------------------------------------------------
# Data model for results
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    task: str
    category: str
    score: float = 0.0
    aider_ok: bool = False
    tests_ok: bool = False
    lint_ok: bool = False
    pr_url: str | None = None
    branch_name: str | None = None
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class EvalReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    score: float = 0.0
    results: list[TaskResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_APP_DIR = Path(__file__).resolve().parent / "dummy_react_app"
_DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def _run(cmd: list[str], cwd: str, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, **kwargs)


def _prepare_repo(work_dir: str) -> None:
    """Copy dummy app into *work_dir*, init git with a bare origin, and npm install."""
    src = str(_DUMMY_APP_DIR)
    # Copy everything except node_modules
    shutil.copytree(src, work_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("node_modules"))

    # Create a bare "origin" repo so that `git push origin <branch>` works.
    bare_dir = os.path.join(work_dir, "..", "origin.git")
    os.makedirs(bare_dir, exist_ok=True)
    _run(["git", "init", "--bare"], cwd=bare_dir)

    # Init the working repo
    _run(["git", "init", "-b", "main"], cwd=work_dir)
    _run(["git", "remote", "add", "origin", bare_dir], cwd=work_dir)

    # npm install so tests and lint work
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm not found on PATH")
    result = _run([npm, "install"], cwd=work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"npm install failed:\n{result.stderr}")

    # Initial commit so the repo is not empty
    _run(["git", "add", "."], cwd=work_dir)
    _run(["git", "commit", "-m", "initial commit"], cwd=work_dir)
    _run(["git", "push", "-u", "origin", "main"], cwd=work_dir)


def _check_tests(work_dir: str) -> bool:
    npm = shutil.which("npm") or "npm"
    r = _run([npm, "test"], cwd=work_dir)
    if r.returncode != 0:
        logger.warning("Tests failed:\n%s\n%s", r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode == 0


def _check_lint(work_dir: str) -> bool:
    npm = shutil.which("npm") or "npm"
    r = _run([npm, "run", "lint"], cwd=work_dir)
    if r.returncode != 0:
        logger.warning("Lint failed:\n%s\n%s", r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------

async def _run_single_task(task_entry: dict, keep_dirs: bool = False, aider_bin: str | None = None) -> TaskResult:
    """Run one eval task end-to-end and return the scored result."""
    task_id = task_entry["id"]
    task_text = task_entry["task"]
    category = task_entry.get("category", "unknown")
    result = TaskResult(task_id=task_id, task=task_text, category=category)

    tmp_root = tempfile.mkdtemp(prefix=f"eval-{task_id}-")
    work_dir = os.path.join(tmp_root, "repo")
    os.makedirs(work_dir, exist_ok=True)

    try:
        logger.info("[%s] Preparing repo in %s", task_id, work_dir)
        _prepare_repo(work_dir)

        # Build the workflow with stubs
        slack = _NoOpSlack()
        git = _NoOpGit()
        llm = _BypassLLM(task_text, category=category)

        # The bare origin created by _prepare_repo lives alongside work_dir.
        bare_dir = os.path.join(work_dir, "..", "origin.git")

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm,
            repo_map=["eval/dummy-react-app"],
            supabase=None,
            aider_bin=aider_bin,
            github_token="eval-token",
        )

        # Monkey-patch _build_clone_url so the workflow clones from the local
        # bare repo instead of trying to reach GitHub.
        workflow._build_clone_url = lambda repository: os.path.abspath(bare_dir)

        request = ActionRequest(
            channel="eval-channel",
            thread_ts="eval-thread",
            request=task_text,
            user_id="eval-user",
        )

        t0 = time.monotonic()
        route_result: ActionRouteResult = await workflow.run(request)
        result.duration_s = round(time.monotonic() - t0, 2)

        # Check 1: Aider success (PR opened)
        result.aider_ok = route_result.status == "completed"
        result.pr_url = route_result.pr_url
        result.branch_name = route_result.branch_name

        if not result.aider_ok:
            result.error = f"Workflow status={route_result.status}: {route_result.message[:500]}"
            return result

        # Fetch from the bare origin and checkout the branch to verify tests & lint.
        # The workflow pushed to origin from its own ephemeral clone, so the
        # eval's work_dir must fetch first to see the new branch.
        branch = route_result.branch_name
        if branch:
            _run(["git", "fetch", "origin"], cwd=work_dir)
            _run(["git", "checkout", branch], cwd=work_dir)

        # Check 2: Tests pass
        result.tests_ok = _check_tests(work_dir)

        # Check 3: Lint passes
        result.lint_ok = _check_lint(work_dir)

        # Final score
        result.score = 1.0 if (result.aider_ok and result.tests_ok and result.lint_ok) else 0.0

    except Exception as exc:
        result.error = str(exc)
        logger.exception("[%s] Unexpected error", task_id)
    finally:
        if not keep_dirs:
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            logger.info("[%s] Keeping temp dir: %s", task_id, tmp_root)

    return result


async def run_eval(
    task_ids: list[str] | None = None,
    keep_dirs: bool = False,
    aider_bin: str | None = None,
) -> EvalReport:
    """Run the full eval suite (or a subset) and return the report."""
    with open(_DATASET_PATH) as f:
        dataset = json.load(f)

    if task_ids:
        dataset = [t for t in dataset if t["id"] in task_ids]

    report = EvalReport(total=len(dataset))
    for entry in dataset:
        logger.info("=" * 60)
        logger.info("Running task: %s", entry["id"])
        logger.info("=" * 60)
        result = await _run_single_task(entry, keep_dirs=keep_dirs, aider_bin=aider_bin)
        report.results.append(result)
        if result.score == 1.0:
            report.passed += 1
        else:
            report.failed += 1
        logger.info(
            "[%s] score=%.1f  aider=%s  tests=%s  lint=%s  (%.1fs)",
            result.task_id, result.score, result.aider_ok, result.tests_ok,
            result.lint_ok, result.duration_s,
        )

    report.score = report.passed / report.total if report.total else 0.0
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ActionWorkflow code-editing eval suite")
    parser.add_argument("--ids", nargs="*", help="Run only these task IDs")
    parser.add_argument("--out", type=str, help="Write JSON results to this file")
    parser.add_argument("--keep-dirs", action="store_true", help="Keep temp directories for debugging")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--aider-bin", type=str, help="Path to the aider binary (overrides auto-detect)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    report = asyncio.run(run_eval(task_ids=args.ids, keep_dirs=args.keep_dirs, aider_bin=args.aider_bin))

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVAL COMPLETE — {report.passed}/{report.total} passed  (score: {report.score:.2f})")
    print("=" * 60)
    for r in report.results:
        status = "✅" if r.score == 1.0 else "❌"
        print(f"  {status}  {r.task_id:<30}  aider={r.aider_ok}  tests={r.tests_ok}  lint={r.lint_ok}  ({r.duration_s}s)")
        if r.error:
            print(f"      ↳ {r.error[:120]}")
    print()

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(asdict(report), indent=2, default=str))
        print(f"Results written to {out_path}")

    # Exit with non-zero if any task failed
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()

