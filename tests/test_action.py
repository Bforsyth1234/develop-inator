import base64
import json
import subprocess
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from slack_bot_backend.config import Settings
from slack_bot_backend.dependencies import ServiceContainer, get_container
from slack_bot_backend.main import create_app
from slack_bot_backend.models.action import AiderResult, ActionExecution, ActionExecutionStatus, ActionRequest, ActionRouteResult, ProposedFileChange, RepositorySearchResult
from slack_bot_backend.models.persistence import ActivePullRequestRecord
from slack_bot_backend.services.github import GitHubGitService
from slack_bot_backend.services.interfaces import LLMResult, PullRequestDraft
from slack_bot_backend.services.stubs import StubGitService, StubLanguageModel, StubSlackGateway, StubSupabaseRepository
from slack_bot_backend.services.supabase_persistence import RepositoryConfig
from slack_bot_backend.workflows import ActionWorkflow, IntentWorkflow, QuestionWorkflow


class FakeSlackGateway:
    def __init__(self) -> None:
        self.messages: list[dict[str, str | None]] = []
        self.blocks_calls: list[dict] = []
        self.update_calls: list[dict] = []

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        self.messages.append({"channel": channel, "text": text, "thread_ts": thread_ts})

    async def post_blocks(
        self, channel: str, blocks: list[dict], text: str = "", thread_ts: str | None = None,
    ) -> None:
        self.blocks_calls.append({"channel": channel, "blocks": blocks, "text": text, "thread_ts": thread_ts})

    async def update_message(
        self, channel: str, ts: str, text: str, blocks: list[dict] | None = None,
    ) -> None:
        self.update_calls.append({"channel": channel, "ts": ts, "text": text, "blocks": blocks})

    async def fetch_replies(
        self, channel: str, thread_ts: str, *, oldest: str | None = None, limit: int = 100,
    ) -> list[dict]:
        return []


class FakeGitService:
    """Records create_pull_request calls and returns a canned PR URL."""

    def __init__(self, *, pr_url: str = "https://example.invalid/pr/42", fail: bool = False) -> None:
        self.pr_url = pr_url
        self.fail = fail
        self.pr_calls: list[PullRequestDraft] = []
        self.resolved_threads: list[tuple[str, str]] = []

    async def create_pull_request(self, draft: PullRequestDraft, *, repository: str | None = None) -> str:
        self.pr_calls.append(draft)
        if self.fail:
            raise RuntimeError("GitHub unavailable")
        return self.pr_url

    async def resolve_review_thread(self, pr_url: str, comment_node_id: str) -> None:
        self.resolved_threads.append((pr_url, comment_node_id))


class FakeLanguageModel:
    """Configurable fake LLM for evaluator tests."""

    def __init__(self, *, evaluator_json: dict | None = None, raise_on_generate: bool = False) -> None:
        self._evaluator_json = evaluator_json or {
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Optimized: do the thing in FooComponent",
            "complexity_tier": "standard",
            "target_repository": "owner/repo",
        }
        self._raise = raise_on_generate
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> LLMResult:
        self.prompts.append(prompt)
        if self._raise:
            raise RuntimeError("LLM unavailable")
        return LLMResult(content=json.dumps(self._evaluator_json), provider="fake")

    async def embed(self, text: str) -> object:
        return object()


class FakeSupabaseRepository:
    """Records save_repository_config calls and returns canned get results."""

    def __init__(
        self,
        *,
        stored_config: RepositoryConfig | None = None,
        fail_save: bool = False,
        pr_mapping: ActivePullRequestRecord | None = None,
        stored_execution: ActionExecution | None = None,
        thread_context: dict[tuple[str, str], str] | None = None,
        pending_requests: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self._stored_config = stored_config
        self._fail_save = fail_save
        self._pr_mapping = pr_mapping
        self._stored_execution = stored_execution
        self._thread_contexts: dict[tuple[str, str], str] = dict(thread_context or {})
        self._pending_requests: dict[tuple[str, str], str] = dict(pending_requests or {})
        self.save_calls: list[dict[str, str]] = []
        self.pr_mapping_saves: list[ActivePullRequestRecord] = []
        self.execution_status_updates: list[tuple[str, ActionExecutionStatus]] = []
        self.thread_context_upserts: list[dict[str, str]] = []

    async def healthcheck(self) -> bool:
        return True

    async def get_thread_messages(self, *, channel_id, thread_ts, limit=50):
        return []

    async def get_repository_config(self) -> RepositoryConfig | None:
        return self._stored_config

    async def save_repository_config(self, *, github_repository: str) -> None:
        if self._fail_save:
            raise RuntimeError("Supabase unavailable")
        self.save_calls.append({"github_repository": github_repository})

    # -- Action execution persistence stubs --

    async def save_action_execution(self, execution: ActionExecution) -> None:
        pass

    async def get_action_execution(self, execution_id: str) -> ActionExecution | None:
        return self._stored_execution

    async def get_pending_execution_for_thread(
        self, *, channel: str, thread_ts: str
    ) -> ActionExecution | None:
        return None

    async def update_action_execution_status(
        self, execution_id: str, status: ActionExecutionStatus
    ) -> None:
        self.execution_status_updates.append((execution_id, status))

    # -- PR mapping stubs --

    async def save_pr_mapping(self, record: ActivePullRequestRecord) -> None:
        self.pr_mapping_saves.append(record)

    async def get_pr_mapping_by_url(self, pr_url: str) -> ActivePullRequestRecord | None:
        if self._pr_mapping and self._pr_mapping.pr_url == pr_url:
            return self._pr_mapping
        return None

    async def get_pr_mapping_by_thread(
        self, *, channel_id: str, thread_ts: str
    ) -> ActivePullRequestRecord | None:
        if (
            self._pr_mapping
            and self._pr_mapping.channel_id == channel_id
            and self._pr_mapping.thread_ts == thread_ts
        ):
            return self._pr_mapping
        return None

    # -- Thread context (thread memory) --

    async def get_thread_context(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return self._thread_contexts.get((channel_id, thread_ts))

    async def upsert_thread_context(
        self, *, channel_id: str, thread_ts: str, target_repository: str
    ) -> None:
        self._thread_contexts[(channel_id, thread_ts)] = target_repository
        self.thread_context_upserts.append({
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "target_repository": target_repository,
        })

    # -- Pending request persistence --

    async def get_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> str | None:
        return self._pending_requests.get((channel_id, thread_ts))

    async def save_pending_request(
        self, *, channel_id: str, thread_ts: str, pending_request: str
    ) -> None:
        self._pending_requests[(channel_id, thread_ts)] = pending_request

    async def clear_pending_request(
        self, *, channel_id: str, thread_ts: str
    ) -> None:
        self._pending_requests.pop((channel_id, thread_ts), None)


class FakeActionWorkflow:
    def __init__(self, result: ActionRouteResult) -> None:
        self.result = result
        self.requests: list[ActionRequest] = []

    async def run(self, request: ActionRequest) -> ActionRouteResult:
        self.requests.append(request)
        return self.result


def _make_proc(*, returncode: int, stdout: str = "", stderr: str = "") -> mock.Mock:
    """Build a mock subprocess.CompletedProcess-like object."""
    proc = mock.Mock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def _make_aider_result(
    stdout: str = "Aider applied changes.",
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess:
    """Build a CompletedProcess for mocking _run_aider_subprocess."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _make_git_aware_side_effect(
    aider_makes_commits: bool = True,
    test_returncode: int = 0,
    test_stdout: str = "",
    test_stderr: str = "",
):
    """Return a side_effect for asyncio.to_thread that handles git + test commands.

    Aider commands now flow through _run_aider_subprocess (mocked separately).

    The git flow:
      rev-parse --abbrev-ref HEAD → stash → fetch origin main →
      rev-parse origin/main (base SHA) → checkout -b ... origin/main →
      [aider via _run_aider_subprocess] →
      rev-parse HEAD (current SHA) → [test] → push → checkout base → stash pop
    """
    checkout_b_done = False

    async def side_effect(_fn, cmd, **kwargs):
        nonlocal checkout_b_done

        # Non-git command = test runner or aider retry (via to_thread)
        if cmd[0] != "git":
            return _make_proc(returncode=test_returncode, stdout=test_stdout, stderr=test_stderr)

        subcmd = cmd[1] if len(cmd) > 1 else ""

        if subcmd == "checkout" and "-b" in cmd:
            checkout_b_done = True
            return _make_proc(returncode=0)

        if subcmd == "rev-parse":
            if "--abbrev-ref" in cmd:
                return _make_proc(returncode=0, stdout="main")
            if checkout_b_done and aider_makes_commits:
                return _make_proc(returncode=0, stdout="new-sha-111")
            return _make_proc(returncode=0, stdout="base-sha-000")

        if subcmd == "stash":
            return _make_proc(returncode=0, stdout="No local changes to save")

        # fetch, checkout (non -b), push, branch -D, etc.
        return _make_proc(returncode=0)

    return side_effect


# Keep backward-compat alias
_make_successful_side_effect = _make_git_aware_side_effect


_DEFAULT_TRIVIAL_MODEL = "groq/llama-3.3-70b-versatile"
_DEFAULT_STANDARD_MODEL = "anthropic/claude-sonnet-4-6"


class ActionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Disable test detection by default so existing tests aren't affected.
        patcher = mock.patch.object(
            ActionWorkflow, "_detect_test_command", return_value=None,
        )
        self._no_tests = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_workflow(
        self,
        *,
        slack: FakeSlackGateway,
        git: FakeGitService,
        llm: FakeLanguageModel | None = None,
        github_token: str = "stub",
        repo_map: list[str] | None = None,
    ) -> ActionWorkflow:
        return ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm or FakeLanguageModel(),
            github_token=github_token,
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=repo_map or ["owner/repo"],
        )

    async def test_success_opens_pr_and_posts_link_to_slack(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update the deploy docs")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.provider, _DEFAULT_STANDARD_MODEL)
        self.assertTrue(result.branch_name.startswith("ai-update-"))
        self.assertEqual(result.pr_url, "https://example.invalid/pr/42")
        self.assertEqual(len(git.pr_calls), 1)
        self.assertTrue(git.pr_calls[0].branch_name.startswith("ai-update-"))
        self.assertIn("ai-update-", slack.messages[0]["text"])
        self.assertIn("https://example.invalid/pr/42", slack.messages[0]["text"])
        self.assertEqual(slack.messages[0]["channel"], "C123")
        self.assertEqual(slack.messages[0]["thread_ts"], "1710.2")

    async def test_subprocess_failure_posts_stderr_to_slack(self) -> None:
        """Aider subprocess fails — stderr is posted to Slack."""
        slack = FakeSlackGateway()
        git = FakeGitService()

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result(
                returncode=1, stderr="fatal: not a git repository", stdout="",
            )),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update the deploy docs")
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.provider, "aider")
        posted = slack.messages[0]["text"]
        self.assertIn("exit 1", posted)
        self.assertIn("fatal: not a git repository", posted)
        self.assertEqual(len(git.pr_calls), 0)

    async def test_github_pr_failure_is_surfaced_as_unexpected_error(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService(fail=True)

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update the deploy docs")
            )

        self.assertEqual(result.status, "error")
        self.assertIn("PR creation failed", result.message)

    async def test_aider_command_includes_model_and_message(self) -> None:
        """The aider subprocess receives --model, --yes-always, and --message with the optimized prompt."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Add a dark-mode toggle to the AppComponent",
            "complexity_tier": "standard",
            "target_repository": "owner/repo",
        })
        captured_git_cmd: list[list[str]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_git_cmd.append(cmd)
            return await _delegate(_fn, cmd, **kwargs)

        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=capture_to_thread,
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Add a dark-mode toggle")
            )

        # Find specific commands by type
        checkout_cmd = [c for c in captured_git_cmd if c[0] == "git" and len(c) > 2 and c[1] == "checkout" and "-b" in c][0]
        push_cmd = [c for c in captured_git_cmd if c[0] == "git" and len(c) > 1 and c[1] == "push"][0]

        # aider cmd is captured via the mock
        aider_cmd = mock_aider.call_args[0][0]  # first positional arg

        # checkout creates branch
        self.assertTrue(checkout_cmd[3].startswith("ai-update-"))

        # aider gets model, yes-always, and message
        self.assertTrue(aider_cmd[0].endswith("aider"), aider_cmd[0])
        self.assertIn("--model", aider_cmd)
        self.assertIn("--yes-always", aider_cmd)
        self.assertIn("--message", aider_cmd)
        msg_idx = aider_cmd.index("--message") + 1
        self.assertIn("Add a dark-mode toggle", aider_cmd[msg_idx])

        # push uses the same branch (git push --force-with-lease origin <branch>)
        self.assertEqual(push_cmd[0], "git")
        self.assertEqual(push_cmd[1], "push")
        self.assertIn("--force-with-lease", push_cmd)
        self.assertEqual(push_cmd[-1], checkout_cmd[3])

    # ------------------------------------------------------------------
    # Evaluator routing tests
    # ------------------------------------------------------------------

    async def test_evaluator_actionable_uses_optimized_prompt_for_subprocess(self) -> None:
        """When evaluator says actionable, Aider receives the optimized prompt."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "In LoginComponent set the button colour to #E53935",
            "complexity_tier": "standard",
            "target_repository": "owner/repo",
        })
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Make the button red")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        msg_idx = aider_cmd.index("--message") + 1
        self.assertIn("LoginComponent", aider_cmd[msg_idx])
        self.assertIn("#E53935", aider_cmd[msg_idx])
        self.assertNotIn("Make the button red", aider_cmd[msg_idx])

    async def test_evaluator_needs_clarification_posts_question_and_skips_aider(self) -> None:
        """When evaluator says not actionable, question is posted and Aider never runs."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": False,
            "clarifying_question": "Which component should the button colour change apply to?",
            "optimized_prompt": None,
            "complexity_tier": "standard",
        })

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
        ) as mock_thread:
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Make the button red")
            )

        self.assertEqual(result.status, "needs_clarification")
        self.assertEqual(result.provider, "evaluator")
        mock_thread.assert_not_called()
        self.assertEqual(len(git.pr_calls), 0)
        self.assertEqual(len(slack.messages), 1)
        self.assertIn("Which component", slack.messages[0]["text"])
        self.assertEqual(slack.messages[0]["channel"], "C123")
        self.assertEqual(slack.messages[0]["thread_ts"], "1710.2")

    async def test_repo_name_reply_saves_thread_context_and_returns_repo_selected(self) -> None:
        """When thread has no repo context and user message is a repo name, save it."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()
        wf = self._make_workflow(
            slack=slack, git=git,
            repo_map=["owner/repo", "owner/other-repo"],
        )
        wf.supabase = supabase

        result = await wf.run(
            ActionRequest(channel="C123", thread_ts="1710.2", request="owner/repo")
        )

        self.assertEqual(result.status, "repo_selected")
        self.assertEqual(result.provider, "system")
        self.assertEqual(len(supabase.thread_context_upserts), 1)
        self.assertEqual(supabase.thread_context_upserts[0]["target_repository"], "owner/repo")
        self.assertIn("owner/repo", slack.messages[0]["text"])

    async def test_repo_name_reply_with_pending_request_re_executes(self) -> None:
        """When user replies with repo name and there's a stored pending request, auto re-execute."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository(
            pending_requests={("C123", "1710.2"): "Change the color scheme to rainbow"},
        )
        wf = self._make_workflow(
            slack=slack, git=git,
            repo_map=["owner/repo", "owner/other-repo"],
        )
        wf.supabase = supabase

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await wf.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="owner/repo")
            )

        # Should have re-executed (completed), not just returned repo_selected
        self.assertEqual(result.status, "completed")
        # Thread context should have been saved
        self.assertTrue(any(
            u["target_repository"] == "owner/repo"
            for u in supabase.thread_context_upserts
        ))
        # Pending request should have been cleared
        self.assertIsNone(
            await supabase.get_pending_request(channel_id="C123", thread_ts="1710.2")
        )
        # Confirmation message should mention the repo
        self.assertTrue(any("owner/repo" in m["text"] for m in slack.messages))

    async def test_repo_clarification_saves_pending_request(self) -> None:
        """When bot asks 'which repo?', it should save the original request."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()
        # Evaluator returns actionable but no target_repository
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Change the color scheme",
            "complexity_tier": "standard",
            "target_repository": None,
        })
        wf = self._make_workflow(
            slack=slack, git=git, llm=llm,
            repo_map=["owner/repo", "owner/other-repo"],
        )
        wf.supabase = supabase

        result = await wf.run(
            ActionRequest(channel="C123", thread_ts="1710.2", request="Change the color scheme")
        )

        self.assertEqual(result.status, "needs_clarification")
        # The original request should be persisted
        stored = await supabase.get_pending_request(
            channel_id="C123", thread_ts="1710.2"
        )
        self.assertEqual(stored, "Change the color scheme")

    async def test_repo_name_reply_skipped_when_thread_already_has_context(self) -> None:
        """If thread already has a repo, don't treat message as repo selection."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository(
            thread_context={("C123", "1710.2"): "owner/repo"},
        )
        wf = self._make_workflow(
            slack=slack, git=git,
            repo_map=["owner/repo", "owner/other-repo"],
        )
        wf.supabase = supabase

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await wf.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        # Should proceed normally, not return repo_selected
        self.assertNotEqual(result.status, "repo_selected")

    async def test_thread_context_provides_repo_when_evaluator_does_not(self) -> None:
        """When evaluator omits target_repository but thread context has it, use thread context."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository(
            thread_context={("C123", "1710.2"): "owner/repo"},
        )
        # Evaluator returns actionable but without target_repository
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Fix the bug",
            "complexity_tier": "standard",
            "target_repository": None,
        })
        wf = self._make_workflow(
            slack=slack, git=git, llm=llm,
            repo_map=["owner/repo", "owner/other-repo"],
        )
        wf.supabase = supabase

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await wf.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        # Should complete successfully using thread context repo, NOT ask for clarification
        self.assertEqual(result.status, "completed")
        self.assertNotEqual(result.status, "needs_clarification")

    async def test_evaluator_invalid_json_falls_back_to_original_request(self) -> None:
        """If the evaluator returns garbage JSON, the workflow runs with the raw request."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel()
        llm._evaluator_json = {}  # will produce JSON that fails EvaluationResult validation

        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fallback request text")
            )

        # Should still complete using the original request
        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        msg_idx = aider_cmd.index("--message") + 1
        self.assertIn("Fallback request text", aider_cmd[msg_idx])

    async def test_evaluator_llm_exception_falls_back_to_original_request(self) -> None:
        """If the evaluator LLM raises, the workflow runs with the raw request (fail-open)."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(raise_on_generate=True)

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Exception fallback test")
            )

        self.assertEqual(result.status, "completed")

    # ------------------------------------------------------------------
    # Complexity-tier model routing tests
    # ------------------------------------------------------------------

    async def test_trivial_tier_routes_to_trivial_model(self) -> None:
        """A 'trivial' complexity_tier sends the trivial-tier model (Groq) to Aider."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Change the button text in HeaderComponent to 'Sign In'",
            "complexity_tier": "trivial",
            "target_repository": "owner/repo",
        })
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Change button text")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], _DEFAULT_TRIVIAL_MODEL)

    async def test_standard_tier_routes_to_standard_model(self) -> None:
        """A 'standard' complexity_tier sends the standard-tier model (Sonnet) to Aider."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Refactor LoginComponent to use reactive forms",
            "complexity_tier": "standard",
            "target_repository": "owner/repo",
        })
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Refactor login")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], _DEFAULT_STANDARD_MODEL)

    async def test_complex_tier_routes_to_openhands_when_enabled(self) -> None:
        """A 'complex' complexity_tier invokes OpenHands when openhands_enabled=True."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Refactor AuthService to use NgRx store for session state",
            "complexity_tier": "complex",
            "target_repository": "owner/repo",
        })

        wf = ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm,
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
            openhands_enabled=True,
            openhands_model="anthropic/claude-sonnet-4-6",
            openhands_url="https://app.all-hands.dev",
            openhands_api_key="test-api-key",
        )

        fake_oh_result = AiderResult(
            returncode=0,
            stdout="OpenHands completed",
            stderr="",
            branch_name="ai-update-12345",
        )

        with mock.patch.object(wf, "_run_openhands", return_value=fake_oh_result) as mock_oh:
            result = await wf.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Refactor auth to use NgRx")
            )

        mock_oh.assert_called_once()
        self.assertEqual(result.status, "completed")

    # ------------------------------------------------------------------
    # Tier-flag override tests  (--trivial / --standard / --complex)
    # ------------------------------------------------------------------

    async def test_trivial_flag_forces_trivial_model_regardless_of_evaluator(self) -> None:
        """--trivial in the message forces the trivial model even if evaluator says standard."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        # Evaluator would normally say "standard"
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Fix the typo",
            "complexity_tier": "standard",
            "target_repository": "owner/repo",
        })
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the typo --trivial")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], _DEFAULT_TRIVIAL_MODEL)
        # Flag must be stripped from the prompt sent to Aider
        msg_idx = aider_cmd.index("--message") + 1
        self.assertNotIn("--trivial", aider_cmd[msg_idx])

    async def test_standard_flag_forces_standard_model(self) -> None:
        """--standard in the message forces the standard model."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Add dark mode",
            "complexity_tier": "trivial",
            "target_repository": "owner/repo",
        })
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Add dark mode --standard")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = mock_aider.call_args[0][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], _DEFAULT_STANDARD_MODEL)
        msg_idx = aider_cmd.index("--message") + 1
        self.assertNotIn("--standard", aider_cmd[msg_idx])

    async def test_complex_flag_routes_to_openhands_when_enabled(self) -> None:
        """--complex in the message forces OpenHands routing when openhands_enabled=True."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        # Evaluator says trivial — flag should override it
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Fix the typo",
            "complexity_tier": "trivial",
            "target_repository": "owner/repo",
        })
        wf = ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm,
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
            openhands_enabled=True,
            openhands_model="anthropic/claude-sonnet-4-6",
            openhands_url="https://app.all-hands.dev",
            openhands_api_key="test-api-key",
        )
        fake_oh_result = AiderResult(
            returncode=0, stdout="OpenHands completed", stderr="", branch_name="ai-update-12345",
        )
        with mock.patch.object(wf, "_run_openhands", return_value=fake_oh_result) as mock_oh:
            result = await wf.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the typo --complex")
            )

        mock_oh.assert_called_once()
        self.assertEqual(result.status, "completed")

    async def test_aider_subprocess_receives_env_with_api_keys(self) -> None:
        """The Aider subprocess env includes GROQ_API_KEY and ANTHROPIC_API_KEY."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        mock_aider = mock.AsyncMock(return_value=_make_aider_result())

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", mock_aider),
            mock.patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "gsk_test123", "ANTHROPIC_API_KEY": "sk-ant-test456"},
            ),
        ):
            await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update deploy docs")
            )

        # _run_aider_subprocess should have been called with env kwarg
        mock_aider.assert_called_once()
        aider_kwargs = mock_aider.call_args[1]  # keyword args
        self.assertIn("env", aider_kwargs)
        self.assertEqual(aider_kwargs["env"]["GROQ_API_KEY"], "gsk_test123")
        self.assertEqual(aider_kwargs["env"]["ANTHROPIC_API_KEY"], "sk-ant-test456")


class ActionWorkflowRepoConfigTests(unittest.IsolatedAsyncioTestCase):
    """Tests for saving repository config to Supabase after successful ACTION runs."""

    def setUp(self) -> None:
        patcher = mock.patch.object(
            ActionWorkflow, "_detect_test_command", return_value=None,
        )
        self._no_tests = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_success_saves_repo_config_to_supabase(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            supabase=supabase,
            repo_map=["owner/repo"],
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
        )

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await workflow.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(supabase.save_calls), 1)
        self.assertEqual(supabase.save_calls[0]["github_repository"], "owner/repo")

    async def test_success_without_supabase_skips_config_save(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await workflow.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        self.assertEqual(result.status, "completed")
        # No supabase → no crash, no save

    async def test_config_save_failure_does_not_break_workflow(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository(fail_save=True)

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            supabase=supabase,
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await workflow.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        # Workflow still completes even though save failed
        self.assertEqual(result.status, "completed")
        self.assertIn("https://example.invalid/pr/42", result.pr_url)

    async def test_failure_does_not_save_repo_config(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            supabase=supabase,
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result(
                returncode=1, stderr="error", stdout="",
            )),
        ):
            result = await workflow.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(len(supabase.save_calls), 0)


class ActionEndpointTests(unittest.TestCase):
    def test_action_endpoint_returns_workflow_result(self) -> None:
        action = FakeActionWorkflow(
            ActionRouteResult(
                status="completed",
                provider="fake-action",
                message="Created a pull request.",
                pr_url="https://example.invalid/pr/200",
                branch_name="feature/deploy-docs-update",
                searched_files=2,
                change_count=1,
            )
        )
        question = QuestionWorkflow(
            slack=StubSlackGateway(),
            supabase=StubSupabaseRepository(),
            llm=StubLanguageModel(),
        )
        container = ServiceContainer(
            settings=Settings(environment="testing"),
            slack=StubSlackGateway(),
            supabase=StubSupabaseRepository(),
            llm=StubLanguageModel(),
            git=StubGitService(),
            action=action,
            intent=IntentWorkflow(
                slack=StubSlackGateway(),
                supabase=StubSupabaseRepository(),
                llm=StubLanguageModel(),
                question=question,
                action=action,
            ),
            question=question,
        )
        app = create_app(Settings(environment="testing"))
        app.dependency_overrides[get_container] = lambda: container

        with TestClient(app) as client:
            response = client.post(
                "/api/action",
                json={"channel": "C321", "thread_ts": "1800.1", "request": "Update deploy docs"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "completed")
        self.assertEqual(action.requests[0].request, "Update deploy docs")


class GitHubGitServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_repository_returns_matches_with_file_excerpt(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path == "/search/code":
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "path": "src/app.py",
                                "html_url": "https://github.com/owner/repo/blob/main/src/app.py",
                            }
                        ]
                    },
                )
            if request.url.path == "/repos/owner/repo/contents/src/app.py":
                return httpx.Response(
                    200,
                    json={
                        "encoding": "base64",
                        "content": base64.b64encode(b"print('hello from github')\n").decode("utf-8"),
                    },
                )
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ) as client:
            service = GitHubGitService(token="token", repository="owner/repo", client=client)
            results = await service.search_repository("app", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].path, "src/app.py")
        self.assertIn("hello from github", results[0].snippet)
        self.assertEqual(requests, [("GET", "/search/code"), ("GET", "/repos/owner/repo/contents/src/app.py")])

    async def test_apply_changes_and_open_pull_request_creates_branch_updates_file_and_opens_pr(self) -> None:
        calls: list[tuple[str, str, object | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8")) if request.content else None
            calls.append((request.method, request.url.path, payload))
            if request.method == "GET" and request.url.path == "/repos/owner/repo/git/ref/heads/main":
                return httpx.Response(200, json={"object": {"sha": "base-sha"}})
            if request.method == "POST" and request.url.path == "/repos/owner/repo/git/refs":
                return httpx.Response(201, json={"ref": "refs/heads/feature/deploy-docs-update"})
            if request.method == "GET" and request.url.path == "/repos/owner/repo/contents/docs/deploy.md":
                return httpx.Response(404, json={"message": "Not Found"})
            if request.method == "PUT" and request.url.path == "/repos/owner/repo/contents/docs/deploy.md":
                return httpx.Response(201, json={"content": {"path": "docs/deploy.md"}})
            if request.method == "POST" and request.url.path == "/repos/owner/repo/pulls":
                return httpx.Response(201, json={"html_url": "https://github.com/owner/repo/pull/10"})
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.github.com",
        ) as client:
            service = GitHubGitService(token="token", repository="owner/repo", client=client)
            pr_url = await service.apply_changes_and_open_pull_request(
                changes=[
                    ProposedFileChange(
                        path="docs/deploy.md",
                        content="# Deploy\n\nUpdated steps.\n",
                        summary="Refresh deploy steps.",
                    )
                ],
                draft=PullRequestDraft(
                    title="Update deploy docs",
                    body="## Summary\nRefresh deploy docs.",
                    branch_name="feature/deploy-docs-update",
                ),
                base_branch="main",
            )

        self.assertEqual(pr_url, "https://github.com/owner/repo/pull/10")
        self.assertEqual(calls[1][2], {"ref": "refs/heads/feature/deploy-docs-update", "sha": "base-sha"})
        self.assertEqual(calls[3][2]["branch"], "feature/deploy-docs-update")
        self.assertEqual(calls[4][2]["head"], "feature/deploy-docs-update")


class TestValidationLoopTests(unittest.IsolatedAsyncioTestCase):        
    """Tests for the test validation loop added to _run_aider."""

    def _make_workflow(
        self,
        *,
        slack: FakeSlackGateway,
        git: FakeGitService,
        llm: FakeLanguageModel | None = None,
        github_token: str = "stub",
    ) -> ActionWorkflow:
        return ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm or FakeLanguageModel(),
            github_token=github_token,
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

    async def test_tests_pass_on_first_attempt_proceeds_to_pr(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        with (
            mock.patch.object(ActionWorkflow, "_detect_test_command", return_value=["pytest"]),
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(
                    test_returncode=0, test_stdout="All tests passed",
                ),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C1", thread_ts="1.1", request="Fix it")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(git.pr_calls), 1)

    async def test_tests_fail_all_attempts_returns_error(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        with (
            mock.patch.object(ActionWorkflow, "_detect_test_command", return_value=["pytest"]),
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_git_aware_side_effect(
                    test_returncode=1,
                    test_stdout="FAILED test_foo.py::test_bar",
                    test_stderr="AssertionError",
                ),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C1", thread_ts="1.1", request="Fix it")
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(len(git.pr_calls), 0)
        posted = slack.messages[0]["text"]
        self.assertIn("tests failed after 3 attempts", posted)
        self.assertIn("FAILED test_foo.py", posted)

    async def test_no_test_framework_skips_validation(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        with (
            mock.patch.object(ActionWorkflow, "_detect_test_command", return_value=None),
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=_make_successful_side_effect(),
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", return_value=_make_aider_result()),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C1", thread_ts="1.1", request="Fix it")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(git.pr_calls), 1)

    async def test_tests_fail_then_pass_after_retry(self) -> None:
        """Tests fail on first attempt but pass on second (after Aider fix)."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        test_call_count = 0
        aider_called = False

        async def custom_side_effect(_fn, cmd, **kwargs):
            nonlocal test_call_count
            if cmd[0] != "git":
                # Test runner commands only (aider goes through _run_aider_subprocess)
                if cmd[0] in ("npm", "pytest"):
                    test_call_count += 1
                    if test_call_count == 1:
                        return _make_proc(returncode=1, stdout="FAILED", stderr="err")
                    return _make_proc(returncode=0, stdout="All passed", stderr="")
                return _make_proc(returncode=0, stdout="OK", stderr="")

            # Git commands
            subcmd = cmd[1] if len(cmd) > 1 else ""
            if subcmd == "rev-parse":
                if "--abbrev-ref" in cmd:
                    return _make_proc(returncode=0, stdout="main")
                # After aider ran, return different SHA to indicate commits
                if aider_called:
                    return _make_proc(returncode=0, stdout="new-sha-111")
                return _make_proc(returncode=0, stdout="base-sha-000")
            if subcmd == "stash":
                return _make_proc(returncode=0, stdout="No local changes to save")
            return _make_proc(returncode=0)

        async def aider_side_effect(*args, **kwargs):
            nonlocal aider_called
            aider_called = True
            return _make_aider_result()

        with (
            mock.patch.object(ActionWorkflow, "_detect_test_command", return_value=["pytest"]),
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=custom_side_effect,
            ),
            mock.patch.object(ActionWorkflow, "_run_aider_subprocess", side_effect=aider_side_effect),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C1", thread_ts="1.1", request="Fix it")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(git.pr_calls), 1)
        self.assertEqual(test_call_count, 2)


class TestDetectTestCommand(unittest.TestCase):
    """Unit tests for ActionWorkflow._detect_test_command."""

    def _make_workflow(self, work_dir: str = "") -> ActionWorkflow:
        return ActionWorkflow(
            slack=FakeSlackGateway(),
            git=FakeGitService(),
            llm=FakeLanguageModel(),
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

    def test_detects_npm_from_package_json(self) -> None:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "package.json"), "w").close()
            wf = self._make_workflow(d)
            self.assertEqual(wf._detect_test_command(d), ["npm", "test"])

    def test_detects_pytest_from_pytest_ini(self) -> None:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "pytest.ini"), "w").close()
            wf = self._make_workflow(d)
            self.assertEqual(wf._detect_test_command(d), ["pytest"])

    def test_detects_pytest_from_pyproject_toml(self) -> None:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "pyproject.toml"), "w") as f:
                f.write("[tool.pytest.ini_options]\n")
            wf = self._make_workflow(d)
            self.assertEqual(wf._detect_test_command(d), ["pytest"])

    def test_detects_pytest_from_tests_directory(self) -> None:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tests"))
            wf = self._make_workflow(d)
            self.assertEqual(wf._detect_test_command(d), ["pytest"])

    def test_returns_none_for_empty_repo(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            wf = self._make_workflow(d)
            self.assertIsNone(wf._detect_test_command(d))


class SubprocessTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Tests for subprocess timeout handling in ActionWorkflow."""

    def _make_workflow(self) -> ActionWorkflow:
        return ActionWorkflow(
            slack=FakeSlackGateway(),
            git=FakeGitService(),
            llm=FakeLanguageModel(),
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            repo_map=["owner/repo"],
        )

    async def test_git_timeout_returns_error_result(self) -> None:
        """When a git command times out, _git returns a synthetic error."""
        wf = self._make_workflow()

        async def timeout_side_effect(_fn, cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=120)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=timeout_side_effect,
        ):
            result = await wf._git("push", "origin", "main")

        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)

    async def test_test_command_timeout_returns_error_result(self) -> None:
        """When a test command times out, _run_test_command returns a synthetic error."""
        wf = self._make_workflow()

        async def timeout_side_effect(_fn, cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=timeout_side_effect,
        ):
            result = await wf._run_test_command(["pytest"], {})

        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)


class TestRepoKeyFromPrUrl(unittest.TestCase):
    """Unit tests for ActionWorkflow._repo_key_from_pr_url."""

    def test_standard_pr_url(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url(
            "https://github.com/Bforsyth1234/develop-inator/pull/42"
        )
        self.assertEqual(result, "Bforsyth1234/develop-inator")

    def test_pr_url_with_trailing_slash(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url(
            "https://github.com/owner/repo/pull/99/"
        )
        self.assertEqual(result, "owner/repo")

    def test_non_pr_github_url(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url(
            "https://github.com/owner/repo/issues/5"
        )
        self.assertEqual(result, "owner/repo")

    def test_non_github_url_returns_none(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url("https://example.com/foo/bar")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url("")
        self.assertIsNone(result)

    def test_only_owner_returns_none(self) -> None:
        result = ActionWorkflow._repo_key_from_pr_url("https://github.com/owner")
        self.assertIsNone(result)


class TestHandlePrCommentPassesTargetRepo(unittest.IsolatedAsyncioTestCase):
    """Verify handle_pr_comment resolves and passes target_repo_key to _run_aider."""

    def setUp(self) -> None:
        # Disable test detection
        patcher = mock.patch.object(
            ActionWorkflow, "_detect_test_command", return_value=None,
        )
        self._no_tests = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_handle_pr_comment_passes_target_repo_key(self) -> None:
        pr_url = "https://github.com/acme/widgets/pull/7"
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository(
            pr_mapping=ActivePullRequestRecord(
                pr_url=pr_url,
                branch_name="ai-update-123",
                channel_id="C999",
                thread_ts="111.222",
            ),
        )
        llm = FakeLanguageModel()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm,
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            supabase=supabase,
            repo_map=["acme/widgets"],
        )

        captured_kwargs: dict = {}

        async def fake_run_aider(request, **kwargs):
            captured_kwargs.update(kwargs)
            return AiderResult(
                branch_name="ai-update-123",
                stdout="ok",
                stderr="",
                returncode=0,
            )

        with mock.patch.object(workflow, "_run_aider", side_effect=fake_run_aider):
            await workflow.handle_pr_comment(
                pr_url=pr_url,
                comment_body="Please fix the typo",
                sender="reviewer",
            )

        self.assertEqual(captured_kwargs.get("target_repo_key"), "acme/widgets")
        self.assertEqual(captured_kwargs.get("existing_branch"), "ai-update-123")

    async def test_handle_pr_comment_no_mapping_is_noop(self) -> None:
        """When no PR mapping exists, handle_pr_comment should return early."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()  # no pr_mapping

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            aider_model_trivial=_DEFAULT_TRIVIAL_MODEL,
            aider_model_standard=_DEFAULT_STANDARD_MODEL,
            supabase=supabase,
        )

        await workflow.handle_pr_comment(
            pr_url="https://github.com/acme/widgets/pull/99",
            comment_body="Fix it",
            sender="someone",
        )

        # No Slack messages should be posted
        self.assertEqual(len(slack.messages), 0)


if __name__ == "__main__":
    unittest.main()