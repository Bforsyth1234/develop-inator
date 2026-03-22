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


class FakeGitService:
    """Records create_pull_request calls and returns a canned PR URL."""

    def __init__(self, *, pr_url: str = "https://example.invalid/pr/42", fail: bool = False) -> None:
        self.pr_url = pr_url
        self.fail = fail
        self.pr_calls: list[PullRequestDraft] = []
        self.resolved_threads: list[tuple[str, str]] = []

    async def create_pull_request(self, draft: PullRequestDraft) -> str:
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
            "complexity_tier": "simple",
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
    ) -> None:
        self._stored_config = stored_config
        self._fail_save = fail_save
        self._pr_mapping = pr_mapping
        self._stored_execution = stored_execution
        self.save_calls: list[dict[str, str]] = []
        self.pr_mapping_saves: list[ActivePullRequestRecord] = []
        self.execution_status_updates: list[tuple[str, ActionExecutionStatus]] = []

    async def healthcheck(self) -> bool:
        return True

    async def get_thread_messages(self, *, channel_id, thread_ts, limit=50):
        return []

    async def match_chunks(self, query_embedding, *, query_text="", limit=5, min_similarity=0.0, metadata_filter=None):
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


def _make_git_aware_side_effect(
    aider_stdout: str = "Aider applied changes.",
    aider_returncode: int = 0,
    aider_stderr: str = "",
    aider_makes_commits: bool = True,
    test_returncode: int = 0,
    test_stdout: str = "",
    test_stderr: str = "",
):
    """Return a side_effect for asyncio.to_thread that understands the full git workflow.

    The _run_aider flow:
      rev-parse --abbrev-ref HEAD → stash → fetch origin main →
      rev-parse origin/main (base SHA) → checkout -b ... origin/main →
      aider → rev-parse HEAD (current SHA) → [test] → push → checkout base → stash pop
    """
    aider_ran = False

    async def side_effect(_fn, cmd, **kwargs):
        nonlocal aider_ran

        # Non-git command = aider or test runner
        if cmd[0] != "git":
            if not aider_ran:
                # First non-git command is aider
                aider_ran = True
                return _make_proc(returncode=aider_returncode, stdout=aider_stdout, stderr=aider_stderr)
            # Subsequent non-git commands: test runner or aider retry
            cmd_name = cmd[0] if cmd else ""
            if cmd_name in ("npm", "pytest"):
                return _make_proc(returncode=test_returncode, stdout=test_stdout, stderr=test_stderr)
            # Aider retry
            return _make_proc(returncode=0, stdout="Aider fixed tests.", stderr="")

        subcmd = cmd[1] if len(cmd) > 1 else ""

        if subcmd == "rev-parse":
            if "--abbrev-ref" in cmd:
                return _make_proc(returncode=0, stdout="main")
            # After aider ran: return different SHA if aider made commits
            if aider_ran:
                if aider_makes_commits:
                    return _make_proc(returncode=0, stdout="new-sha-111")
                return _make_proc(returncode=0, stdout="base-sha-000")
            # Before aider: base SHA (for origin/main)
            return _make_proc(returncode=0, stdout="base-sha-000")

        if subcmd == "stash":
            return _make_proc(returncode=0, stdout="No local changes to save")

        # fetch, checkout, push, branch -D, etc.
        return _make_proc(returncode=0)

    return side_effect


# Keep backward-compat alias
_make_successful_side_effect = _make_git_aware_side_effect


_DEFAULT_TIER_MAP = {
    "simple": "anthropic/claude-sonnet-4-20250514",
    "complex": "anthropic/claude-opus-4-20250514",
}


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
        github_repository: str = "owner/repo",
        model_tier_map: dict[str, str] | None = None,
    ) -> ActionWorkflow:
        return ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm or FakeLanguageModel(),
            github_token=github_token,
            github_repository=github_repository,
            model_tier_map=model_tier_map or _DEFAULT_TIER_MAP,
        )

    async def test_success_opens_pr_and_posts_link_to_slack(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update the deploy docs")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.provider, _DEFAULT_TIER_MAP["simple"])
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

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_git_aware_side_effect(
                aider_returncode=1,
                aider_stderr="fatal: not a git repository",
                aider_stdout="",
            ),
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

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
        ):
            result = await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update the deploy docs")
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.provider, "error")
        self.assertIn("internal error", result.message)

    async def test_aider_command_includes_model_and_message(self) -> None:
        """The aider subprocess receives --model, --yes-always, and --message with the optimized prompt."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Add a dark-mode toggle to the AppComponent",
            "complexity_tier": "simple",
        })
        captured_cmd: list[list[str]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_cmd.append(cmd)
            return await _delegate(_fn, cmd, **kwargs)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Add a dark-mode toggle")
            )

        # Find specific commands by type
        checkout_cmd = [c for c in captured_cmd if c[0] == "git" and len(c) > 2 and c[1] == "checkout" and "-b" in c][0]
        aider_cmd = [c for c in captured_cmd if c[0] != "git"][0]
        push_cmd = [c for c in captured_cmd if c[0] == "git" and len(c) > 1 and c[1] == "push"][0]

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
            "complexity_tier": "simple",
        })
        captured_cmd: list[list[str]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_cmd.append(cmd)
            return await _delegate(_fn, cmd, **kwargs)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Make the button red")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = [c for c in captured_cmd if c[0] != "git"][0]
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
            "complexity_tier": "simple",
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

    async def test_evaluator_invalid_json_falls_back_to_original_request(self) -> None:
        """If the evaluator returns garbage JSON, the workflow runs with the raw request."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel()
        llm._evaluator_json = {}  # will produce JSON that fails EvaluationResult validation

        captured_cmd: list[list[str]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_cmd.append(cmd)
            return await _delegate(_fn, cmd, **kwargs)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fallback request text")
            )

        # Should still complete using the original request
        self.assertEqual(result.status, "completed")
        aider_cmd = [c for c in captured_cmd if c[0] != "git"][0]
        msg_idx = aider_cmd.index("--message") + 1
        self.assertIn("Fallback request text", aider_cmd[msg_idx])

    async def test_evaluator_llm_exception_falls_back_to_original_request(self) -> None:
        """If the evaluator LLM raises, the workflow runs with the raw request (fail-open)."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(raise_on_generate=True)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Exception fallback test")
            )

        self.assertEqual(result.status, "completed")

    # ------------------------------------------------------------------
    # Complexity-tier model routing tests
    # ------------------------------------------------------------------

    async def test_simple_tier_routes_to_correct_model(self) -> None:
        """A 'simple' complexity_tier sends the simple-tier model to Aider."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Change the button text in HeaderComponent to 'Sign In'",
            "complexity_tier": "simple",
        })
        captured_cmd: list[list[str]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_cmd.append(cmd)
            return await _delegate(_fn, cmd, **kwargs)

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=capture_to_thread,
        ):
            result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Change button text")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = [c for c in captured_cmd if c[0] != "git"][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], _DEFAULT_TIER_MAP["simple"])

    async def test_complex_tier_routes_to_planner(self) -> None:
        """A 'complex' complexity_tier invokes the Planner instead of Aider directly."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Refactor AuthService to use NgRx store for session state",
            "complexity_tier": "complex",
        })

        # The planner LLM call (second generate call) returns a spec string
        call_count = 0
        original_generate = llm.generate

        async def patched_generate(prompt: str) -> LLMResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is the evaluator
                return await original_generate(prompt)
            # Second call is the planner — return a markdown spec
            return LLMResult(content="## Implementation Spec\n1. Refactor AuthService", provider="fake")

        llm.generate = patched_generate

        result = await self._make_workflow(slack=slack, git=git, llm=llm).run(
            ActionRequest(channel="C123", thread_ts="1710.2", request="Refactor auth to use NgRx")
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.provider, "planner")
        # Should have posted a "generating spec" message + Block Kit blocks
        self.assertTrue(len(slack.messages) >= 1)
        self.assertIn("complex", slack.messages[0]["text"])
        self.assertEqual(len(slack.blocks_calls), 1)
        blocks = slack.blocks_calls[0]["blocks"]
        # Should have approve/reject buttons
        action_block = [b for b in blocks if b.get("type") == "actions"][0]
        action_ids = [e["action_id"] for e in action_block["elements"]]
        self.assertIn("approve_spec", action_ids)
        self.assertIn("reject_spec", action_ids)

    async def test_aider_subprocess_receives_env_with_api_keys(self) -> None:
        """The Aider subprocess env includes GROQ_API_KEY and ANTHROPIC_API_KEY."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        captured_calls: list[tuple[list[str], dict]] = []
        _delegate = _make_git_aware_side_effect()

        async def capture_to_thread(_fn, cmd, **kwargs):
            captured_calls.append((cmd, kwargs))
            return await _delegate(_fn, cmd, **kwargs)

        with (
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=capture_to_thread,
            ),
            mock.patch.dict(
                "os.environ",
                {"GROQ_API_KEY": "gsk_test123", "ANTHROPIC_API_KEY": "sk-ant-test456"},
            ),
        ):
            await self._make_workflow(slack=slack, git=git).run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Update deploy docs")
            )

        # Find the aider call (non-git command) — it should have env kwarg
        aider_call = [(cmd, kw) for cmd, kw in captured_calls if cmd[0] != "git"]
        self.assertEqual(len(aider_call), 1)
        aider_kwargs = aider_call[0][1]
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
            github_repository="owner/my-repo",
            supabase=supabase,
            model_tier_map=_DEFAULT_TIER_MAP,
        )

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
        ):
            result = await workflow.run(
                ActionRequest(channel="C123", thread_ts="1710.2", request="Fix the bug")
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(supabase.save_calls), 1)
        self.assertEqual(supabase.save_calls[0]["github_repository"], "owner/my-repo")

    async def test_success_without_supabase_skips_config_save(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            github_token="stub",
            github_repository="owner/repo",
            model_tier_map=_DEFAULT_TIER_MAP,
        )

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
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
            github_repository="owner/repo",
            supabase=supabase,
            model_tier_map=_DEFAULT_TIER_MAP,
        )

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_successful_side_effect(),
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
            github_repository="owner/repo",
            supabase=supabase,
            model_tier_map=_DEFAULT_TIER_MAP,
        )

        with mock.patch(
            "slack_bot_backend.workflows.action.asyncio.to_thread",
            side_effect=_make_git_aware_side_effect(
                aider_returncode=1, aider_stderr="error", aider_stdout="",
            ),
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
        github_repository: str = "owner/repo",
    ) -> ActionWorkflow:
        return ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm or FakeLanguageModel(),
            github_token=github_token,
            github_repository=github_repository,
            model_tier_map=_DEFAULT_TIER_MAP,
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
        aider_ran = False

        async def custom_side_effect(_fn, cmd, **kwargs):
            nonlocal test_call_count, aider_ran
            if cmd[0] != "git":
                if not aider_ran:
                    aider_ran = True
                    return _make_proc(returncode=0, stdout="Aider applied changes.", stderr="")
                # Check if this is a test command
                if cmd[0] in ("npm", "pytest"):
                    test_call_count += 1
                    if test_call_count == 1:
                        return _make_proc(returncode=1, stdout="FAILED", stderr="err")
                    return _make_proc(returncode=0, stdout="All passed", stderr="")
                # Aider retry
                return _make_proc(returncode=0, stdout="Fixed.", stderr="")

            # Git commands
            subcmd = cmd[1] if len(cmd) > 1 else ""
            if subcmd == "rev-parse":
                if "--abbrev-ref" in cmd:
                    return _make_proc(returncode=0, stdout="main")
                # After aider ran, return different SHA to indicate commits
                if aider_ran:
                    return _make_proc(returncode=0, stdout="new-sha-111")
                return _make_proc(returncode=0, stdout="base-sha-000")
            if subcmd == "stash":
                return _make_proc(returncode=0, stdout="No local changes to save")
            return _make_proc(returncode=0)

        with (
            mock.patch.object(ActionWorkflow, "_detect_test_command", return_value=["pytest"]),
            mock.patch(
                "slack_bot_backend.workflows.action.asyncio.to_thread",
                side_effect=custom_side_effect,
            ),
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
            github_repository="owner/repo",
            model_tier_map=_DEFAULT_TIER_MAP,
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


class ExecuteApprovedTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the _execute_approved background task in the approval flow."""

    def setUp(self) -> None:
        patcher = mock.patch.object(
            ActionWorkflow, "_detect_test_command", return_value=None,
        )
        self._no_tests = patcher.start()
        self.addCleanup(patcher.stop)

    def _make_execution(self) -> ActionExecution:
        return ActionExecution(
            id="exec-001",
            channel="C123",
            thread_ts="1710.2",
            user_id="U456",
            original_request="Refactor auth module",
            generated_spec="## Spec\n1. Refactor AuthService",
            status="approved",
            model="anthropic/claude-sonnet-4-20250514",
        )

    def _make_container(
        self,
        *,
        slack: FakeSlackGateway,
        git: FakeGitService,
        supabase: FakeSupabaseRepository,
    ) -> ServiceContainer:
        llm = FakeLanguageModel()
        action = ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm,
            github_token="stub",
            github_repository="owner/repo",
            supabase=supabase,
            model_tier_map=_DEFAULT_TIER_MAP,
        )
        question = QuestionWorkflow(slack=slack, supabase=supabase, llm=llm)
        return ServiceContainer(
            settings=Settings(environment="testing"),
            slack=slack,
            supabase=supabase,
            llm=llm,
            git=git,
            action=action,
            intent=IntentWorkflow(
                slack=slack, supabase=supabase, llm=llm,
                question=question, action=action,
            ),
            question=question,
        )

    async def test_approved_spec_creates_pr_and_posts_success(self) -> None:
        """After approval, _handle_spec_action dispatches to Celery."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        execution = self._make_execution()
        supabase = FakeSupabaseRepository(stored_execution=execution)
        container = self._make_container(slack=slack, git=git, supabase=supabase)

        with mock.patch(
            "slack_bot_backend.api.routes.process_spec_approval_task"
        ) as mock_task:
            from slack_bot_backend.api.routes import _handle_spec_action
            payload = {
                "actions": [{"action_id": "approve_spec", "value": "exec-001"}],
                "container": {"message_ts": "1710.3"},
            }
            result = await _handle_spec_action(container, payload)

        self.assertEqual(result, {"ok": True})
        mock_task.apply_async.assert_called_once_with(
            kwargs={"execution_id": "exec-001"},
        )

    async def test_approved_spec_with_existing_branch_uses_it(self) -> None:
        """When a PR mapping exists, _handle_spec_action still dispatches to Celery."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        execution = self._make_execution()
        pr_mapping = ActivePullRequestRecord(
            pr_url="https://github.com/owner/repo/pull/10",
            branch_name="ai-update-existing",
            channel_id="C123",
            thread_ts="1710.2",
        )
        supabase = FakeSupabaseRepository(stored_execution=execution, pr_mapping=pr_mapping)
        container = self._make_container(slack=slack, git=git, supabase=supabase)

        with mock.patch(
            "slack_bot_backend.api.routes.process_spec_approval_task"
        ) as mock_task:
            from slack_bot_backend.api.routes import _handle_spec_action
            payload = {
                "actions": [{"action_id": "approve_spec", "value": "exec-001"}],
                "container": {"message_ts": "1710.3"},
            }
            result = await _handle_spec_action(container, payload)

        self.assertEqual(result, {"ok": True})
        mock_task.apply_async.assert_called_once_with(
            kwargs={"execution_id": "exec-001"},
        )

    async def test_approved_spec_aider_failure_posts_error(self) -> None:
        """When approval is dispatched, Celery task is invoked."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        execution = self._make_execution()
        supabase = FakeSupabaseRepository(stored_execution=execution)
        container = self._make_container(slack=slack, git=git, supabase=supabase)

        with mock.patch(
            "slack_bot_backend.api.routes.process_spec_approval_task"
        ) as mock_task:
            from slack_bot_backend.api.routes import _handle_spec_action
            payload = {
                "actions": [{"action_id": "approve_spec", "value": "exec-001"}],
                "container": {"message_ts": "1710.3"},
            }
            result = await _handle_spec_action(container, payload)

        self.assertEqual(result, {"ok": True})
        mock_task.apply_async.assert_called_once()

    async def test_approved_spec_exception_posts_error_to_slack(self) -> None:
        """Rejection updates execution status and updates the Slack message."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        execution = self._make_execution()
        supabase = FakeSupabaseRepository(stored_execution=execution)
        container = self._make_container(slack=slack, git=git, supabase=supabase)

        from slack_bot_backend.api.routes import _handle_spec_action
        payload = {
            "actions": [{"action_id": "reject_spec", "value": "exec-001"}],
            "container": {"message_ts": "1710.3"},
        }
        result = await _handle_spec_action(container, payload)

        self.assertEqual(result, {"ok": True})
        # Rejection message should have been sent via update_message
        rejection_updates = [u for u in slack.update_calls if "rejected" in (u["text"] or "").lower()]
        self.assertTrue(len(rejection_updates) >= 1, f"Expected rejection update, got: {slack.update_calls}")


class SubprocessTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Tests for subprocess timeout handling in ActionWorkflow."""

    def _make_workflow(self) -> ActionWorkflow:
        return ActionWorkflow(
            slack=FakeSlackGateway(),
            git=FakeGitService(),
            llm=FakeLanguageModel(),
            github_token="stub",
            github_repository="owner/repo",
            model_tier_map=_DEFAULT_TIER_MAP,
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


if __name__ == "__main__":
    unittest.main()