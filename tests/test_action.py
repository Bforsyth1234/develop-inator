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
from slack_bot_backend.models.action import ActionRequest, ActionRouteResult, ProposedFileChange, RepositorySearchResult
from slack_bot_backend.services.github import GitHubGitService
from slack_bot_backend.services.interfaces import LLMResult, PullRequestDraft
from slack_bot_backend.services.stubs import StubGitService, StubLanguageModel, StubSlackGateway, StubSupabaseRepository
from slack_bot_backend.services.supabase_persistence import RepositoryConfig
from slack_bot_backend.workflows import ActionWorkflow, IntentWorkflow, QuestionWorkflow


class FakeSlackGateway:
    def __init__(self) -> None:
        self.messages: list[dict[str, str | None]] = []

    async def post_message(self, channel: str, text: str, thread_ts: str | None = None) -> None:
        self.messages.append({"channel": channel, "text": text, "thread_ts": thread_ts})


class FakeGitService:
    """Records create_pull_request calls and returns a canned PR URL."""

    def __init__(self, *, pr_url: str = "https://example.invalid/pr/42", fail: bool = False) -> None:
        self.pr_url = pr_url
        self.fail = fail
        self.pr_calls: list[PullRequestDraft] = []

    async def create_pull_request(self, draft: PullRequestDraft) -> str:
        self.pr_calls.append(draft)
        if self.fail:
            raise RuntimeError("GitHub unavailable")
        return self.pr_url


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

    def __init__(self, *, stored_config: RepositoryConfig | None = None, fail_save: bool = False) -> None:
        self._stored_config = stored_config
        self._fail_save = fail_save
        self.save_calls: list[dict[str, str]] = []

    async def healthcheck(self) -> bool:
        return True

    async def get_thread_messages(self, *, channel_id, thread_ts, limit=50):
        return []

    async def match_chunks(self, query_embedding, *, query_text="", limit=5, min_similarity=0.0, metadata_filter=None):
        return []

    async def get_repository_config(self) -> RepositoryConfig | None:
        return self._stored_config

    async def save_repository_config(self, *, repo_path: str, github_repository: str) -> None:
        if self._fail_save:
            raise RuntimeError("Supabase unavailable")
        self.save_calls.append({"repo_path": repo_path, "github_repository": github_repository})


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
):
    """Return a side_effect for asyncio.to_thread that understands the full git workflow.

    The _run_aider flow:
      rev-parse --abbrev-ref HEAD → stash → fetch origin main →
      rev-parse origin/main (base SHA) → checkout -b ... origin/main →
      aider → rev-parse HEAD (current SHA) → push → checkout base → stash pop
    """
    aider_ran = False

    async def side_effect(_fn, cmd, **kwargs):
        nonlocal aider_ran

        # Non-git command = aider
        if cmd[0] != "git":
            aider_ran = True
            return _make_proc(returncode=aider_returncode, stdout=aider_stdout, stderr=aider_stderr)

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
    "simple": "groq/llama-3.3-70b-versatile",
    "complex": "anthropic/claude-sonnet-4-20250514",
}


class ActionWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def _make_workflow(
        self,
        *,
        slack: FakeSlackGateway,
        git: FakeGitService,
        llm: FakeLanguageModel | None = None,
        repo_path: str = "/tmp/repo",
        model_tier_map: dict[str, str] | None = None,
    ) -> ActionWorkflow:
        return ActionWorkflow(
            slack=slack,
            git=git,
            llm=llm or FakeLanguageModel(),
            repo_path=repo_path,
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
        self.assertEqual(result.provider, "groq/llama-3.3-70b-versatile")
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

        # push uses the same branch
        self.assertEqual(push_cmd[0], "git")
        self.assertEqual(push_cmd[1], "push")
        self.assertEqual(push_cmd[3], checkout_cmd[3])

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

    async def test_simple_tier_routes_to_groq_model(self) -> None:
        """A 'simple' complexity_tier sends the Groq model to Aider."""
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
        self.assertEqual(aider_cmd[model_idx], "groq/llama-3.3-70b-versatile")

    async def test_complex_tier_routes_to_anthropic_model(self) -> None:
        """A 'complex' complexity_tier sends the Anthropic model to Aider."""
        slack = FakeSlackGateway()
        git = FakeGitService()
        llm = FakeLanguageModel(evaluator_json={
            "is_actionable": True,
            "clarifying_question": None,
            "optimized_prompt": "Refactor AuthService to use NgRx store for session state",
            "complexity_tier": "complex",
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
                ActionRequest(channel="C123", thread_ts="1710.2", request="Refactor auth to use NgRx")
            )

        self.assertEqual(result.status, "completed")
        aider_cmd = [c for c in captured_cmd if c[0] != "git"][0]
        model_idx = aider_cmd.index("--model") + 1
        self.assertEqual(aider_cmd[model_idx], "anthropic/claude-sonnet-4-20250514")

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

    async def test_success_saves_repo_config_to_supabase(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()
        supabase = FakeSupabaseRepository()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            repo_path="/home/user/my-repo",
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
        self.assertEqual(supabase.save_calls[0]["repo_path"], "/home/user/my-repo")
        self.assertEqual(supabase.save_calls[0]["github_repository"], "owner/my-repo")

    async def test_success_without_supabase_skips_config_save(self) -> None:
        slack = FakeSlackGateway()
        git = FakeGitService()

        workflow = ActionWorkflow(
            slack=slack,
            git=git,
            llm=FakeLanguageModel(),
            repo_path="/tmp/repo",
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
            repo_path="/tmp/repo",
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
            repo_path="/tmp/repo",
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


if __name__ == "__main__":
    unittest.main()