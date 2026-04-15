"""Tests for advisory_pre_review observability, model-drift fix, and budget gate.

Split from test_commit_gate.py to keep each test module within the ~1000-line limit (P5).
"""
import importlib
import importlib.util as _ilu
import json
import os
import sys
import types

import asyncio

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_sdk_mock():
    """Install a lightweight mock of claude_agent_sdk only when truly absent."""
    try:
        spec = _ilu.find_spec("claude_agent_sdk")
        sdk_available = spec is not None
    except (ValueError, ModuleNotFoundError):
        sdk_available = "claude_agent_sdk" in sys.modules
    if not sdk_available:
        mock_sdk = types.ModuleType("claude_agent_sdk")
        mock_sdk.ClaudeAgentOptions = type("ClaudeAgentOptions", (), {})
        mock_sdk.ClaudeSDKClient = type("ClaudeSDKClient", (), {})
        mock_sdk.HookMatcher = type("HookMatcher", (), {"__init__": lambda self, **kw: None})
        mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
        mock_sdk.ResultMessage = type("ResultMessage", (), {})
        mock_sdk.query = lambda **kw: None
        sys.modules["claude_agent_sdk"] = mock_sdk


_ensure_sdk_mock()


def _get_advisory_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.tools.claude_advisory_review")


# ---------------------------------------------------------------------------
# Model-drift: resolve_claude_code_model
# ---------------------------------------------------------------------------

def test_resolve_claude_code_model_returns_env_value(monkeypatch):
    """resolve_claude_code_model must return CLAUDE_CODE_MODEL env var value."""
    sys.path.insert(0, REPO)
    gw = importlib.import_module("ouroboros.gateways.claude_code")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "sonnet")
    assert gw.resolve_claude_code_model() == "sonnet"


def test_resolve_claude_code_model_falls_back_to_opus(monkeypatch):
    """resolve_claude_code_model defaults to 'opus' when env var is absent."""
    sys.path.insert(0, REPO)
    gw = importlib.import_module("ouroboros.gateways.claude_code")
    monkeypatch.delenv("CLAUDE_CODE_MODEL", raising=False)
    assert gw.resolve_claude_code_model() == "opus"


def test_resolve_claude_code_model_strips_whitespace(monkeypatch):
    """resolve_claude_code_model strips leading/trailing whitespace."""
    sys.path.insert(0, REPO)
    gw = importlib.import_module("ouroboros.gateways.claude_code")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "  claude-opus-4.6  ")
    assert gw.resolve_claude_code_model() == "claude-opus-4.6"


def test_shell_edit_uses_resolve_claude_code_model_helper():
    """claude_code_edit path must use resolve_claude_code_model(), not raw os.environ.get."""
    import inspect
    sys.path.insert(0, REPO)
    shell_mod = importlib.import_module("ouroboros.tools.shell")
    source = inspect.getsource(shell_mod._claude_code_edit)
    assert "resolve_claude_code_model" in source
    assert 'os.environ.get("CLAUDE_CODE_MODEL"' not in source


def test_advisory_uses_resolve_claude_code_model_helper():
    """_run_claude_advisory must call resolve_claude_code_model() — no hardcoded 'opus'."""
    import inspect
    adv_mod = _get_advisory_module()
    source = inspect.getsource(adv_mod._run_claude_advisory)
    assert "resolve_claude_code_model" in source


# ---------------------------------------------------------------------------
# Observability: _format_advisory_error / _get_runtime_diagnostics
# ---------------------------------------------------------------------------

def test_advisory_error_message_includes_diagnostic_fields():
    """_format_advisory_error must include all required diagnostic fields."""
    adv_mod = _get_advisory_module()
    diag = {
        "model": "opus",
        "sdk_version": "0.1.56",
        "cli_version": "2.1.92",
        "cli_path": "/app/claude",
        "python": "/usr/bin/python3",
        "prompt_chars": 120000,
        "prompt_tokens_approx": 30000,
        "touched_paths": ["ouroboros/tools/foo.py"],
    }
    msg = adv_mod._format_advisory_error(
        prefix="test failure",
        result_error="exit code 1",
        stderr_tail="some stderr line",
        session_id="sess-123",
        diag=diag,
    )
    assert "⚠️ ADVISORY_ERROR:" in msg
    assert "opus" in msg
    assert "0.1.56" in msg
    assert "2.1.92" in msg
    assert "/app/claude" in msg
    assert "120000" in msg
    assert "30000" in msg or "30,000" in msg
    assert "sess-123" in msg
    assert "some stderr line" in msg
    assert "ouroboros/tools/foo.py" in msg


def test_get_runtime_diagnostics_never_raises():
    """_get_runtime_diagnostics must return partial data on any error, never raise."""
    adv_mod = _get_advisory_module()
    diag = adv_mod._get_runtime_diagnostics("opus", 50000, ["file.py"])
    assert isinstance(diag, dict)
    assert diag["model"] == "opus"
    assert diag["prompt_chars"] == 50000
    assert diag["prompt_tokens_approx"] == 12500
    assert diag["touched_paths"] == ["file.py"]
    assert "sdk_version" in diag


def test_get_runtime_diagnostics_reads_runtime_state_attributes(monkeypatch):
    """Runtime diagnostics must read cli_path/cli_version from ClaudeRuntimeState attributes."""
    adv_mod = _get_advisory_module()
    from ouroboros.platform_layer import ClaudeRuntimeState

    monkeypatch.setattr(
        "ouroboros.platform_layer.resolve_claude_runtime",
        lambda: ClaudeRuntimeState(
            cli_path="/app/claude",
            cli_version="2.1.92",
        ),
    )
    diag = adv_mod._get_runtime_diagnostics("opus", 1234, ["file.py"])

    assert diag["cli_path"] == "/app/claude"
    assert diag["cli_version"] == "2.1.92"


# ---------------------------------------------------------------------------
# Budget gate: skip path and durable state
# ---------------------------------------------------------------------------

def _make_minimal_git_repo(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "BIBLE.md").write_text("bible", encoding="utf-8")
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text("# Repo Commit Checklist\n", encoding="utf-8")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)


def test_advisory_budget_gate_returns_skipped_on_large_prompt(monkeypatch, tmp_path):
    """_run_claude_advisory must return ADVISORY_SKIPPED when prompt exceeds budget gate."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    try:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
        from types import SimpleNamespace
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path,
                              emit_progress_fn=lambda _: None, pending_events=[])
        items, raw = adv_mod._run_claude_advisory(tmp_path, "test commit", ctx)
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit

    assert items == []
    assert raw.startswith("⚠️ ADVISORY_SKIPPED:")
    assert "chars" in raw


def test_handle_advisory_pre_review_returns_skipped_status_on_budget_gate(
    monkeypatch, tmp_path
):
    """_handle_advisory_pre_review must surface ADVISORY_SKIPPED as status='skipped'."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    try:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
        from types import SimpleNamespace
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-test",
                              emit_progress_fn=lambda _: None, pending_events=[])
        raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="test commit")
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit

    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    assert "ADVISORY_SKIPPED" in result["message"]


def test_budget_gate_skip_persists_durable_state(monkeypatch, tmp_path):
    """Budget-gate skip must write status='skipped' to state; is_fresh() must return True."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    try:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
        from types import SimpleNamespace
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-bg",
                              emit_progress_fn=lambda _: None, pending_events=[])
        raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="budget gate test")
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit

    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    snapshot_hash = result["snapshot_hash"]

    from ouroboros.review_state import load_state
    state = load_state(tmp_path)
    assert state.is_fresh(snapshot_hash), (
        "is_fresh() must be True after budget-gate skip so commit gate does not re-block"
    )
    run = state.find_by_hash(snapshot_hash)
    assert run is not None
    assert run.status == "skipped"


def test_next_step_guidance_for_skipped_advisory():
    """_next_step_guidance must return a distinct message for status='skipped' runs."""
    adv_mod = _get_advisory_module()
    from ouroboros.review_state import AdvisoryRunRecord, AdvisoryReviewState

    skipped_run = AdvisoryRunRecord(
        snapshot_hash="abc123",
        commit_message="test",
        status="skipped",
        ts="2026-01-01T00:00:00",
    )
    state = AdvisoryReviewState(advisory_runs=[skipped_run])
    msg = adv_mod._next_step_guidance(
        latest=skipped_run,
        state=state,
        stale_from_edit=False,
        stale_from_edit_ts=None,
        open_obs=[],
        effective_is_fresh=True,
    )
    # Must NOT say "fresh" or "no critical findings" — that would mislead
    assert "skip" in msg.lower() or "budget" in msg.lower(), (
        "skipped advisory must produce a distinct message, not the generic fresh-pass message"
    )
    assert "repo_commit" in msg, "message should still indicate commit can proceed"


def test_skipped_run_hash_mismatch_reported_as_stale(monkeypatch, tmp_path):
    """A skipped run with a different snapshot hash must be reported as stale (hash_mismatch path)."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    import subprocess
    # Commit BIBLE.md so git has a real HEAD
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True)

    # Write a skipped run with a fake (stale) hash directly into state
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState, load_state, save_state,
    )
    old_hash = "000000000000000000000000000000000000000000000000"
    run = AdvisoryRunRecord(
        snapshot_hash=old_hash,
        commit_message="skipped test",
        status="skipped",
        ts="2026-01-01T00:00:00",
    )
    state = AdvisoryReviewState(advisory_runs=[run])
    save_state(tmp_path, state)

    # Now add a file to the worktree so the real snapshot hash differs from old_hash
    (tmp_path / "new_file.py").write_text("x = 1\n", encoding="utf-8")

    # review_status must report stale (hash mismatch), not fresh
    raw_json = adv_mod._handle_review_status(
        ctx=__import__("types").SimpleNamespace(
            repo_dir=tmp_path, drive_root=tmp_path,
            emit_progress_fn=lambda _: None, pending_events=[],
        )
    )
    import json as _json
    result = _json.loads(raw_json)
    latest_status = result.get("latest_advisory_status", "")
    assert latest_status in ("stale", "no_advisory"), (
        f"Expected stale/no_advisory for skipped run with hash mismatch, got: {latest_status!r}\n"
        f"Full result: {result}"
    )


def test_advisory_context_build_failure_is_surfaced(monkeypatch, tmp_path):
    """Phase 4: changed-file context build failures must surface as explicit advisory errors."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    monkeypatch.setattr(adv_mod, "_get_staged_diff", lambda *args, **kwargs: "(no diff)")
    monkeypatch.setattr(adv_mod, "_get_changed_file_list", lambda *args, **kwargs: "M foo.py")
    monkeypatch.setattr(
        adv_mod,
        "build_advisory_changed_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("context pack exploded")),
    )

    from types import SimpleNamespace
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        emit_progress_fn=lambda _: None,
        pending_events=[],
        task_id="ctx-fail",
    )
    items, raw = adv_mod._run_claude_advisory(tmp_path, "test commit", ctx)
    assert items == []
    assert raw.startswith("⚠️ ADVISORY_ERROR:")
    assert "failed to build advisory prompt" in raw


def test_budget_gate_skip_becomes_stale_after_edit(monkeypatch, tmp_path):
    """A budget-gate skip must be invalidated (marked stale) by a subsequent worktree edit."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    try:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
        from types import SimpleNamespace
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-stale",
                              emit_progress_fn=lambda _: None, pending_events=[])
        raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="skip stale test")
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit

    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    snapshot_hash = result["snapshot_hash"]

    # Simulate a worktree edit invalidating the advisory
    from ouroboros.review_state import load_state, mark_advisory_stale_after_edit
    mark_advisory_stale_after_edit(tmp_path)

    state = load_state(tmp_path)
    assert not state.is_fresh(snapshot_hash), (
        "is_fresh() must be False after mark_advisory_stale_after_edit() — edit invalidates skip"
    )
    run = state.find_by_hash(snapshot_hash)
    assert run is not None
    assert run.status == "stale"


# ---------------------------------------------------------------------------
# SDK break-after-ResultMessage fix (spurious exit code 1 prevention)
# ---------------------------------------------------------------------------

def test_run_readonly_async_breaks_after_result_message():
    """_run_readonly_async must stop iterating after ResultMessage.

    Root cause of the spurious 'exit code 1' error: the SDK's query() generator
    raises when iterated past the ResultMessage because the CLI subprocess has
    already exited and the message reader tries to read from a closed pipe.

    The fix adds a `break` after processing ResultMessage. This test verifies
    that the break prevents the post-ResultMessage Exception from reaching the
    caller as a failure.
    """
    import sys
    import types

    sys.path.insert(0, REPO)

    # Build realistic mock message types
    AssistantMsg = type("AssistantMessage", (), {})
    ResultMsg = type("ResultMessage", (), {})

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeAssistantMessage(AssistantMsg):
        def __init__(self):
            self.content = [FakeTextBlock("Hello")]

    class FakeResultMessage(ResultMsg):
        session_id = "test-session-123"
        total_cost_usd = 0.001
        usage = {"input_tokens": 10, "output_tokens": 5}
        subtype = "success"

    async def fake_query_raises_after_result(prompt, options):
        """Simulates SDK: yields AssistantMessage + ResultMessage, then raises on next iteration."""
        yield FakeAssistantMessage()
        yield FakeResultMessage()
        # This raise simulates the CLI pipe-closed error that happened WITHOUT the break fix
        raise Exception("Command failed with exit code 1 (exit code: 1)\nError output: Check stderr output for details")

    # Patch claude_agent_sdk in the gateway module
    import ouroboros.gateways.claude_code as gw

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs):
            pass  # accept all kwargs from _run_readonly_async

    orig_query = gw.query
    orig_AssistantMessage = gw.AssistantMessage
    orig_ResultMessage = gw.ResultMessage
    orig_ClaudeAgentOptions = gw.ClaudeAgentOptions
    try:
        gw.query = fake_query_raises_after_result
        gw.AssistantMessage = FakeAssistantMessage
        gw.ResultMessage = FakeResultMessage
        gw.ClaudeAgentOptions = FakeClaudeAgentOptions

        result = asyncio.run(gw._run_readonly_async(
            prompt="test",
            cwd="/tmp",
            model="opus",
            max_turns=1,
            effort=None,
        ))
    finally:
        gw.query = orig_query
        gw.AssistantMessage = orig_AssistantMessage
        gw.ResultMessage = orig_ResultMessage
        gw.ClaudeAgentOptions = orig_ClaudeAgentOptions
    assert result.success, f"Expected success but got error: {result.error}"
    assert result.session_id == "test-session-123"
    assert "Hello" in result.result_text


def test_run_edit_async_breaks_after_result_message():
    """_run_edit_async must stop iterating after ResultMessage (edit/ClaudeSDKClient path).

    Companion to test_run_readonly_async_breaks_after_result_message.
    Verifies the same break-after-ResultMessage fix on the ClaudeSDKClient+receive_response path.
    """
    import sys

    sys.path.insert(0, REPO)

    AssistantMsg = type("AssistantMessage", (), {})
    ResultMsg = type("ResultMessage", (), {})

    class FakeTextBlock:
        def __init__(self, text):
            self.text = text

    class FakeAssistantMessage(AssistantMsg):
        def __init__(self):
            self.content = [FakeTextBlock("Edit output")]

    class FakeResultMessage(ResultMsg):
        session_id = "edit-session-456"
        total_cost_usd = 0.002
        usage = {"input_tokens": 20, "output_tokens": 10}
        subtype = "success"

    class FakeSDKClient:
        """Mock ClaudeSDKClient context manager."""
        def __init__(self, options=None):
            self.options = options
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def query(self, prompt):
            pass
        async def receive_response(self):
            yield FakeAssistantMessage()
            yield FakeResultMessage()
            # This simulates the CLI pipe-closed error WITHOUT the break fix
            raise Exception("Command failed with exit code 1 (exit code: 1)\nError output: Check stderr output for details")

    import ouroboros.gateways.claude_code as gw

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs):
            pass

    orig_ClaudeSDKClient = gw.ClaudeSDKClient
    orig_AssistantMessage = gw.AssistantMessage
    orig_ResultMessage = gw.ResultMessage
    orig_ClaudeAgentOptions = gw.ClaudeAgentOptions
    orig_HookMatcher = gw.HookMatcher

    class FakeHookMatcher:
        def __init__(self, **kwargs):
            pass

    try:
        gw.ClaudeSDKClient = FakeSDKClient
        gw.AssistantMessage = FakeAssistantMessage
        gw.ResultMessage = FakeResultMessage
        gw.ClaudeAgentOptions = FakeClaudeAgentOptions
        gw.HookMatcher = FakeHookMatcher

        result = asyncio.run(gw._run_edit_async(
            prompt="test edit",
            cwd="/tmp",
            model="opus",
            max_turns=1,
        ))
    finally:
        gw.ClaudeSDKClient = orig_ClaudeSDKClient
        gw.AssistantMessage = orig_AssistantMessage
        gw.ResultMessage = orig_ResultMessage
        gw.ClaudeAgentOptions = orig_ClaudeAgentOptions
        gw.HookMatcher = orig_HookMatcher
    assert result.success, f"Expected success but got error: {result.error}"
    assert result.session_id == "edit-session-456"
    assert "Edit output" in result.result_text


@pytest.mark.parametrize(
    ("cwd", "expected_repo_name"),
    [
        ("", None),          # self repo root
        ("external", "external"),  # nested external git root
    ],
)
def test_claude_code_edit_invalidates_target_repo_root(monkeypatch, tmp_path, cwd, expected_repo_name):
    """Phase 3: claude_code_edit should invalidate advisory for the nearest git root."""
    from types import SimpleNamespace

    sys.path.insert(0, REPO)
    shell_mod = importlib.import_module("ouroboros.tools.shell")
    git_mod = importlib.import_module("ouroboros.tools.git")
    gw = importlib.import_module("ouroboros.gateways.claude_code")

    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
    target_root = tmp_path
    if expected_repo_name:
        target_root = tmp_path / expected_repo_name
        (target_root / ".git").mkdir(parents=True, exist_ok=True)

    class FakeResult:
        def __init__(self):
            self.success = True
            self.result_text = "ok"
            self.session_id = "sess-1"
            self.cost_usd = 0.0
            self.usage = {}
            self.changed_files = []
            self.diff_stat = ""
            self.validation_summary = ""
            self.error = ""

        def to_tool_output(self):
            return json.dumps({"success": True})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(gw, "resolve_claude_code_model", lambda: "opus")
    monkeypatch.setattr(gw, "run_edit", lambda **kwargs: FakeResult())
    monkeypatch.setattr(git_mod, "_acquire_git_lock", lambda ctx: object())
    monkeypatch.setattr(git_mod, "_release_git_lock", lambda lock: None)
    monkeypatch.setattr(shell_mod, "_load_project_context", lambda repo_dir: "")
    monkeypatch.setattr(shell_mod, "_get_diff_stat", lambda repo_dir: "")
    monkeypatch.setattr(shell_mod, "run_cmd", lambda *args, **kwargs: "")

    change_calls = iter([[], ["foo.py"], ["foo.py"]])
    monkeypatch.setattr(shell_mod, "_get_changed_files", lambda repo_dir: next(change_calls))
    invalidate_calls = []
    monkeypatch.setattr(
        shell_mod,
        "_invalidate_advisory",
        lambda ctx, **kwargs: invalidate_calls.append(kwargs),
    )

    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        emit_progress_fn=lambda *_: None,
        pending_events=[],
    )

    raw = shell_mod._claude_code_edit(ctx, prompt="edit something", cwd=cwd)
    assert json.loads(raw)["success"] is True
    assert len(invalidate_calls) == 1
    mutation_root = invalidate_calls[0]["mutation_root"]
    assert mutation_root == target_root
    assert invalidate_calls[0]["source_tool"] == "claude_code_edit"
    assert invalidate_calls[0]["changed_paths"] == ["foo.py"]
