"""Tests for ouroboros/tools/ci.py — CI trigger and monitoring tool."""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx(tmp_path):
    """Minimal ToolContext mock."""
    c = MagicMock()
    c.repo_dir = str(tmp_path)
    c.pending_events = []
    return c


@pytest.fixture
def _gh_settings(monkeypatch):
    """Ensure GITHUB_TOKEN and GITHUB_REPO are set."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_123")
    monkeypatch.setenv("GITHUB_REPO", "joi-lab/ouroboros-desktop")
    # Also patch load_settings to return them
    with patch("ouroboros.tools.ci.load_settings", return_value={
        "GITHUB_TOKEN": "ghp_test_token_123",
        "GITHUB_REPO": "joi-lab/ouroboros-desktop",
    }):
        yield


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestGetGithubConfig:
    def test_missing_token_raises(self):
        from ouroboros.tools.ci import _get_github_config
        with patch("ouroboros.tools.ci.load_settings", return_value={}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="GITHUB_TOKEN"):
                    _get_github_config()

    def test_missing_repo_raises(self):
        from ouroboros.tools.ci import _get_github_config
        with patch("ouroboros.tools.ci.load_settings", return_value={"GITHUB_TOKEN": "tok"}):
            with patch.dict(os.environ, {"GITHUB_TOKEN": "tok"}, clear=True):
                with pytest.raises(ValueError, match="GITHUB_REPO"):
                    _get_github_config()

    def test_valid_config(self, _gh_settings):
        from ouroboros.tools.ci import _get_github_config
        token, repo = _get_github_config()
        assert token == "ghp_test_token_123"
        assert repo == "joi-lab/ouroboros-desktop"


class TestExtractOs:
    def test_ubuntu(self):
        from ouroboros.tools.ci import _extract_os
        assert _extract_os("full-test (ubuntu-latest)") == "ubuntu"

    def test_windows(self):
        from ouroboros.tools.ci import _extract_os
        assert _extract_os("full-test (windows-latest)") == "windows"

    def test_macos(self):
        from ouroboros.tools.ci import _extract_os
        assert _extract_os("full-test (macos-latest)") == "macos"

    def test_unknown(self):
        from ouroboros.tools.ci import _extract_os
        assert _extract_os("some-job") == "unknown"


class TestGhApi:
    def test_success(self):
        from ouroboros.tools.ci import _gh_api
        response_data = json.dumps({"id": 1}).encode()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            status, data = _gh_api("GET", "/repos/test/test", "token123")
            assert status == 200
            assert data["id"] == 1

    def test_http_error(self):
        from ouroboros.tools.ci import _gh_api
        import urllib.error
        err = urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=MagicMock(read=lambda: b"not found"),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            status, data = _gh_api("GET", "/repos/test/test", "token123")
            assert status == 404
            assert "error" in data


# ---------------------------------------------------------------------------
# Integration tests: run_ci_tests tool handler
# ---------------------------------------------------------------------------

class TestRunCiTests:
    def test_no_token_returns_unavailable(self, ctx):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci.load_settings", return_value={}):
            with patch.dict(os.environ, {}, clear=True):
                result = _run_ci_tests(ctx)
                assert "CI_UNAVAILABLE" in result

    def test_detached_head_returns_invalid(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="HEAD"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"):
            result = _run_ci_tests(ctx)
            assert "CI_BRANCH_INVALID" in result
            assert "detached HEAD" in result

    def test_remote_mismatch_returns_error(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci.run_cmd", side_effect=lambda cmd, **kw:
                   "https://github.com/other-org/other-repo.git\n" if "get-url" in cmd else "ouroboros\n"):
            result = _run_ci_tests(ctx)
            assert "CI_REMOTE_MISMATCH" in result

    def test_non_github_remote_fails_closed(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci.run_cmd", side_effect=lambda cmd, **kw:
                   "https://gitlab.com/user/repo.git\n" if "get-url" in cmd else "ouroboros\n"):
            result = _run_ci_tests(ctx)
            assert "CI_REMOTE_MISMATCH" in result
            assert "not a GitHub remote" in result

    def test_repo_with_dots_matches(self, ctx):
        """Repos with dots in their name should match correctly."""
        from ouroboros.tools.ci import _run_ci_tests
        settings = {"GITHUB_TOKEN": "test", "GITHUB_REPO": "owner/my.repo.name"}
        with patch("ouroboros.tools.ci.load_settings", return_value=settings), \
             patch.dict(os.environ, {"GITHUB_TOKEN": "test", "GITHUB_REPO": "owner/my.repo.name"}), \
             patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci.run_cmd", side_effect=lambda cmd, **kw:
                   "https://github.com/owner/my.repo.name.git\n" if "get-url" in cmd else "ouroboros\n"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._poll_workflow_run", return_value={
                 "status": "completed", "conclusion": "success",
                 "url": "https://github.com/owner/my.repo.name/actions/runs/1", "run_id": 1}):
            result = _run_ci_tests(ctx)
            assert "CI PASSED" in result  # Should not hit CI_REMOTE_MISMATCH

    def test_push_failure(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(False, "remote rejected")):
            result = _run_ci_tests(ctx)
            assert "CI_PUSH_FAILED" in result

    def test_workflow_not_found(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=None):
            result = _run_ci_tests(ctx)
            assert "CI_WORKFLOW_NOT_FOUND" in result

    def test_trigger_no_wait(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")):
            result = _run_ci_tests(ctx, wait=False)
            assert "CI triggered" in result
            assert "ouroboros" in result

    def test_trigger_failure(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(False, "HTTP 403")):
            result = _run_ci_tests(ctx)
            assert "CI_TRIGGER_FAILED" in result

    def test_ci_success(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        poll_result = {
            "status": "completed",
            "conclusion": "success",
            "url": "https://github.com/test/actions/runs/1",
            "run_id": 1,
        }
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._poll_workflow_run", return_value=poll_result):
            result = _run_ci_tests(ctx)
            assert "CI PASSED" in result
            assert "3 platforms" in result

    def test_ci_failure_with_details(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        poll_result = {
            "status": "completed",
            "conclusion": "failure",
            "url": "https://github.com/test/actions/runs/1",
            "run_id": 1,
        }
        failed_jobs = [
            {"id": 99, "name": "full-test (windows-latest)", "os": "windows",
             "url": "https://github.com/test/jobs/1", "failed_steps": ["Run tests"]},
        ]
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._poll_workflow_run", return_value=poll_result), \
             patch("ouroboros.tools.ci._get_failed_jobs", return_value=failed_jobs), \
             patch("ouroboros.tools.ci._get_job_logs", return_value="FAILED test_x.py::test_foo"):
            result = _run_ci_tests(ctx)
            assert "CI FAILED" in result
            assert "windows" in result
            assert "FAILED test_x.py" in result  # verify log download path exercised

    def test_ci_timeout(self, ctx, _gh_settings):
        from ouroboros.tools.ci import _run_ci_tests
        poll_result = {
            "status": "timeout",
            "conclusion": None,
            "url": "",
            "run_id": None,
        }
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._poll_workflow_run", return_value=poll_result):
            result = _run_ci_tests(ctx, timeout_minutes=1)
            assert "CI_TIMEOUT" in result


class TestNetworkErrorHandling:
    def test_url_error_returns_zero(self):
        from ouroboros.tools.ci import _gh_api
        import urllib.error
        err = urllib.error.URLError("DNS lookup failed")
        with patch("urllib.request.urlopen", side_effect=err):
            status, data = _gh_api("GET", "/repos/test/test", "token123")
            assert status == 0
            assert "Network error" in data["error"]

    def test_timeout_error_returns_zero(self):
        from ouroboros.tools.ci import _gh_api
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            status, data = _gh_api("GET", "/repos/test/test", "token123")
            assert status == 0
            assert "Network error" in data["error"]


class TestToolRegistration:
    def test_get_tools_returns_entry(self):
        from ouroboros.tools.ci import get_tools
        tools = get_tools()
        assert len(tools) == 1
        assert tools[0].name == "run_ci_tests"
        schema = tools[0].schema
        assert "parameters" in schema
        assert "wait" in schema["parameters"]["properties"]
        assert "timeout_minutes" in schema["parameters"]["properties"]


class TestRedirectHandler:
    """Verify _NoAuthRedirectHandler strips Authorization on cross-domain redirects."""

    def test_strips_auth_on_cross_domain(self):
        from ouroboros.tools.ci import _NoAuthRedirectHandler
        import urllib.request

        handler = _NoAuthRedirectHandler()
        # Original request to github.com
        orig_req = urllib.request.Request("https://api.github.com/repos/o/r/actions/jobs/1/logs")
        orig_req.add_header("Authorization", "token ghp_secret123")

        # Simulate redirect to Azure Blob Storage (different domain)
        new_req = handler.redirect_request(
            orig_req, None, 302, "Found", {},
            "https://productionresults.blob.core.windows.net/logs/job-123?sig=abc"
        )
        assert new_req is not None
        assert new_req.get_header("Authorization") is None

    def test_keeps_auth_on_same_domain(self):
        from ouroboros.tools.ci import _NoAuthRedirectHandler
        import urllib.request

        handler = _NoAuthRedirectHandler()
        orig_req = urllib.request.Request("https://api.github.com/repos/o/r/actions/jobs/1/logs")
        orig_req.add_header("Authorization", "token ghp_secret123")

        # Redirect to same domain
        new_req = handler.redirect_request(
            orig_req, None, 302, "Found", {},
            "https://api.github.com/repos/o/r/actions/jobs/1/logs?redirect=true"
        )
        assert new_req is not None
        assert new_req.get_header("Authorization") == "token ghp_secret123"


class TestProgressEmission:
    def test_progress_events_emitted(self, ctx, _gh_settings):
        """Verify that progress events are emitted during workflow."""
        from ouroboros.tools.ci import _run_ci_tests
        poll_result = {
            "status": "completed",
            "conclusion": "success",
            "url": "https://github.com/test/actions/runs/1",
            "run_id": 1,
        }
        with patch("ouroboros.tools.ci._get_current_branch", return_value="ouroboros"), \
             patch("ouroboros.tools.ci._get_current_sha", return_value="abc1234567890"), \
             patch("ouroboros.tools.ci._push_branch", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._find_workflow_id", return_value=12345), \
             patch("ouroboros.tools.ci._trigger_workflow", return_value=(True, "ok")), \
             patch("ouroboros.tools.ci._poll_workflow_run", return_value=poll_result):
            _run_ci_tests(ctx)
            # At least push and trigger progress events should be emitted
            progress_events = [e for e in ctx.pending_events if e.get("type") == "progress"]
            assert len(progress_events) >= 2
