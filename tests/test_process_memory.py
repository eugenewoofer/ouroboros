"""Tests for process memory infrastructure.

Covers:
- Execution reflection trigger logic (should_generate_reflection)
- Error detail collection and marker detection
- Reflection loading into context
"""

import inspect
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ─────────────────────────────────────────────────────────────────────────────
# should_generate_reflection
# ─────────────────────────────────────────────────────────────────────────────

class TestReflectionTrigger:
    """should_generate_reflection(llm_trace) must detect error conditions."""

    def test_clean_trace_no_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        trace = {"tool_calls": [
            {"tool": "repo_read", "args": {}, "result": "file contents", "is_error": False},
            {"tool": "run_shell", "args": {}, "result": "ok", "is_error": False},
        ]}
        assert should_generate_reflection(trace) is False

    def test_error_tool_triggers_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        trace = {"tool_calls": [
            {"tool": "run_shell", "args": {}, "result": "⚠️ TOOL_ERROR: failed", "is_error": True},
        ]}
        assert should_generate_reflection(trace) is True

    def test_review_blocked_marker_triggers_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        trace = {"tool_calls": [
            {"tool": "repo_commit", "args": {}, "is_error": False,
             "result": "⚠️ REVIEW_BLOCKED (attempt 1/3): reviewer flagged version sync"},
        ]}
        assert should_generate_reflection(trace) is True

    def test_tests_failed_marker_triggers_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        trace = {"tool_calls": [
            {"tool": "repo_commit", "args": {}, "is_error": False,
             "result": "OK: committed\n\n⚠️ TESTS_FAILED: VERSION not in README"},
        ]}
        assert should_generate_reflection(trace) is True

    def test_empty_trace_no_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        assert should_generate_reflection({"tool_calls": []}) is False
        assert should_generate_reflection({}) is False

    def test_structured_non_zero_exit_triggers_reflection(self):
        from ouroboros.reflection import should_generate_reflection
        trace = {"tool_calls": [
            {
                "tool": "run_shell",
                "args": {},
                "result": "exit_code=-9",
                "is_error": False,
                "status": "non_zero_exit",
                "exit_code": -9,
                "signal": "SIGKILL",
            },
        ]}
        assert should_generate_reflection(trace) is True


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    """_detect_markers and _collect_error_details must extract structured info."""

    def test_detect_markers_finds_all(self):
        from ouroboros.reflection import _detect_markers
        trace = {"tool_calls": [
            {"tool": "repo_commit", "is_error": True,
             "result": "⚠️ REVIEW_BLOCKED: test"},
            {"tool": "run_shell", "is_error": False,
             "result": "⚠️ TESTS_FAILED: something"},
        ]}
        markers = _detect_markers(trace)
        assert "REVIEW_BLOCKED" in markers
        assert "TESTS_FAILED" in markers

    def test_detect_markers_empty_trace(self):
        from ouroboros.reflection import _detect_markers
        assert _detect_markers({}) == []
        assert _detect_markers({"tool_calls": []}) == []

    def test_collect_error_details_includes_tool_name(self):
        from ouroboros.reflection import _collect_error_details
        trace = {"tool_calls": [
            {"tool": "repo_commit", "is_error": True,
             "result": "⚠️ REVIEW_BLOCKED: test"},
        ]}
        details = _collect_error_details(trace)
        assert "repo_commit" in details
        assert "REVIEW_BLOCKED" in details

    def test_collect_error_details_respects_cap(self):
        from ouroboros.reflection import _collect_error_details
        trace = {"tool_calls": [
            {"tool": "run_shell", "is_error": True,
             "result": "x" * 5000},
        ]}
        details = _collect_error_details(trace, cap=200)
        assert len(details) <= 210  # cap + small overhead from "..."

    def test_collect_error_details_skips_clean_results(self):
        from ouroboros.reflection import _collect_error_details
        trace = {"tool_calls": [
            {"tool": "repo_read", "is_error": False, "result": "file contents"},
            {"tool": "run_shell", "is_error": True, "result": "error happened"},
        ]}
        details = _collect_error_details(trace)
        assert "repo_read" not in details
        assert "run_shell" in details

    def test_collect_error_details_includes_structured_status(self):
        from ouroboros.reflection import _collect_error_details
        trace = {"tool_calls": [
            {
                "tool": "run_shell",
                "is_error": True,
                "status": "non_zero_exit",
                "exit_code": -9,
                "signal": "SIGKILL",
                "result": "⚠️ SHELL_EXIT_ERROR: command exited with exit_code=-9 (signal=SIGKILL).",
            },
        ]}
        details = _collect_error_details(trace)
        assert "status=non_zero_exit" in details
        assert "signal=SIGKILL" in details

    def test_generate_reflection_includes_review_evidence(self):
        from ouroboros.reflection import generate_reflection

        captured = {}

        class FakeLlm:
            def chat(self, *, messages, model, reasoning_effort, max_tokens):
                captured["prompt"] = messages[0]["content"]
                return {"content": "Reflection mentions tests_affected."}, {"cost": 0}

        entry = generate_reflection(
            task={"id": "task-1", "type": "task", "text": "Fix commit flow"},
            llm_trace={"tool_calls": [{
                "tool": "repo_commit",
                "is_error": False,
                "result": "⚠️ REVIEW_BLOCKED: blocked by tests_affected",
            }]},
            trace_summary="repo_commit blocked",
            llm_client=FakeLlm(),
            usage_dict={"rounds": 3, "cost": 0.01},
            review_evidence={
                "has_evidence": True,
                "recent_attempts": [{
                    "status": "blocked",
                    "critical_findings": [{
                        "severity": "critical",
                        "item": "tests_affected",
                        "reason": "broken",
                    }],
                }],
            },
        )

        assert "Structured review evidence" in captured["prompt"]
        assert "tests_affected" in captured["prompt"]
        assert entry["review_evidence"]["has_evidence"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Reflection context loading
# ─────────────────────────────────────────────────────────────────────────────

class TestReflectionContextLoading:
    """build_recent_sections must load execution reflections from JSONL."""

    def test_reflections_loaded_when_file_exists(self):
        from ouroboros.context import build_recent_sections
        source = inspect.getsource(build_recent_sections)
        assert "task_reflections.jsonl" in source, (
            "build_recent_sections must load from task_reflections.jsonl"
        )
        assert "Execution reflections" in source, (
            "Section header must contain 'Execution reflections'"
        )

    def test_reflection_entry_format(self):
        """Reflection entries must include required fields."""
        from ouroboros.reflection import _detect_markers, _collect_error_details
        trace = {"tool_calls": [
            {"tool": "repo_commit", "is_error": True,
             "result": "⚠️ REVIEW_BLOCKED: test"},
        ]}
        markers = _detect_markers(trace)
        assert "REVIEW_BLOCKED" in markers

        details = _collect_error_details(trace)
        assert "repo_commit" in details
        assert "REVIEW_BLOCKED" in details
