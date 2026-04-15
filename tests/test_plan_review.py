"""Tests for the plan_task (plan_review.py) pre-implementation design review tool.

Tests cover:
- Tool is registered and callable
- Input validation (missing plan, missing goal)
- Budget gate fires when prompt is oversized
- _get_review_models fallback when OUROBOROS_REVIEW_MODELS not set
- _load_plan_checklist returns non-empty text (section exists in CHECKLISTS.md)
- _format_output aggregate signal logic (GREEN / REVIEW_REQUIRED / REVISE_PLAN)
- Output structure: all reviewer sections present
"""

from __future__ import annotations

import os
import pathlib
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path: pathlib.Path | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.repo_dir = tmp_path or pathlib.Path(".")
    ctx.drive_root = pathlib.Path(".")
    ctx.emit_progress_fn = MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPlanReviewInputValidation(unittest.TestCase):
    def setUp(self):
        from ouroboros.tools.plan_review import _handle_plan_task
        self.handler = _handle_plan_task
        self.ctx = _make_ctx()

    def test_missing_plan_returns_error(self):
        result = self.handler(self.ctx, plan="", goal="some goal")
        self.assertIn("ERROR", result)
        self.assertIn("plan", result.lower())

    def test_missing_goal_returns_error(self):
        result = self.handler(self.ctx, plan="some plan", goal="")
        self.assertIn("ERROR", result)
        self.assertIn("goal", result.lower())

    def test_whitespace_plan_returns_error(self):
        result = self.handler(self.ctx, plan="   ", goal="some goal")
        self.assertIn("ERROR", result)

    def test_whitespace_goal_returns_error(self):
        result = self.handler(self.ctx, plan="some plan", goal="   ")
        self.assertIn("ERROR", result)


class TestPlanReviewModels(unittest.TestCase):
    def test_falls_back_to_main_model_when_not_configured(self):
        from ouroboros.tools.plan_review import _get_review_models
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": "", "OUROBOROS_MODEL": "test/model-x"}, clear=False):
            models = _get_review_models()
        self.assertEqual(len(models), 3)
        self.assertTrue(all(m == "test/model-x" for m in models))

    def test_returns_configured_models(self):
        from ouroboros.tools.plan_review import _get_review_models
        configured = "openai/gpt-5.4,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.6"
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": configured}, clear=False):
            models = _get_review_models()
        self.assertEqual(models, [
            "openai/gpt-5.4",
            "google/gemini-3.1-pro-preview",
            "anthropic/claude-opus-4.6",
        ])

    def test_capped_at_three_models(self):
        from ouroboros.tools.plan_review import _get_review_models
        too_many = "a/1,b/2,c/3,d/4,e/5"
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": too_many}, clear=False):
            models = _get_review_models()
        self.assertEqual(len(models), 3)

    def test_pads_to_three_when_one_model_configured(self):
        """One model configured → pad to exactly 3 by repeating."""
        from ouroboros.tools.plan_review import _get_review_models
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": "only/one"}, clear=False):
            models = _get_review_models()
        self.assertEqual(len(models), 3)
        self.assertTrue(all(m == "only/one" for m in models))

    def test_pads_to_three_when_two_models_configured(self):
        """Two models configured → pad last to reach 3."""
        from ouroboros.tools.plan_review import _get_review_models
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": "model/a,model/b"}, clear=False):
            models = _get_review_models()
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0], "model/a")
        self.assertEqual(models[1], "model/b")
        self.assertEqual(models[2], "model/b")  # last model padded


class TestPlanReviewChecklist(unittest.TestCase):
    def test_checklist_section_exists_and_non_empty(self):
        """Plan Review Checklist section must exist in CHECKLISTS.md."""
        from ouroboros.tools.plan_review import _load_plan_checklist
        checklist = _load_plan_checklist()
        self.assertIsInstance(checklist, str)
        self.assertGreater(len(checklist), 100)
        # Verify key items are present
        self.assertIn("completeness", checklist)
        self.assertIn("correctness", checklist)
        self.assertIn("minimalism", checklist)
        self.assertIn("bible_alignment", checklist)


class TestPlanReviewFormatOutput(unittest.TestCase):
    def _run(self, raw_results):
        from ouroboros.tools.plan_review import _format_output
        return _format_output(raw_results, ["model-a", "model-b", "model-c"], "test goal", 12345)

    def test_green_when_no_fails_or_risks(self):
        results = [
            {"model": "model-a", "text": "PASS on all items.\nAGGREGATE: GREEN", "error": None},
            {"model": "model-b", "text": "Everything looks good.\nAGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "No issues found.\nAGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("GREEN", out)
        self.assertNotIn("REVISE_PLAN", out.split("## Aggregate")[1])

    def test_review_required_when_risk_present(self):
        results = [
            {"model": "model-a", "text": "Some RISK items.\nAGGREGATE: REVIEW_REQUIRED", "error": None},
            {"model": "model-b", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("REVIEW_REQUIRED", out)

    def test_revise_plan_when_fail_present(self):
        results = [
            {"model": "model-a", "text": "Critical FAIL: missing tests.\nAGGREGATE: REVISE_PLAN", "error": None},
            {"model": "model-b", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("REVISE_PLAN", out)

    def test_error_result_does_not_crash(self):
        results = [
            {"model": "model-a", "text": "", "error": "Timeout after 120s"},
            {"model": "model-b", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("ERROR", out)

    def test_error_does_not_downgrade_revise_plan(self):
        """An error from a later reviewer must not downgrade REVISE_PLAN to REVIEW_REQUIRED."""
        results = [
            {"model": "model-a", "text": "Critical FAIL.\nAGGREGATE: REVISE_PLAN", "error": None},
            {"model": "model-b", "text": "", "error": "Timeout after 120s"},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVISE_PLAN", aggregate_section)
        self.assertNotIn("REVIEW_REQUIRED", aggregate_section)

    def test_missing_aggregate_line_yields_review_required(self):
        """A non-error response with no AGGREGATE: line → REVIEW_REQUIRED (not GREEN)."""
        results = [
            {"model": "model-a", "text": "Looks generally fine but some concerns.", "error": None},
            {"model": "model-b", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        # model-a has no aggregate line → should pull aggregate down to REVIEW_REQUIRED
        self.assertIn("REVIEW_REQUIRED", out)
        self.assertNotIn("\n## Aggregate Signal: GREEN", out)

    def test_all_reviewer_sections_present(self):
        results = [
            {"model": "model-a", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-b", "text": "AGGREGATE: GREEN", "error": None},
            {"model": "model-c", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("Reviewer 1", out)
        self.assertIn("Reviewer 2", out)
        self.assertIn("Reviewer 3", out)

    def test_goal_and_token_estimate_in_output(self):
        results = [
            {"model": "model-a", "text": "AGGREGATE: GREEN", "error": None},
        ]
        out = self._run(results)
        self.assertIn("test goal", out)
        self.assertIn("12,345", out)


class TestPlanReviewBudgetGate(unittest.IsolatedAsyncioTestCase):
    async def test_budget_gate_skips_when_oversized(self):
        """When assembled prompt exceeds token limit, returns PLAN_REVIEW_SKIPPED."""
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")

        with (
            patch.object(pr, "build_full_repo_pack", return_value=("x" * 1_000_000, [])),
            patch.object(pr, "build_head_snapshot_section", return_value=""),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "_load_bible", return_value=""),
            patch.object(pr, "_load_doc", return_value=""),
            patch.object(pr, "_get_review_models", return_value=["model-a"]),
            # estimate_tokens returns a large number
            patch("ouroboros.tools.plan_review.estimate_tokens", return_value=900_000),
        ):
            result = await pr._run_plan_review_async(ctx, "my plan", "my goal", [])

        self.assertIn("PLAN_REVIEW_SKIPPED", result)

    async def test_proceeds_when_within_budget(self):
        """When prompt is within budget, reviewers are called."""
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")

        mock_result = {
            "model": "model-a",
            "text": "All good.\nAGGREGATE: GREEN",
            "error": None,
            "tokens_in": 100,
            "tokens_out": 50,
        }

        with (
            patch.object(pr, "build_full_repo_pack", return_value=("small pack", [])),
            patch.object(pr, "build_head_snapshot_section", return_value=""),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "_load_bible", return_value=""),
            patch.object(pr, "_load_doc", return_value=""),
            patch.object(pr, "_get_review_models", return_value=["model-a"]),
            patch("ouroboros.tools.plan_review.estimate_tokens", return_value=10_000),
            patch.object(pr, "_query_reviewer", return_value=mock_result),
        ):
            result = await pr._run_plan_review_async(ctx, "my plan", "my goal", [])

        self.assertIn("Plan Review Results", result)
        self.assertIn("GREEN", result)


class TestParseAggregateSignal(unittest.TestCase):
    def setUp(self):
        from ouroboros.tools.plan_review import _parse_aggregate_signal
        self.parse = _parse_aggregate_signal

    def test_detects_green(self):
        self.assertEqual(self.parse("AGGREGATE: GREEN"), "GREEN")

    def test_detects_review_required(self):
        self.assertEqual(self.parse("AGGREGATE: REVIEW_REQUIRED"), "REVIEW_REQUIRED")

    def test_detects_revise_plan(self):
        self.assertEqual(self.parse("AGGREGATE: REVISE_PLAN"), "REVISE_PLAN")

    def test_case_insensitive(self):
        self.assertEqual(self.parse("aggregate: green"), "GREEN")

    def test_allows_leading_whitespace(self):
        self.assertEqual(self.parse("  AGGREGATE: REVISE_PLAN"), "REVISE_PLAN")

    def test_returns_empty_when_no_aggregate_line(self):
        text = "This is not a REVISE_PLAN case — the situation is fine.\nLooks GREEN to me overall."
        self.assertEqual(self.parse(text), "")

    def test_body_text_does_not_false_positive(self):
        """Reviewer explaining 'This would be REVISE_PLAN if X' should not trigger signal."""
        text = "Normally this would be REVISE_PLAN but in this case it is acceptable.\nAGGREGATE: REVIEW_REQUIRED"
        self.assertEqual(self.parse(text), "REVIEW_REQUIRED")

    def test_last_valid_aggregate_line_wins(self):
        """When multiple AGGREGATE lines exist, LAST one wins (self-correction semantics)."""
        text = "AGGREGATE: GREEN\nAGGREGATE: REVISE_PLAN"
        self.assertEqual(self.parse(text), "REVISE_PLAN")

    def test_last_aggregate_line_wins_when_model_self_corrects(self):
        """Model says REVIEW_REQUIRED, then corrects to REVISE_PLAN — final verdict wins."""
        text = "Initial thought: AGGREGATE: REVIEW_REQUIRED\nAfter reconsideration:\nAGGREGATE: REVISE_PLAN"
        self.assertEqual(self.parse(text), "REVISE_PLAN")


class TestPlanReviewToolRegistration(unittest.TestCase):
    def test_get_tools_returns_plan_task(self):
        from ouroboros.tools.plan_review import get_tools
        tools = get_tools()
        names = [t.name for t in tools]
        self.assertIn("plan_task", names)

    def test_plan_task_schema_has_required_fields(self):
        from ouroboros.tools.plan_review import get_tools
        tool = next(t for t in get_tools() if t.name == "plan_task")
        params = tool.schema["parameters"]["properties"]
        self.assertIn("plan", params)
        self.assertIn("goal", params)
        self.assertIn("files_to_touch", params)
        self.assertEqual(tool.schema["parameters"]["required"], ["plan", "goal"])

    def test_plan_task_description_mentions_pre_implementation(self):
        from ouroboros.tools.plan_review import get_tools
        tool = next(t for t in get_tools() if t.name == "plan_task")
        desc = tool.schema["description"].lower()
        self.assertIn("before", desc)
        self.assertIn("code", desc)


if __name__ == "__main__":
    unittest.main()
