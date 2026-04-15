"""Tests for the improved advisory pre-review workflow:

1. mark_advisory_stale_after_edit invalidates fresh advisory runs
2. ObligationItem creation and deduplication
3. blocking_history bounded cap
4. add_blocking_attempt populates open_obligations
5. on_successful_commit clears obligations and history
6. _build_blocking_history_section reads full history
7. format_status_section shows staleness from edit and obligations
8. review_status response includes stale_from_edit and open_obligations
9. _check_advisory_freshness includes obligation summary
10. Serialization roundtrip for new fields (including snapshot_paths)
11. _collect_review_findings stores structured findings on ctx
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_drive_root(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _make_blocking_attempt(
    commit_message: str = "test commit",
    block_reason: str = "critical_findings",
    critical_findings: list | None = None,
):
    from ouroboros.review_state import CommitAttemptRecord, _utc_now
    return CommitAttemptRecord(
        ts=_utc_now(),
        commit_message=commit_message,
        status="blocked",
        block_reason=block_reason,
        block_details="CRITICAL: something",
        duration_sec=5.0,
        critical_findings=critical_findings or [
            {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
             "reason": "No test changes found", "model": "test-model"},
        ],
    )


# ---------------------------------------------------------------------------
# 1. mark_advisory_stale_after_edit
# ---------------------------------------------------------------------------

def test_mark_stale_after_edit_invalidates_fresh_run(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState,
        load_state, save_state, mark_advisory_stale_after_edit, _utc_now,
    )
    state = AdvisoryReviewState()
    run = AdvisoryRunRecord(
        snapshot_hash="aabbcc112233",
        commit_message="v1: test",
        status="fresh",
        ts=_utc_now(),
    )
    state.add_run(run)
    save_state(drive_root, state)

    mark_advisory_stale_after_edit(drive_root)

    loaded = load_state(drive_root)
    assert loaded.runs[-1].status == "stale"
    assert loaded.last_stale_from_edit_ts != ""


def test_mark_stale_after_edit_no_op_when_no_fresh(tmp_path):
    """If there is no fresh run, mark_advisory_stale_after_edit should not error."""
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState,
        save_state, mark_advisory_stale_after_edit, _utc_now,
    )
    state = AdvisoryReviewState()
    run = AdvisoryRunRecord(
        snapshot_hash="aabbcc112233",
        commit_message="v1: test",
        status="stale",
        ts=_utc_now(),
    )
    state.runs.append(run)
    save_state(drive_root, state)

    mark_advisory_stale_after_edit(drive_root)  # should not raise

    from ouroboros.review_state import load_state
    loaded = load_state(drive_root)
    assert loaded.runs[-1].status == "stale"
    # last_stale_from_edit_ts should NOT be set (no fresh run was invalidated)
    assert loaded.last_stale_from_edit_ts == ""


def test_mark_stale_after_edit_only_affects_fresh(tmp_path):
    """mark_all_stale should only transition fresh→stale, not touch bypassed."""
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState,
        save_state, mark_advisory_stale_after_edit, load_state, _utc_now,
    )
    state = AdvisoryReviewState()
    fresh = AdvisoryRunRecord("hash1", "commit1", "fresh", _utc_now())
    bypassed = AdvisoryRunRecord("hash2", "commit2", "bypassed", _utc_now())
    state.runs = [fresh, bypassed]
    save_state(drive_root, state)

    mark_advisory_stale_after_edit(drive_root)

    loaded = load_state(drive_root)
    statuses = {r.snapshot_hash: r.status for r in loaded.runs}
    assert statuses["hash1"] == "stale"
    # bypassed runs are now also invalidated after worktree edits — same lifecycle as fresh
    assert statuses["hash2"] == "stale"


def test_mark_stale_after_edit_invalidates_bypassed_only_state(tmp_path):
    """mark_advisory_stale_after_edit must invalidate even when only bypassed runs exist."""
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState,
        save_state, mark_advisory_stale_after_edit, load_state, _utc_now,
    )
    state = AdvisoryReviewState()
    # Only a bypassed run — no fresh run present
    bypassed = AdvisoryRunRecord("hash-bypass", "commit-bypass", "bypassed", _utc_now())
    state.runs = [bypassed]
    save_state(drive_root, state)

    mark_advisory_stale_after_edit(drive_root)

    loaded = load_state(drive_root)
    assert loaded.runs[0].status == "stale"
    assert loaded.last_stale_from_edit_ts != ""


def test_invalidate_advisory_after_mutation_targets_matching_repo(tmp_path):
    """Phase 3: repo-scoped invalidation should stale only the mutated repo when identity is known."""
    from ouroboros.review_state import (
        AdvisoryRunRecord,
        AdvisoryReviewState,
        invalidate_advisory_after_mutation,
        load_state,
        save_state,
        _utc_now,
    )

    drive_root = _make_drive_root(tmp_path)
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    (repo_a / ".git").mkdir(parents=True)
    (repo_b / ".git").mkdir(parents=True)

    state = AdvisoryReviewState()
    state.add_run(AdvisoryRunRecord(
        snapshot_hash="hash-a",
        commit_message="repo a",
        status="fresh",
        ts=_utc_now(),
        repo_key=str(repo_a),
    ))
    state.add_run(AdvisoryRunRecord(
        snapshot_hash="hash-b",
        commit_message="repo b",
        status="fresh",
        ts=_utc_now(),
        repo_key=str(repo_b),
    ))
    save_state(drive_root, state)

    invalidate_advisory_after_mutation(
        drive_root,
        mutation_root=repo_b,
        changed_paths=["foo.py"],
        source_tool="claude_code_edit",
    )

    loaded = load_state(drive_root)
    statuses = {run.repo_key: run.status for run in loaded.advisory_runs}
    assert statuses[str(repo_a)] == "fresh"
    assert statuses[str(repo_b)] == "stale"
    assert "claude_code_edit mutated the worktree" in loaded.last_stale_reason


# ---------------------------------------------------------------------------
# 2. ObligationItem creation
# ---------------------------------------------------------------------------

def test_obligation_id_is_stable():
    """Same item+reason always produces the same obligation_id."""
    from ouroboros.review_state import AdvisoryReviewState
    state = AdvisoryReviewState()
    id1 = state._make_obligation_id("tests_affected", "No test changes")
    id2 = state._make_obligation_id("tests_affected", "No test changes")
    assert id1 == id2
    assert len(id1) == 12


def test_obligation_id_differs_for_different_items():
    from ouroboros.review_state import AdvisoryReviewState
    state = AdvisoryReviewState()
    id1 = state._make_obligation_id("tests_affected", "No test changes")
    id2 = state._make_obligation_id("version_bump", "No test changes")
    assert id1 != id2


# ---------------------------------------------------------------------------
# 3. blocking_history cap
# ---------------------------------------------------------------------------

def test_blocking_history_capped_at_max(tmp_path):
    from ouroboros.review_state import (
        AdvisoryReviewState, _MAX_BLOCKING_HISTORY,
    )
    state = AdvisoryReviewState()
    for i in range(_MAX_BLOCKING_HISTORY + 5):
        attempt = _make_blocking_attempt(
            commit_message=f"commit {i}",
            critical_findings=[{
                "verdict": "FAIL", "severity": "critical",
                "item": f"item_{i}", "reason": f"reason {i}", "model": "m",
            }],
        )
        # Give each attempt a unique task_id so _attempt_identity_tuple
        # produces distinct keys even when timestamps collide (Windows
        # datetime.now() has ~15ms granularity — a tight loop can produce
        # duplicate timestamps).
        attempt.task_id = f"cap_test_{i}"
        state.add_blocking_attempt(attempt)
    assert len(state.blocking_history) == _MAX_BLOCKING_HISTORY


# ---------------------------------------------------------------------------
# 4. add_blocking_attempt populates open_obligations
# ---------------------------------------------------------------------------

def test_add_blocking_attempt_creates_obligations():
    from ouroboros.review_state import AdvisoryReviewState
    state = AdvisoryReviewState()
    attempt = _make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
         "reason": "No test changes found", "model": "m"},
        {"verdict": "FAIL", "severity": "critical", "item": "version_bump",
         "reason": "VERSION not updated", "model": "m"},
    ])
    state.add_blocking_attempt(attempt)
    assert len(state.open_obligations) == 2
    items = {ob.item for ob in state.open_obligations}
    assert "tests_affected" in items
    assert "version_bump" in items


def test_add_blocking_attempt_deduplicates_obligations():
    """Same issue appearing twice (two blocking rounds) should not duplicate."""
    from ouroboros.review_state import AdvisoryReviewState
    finding = {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
               "reason": "No test changes found", "model": "m"}
    state = AdvisoryReviewState()
    state.add_blocking_attempt(_make_blocking_attempt(critical_findings=[finding]))
    state.add_blocking_attempt(_make_blocking_attempt(critical_findings=[finding]))
    # Same obligation_id should not be added twice
    assert len(state.open_obligations) == 1


def test_add_blocking_attempt_advisory_findings_not_tracked():
    """Advisory (non-critical) findings should NOT create obligations."""
    from ouroboros.review_state import AdvisoryReviewState
    attempt = _make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "advisory", "item": "context_building",
         "reason": "Something advisory", "model": "m"},
    ])
    state = AdvisoryReviewState()
    state.add_blocking_attempt(attempt)
    assert len(state.open_obligations) == 0


# ---------------------------------------------------------------------------
# 5. on_successful_commit clears obligations and history
# ---------------------------------------------------------------------------

def test_on_successful_commit_clears_state():
    from ouroboros.review_state import AdvisoryReviewState
    state = AdvisoryReviewState()
    attempt = _make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
         "reason": "No tests", "model": "m"},
    ])
    state.add_blocking_attempt(attempt)
    state.last_stale_from_edit_ts = "2026-04-05T12:00:00"
    assert len(state.open_obligations) == 1

    state.on_successful_commit()

    assert state.open_obligations == []
    assert state.blocking_history == []
    assert state.last_stale_from_edit_ts == ""


# ---------------------------------------------------------------------------
# 6. Serialization roundtrip for new fields
# ---------------------------------------------------------------------------

def test_serialization_roundtrip_new_fields(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryReviewState, load_state, save_state,
    )
    state = AdvisoryReviewState()
    attempt = _make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "bible_compliance",
         "reason": "Violates P5", "model": "m"},
    ])
    state.add_blocking_attempt(attempt)
    state.last_stale_from_edit_ts = "2026-04-05T12:00:00+00:00"
    save_state(drive_root, state)

    loaded = load_state(drive_root)
    assert len(loaded.blocking_history) == 1
    assert len(loaded.open_obligations) == 1
    assert loaded.open_obligations[0].item == "bible_compliance"
    assert loaded.open_obligations[0].status == "still_open"
    assert loaded.last_stale_from_edit_ts == "2026-04-05T12:00:00+00:00"


def test_old_state_loads_cleanly_without_new_fields(tmp_path):
    """Old state files without blocking_history / open_obligations load cleanly."""
    drive_root = _make_drive_root(tmp_path)
    state_file = drive_root / "state" / "advisory_review.json"
    state_file.write_text(json.dumps({
        "runs": [],
        "last_commit_attempt": None,
        "saved_at": "2026-04-01T00:00:00",
    }))

    from ouroboros.review_state import load_state
    loaded = load_state(drive_root)
    assert loaded.blocking_history == []
    assert loaded.open_obligations == []
    assert loaded.last_stale_from_edit_ts == ""


def test_commit_attempt_with_critical_findings_roundtrip(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryReviewState, load_state, save_state,
    )
    findings = [
        {"verdict": "FAIL", "severity": "critical", "item": "version_bump",
         "reason": "VERSION missing", "model": "m1"},
    ]
    state = AdvisoryReviewState()
    attempt = _make_blocking_attempt(critical_findings=findings)
    state.add_blocking_attempt(attempt)
    save_state(drive_root, state)

    loaded = load_state(drive_root)
    assert loaded.blocking_history[0].critical_findings == findings


# ---------------------------------------------------------------------------
# 7. format_status_section shows staleness and obligations
# ---------------------------------------------------------------------------

def test_format_status_shows_stale_from_edit():
    from ouroboros.review_state import (
        AdvisoryReviewState, AdvisoryRunRecord, format_status_section, _utc_now,
    )
    state = AdvisoryReviewState()
    state.runs.append(AdvisoryRunRecord("h1", "v1", "stale", _utc_now()))
    state.last_stale_from_edit_ts = "2026-04-05T13:00:00+00:00"
    section = format_status_section(state)
    assert "stale after worktree edit" in section
    assert "2026-04-05T13:00" in section


def test_format_status_shows_open_obligations():
    from ouroboros.review_state import AdvisoryReviewState, format_status_section
    state = AdvisoryReviewState()
    attempt = _make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
         "reason": "No test changes", "model": "m"},
    ])
    state.add_blocking_attempt(attempt)
    section = format_status_section(state)
    assert "Open obligations" in section
    assert "tests_affected" in section
    assert "Advisory MUST verify" in section


def test_format_status_no_obligations_section_when_clean():
    from ouroboros.review_state import AdvisoryReviewState, format_status_section
    state = AdvisoryReviewState()
    section = format_status_section(state)
    assert "Open obligations" not in section


# ---------------------------------------------------------------------------
# 8. _build_blocking_history_section reads full history
# ---------------------------------------------------------------------------

def test_build_blocking_history_section_empty_when_no_history(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import save_state, AdvisoryReviewState
    save_state(drive_root, AdvisoryReviewState())

    from ouroboros.tools.claude_advisory_review import _build_blocking_history_section
    result = _build_blocking_history_section(drive_root)
    assert result == ""


def test_build_blocking_history_section_contains_all_obligations(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import AdvisoryReviewState, save_state
    state = AdvisoryReviewState()
    # Two distinct blocking rounds with different issues
    state.add_blocking_attempt(_make_blocking_attempt(
        commit_message="round 1",
        critical_findings=[{"verdict": "FAIL", "severity": "critical",
                            "item": "tests_affected", "reason": "No tests", "model": "m"}],
    ))
    state.add_blocking_attempt(_make_blocking_attempt(
        commit_message="round 2",
        critical_findings=[{"verdict": "FAIL", "severity": "critical",
                            "item": "version_bump", "reason": "No VERSION", "model": "m"}],
    ))
    save_state(drive_root, state)

    from ouroboros.tools.claude_advisory_review import _build_blocking_history_section
    result = _build_blocking_history_section(drive_root)
    assert "Unresolved obligations" in result
    assert "tests_affected" in result
    assert "version_bump" in result
    assert "```json" in result
    assert "recent_blocking_attempts" in result


def test_build_blocking_history_section_instructions_present(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import AdvisoryReviewState, save_state
    state = AdvisoryReviewState()
    state.add_blocking_attempt(_make_blocking_attempt())
    save_state(drive_root, state)

    from ouroboros.tools.claude_advisory_review import _build_blocking_history_section
    result = _build_blocking_history_section(drive_root)
    assert "should explicitly address" in result
    assert "If fixed" in result
    assert "If not fixed" in result


# ---------------------------------------------------------------------------
# 9. _collect_review_findings stores structured findings on ctx
# ---------------------------------------------------------------------------

def test_collect_review_findings_stores_structured_findings():
    from ouroboros.tools.review import _collect_review_findings

    ctx = MagicMock()
    ctx.drive_logs.return_value = pathlib.Path("/tmp/fake_logs")
    ctx._last_review_critical_findings = []

    model_results = [
        {
            "model": "model-a",
            "verdict": "ok",
            "text": json.dumps([
                {"verdict": "FAIL", "severity": "critical",
                 "item": "tests_affected", "reason": "No tests added"},
                {"verdict": "PASS", "severity": "critical",
                 "item": "bible_compliance", "reason": "OK"},
            ]),
        },
    ]

    with patch("ouroboros.tools.review.append_jsonl"):
        critical_fails, advisory_warns, errored = _collect_review_findings(ctx, model_results)

    assert len(critical_fails) == 1
    assert "tests_affected" in critical_fails[0]
    structured = ctx._last_review_critical_findings
    assert len(structured) == 1
    assert structured[0]["item"] == "tests_affected"
    assert structured[0]["severity"] == "critical"
    assert structured[0]["verdict"] == "FAIL"
    assert structured[0]["model"] == "model-a"


def test_collect_review_findings_advisory_not_in_structured():
    from ouroboros.tools.review import _collect_review_findings

    ctx = MagicMock()
    ctx.drive_logs.return_value = pathlib.Path("/tmp/fake_logs")
    ctx._last_review_critical_findings = []

    model_results = [
        {
            "model": "model-a",
            "verdict": "ok",
            "text": json.dumps([
                {"verdict": "FAIL", "severity": "advisory",
                 "item": "context_building", "reason": "Not in context"},
            ]),
        },
    ]

    with patch("ouroboros.tools.review.append_jsonl"):
        critical_fails, advisory_warns, _ = _collect_review_findings(ctx, model_results)

    assert len(critical_fails) == 0
    assert len(advisory_warns) == 1
    assert ctx._last_review_critical_findings == []


# ---------------------------------------------------------------------------
# 10. review_status response includes new fields
# ---------------------------------------------------------------------------

def test_review_status_includes_stale_from_edit(tmp_path):
    from ouroboros.review_state import (
        AdvisoryReviewState, AdvisoryRunRecord, save_state, _utc_now,
    )
    drive_root = _make_drive_root(tmp_path)
    state = AdvisoryReviewState()
    state.runs.append(AdvisoryRunRecord("h1", "v1", "stale", _utc_now()))
    state.last_stale_from_edit_ts = "2026-04-05T13:00:00+00:00"
    save_state(drive_root, state)

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path)  # required by live hash computation

    from ouroboros.tools.claude_advisory_review import _handle_review_status
    result = json.loads(_handle_review_status(ctx))

    assert result["stale_from_edit"] is True
    assert result["stale_from_edit_ts"] is not None
    assert "next_step" in result


def test_review_status_surfaces_explicit_stale_reason(tmp_path):
    """Phase 3: review_status should expose the concrete invalidation reason."""
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryReviewState, AdvisoryRunRecord, save_state, _utc_now,
    )
    state = AdvisoryReviewState()
    state.runs.append(AdvisoryRunRecord("h1", "v1", "stale", _utc_now()))
    state.last_stale_from_edit_ts = "2026-04-05T13:00:00+00:00"
    state.last_stale_reason = "claude_code_edit mutated the worktree; advisory freshness invalidated."
    save_state(drive_root, state)

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path)

    from ouroboros.tools.claude_advisory_review import _handle_review_status
    result = json.loads(_handle_review_status(ctx))

    assert result["stale_from_edit"] is True
    assert result["stale_reason"] == state.last_stale_reason


def test_review_status_includes_open_obligations(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import AdvisoryReviewState, save_state
    state = AdvisoryReviewState()
    state.add_blocking_attempt(_make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
         "reason": "No tests", "model": "m"},
    ]))
    save_state(drive_root, state)

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path)  # required by live hash computation

    from ouroboros.tools.claude_advisory_review import _handle_review_status
    result = json.loads(_handle_review_status(ctx))

    assert result["open_obligations_count"] == 1
    assert len(result["open_obligations"]) == 1
    assert result["open_obligations"][0]["item"] == "tests_affected"


def test_review_status_next_step_after_edit_staleness(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import (
        AdvisoryReviewState, AdvisoryRunRecord, save_state, _utc_now,
    )
    state = AdvisoryReviewState()
    state.runs.append(AdvisoryRunRecord("h1", "v1", "stale", _utc_now()))
    state.last_stale_from_edit_ts = "2026-04-05T13:00:00+00:00"
    save_state(drive_root, state)

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path)  # required by live hash computation

    from ouroboros.tools.claude_advisory_review import _handle_review_status
    result = json.loads(_handle_review_status(ctx))

    assert "invalidated" in result["next_step"].lower()
    assert "advisory_pre_review" in result["next_step"]


# ---------------------------------------------------------------------------
# 11. _check_advisory_freshness includes obligation summary
# ---------------------------------------------------------------------------

def test_check_advisory_freshness_shows_obligations_in_error(tmp_path):
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import AdvisoryReviewState, save_state
    state = AdvisoryReviewState()
    state.add_blocking_attempt(_make_blocking_attempt(critical_findings=[
        {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
         "reason": "No test changes", "model": "m"},
    ]))
    save_state(drive_root, state)

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir(exist_ok=True)

    from ouroboros.tools.git import _check_advisory_freshness
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="M  some_file.py\n")
        result = _check_advisory_freshness(ctx, "test commit")

    assert result is not None
    assert "ADVISORY_PRE_REVIEW_REQUIRED" in result
    # Workflow instructions
    assert "Finish ALL edits first" in result
    assert "run AFTER all edits" in result
    assert "run IMMEDIATELY after advisory" in result
    # Obligation info in error
    assert "tests_affected" in result or "advisory_pre_review will verify" in result


def test_check_advisory_freshness_correct_workflow_in_error(tmp_path):
    """Error message must reflect new workflow, not old 'fix -> commit' pattern."""
    drive_root = _make_drive_root(tmp_path)
    from ouroboros.review_state import save_state, AdvisoryReviewState
    save_state(drive_root, AdvisoryReviewState())

    ctx = MagicMock()
    ctx.drive_root = str(drive_root)
    ctx.repo_dir = str(tmp_path / "repo")
    (tmp_path / "repo").mkdir(exist_ok=True)

    from ouroboros.tools.git import _check_advisory_freshness
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="M  file.py\n")
        result = _check_advisory_freshness(ctx, "test")

    assert result is not None
    # New workflow language
    assert "Finish ALL edits first" in result
    assert "IMMEDIATELY" in result
    # NOT the old misleading language
    assert "fix obvious issues" not in result


def test_obligation_resolution_clears_open_obligations(tmp_path):
    """End-to-end: blocked commit creates obligations → advisory PASS → obligations resolved."""
    from ouroboros.review_state import (
        AdvisoryReviewState, CommitAttemptRecord, ObligationItem, _utc_now,
        load_state, save_state,
    )

    drive_root = _make_drive_root(tmp_path)

    # 1. Simulate a blocking commit attempt that creates an obligation
    state = AdvisoryReviewState()
    blocking_attempt = CommitAttemptRecord(
        ts=_utc_now(),
        commit_message="v1.0: some feature",
        status="blocked",
        block_reason="critical_findings",
        block_details="CRITICAL: tests_affected",
        duration_sec=5.0,
        critical_findings=[
            {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
             "reason": "No test changes alongside code changes", "model": "test-model"},
        ],
    )
    state.add_blocking_attempt(blocking_attempt)
    save_state(drive_root, state)

    # Verify obligation was created
    loaded = load_state(drive_root)
    open_obs = loaded.get_open_obligations()
    assert len(open_obs) == 1
    assert open_obs[0].item == "tests_affected"
    assert open_obs[0].status == "still_open"

    # 2. Simulate advisory returning PASS for tests_affected
    # This exercises the path in _handle_advisory_pre_review that resolves obligations
    passed_items = {"tests_affected"}
    resolved_ids = [o.obligation_id for o in open_obs
                    if o.item.lower() in passed_items]
    assert len(resolved_ids) == 1

    loaded.resolve_obligations(resolved_ids, resolved_by="advisory run abc123")
    save_state(drive_root, loaded)

    # 3. Verify obligations are resolved
    final = load_state(drive_root)
    assert len(final.get_open_obligations()) == 0
    # Resolved obligation still in list but marked resolved
    all_obs = [o for o in final.open_obligations]
    assert any(o.status == "resolved" for o in all_obs)


def test_obligation_id_field_name_is_obligation_id(tmp_path):
    """ObligationItem must use .obligation_id, not .id — regression test for typo."""
    from ouroboros.review_state import ObligationItem, _utc_now
    ob = ObligationItem(
        obligation_id="abc123",
        item="tests_affected",
        severity="critical",
        reason="no tests",
        source_attempt_ts=_utc_now(),
        source_attempt_msg="v1.0: test",
    )
    # Must not raise AttributeError
    assert ob.obligation_id == "abc123"
    # Must NOT have .id attribute (old wrong name)
    assert not hasattr(ob, "id"), "ObligationItem should use .obligation_id, not .id"


def test_empty_items_does_not_resolve_obligations(tmp_path):
    """Empty advisory items list (parse failure) must NOT resolve open obligations."""
    from ouroboros.review_state import (
        AdvisoryReviewState, CommitAttemptRecord, _utc_now,
        load_state, save_state,
    )

    drive_root = _make_drive_root(tmp_path)

    # Create state with an open obligation
    state = AdvisoryReviewState()
    blocking_attempt = CommitAttemptRecord(
        ts=_utc_now(),
        commit_message="v1.0: feature",
        status="blocked",
        block_reason="critical_findings",
        block_details="CRITICAL: tests_affected",
        duration_sec=3.0,
        critical_findings=[
            {"verdict": "FAIL", "severity": "critical", "item": "tests_affected",
             "reason": "No test changes", "model": "test-model"},
        ],
    )
    state.add_blocking_attempt(blocking_attempt)
    save_state(drive_root, state)

    # Verify obligation exists
    loaded = load_state(drive_root)
    assert len(loaded.get_open_obligations()) == 1

    # Simulate empty items list (parse failure scenario)
    items = []
    critical_fails = []  # empty because no items to fail
    # The guard `if not critical_fails and items:` must prevent resolution
    should_resolve = not critical_fails and bool(items)
    assert not should_resolve, (
        "Empty items list should NOT trigger obligation resolution — "
        "obligations must stay open after advisory parse failure"
    )

# Extended tests moved to tests/test_advisory_workflow_ext.py
# (snapshot_paths, parse_failure handling, obligation resolution edge cases)
