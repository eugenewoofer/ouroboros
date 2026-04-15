"""Structured review-evidence collection for summaries, reflections, and UX."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List


def collect_review_evidence(
    drive_root: Any,
    *,
    task_id: str = "",
    repo_dir: Any = None,
    max_attempts: int = 3,
    max_runs: int = 3,
    max_obligations: int = 6,
    max_continuations: int = 3,
) -> Dict[str, Any]:
    from ouroboros.review_state import (
        _LEGACY_CURRENT_REPO_KEY,
        compute_snapshot_hash,
        load_state,
        make_repo_key,
    )
    from ouroboros.task_continuation import list_review_continuations

    drive_root_path = pathlib.Path(drive_root)
    repo_dir_path = pathlib.Path(repo_dir) if repo_dir else None
    repo_key = make_repo_key(repo_dir_path) if repo_dir_path else ""
    snapshot_hash = compute_snapshot_hash(repo_dir_path) if repo_dir_path else ""

    state = load_state(drive_root_path)
    all_runs = list(state.advisory_runs or [])
    all_attempts = list(state.attempts or [])

    if repo_key:
        repo_runs = state.filter_advisory_runs(repo_key=repo_key)
    else:
        repo_runs = all_runs

    if task_id:
        scoped_attempts = state.filter_attempts(task_id=task_id)
    elif repo_key:
        scoped_attempts = state.filter_attempts(repo_key=repo_key)
    else:
        scoped_attempts = all_attempts

    current_run = None
    if snapshot_hash:
        current_run = state.find_by_hash(snapshot_hash, repo_key=repo_key or None)

    open_obligations = state.get_open_obligations(repo_key=repo_key or None)
    continuations, corrupt = list_review_continuations(drive_root_path)
    if task_id:
        scoped_continuations = [item for item in continuations if item.task_id == task_id]
    elif repo_key:
        scoped_continuations = [
            item for item in continuations
            if item.repo_key in ("", repo_key, _LEGACY_CURRENT_REPO_KEY)
        ]
    else:
        scoped_continuations = continuations
    scoped_continuations.sort(key=lambda item: str(item.updated_ts or item.created_ts or ""), reverse=True)
    stale_matches_repo = not repo_key or state.last_stale_repo_key in ("", repo_key)

    evidence = {
        "task_id": task_id,
        "repo_key": repo_key,
        "current_repo": {
            "snapshot_hash": snapshot_hash[:12] if snapshot_hash else "",
            "advisory_status": str(getattr(current_run, "status", "") or "missing"),
            "repo_commit_ready": bool(
                current_run is not None
                and current_run.status in ("fresh", "bypassed", "skipped")
                and not open_obligations
            ),
            "bypass_reason": str(getattr(current_run, "bypass_reason", "") or ""),
            "stale_reason": str(getattr(state, "last_stale_reason", "") or "") if stale_matches_repo else "",
            "stale_ts": str(getattr(state, "last_stale_from_edit_ts", "") or "") if stale_matches_repo else "",
        },
        "recent_attempts": [_attempt_to_dict(item) for item in scoped_attempts[-max_attempts:]],
        "recent_advisory_runs": [_run_to_dict(item) for item in repo_runs[-max_runs:]],
        "open_obligations": [_obligation_to_dict(item) for item in open_obligations[:max_obligations]],
        "continuations": [_continuation_to_dict(item) for item in scoped_continuations[:max_continuations]],
        "corrupt_continuations": [str(item) for item in corrupt[:3]],
    }
    evidence["has_evidence"] = any([
        evidence["recent_attempts"],
        evidence["recent_advisory_runs"],
        evidence["open_obligations"],
        evidence["continuations"],
        evidence["corrupt_continuations"],
        evidence["current_repo"]["advisory_status"] not in ("", "missing"),
    ])
    return evidence


def format_review_evidence_for_prompt(evidence: Dict[str, Any], *, max_chars: int = 2500) -> str:
    if not evidence or not evidence.get("has_evidence"):
        return "(no structured review evidence)"
    raw = json.dumps(evidence, ensure_ascii=False, indent=2)
    if len(raw) <= max_chars:
        return raw
    omitted = len(raw) - max_chars
    return raw[:max_chars] + f"\n... [truncated structured review evidence; omitted {omitted} chars]"


def _attempt_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "ts": str(getattr(item, "ts", "") or ""),
        "tool_name": str(getattr(item, "tool_name", "") or ""),
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "status": str(getattr(item, "status", "") or ""),
        "phase": str(getattr(item, "phase", "") or ""),
        "block_reason": str(getattr(item, "block_reason", "") or ""),
        "late_result_pending": bool(getattr(item, "late_result_pending", False)),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
        "obligation_ids": [str(x) for x in (getattr(item, "obligation_ids", []) or [])],
        "degraded_reasons": [str(x) for x in (getattr(item, "degraded_reasons", []) or [])],
    }


def _run_to_dict(item: Any) -> Dict[str, Any]:
    fail_items: List[Dict[str, Any]] = []
    for entry in list(getattr(item, "items", []) or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("verdict", "")).upper() != "FAIL":
            continue
        fail_items.append({
            "severity": str(entry.get("severity", "") or "advisory"),
            "item": str(entry.get("item", "") or ""),
            "reason": str(entry.get("reason", "") or ""),
        })
    return {
        "ts": str(getattr(item, "ts", "") or ""),
        "status": str(getattr(item, "status", "") or ""),
        "repo_key": str(getattr(item, "repo_key", "") or ""),
        "bypass_reason": str(getattr(item, "bypass_reason", "") or ""),
        "snapshot_summary": str(getattr(item, "snapshot_summary", "") or ""),
        "findings": fail_items,
    }


def _obligation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "obligation_id": str(getattr(item, "obligation_id", "") or ""),
        "item": str(getattr(item, "item", "") or ""),
        "severity": str(getattr(item, "severity", "") or ""),
        "reason": str(getattr(item, "reason", "") or ""),
        "status": str(getattr(item, "status", "") or ""),
    }


def _continuation_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "task_id": str(getattr(item, "task_id", "") or ""),
        "source": str(getattr(item, "source", "") or ""),
        "stage": str(getattr(item, "stage", "") or ""),
        "tool_name": str(getattr(item, "tool_name", "") or ""),
        "attempt": int(getattr(item, "attempt", 0) or 0),
        "block_reason": str(getattr(item, "block_reason", "") or ""),
        "critical_findings": list(getattr(item, "critical_findings", []) or []),
        "advisory_findings": list(getattr(item, "advisory_findings", []) or []),
        "readiness_warnings": [str(x) for x in (getattr(item, "readiness_warnings", []) or [])],
        "updated_ts": str(getattr(item, "updated_ts", "") or ""),
    }
