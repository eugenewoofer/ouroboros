"""commit_gate.py — Advisory freshness gate and commit-attempt recording.

Extracted from git.py to relieve module-size pressure under P5 Minimalism.
Provides:
  _record_commit_attempt(ctx, commit_message, status, ...)
  _invalidate_advisory(ctx)
  _check_advisory_freshness(ctx, commit_message, skip, paths) -> Optional[str]
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext
from ouroboros.utils import (
    truncate_review_artifact as _truncate_review_artifact,
    truncate_review_reason as _truncate_review_reason,
)

log = logging.getLogger(__name__)


def _current_review_tool_name(ctx: ToolContext) -> str:
    return str(getattr(ctx, "_current_review_tool_name", "") or "repo_commit")


def _attempt_phase(status: str, block_reason: str = "") -> str:
    if status == "reviewing":
        return "review"
    if status == "blocked":
        if block_reason == "no_advisory":
            return "advisory_gate"
        if block_reason == "preflight":
            return "preflight"
        return "blocking_review"
    if status == "succeeded":
        return "commit"
    if status == "failed":
        return "infra"
    return "review"


def _normalize_advisory_entries(items: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in list(items or []):
        if isinstance(item, dict):
            normalized.append(item)
        elif item:
            normalized.append({"reason": str(item), "severity": "advisory"})
    return normalized


def _list_or_default(items: Optional[List[Any]], fallback: List[Any]) -> List[Any]:
    if items is None:
        return list(fallback)
    return list(items)


def _continuation_source(status: str, *, late_result_pending: bool) -> str:
    if status == "blocked":
        return "blocked_review"
    if late_result_pending:
        return "late_result_pending"
    if status == "failed":
        return "review_failure"
    return ""


def _attempt_accepts_reviewing_update(existing: Any) -> bool:
    if existing is None:
        return False
    return bool(existing.status == "reviewing" or existing.late_result_pending)


def _record_commit_attempt(ctx: ToolContext, commit_message: str, status: str,
                           block_reason: str = "", block_details: str = "",
                           duration_sec: float = 0.0, snapshot_hash: str = "",
                           critical_findings: Optional[List[Dict[str, Any]]] = None,
                           advisory_findings: Optional[List[Dict[str, Any]]] = None,
                           readiness_warnings: Optional[List[str]] = None,
                           late_result_pending: bool = False,
                           phase: Optional[str] = None,
                           pre_review_fingerprint: str = "",
                           post_review_fingerprint: str = "",
                           fingerprint_status: str = "",
                           degraded_reasons: Optional[List[str]] = None) -> None:
    try:
        from ouroboros.review_state import (
            CommitAttemptRecord,
            make_repo_key,
            update_state,
            _utc_now,
        )
        dr = pathlib.Path(ctx.drive_root)
        repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
        tool_name = _current_review_tool_name(ctx)
        task_id = str(getattr(ctx, "task_id", "") or "")

        def _mutate(state):
            state.expire_stale_attempts()
            attempt_no = int(getattr(ctx, "_current_review_attempt_number", 0) or 0)
            existing = (
                state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                    attempt=attempt_no,
                )
                if attempt_no > 0
                else None
            )
            if status == "reviewing":
                if not _attempt_accepts_reviewing_update(existing):
                    attempt_no = state.next_attempt_number(repo_key, tool_name, task_id)
                    existing = None
                ctx._current_review_attempt_number = attempt_no
            elif attempt_no <= 0:
                existing = state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                )
                if existing and existing.status == "reviewing" and not existing.finished_ts:
                    attempt_no = int(existing.attempt or 0)
                else:
                    attempt_no = state.next_attempt_number(repo_key, tool_name, task_id)
                ctx._current_review_attempt_number = attempt_no
            else:
                existing = state.latest_attempt_for(
                    repo_key=repo_key,
                    tool_name=tool_name,
                    task_id=task_id,
                    attempt=attempt_no,
                )

            attempt = CommitAttemptRecord(
                ts=_utc_now(),
                commit_message=commit_message,  # full message — no [:200] truncation
                status=status,
                snapshot_hash=snapshot_hash,
                block_reason=block_reason,
                block_details=_truncate_review_artifact(block_details),
                duration_sec=duration_sec,
                task_id=task_id,
                critical_findings=_list_or_default(
                    critical_findings,
                    list(getattr(existing, "critical_findings", []) or []),
                ),
                repo_key=repo_key,
                tool_name=tool_name,
                attempt=attempt_no,
                phase=phase or _attempt_phase(status, block_reason),
                blocked=(status == "blocked"),
                advisory_findings=_normalize_advisory_entries(
                    _list_or_default(
                        advisory_findings,
                        getattr(existing, "advisory_findings", None)
                        or getattr(ctx, "_review_advisory", []),
                    )
                ),
                readiness_warnings=[
                    str(x) for x in _list_or_default(
                        readiness_warnings,
                        list(getattr(existing, "readiness_warnings", []) or []),
                    ) if str(x).strip()
                ],
                late_result_pending=late_result_pending,
                pre_review_fingerprint=pre_review_fingerprint or getattr(existing, "pre_review_fingerprint", ""),
                post_review_fingerprint=post_review_fingerprint or getattr(existing, "post_review_fingerprint", ""),
                fingerprint_status=fingerprint_status or getattr(existing, "fingerprint_status", ""),
                degraded_reasons=[
                    str(x) for x in _list_or_default(
                        degraded_reasons,
                        list(getattr(existing, "degraded_reasons", []) or []),
                    ) if str(x).strip()
                ],
                started_ts=str(getattr(existing, "started_ts", "") or ""),
            )
            state.record_attempt(attempt)

        update_state(dr, _mutate)
        try:
            from ouroboros.review_state import load_state
            from ouroboros.task_continuation import (
                build_review_continuation,
                clear_review_continuation,
                save_review_continuation,
            )

            if task_id:
                if status == "succeeded":
                    clear_review_continuation(dr, task_id)
                else:
                    source = _continuation_source(status, late_result_pending=late_result_pending)
                    if source:
                        latest_state = load_state(dr)
                        latest_attempt = latest_state.latest_attempt_for(
                            repo_key=repo_key,
                            tool_name=tool_name,
                            task_id=task_id,
                            attempt=int(getattr(ctx, "_current_review_attempt_number", 0) or 0) or None,
                        )
                        continuation = build_review_continuation(
                            {
                                "id": task_id,
                                "type": str(getattr(ctx, "current_task_type", "") or ""),
                                "parent_task_id": str(getattr(ctx, "parent_task_id", "") or ""),
                            },
                            latest_attempt,
                            latest_state.get_open_obligations(repo_key=repo_key),
                            source=source,
                        )
                        if continuation is not None:
                            save_review_continuation(dr, continuation, expect_task_id=task_id)
        except Exception as e:
            log.warning("Failed to sync review continuation: %s", e)
        if status in ("blocked", "failed", "succeeded") and not late_result_pending:
            ctx._current_review_attempt_number = None
    except Exception as e:
        log.warning("Failed to record commit attempt: %s", e)


def _invalidate_advisory(
    ctx: ToolContext,
    *,
    changed_paths: Optional[List[str]] = None,
    mutation_root: Optional[pathlib.Path] = None,
    source_tool: str = "",
) -> None:
    try:
        from ouroboros.review_state import invalidate_advisory_after_mutation
        invalidate_advisory_after_mutation(
            pathlib.Path(ctx.drive_root),
            mutation_root=mutation_root or pathlib.Path(ctx.repo_dir),
            changed_paths=changed_paths,
            source_tool=source_tool or _current_review_tool_name(ctx),
        )
    except Exception:
        pass


def _mark_review_attempt_late(
    ctx: ToolContext,
    *,
    soft_timeout_sec: int,
    duration_sec: float,
) -> None:
    warning = (
        f"Soft timeout exceeded {soft_timeout_sec}s; waiting for a possible late reviewed result."
    )
    _record_commit_attempt(
        ctx,
        commit_message=str(getattr(ctx, "_current_review_commit_message", "") or ""),
        status="reviewing",
        duration_sec=duration_sec,
        readiness_warnings=[warning],
        late_result_pending=True,
        phase="late_wait",
    )


def _check_overlapping_review_attempt(ctx: ToolContext) -> Optional[str]:
    from ouroboros.review_state import (
        _REVIEW_ATTEMPT_GRACE_SEC,
        _REVIEW_ATTEMPT_TTL_SEC,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.tool_capabilities import REVIEWED_MUTATIVE_TOOLS

    repo_key = make_repo_key(pathlib.Path(ctx.repo_dir))
    expiration_window = _REVIEW_ATTEMPT_TTL_SEC + _REVIEW_ATTEMPT_GRACE_SEC

    def _mutate(state):
        state.expire_stale_attempts(now_ts=_utc_now())
        return [
            item for item in state.get_active_attempts(repo_key=repo_key)
            if item.tool_name in REVIEWED_MUTATIVE_TOOLS
        ]

    try:
        active_attempts = update_state(pathlib.Path(ctx.drive_root), _mutate)
    except Exception as e:
        log.warning("Failed to check overlapping review attempts: %s", e)
        return None
    if not active_attempts:
        return None

    active = active_attempts[-1]
    attempt_label = (
        f"{active.tool_name}#{active.attempt}"
        if int(active.attempt or 0) > 0
        else active.tool_name
    )
    return (
        f"⚠️ REVIEWED_ATTEMPT_IN_PROGRESS: {attempt_label} is still active "
        f"(status={active.status}, late_result_pending={bool(active.late_result_pending)}, "
        f"started={active.started_ts or active.ts}). "  # full ts — no [:19] truncation
        f"Do not start another reviewed attempt for this repo until it finishes or auto-expires "
        f"after {expiration_window}s TTL+grace. Check review_status for current state."
    )


def _check_advisory_freshness(ctx: ToolContext, commit_message: str,
                              skip_advisory_pre_review: bool = False,
                              paths: Optional[List[str]] = None) -> Optional[str]:
    from ouroboros.review_state import (
        AdvisoryRunRecord,
        compute_snapshot_hash,
        load_state,
        make_repo_key,
        update_state,
        _utc_now,
    )
    from ouroboros.utils import append_jsonl
    drive_root = pathlib.Path(ctx.drive_root)
    repo_dir = pathlib.Path(ctx.repo_dir)
    repo_key = make_repo_key(repo_dir)

    snapshot_hash = compute_snapshot_hash(repo_dir, commit_message, paths=paths)
    state = load_state(drive_root)
    # Pass only when snapshot is fresh AND no open obligations remain.
    if state.is_fresh(snapshot_hash, repo_key=repo_key) and not state.get_open_obligations(repo_key=repo_key):
        return None

    if skip_advisory_pre_review:
        task_id = str(getattr(ctx, "task_id", "") or "")
        reason = "skip_advisory_pre_review=True passed to repo_commit"
        try:
            append_jsonl(ctx.drive_logs() / "events.jsonl", {
                "ts": _utc_now(), "type": "advisory_pre_review_bypassed",
                "snapshot_hash": snapshot_hash, "commit_message": commit_message,  # full — no [:200]
                "bypass_reason": reason, "task_id": task_id,
            })
        except Exception:
            pass

        def _mutate(bypass_state):
            next_run_attempt = len(
                bypass_state.filter_advisory_runs(
                    repo_key=repo_key,
                    tool_name="advisory_pre_review",
                    task_id=task_id,
                )
            ) + 1
            bypass_state.add_run(AdvisoryRunRecord(
                snapshot_hash=snapshot_hash,
                commit_message=commit_message,
                status="bypassed",
                ts=_utc_now(),
                bypass_reason=reason,
                bypassed_by_task=task_id,
                snapshot_paths=paths,
                repo_key=repo_key,
                tool_name="advisory_pre_review",
                task_id=task_id,
                attempt=next_run_attempt,
            ))

        update_state(drive_root, _mutate)
        return None  # audited bypass

    # Advisory is fresh for this snapshot — check if obligations remain
    open_obs = state.get_open_obligations(repo_key=repo_key)
    if state.is_fresh(snapshot_hash, repo_key=repo_key) and open_obs:
        lines = [f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Advisory is current (hash={snapshot_hash[:12]}) "
                 f"but {len(open_obs)} open obligation(s) from previous blocking rounds must be resolved.\n",
                 "Unresolved obligations:"]
        # No [:N] cap — show all obligations so the agent sees every unresolved item.
        lines += [f"  [{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=80)}"
                  for o in open_obs]
        lines.append("\nFix the flagged issues and re-run advisory_pre_review so it can mark them PASS.")
        lines.append("Or bypass: repo_commit(commit_message='...', skip_advisory_pre_review=True) (audited).")
        return "\n".join(lines)

    matching_run = state.find_by_hash(snapshot_hash, repo_key=repo_key)
    scoped_runs = state.filter_advisory_runs(repo_key=repo_key)
    latest = scoped_runs[-1] if scoped_runs else None

    # Explicit parse_failure branch: advisory ran for this snapshot but was unparseable.
    # Must come before the generic stale branch to avoid misleading "snapshot changed" message.
    if matching_run and matching_run.status == "parse_failure":
        obs_section = ""
        if state.get_open_obligations(repo_key=repo_key):
            open_obs = state.get_open_obligations(repo_key=repo_key)
            obs_lines = [f"\nOpen obligations ({len(open_obs)}):"]
            # No [:N] cap — all obligations shown.
            obs_lines += [f"  [{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=80)}"
                          for o in open_obs]
            obs_section = "\n".join(obs_lines)
        return (
            f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Last advisory run for this snapshot returned "
            f"parse_failure (hash={snapshot_hash[:12]}, ts={matching_run.ts}). "  # full ts
            f"The advisory ran but its output could not be parsed — re-run it.{obs_section}\n"
            "Re-run: advisory_pre_review(commit_message='...')\n"
            "Or bypass: repo_commit(commit_message='...', skip_advisory_pre_review=True) (audited)."
        )

    if latest and latest.status == "stale" and state.last_stale_from_edit_ts:
        stale_reason = (f"Advisory invalidated by worktree edit at "
                        f"{state.last_stale_from_edit_ts}. Re-run advisory after all edits.")  # full ts
    elif latest:
        stale_reason = (f"Latest run: status={latest.status}, hash={latest.snapshot_hash[:12]}, "
                        f"ts={latest.ts}. Snapshot changed (files edited after advisory ran).")  # full ts
    else:
        stale_reason = "No advisory runs recorded yet."

    obs_section = ""
    if open_obs:
        lines = [f"\nOpen obligations ({len(open_obs)}):"]
        # No [:N] cap — all obligations shown so nothing is silently hidden.
        lines += [f"  [{o.obligation_id}] {o.item}: {_truncate_review_reason(o.reason, limit=80)}"
                  for o in open_obs]
        lines.append("  → advisory_pre_review will verify each obligation is resolved.")
        obs_section = "\n".join(lines)

    return (
        f"⚠️ ADVISORY_PRE_REVIEW_REQUIRED: No fresh advisory run found for this snapshot "
        f"(hash={snapshot_hash[:12]}).\n"
        f"{stale_reason}\n"
        f"{obs_section}\n\n"
        "Correct workflow:\n"
        "  1. Finish ALL edits first\n"
        "  2. advisory_pre_review(commit_message='your message')   ← run AFTER all edits\n"
        "  3. repo_commit(commit_message='your message')            ← run IMMEDIATELY after advisory\n\n"
        "⚠️ Any edit after step 2 makes the advisory stale and requires re-running it.\n\n"
        "To bypass (will be durably audited):\n"
        "  repo_commit(commit_message='...', skip_advisory_pre_review=True)"
    )
