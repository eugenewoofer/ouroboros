"""Durable review/advisory state with typed attempt ledger and compatibility views.

State persists across task boundaries and restarts in
``~/Ouroboros/data/state/advisory_review.json``.

The modern state model is a ledger:
- ``advisory_runs[]`` — advisory pre-review coverage for snapshots
- ``attempts[]`` — reviewed mutative attempts with lifecycle/status metadata

Compatibility views remain available for the current desktop workflow:
- ``runs`` (alias of ``advisory_runs``)
- ``last_commit_attempt``
- ``blocking_history``
- ``open_obligations``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.utils import (
    truncate_review_artifact as _truncate_review_artifact,
    truncate_review_reason as _truncate_review_reason,
)

log = logging.getLogger(__name__)

_STATE_RELPATH = "state/advisory_review.json"
_LOCK_RELPATH = "locks/advisory_review.lock"
_STATE_SCHEMA_VERSION = 2
_MAX_RUN_HISTORY = 10
_MAX_ATTEMPT_HISTORY = 50
_MAX_BLOCKING_HISTORY = 10
_DEFAULT_TOOL_NAME = "repo_commit"
_DEFAULT_ADVISORY_TOOL_NAME = "advisory_pre_review"
_LEGACY_CURRENT_REPO_KEY = "__legacy_current_repo__"
_REVIEW_ATTEMPT_TTL_SEC = 1800
_REVIEW_ATTEMPT_GRACE_SEC = 120


def _repo_scope_exact_match_exists(records: List[Any], repo_key: str | None) -> bool:
    return repo_key is not None and any(str(getattr(record, "repo_key", "") or "") == repo_key for record in records)
def _repo_scope_matches(
    record_repo_key: str,
    repo_key: str | None,
    *,
    exact_match_exists: bool,
) -> bool:
    if repo_key is None:
        return True
    if record_repo_key == repo_key:
        return True
    return (not exact_match_exists) and record_repo_key in ("", _LEGACY_CURRENT_REPO_KEY)


@dataclass
class ObligationItem:
    """A single unresolved obligation extracted from a blocking commit attempt."""

    obligation_id: str
    item: str
    severity: str
    reason: str
    source_attempt_ts: str
    source_attempt_msg: str
    status: str = "still_open"
    resolved_by: str = ""
    repo_key: str = _LEGACY_CURRENT_REPO_KEY


@dataclass
class AdvisoryRunRecord:
    """A single completed advisory pre-review run."""

    snapshot_hash: str
    commit_message: str
    status: str
    ts: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    snapshot_summary: str = ""
    raw_result: str = ""
    bypass_reason: str = ""
    bypassed_by_task: str = ""
    snapshot_paths: Optional[List[str]] = field(default=None)
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    tool_name: str = _DEFAULT_ADVISORY_TOOL_NAME
    task_id: str = ""
    attempt: int = 0
    phase: str = "advisory"
    created_ts: str = ""
    updated_ts: str = ""

@dataclass
class CommitAttemptRecord:
    """Tracks one reviewed mutative tool attempt across its lifecycle."""

    ts: str
    commit_message: str
    status: str
    snapshot_hash: str = ""
    block_reason: str = ""
    block_details: str = ""
    duration_sec: float = 0.0
    task_id: str = ""
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    repo_key: str = _LEGACY_CURRENT_REPO_KEY
    tool_name: str = _DEFAULT_TOOL_NAME
    attempt: int = 0
    phase: str = "review"
    blocked: bool = False
    advisory_findings: List[Dict[str, Any]] = field(default_factory=list)
    obligation_ids: List[str] = field(default_factory=list)
    readiness_warnings: List[str] = field(default_factory=list)
    late_result_pending: bool = False
    pre_review_fingerprint: str = ""
    post_review_fingerprint: str = ""
    fingerprint_status: str = ""  # "pending" | "matched" | "mismatch" | "unavailable"
    degraded_reasons: List[str] = field(default_factory=list)
    started_ts: str = ""
    updated_ts: str = ""
    finished_ts: str = ""

@dataclass
class AdvisoryReviewState:
    """Top-level durable state container."""

    state_version: int = _STATE_SCHEMA_VERSION
    advisory_runs: List[AdvisoryRunRecord] = field(default_factory=list)
    attempts: List[CommitAttemptRecord] = field(default_factory=list)
    last_commit_attempt: Optional[CommitAttemptRecord] = field(default=None)
    blocking_history: List[CommitAttemptRecord] = field(default_factory=list)
    open_obligations: List[ObligationItem] = field(default_factory=list)
    last_stale_from_edit_ts: str = ""
    last_stale_reason: str = ""
    last_stale_repo_key: str = ""

    @property
    def runs(self) -> List[AdvisoryRunRecord]:
        """Backward-compatible alias used by existing callers/tests."""
        return self.advisory_runs

    @runs.setter
    def runs(self, value: List[AdvisoryRunRecord]) -> None:
        self.advisory_runs = list(value or [])

    def latest(self) -> Optional[AdvisoryRunRecord]:
        return self.advisory_runs[-1] if self.advisory_runs else None

    def latest_attempt(self) -> Optional[CommitAttemptRecord]:
        return self.attempts[-1] if self.attempts else self.last_commit_attempt

    def latest_attempt_for(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> Optional[CommitAttemptRecord]:
        matches = self.filter_attempts(
            repo_key=repo_key,
            tool_name=tool_name,
            task_id=task_id,
            attempt=attempt,
        )
        return matches[-1] if matches else None

    def get_active_attempts(self, *, repo_key: str | None = None) -> List[CommitAttemptRecord]:
        active = [
            item for item in self.attempts
            if item.status == "reviewing" or item.late_result_pending
        ]
        if repo_key is not None:
            exact_match_exists = _repo_scope_exact_match_exists(active, repo_key)
            active = [
                item for item in active
                if _repo_scope_matches(
                    item.repo_key,
                    repo_key,
                    exact_match_exists=exact_match_exists,
                )
            ]
        return active

    def filter_advisory_runs(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> List[AdvisoryRunRecord]:
        results = list(self.advisory_runs)
        if repo_key is not None:
            exact_match_exists = _repo_scope_exact_match_exists(results, repo_key)
            results = [
                run for run in results
                if _repo_scope_matches(
                    run.repo_key,
                    repo_key,
                    exact_match_exists=exact_match_exists,
                )
            ]
        if tool_name is not None:
            results = [run for run in results if run.tool_name == tool_name]
        if task_id is not None:
            results = [run for run in results if run.task_id == task_id]
        if attempt is not None:
            results = [run for run in results if int(run.attempt or 0) == int(attempt)]
        return results

    def filter_attempts(
        self,
        *,
        repo_key: str | None = None,
        tool_name: str | None = None,
        task_id: str | None = None,
        attempt: int | None = None,
    ) -> List[CommitAttemptRecord]:
        results = list(self.attempts)
        if repo_key is not None:
            exact_match_exists = _repo_scope_exact_match_exists(results, repo_key)
            results = [
                item for item in results
                if _repo_scope_matches(
                    item.repo_key,
                    repo_key,
                    exact_match_exists=exact_match_exists,
                )
            ]
        if tool_name is not None:
            results = [item for item in results if item.tool_name == tool_name]
        if task_id is not None:
            results = [item for item in results if item.task_id == task_id]
        if attempt is not None:
            results = [item for item in results if int(item.attempt or 0) == int(attempt)]
        return results

    def next_attempt_number(self, repo_key: str, tool_name: str, task_id: str = "") -> int:
        candidates = self.filter_attempts(repo_key=repo_key, tool_name=tool_name, task_id=task_id)
        latest = max((int(item.attempt or 0) for item in candidates), default=0)
        return latest + 1

    def find_by_hash(
        self,
        snapshot_hash: str,
        repo_key: str | None = None,
    ) -> Optional[AdvisoryRunRecord]:
        exact_match_exists = _repo_scope_exact_match_exists(self.advisory_runs, repo_key)
        for run in reversed(self.advisory_runs):
            if run.snapshot_hash != snapshot_hash:
                continue
            if _repo_scope_matches(
                run.repo_key,
                repo_key,
                exact_match_exists=exact_match_exists,
            ):
                return run
        return None

    def is_fresh(self, snapshot_hash: str, repo_key: str | None = None) -> bool:
        run = self.find_by_hash(snapshot_hash, repo_key=repo_key)
        return run is not None and run.status in ("fresh", "bypassed", "skipped")

    def add_run(self, run: AdvisoryRunRecord) -> None:
        if not run.created_ts:
            run.created_ts = run.ts or _utc_now()
        if not run.updated_ts:
            run.updated_ts = run.created_ts
        self.mark_all_stale_except(run.snapshot_hash, repo_key=run.repo_key)
        self.advisory_runs.append(run)
        if len(self.advisory_runs) > _MAX_RUN_HISTORY:
            self.advisory_runs = self.advisory_runs[-_MAX_RUN_HISTORY:]
        if run.status in ("fresh", "bypassed", "skipped", "parse_failure"):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""

    def mark_stale(self, snapshot_hash: str) -> None:
        for run in self.advisory_runs:
            if run.snapshot_hash == snapshot_hash:
                run.status = "stale"
                run.updated_ts = _utc_now()

    def mark_all_stale_except(self, snapshot_hash: str, repo_key: str = "") -> None:
        for run in self.advisory_runs:
            same_repo = not repo_key or run.repo_key == repo_key
            if same_repo and run.snapshot_hash != snapshot_hash and run.status in ("fresh", "bypassed", "skipped"):
                run.status = "stale"
                run.updated_ts = _utc_now()

    def mark_all_stale(self, reason_ts: str = "", reason: str = "", repo_key: str = "") -> None:
        self.mark_repo_stale(repo_key="", reason_ts=reason_ts, reason=reason, stale_repo_key=repo_key)

    def mark_repo_stale(
        self,
        *,
        repo_key: str = "",
        reason_ts: str = "",
        reason: str = "",
        stale_repo_key: str = "",
    ) -> int:
        """Invalidate advisory runs for a repo, with conservative fallback to full stale."""
        invalidatable = [
            run for run in self.advisory_runs
            if run.status in ("fresh", "bypassed", "skipped")
        ]
        if not invalidatable:
            return 0

        target_runs: List[AdvisoryRunRecord]
        if not repo_key:
            target_runs = invalidatable
        else:
            exact_matches = [run for run in invalidatable if run.repo_key == repo_key]
            legacy_present = any(run.repo_key in ("", _LEGACY_CURRENT_REPO_KEY) for run in invalidatable)
            target_runs = invalidatable if legacy_present and not exact_matches else (exact_matches or invalidatable)

        for run in target_runs:
            run.status = "stale"
            run.updated_ts = reason_ts or _utc_now()
        if target_runs:
            self.last_stale_from_edit_ts = reason_ts or _utc_now()
            self.last_stale_reason = reason
            self.last_stale_repo_key = stale_repo_key or repo_key
        return len(target_runs)

    def add_blocking_attempt(self, attempt: CommitAttemptRecord) -> None:
        """Backward-compatible alias preserved for existing callers/tests."""
        attempt.status = "blocked"
        attempt.blocked = True
        self.record_attempt(attempt)

    def record_attempt(self, attempt: CommitAttemptRecord) -> CommitAttemptRecord:
        """Upsert one reviewed attempt into the durable ledger and compatibility views."""
        now = _utc_now()
        attempt.tool_name = str(attempt.tool_name or _DEFAULT_TOOL_NAME)
        attempt.repo_key = str(attempt.repo_key or _LEGACY_CURRENT_REPO_KEY)
        attempt.blocked = bool(attempt.blocked or attempt.status == "blocked")
        if not attempt.started_ts:
            attempt.started_ts = attempt.ts or now
        if not attempt.ts:
            attempt.ts = attempt.started_ts
        attempt.updated_ts = now
        if attempt.status in ("blocked", "failed", "succeeded") and not attempt.finished_ts:
            attempt.finished_ts = now

        merged = self._upsert_attempt(attempt)
        self.last_commit_attempt = merged

        if merged.status == "blocked" or merged.blocked:
            merged.blocked = True
            merged.obligation_ids = self._update_obligations_from_attempt(merged)
            self._upsert_attempt(merged)
            self._upsert_blocking_history(merged)
        elif merged.status == "succeeded":
            self.on_successful_commit(repo_key=merged.repo_key)

        return merged

    def _upsert_attempt(self, attempt: CommitAttemptRecord) -> CommitAttemptRecord:
        key = _attempt_identity_tuple(attempt)
        for idx, existing in enumerate(self.attempts):
            if _attempt_identity_tuple(existing) == key:
                merged = _merge_attempt(existing, attempt)
                self.attempts[idx] = merged
                return merged
        self.attempts.append(attempt)
        if len(self.attempts) > _MAX_ATTEMPT_HISTORY:
            self.attempts = self.attempts[-_MAX_ATTEMPT_HISTORY:]
        return attempt

    def _upsert_blocking_history(self, attempt: CommitAttemptRecord) -> None:
        key = _attempt_identity_tuple(attempt)
        for idx, existing in enumerate(self.blocking_history):
            if _attempt_identity_tuple(existing) == key:
                self.blocking_history[idx] = attempt
                break
        else:
            self.blocking_history.append(attempt)
        if len(self.blocking_history) > _MAX_BLOCKING_HISTORY:
            self.blocking_history = self.blocking_history[-_MAX_BLOCKING_HISTORY:]

    def _make_obligation_id(self, item: str, reason: str) -> str:
        key = f"{item}:{reason}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def _update_obligations_from_attempt(self, attempt: CommitAttemptRecord) -> List[str]:
        """Merge critical findings into open obligations keyed by per-finding fingerprint.

        Different reasons → different obligations (no moving-target collapse).
        Same reason repeated → merges into existing obligation (reason deduped).
        """
        if not attempt.critical_findings:
            return []

        existing = {ob.obligation_id: ob for ob in self.get_open_obligations(repo_key=attempt.repo_key)}
        touched_ids = []

        for f in attempt.critical_findings:
            if not isinstance(f, dict):
                continue
            if str(f.get("verdict", "")).upper() != "FAIL":
                continue
            if str(f.get("severity", "")).lower() != "critical":
                continue
            item = str(f.get("item", "unknown"))
            reason = str(f.get("reason", ""))
            severity = str(f.get("severity", "critical"))
            ob_id = self._make_obligation_id(item, reason)

            if ob_id in existing:
                # Same finding reappeared — update timestamp, dedup reason text.
                ob = existing[ob_id]
                prev = [s.strip() for s in (ob.reason or "").split(" | ") if s.strip()]
                if reason.strip() not in set(prev):
                    ob.reason = " | ".join(prev + [reason.strip()]) if prev else reason
                ob.source_attempt_ts = attempt.ts
                ob.source_attempt_msg = attempt.commit_message
            else:
                new_ob = ObligationItem(
                    obligation_id=ob_id,
                    item=item,
                    severity=severity,
                    reason=reason,
                    source_attempt_ts=attempt.ts,
                    source_attempt_msg=attempt.commit_message,
                    status="still_open",
                    repo_key=attempt.repo_key,
                )
                self.open_obligations.append(new_ob)
                existing[ob_id] = new_ob

            touched_ids.append(ob_id)

        return touched_ids

    def resolve_obligations(
        self,
        resolved_ids: List[str],
        resolved_by: str = "",
        repo_key: str | None = None,
    ) -> int:
        exact_match_exists = _repo_scope_exact_match_exists(self.open_obligations, repo_key)
        count = 0
        for ob in self.open_obligations:
            if ob.obligation_id not in resolved_ids or ob.status != "still_open":
                continue
            if not _repo_scope_matches(
                ob.repo_key,
                repo_key,
                exact_match_exists=exact_match_exists,
            ):
                continue
            if ob.obligation_id in resolved_ids and ob.status == "still_open":
                ob.status = "resolved"
                ob.resolved_by = resolved_by
                count += 1
        return count

    def clear_resolved_obligations(self) -> None:
        self.open_obligations = [ob for ob in self.open_obligations if ob.status == "still_open"]

    def get_open_obligations(self, repo_key: str | None = None) -> List[ObligationItem]:
        exact_match_exists = _repo_scope_exact_match_exists(self.open_obligations, repo_key)
        return [
            ob for ob in self.open_obligations
            if ob.status == "still_open"
            and _repo_scope_matches(
                ob.repo_key,
                repo_key,
                exact_match_exists=exact_match_exists,
            )
        ]

    def get_blocking_history(self, repo_key: str | None = None) -> List[CommitAttemptRecord]:
        exact_match_exists = _repo_scope_exact_match_exists(self.blocking_history, repo_key)
        return [
            attempt for attempt in self.blocking_history
            if _repo_scope_matches(
                attempt.repo_key,
                repo_key,
                exact_match_exists=exact_match_exists,
            )
        ]

    def on_successful_commit(self, repo_key: str | None = None) -> None:
        if repo_key is None:
            self.open_obligations = []
            self.blocking_history = []
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""
            return

        exact_obligation_match = _repo_scope_exact_match_exists(self.open_obligations, repo_key)
        self.open_obligations = [
            ob for ob in self.open_obligations
            if not _repo_scope_matches(
                ob.repo_key,
                repo_key,
                exact_match_exists=exact_obligation_match,
            )
        ]
        exact_history_match = _repo_scope_exact_match_exists(self.blocking_history, repo_key)
        self.blocking_history = [
            attempt for attempt in self.blocking_history
            if not _repo_scope_matches(
                attempt.repo_key,
                repo_key,
                exact_match_exists=exact_history_match,
            )
        ]
        if self.last_stale_repo_key in ("", repo_key):
            self.last_stale_from_edit_ts = ""
            self.last_stale_reason = ""
            self.last_stale_repo_key = ""

    def expire_stale_attempts(
        self,
        *,
        now_ts: str | None = None,
        ttl_sec: int = _REVIEW_ATTEMPT_TTL_SEC,
        grace_sec: int = _REVIEW_ATTEMPT_GRACE_SEC,
    ) -> List[CommitAttemptRecord]:
        """Auto-expire stale reviewing/late attempts after TTL+grace."""
        now_ts = now_ts or _utc_now()
        now_epoch = _parse_iso_ts(now_ts)
        if now_epoch is None:
            return []

        expired: List[CommitAttemptRecord] = []
        for item in self.attempts:
            if item.status != "reviewing" and not item.late_result_pending:
                continue
            started_epoch = _parse_iso_ts(item.started_ts or item.ts)
            if started_epoch is None:
                continue
            age_sec = max(0.0, now_epoch - started_epoch)
            if age_sec < float(ttl_sec + grace_sec):
                continue

            item.status = "failed"
            item.phase = "expired"
            item.blocked = False
            item.block_reason = "infra_failure"
            item.block_details = (
                f"Auto-expired stale reviewed attempt after {ttl_sec + grace_sec}s TTL+grace."
            )
            item.duration_sec = max(item.duration_sec, round(age_sec, 1))
            item.finished_ts = now_ts
            item.updated_ts = now_ts
            item.late_result_pending = False
            item.readiness_warnings = _dedupe_strings(
                list(item.readiness_warnings or [])
                + ["Previous reviewed attempt auto-expired after exceeding TTL+grace."]
            )
            expired.append(item)

        if expired and self.last_commit_attempt:
            expired_keys = {_attempt_identity_tuple(item) for item in expired}
            if _attempt_identity_tuple(self.last_commit_attempt) in expired_keys:
                replacement = self.latest_attempt_for(
                    repo_key=self.last_commit_attempt.repo_key,
                    tool_name=self.last_commit_attempt.tool_name,
                    task_id=self.last_commit_attempt.task_id,
                    attempt=self.last_commit_attempt.attempt,
                )
                if replacement is not None:
                    self.last_commit_attempt = replacement

        return expired


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _obligation_from_dict(d: Dict[str, Any]) -> ObligationItem:
    return ObligationItem(
        obligation_id=str(d.get("obligation_id", "")),
        item=str(d.get("item", "")),
        severity=str(d.get("severity", "critical")),
        reason=str(d.get("reason", "")),
        source_attempt_ts=str(d.get("source_attempt_ts", "")),
        source_attempt_msg=str(d.get("source_attempt_msg", "")),
        status=str(d.get("status", "still_open")),
        resolved_by=str(d.get("resolved_by", "")),
        repo_key=str(d.get("repo_key", _LEGACY_CURRENT_REPO_KEY)),
    )


def _record_from_dict(d: Dict[str, Any]) -> AdvisoryRunRecord:
    raw_paths = d.get("snapshot_paths")
    ts = str(d.get("ts", ""))
    return AdvisoryRunRecord(
        snapshot_hash=str(d.get("snapshot_hash", "")),
        commit_message=str(d.get("commit_message", "")),
        status=str(d.get("status", "stale")),
        ts=ts,
        items=list(d.get("items") or []),
        snapshot_summary=str(d.get("snapshot_summary", "")),
        raw_result=str(d.get("raw_result", "")),
        bypass_reason=str(d.get("bypass_reason", "")),
        bypassed_by_task=str(d.get("bypassed_by_task", "")),
        snapshot_paths=list(raw_paths) if isinstance(raw_paths, list) else None,
        repo_key=str(d.get("repo_key", _LEGACY_CURRENT_REPO_KEY)),
        tool_name=str(d.get("tool_name", _DEFAULT_ADVISORY_TOOL_NAME)),
        task_id=str(d.get("task_id", d.get("bypassed_by_task", ""))),
        attempt=_coerce_int(d.get("attempt", 0)),
        phase=str(d.get("phase", "advisory")),
        created_ts=str(d.get("created_ts", ts)),
        updated_ts=str(d.get("updated_ts", ts)),
    )


def _commit_attempt_from_dict(d: Dict[str, Any]) -> CommitAttemptRecord:
    ts = str(d.get("ts", ""))
    status = str(d.get("status", "failed"))
    return CommitAttemptRecord(
        ts=ts,
        commit_message=str(d.get("commit_message", "")),
        status=status,
        snapshot_hash=str(d.get("snapshot_hash", "")),
        block_reason=str(d.get("block_reason", "")),
        block_details=str(d.get("block_details", "")),
        duration_sec=float(d.get("duration_sec", 0.0)),
        task_id=str(d.get("task_id", "")),
        critical_findings=list(d.get("critical_findings") or []),
        repo_key=str(d.get("repo_key", _LEGACY_CURRENT_REPO_KEY)),
        tool_name=str(d.get("tool_name", _DEFAULT_TOOL_NAME)),
        attempt=_coerce_int(d.get("attempt", 0)),
        phase=str(d.get("phase", _infer_phase(status, str(d.get("block_reason", ""))))),
        blocked=bool(d.get("blocked", status == "blocked")),
        advisory_findings=_normalize_findings(d.get("advisory_findings") or []),
        obligation_ids=[str(x) for x in (d.get("obligation_ids") or [])],
        readiness_warnings=[str(x) for x in (d.get("readiness_warnings") or [])],
        late_result_pending=bool(d.get("late_result_pending", False)),
        pre_review_fingerprint=str(d.get("pre_review_fingerprint", "")),
        post_review_fingerprint=str(d.get("post_review_fingerprint", "")),
        fingerprint_status=str(d.get("fingerprint_status", "")),
        degraded_reasons=[str(x) for x in (d.get("degraded_reasons") or [])],
        started_ts=str(d.get("started_ts", ts)),
        updated_ts=str(d.get("updated_ts", ts)),
        finished_ts=str(d.get("finished_ts", ts if status in ("blocked", "failed", "succeeded") else "")),
    )


def _load_state_unlocked(drive_root: pathlib.Path) -> AdvisoryReviewState:
    path = drive_root / _STATE_RELPATH
    if not path.exists():
        return AdvisoryReviewState()

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        return AdvisoryReviewState()

    raw_runs = data.get("advisory_runs")
    if raw_runs is None:
        raw_runs = data.get("runs") or []
    advisory_runs = [_record_from_dict(r) for r in (raw_runs or []) if isinstance(r, dict)]

    attempts = [
        _commit_attempt_from_dict(item)
        for item in (data.get("attempts") or [])
        if isinstance(item, dict)
    ]

    last_commit = None
    if isinstance(data.get("last_commit_attempt"), dict):
        last_commit = _commit_attempt_from_dict(data["last_commit_attempt"])

    blocking_history = [
        _commit_attempt_from_dict(item)
        for item in (data.get("blocking_history") or [])
        if isinstance(item, dict)
    ]
    open_obligations = [
        _obligation_from_dict(item)
        for item in (data.get("open_obligations") or [])
        if isinstance(item, dict)
    ]

    state = AdvisoryReviewState(
        state_version=_coerce_int(data.get("state_version", data.get("schema_version", _STATE_SCHEMA_VERSION))),
        advisory_runs=advisory_runs,
        attempts=attempts,
        last_commit_attempt=last_commit,
        blocking_history=blocking_history,
        open_obligations=open_obligations,
        last_stale_from_edit_ts=str(data.get("last_stale_from_edit_ts", "")),
        last_stale_reason=str(data.get("last_stale_reason", "")),
        last_stale_repo_key=str(data.get("last_stale_repo_key", "")),
    )

    if not state.attempts:
        recovered: List[CommitAttemptRecord] = []
        if state.last_commit_attempt:
            recovered.append(state.last_commit_attempt)
        for item in state.blocking_history:
            if _attempt_identity_tuple(item) not in {_attempt_identity_tuple(x) for x in recovered}:
                recovered.append(item)
        state.attempts = recovered

    if state.attempts and state.last_commit_attempt is None:
        state.last_commit_attempt = state.attempts[-1]

    return state


def load_state(drive_root: pathlib.Path) -> AdvisoryReviewState:
    """Load review state from disk. Returns empty state on any error."""
    try:
        return _load_state_unlocked(drive_root)
    except Exception as e:
        path = drive_root / _STATE_RELPATH
        log.warning("Failed to load advisory review state from %s: %s", path, e)
        return AdvisoryReviewState()


def _save_state_unlocked(drive_root: pathlib.Path, state: AdvisoryReviewState) -> None:
    path = drive_root / _STATE_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    _sync_compat_views(state)
    data: Dict[str, Any] = {
        "state_version": _STATE_SCHEMA_VERSION,
        "schema_version": _STATE_SCHEMA_VERSION,
        "advisory_runs": [asdict(r) for r in state.advisory_runs],
        "runs": [asdict(r) for r in state.advisory_runs],
        "attempts": [asdict(r) for r in state.attempts],
        "last_commit_attempt": asdict(state.last_commit_attempt) if state.last_commit_attempt else None,
        "blocking_history": [asdict(r) for r in state.blocking_history],
        "open_obligations": [asdict(o) for o in state.open_obligations],
        "last_stale_from_edit_ts": state.last_stale_from_edit_ts,
        "last_stale_reason": state.last_stale_reason,
        "last_stale_repo_key": state.last_stale_repo_key,
        "saved_at": _utc_now(),
    }
    tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_state(drive_root: pathlib.Path, state: AdvisoryReviewState) -> None:
    """Persist review state atomically with a best-effort lock."""
    lock_path = drive_root / _LOCK_RELPATH
    lock_fd = acquire_review_state_lock(drive_root)
    if lock_fd is None:
        log.warning("Failed to acquire review state lock at %s; skipping save", lock_path)
        return
    try:
        _save_state_unlocked(drive_root, state)
    finally:
        release_review_state_lock(drive_root, lock_fd)


def update_state(
    drive_root: pathlib.Path,
    mutator: Callable[[AdvisoryReviewState], Any],
) -> Any:
    """Run a read-modify-write update under an explicit lock/lockfile."""
    lock_fd = acquire_review_state_lock(drive_root)
    if lock_fd is None:
        raise TimeoutError(f"Could not acquire review state lock for {drive_root / _LOCK_RELPATH}")
    try:
        state = _load_state_unlocked(drive_root)
        result = mutator(state)
        _save_state_unlocked(drive_root, state)
        return state if result is None else result
    finally:
        release_review_state_lock(drive_root, lock_fd)


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------

def acquire_review_state_lock(
    drive_root: pathlib.Path,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
) -> Optional[int]:
    lock_path = drive_root / _LOCK_RELPATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while (time.time() - started) < timeout_sec:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            meta = f"pid={os.getpid()} ts={_utc_now()}\n".encode("utf-8")
            try:
                os.write(fd, meta)
            except Exception:
                log.debug("Failed to write review-state lock metadata", exc_info=True)
            return fd
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_sec:
                    lock_path.unlink()
                    continue
            except Exception:
                log.debug("Failed to inspect/remove stale review-state lock", exc_info=True)
            time.sleep(0.05)
        except Exception:
            log.warning("Failed to acquire review-state lock at %s", lock_path, exc_info=True)
            break
    return None


def release_review_state_lock(drive_root: pathlib.Path, lock_fd: Optional[int]) -> None:
    lock_path = drive_root / _LOCK_RELPATH
    if lock_fd is not None:
        try:
            os.close(lock_fd)
        except Exception:
            log.debug("Failed to close review-state lock fd", exc_info=True)
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        log.debug("Failed to unlink review-state lock", exc_info=True)


# ---------------------------------------------------------------------------
# Snapshot hash / repo identity
# ---------------------------------------------------------------------------

_SNAPSHOT_EXCLUDE_PATHS = frozenset({
    "state/advisory_review.json",
    "state/queue_snapshot.json",
})


def discover_repo_root(path: pathlib.Path) -> pathlib.Path:
    """Return the nearest directory containing `.git`, else the resolved input path."""
    resolved = path.resolve()
    current = resolved if resolved.is_dir() else resolved.parent
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return resolved if resolved.is_dir() else resolved.parent
        current = current.parent


def make_repo_key(repo_dir: pathlib.Path) -> str:
    return str(discover_repo_root(repo_dir))


def compute_snapshot_hash(
    repo_dir: pathlib.Path,
    commit_message: str = "",
    paths: list[str] | None = None,
) -> str:
    """Build a deterministic hash for the current worktree snapshot."""
    if isinstance(paths, list) and len(paths) == 0:
        paths = None

    changed_digests: List[tuple[str, str]] = []

    def _record_digest(relpath: str) -> None:
        relpath = relpath.strip()
        if not relpath or relpath in _SNAPSHOT_EXCLUDE_PATHS:
            return
        file_path = repo_dir / relpath
        try:
            if file_path.is_file():
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
            else:
                digest = "deleted"
        except Exception:
            digest = "unreadable"
        changed_digests.append((relpath, digest))

    if paths is not None:
        for relpath in paths:
            _record_digest(relpath)
    else:
        try:
            from ouroboros.tools.review_helpers import list_changed_paths_from_git_status

            for relpath in list_changed_paths_from_git_status(repo_dir):
                _record_digest(relpath)
        except Exception as e:
            log.debug("compute_snapshot_hash: git status failed: %s", e)

    h = hashlib.sha256()
    for relpath, digest in sorted(changed_digests):
        h.update(f"{relpath}:{digest}\n".encode())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# Advisory staleness from worktree edits
# ---------------------------------------------------------------------------

def mark_advisory_stale_after_edit(drive_root: pathlib.Path) -> None:
    """Mark all fresh advisory runs as stale because the worktree was modified."""
    try:
        updated = update_state(drive_root, lambda state: _mark_advisory_stale_locked(state))
        if isinstance(updated, AdvisoryReviewState):
            log.debug("Advisory state marked stale after worktree edit")
    except Exception as e:
        log.debug("mark_advisory_stale_after_edit failed (non-fatal): %s", e)


def _mark_advisory_stale_locked(state: AdvisoryReviewState) -> None:
    has_invalidatable = any(r.status in ("fresh", "bypassed", "skipped") for r in state.advisory_runs)
    if not has_invalidatable:
        return
    state.mark_all_stale(reason_ts=_utc_now(), reason="Worktree edit invalidated advisory freshness.")


def invalidate_advisory_after_mutation(
    drive_root: pathlib.Path,
    *,
    mutation_root: pathlib.Path | None = None,
    changed_paths: Optional[List[str]] = None,
    source_tool: str = "",
) -> None:
    """Invalidate advisory freshness after a successful mutating tool path.

    Repo identity is inferred from the nearest `.git` root of the mutation root / changed
    paths. If repo resolution is ambiguous, conservatively stale all fresh advisory runs.
    """
    try:
        changed_paths = [str(p).strip() for p in (changed_paths or []) if str(p).strip()]
        resolved_repo_keys = _resolve_mutation_repo_keys(mutation_root, changed_paths)
        reason_ts = _utc_now()
        reason = _build_invalidation_reason(source_tool, mutation_root, changed_paths, resolved_repo_keys)

        def _mutate(state: AdvisoryReviewState) -> None:
            if not resolved_repo_keys or len(resolved_repo_keys) != 1:
                state.mark_all_stale(reason_ts=reason_ts, reason=reason)
                return
            state.mark_repo_stale(
                repo_key=resolved_repo_keys[0],
                reason_ts=reason_ts,
                reason=reason,
                stale_repo_key=resolved_repo_keys[0],
            )

        update_state(drive_root, _mutate)
    except Exception as e:
        log.debug("invalidate_advisory_after_mutation failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

def format_status_section(state: AdvisoryReviewState,
                          repo_dir: Optional[pathlib.Path] = None) -> str:
    """Render a compact historical section for LLM context injection."""
    repo_key = make_repo_key(repo_dir) if repo_dir is not None else None
    advisory_runs = state.filter_advisory_runs(repo_key=repo_key) if repo_key is not None else list(state.advisory_runs)
    attempts = state.filter_attempts(repo_key=repo_key) if repo_key is not None else list(state.attempts)
    last_attempt = state.latest_attempt_for(repo_key=repo_key) if repo_key is not None else state.last_commit_attempt
    open_obs = state.get_open_obligations(repo_key=repo_key)

    if not advisory_runs and last_attempt is None and not open_obs:
        return "## Advisory Pre-Review Status\n\nNo advisory runs recorded yet."

    lines = [
        "## Advisory Pre-Review Status",
        "(Historical — run `review_status` for gate-accurate live freshness)",
    ]

    # No [-3:] cap — include ALL advisory runs so history is preserved.
    for run in advisory_runs:
        status_icon = {
            "fresh": "✅",
            "stale": "⚠️",
            "bypassed": "⏭️",
            "skipped": "⏭️",
            "parse_failure": "🔴",
        }.get(run.status, "❓")
        ts_display = run.ts  # full timestamp — no [:16] truncation
        hash_short = run.snapshot_hash[:12]
        commit_display = run.commit_message  # full message — no [:60] truncation

        lines.append(f"\n{status_icon} **{run.status.upper()}** | hash={hash_short} | {ts_display}")
        lines.append(f"   Commit: {commit_display}")
        if run.bypass_reason:
            lines.append(f"   Bypassed: {run.bypass_reason}")
        if run.snapshot_summary:
            lines.append(f"   Scope: {run.snapshot_summary}")

        findings = [
            item for item in (run.items or [])
            if isinstance(item, dict) and str(item.get("verdict", "")).upper() == "FAIL"
        ]
        if findings:
            lines.append(f"   Findings ({len(findings)}):")
            # No [:N] cap — include ALL findings so advisory receives complete history.
            for item in findings:
                sev = str(item.get("severity", "advisory")).upper()
                name = item.get("item", "?")
                reason = _truncate_review_reason(item.get("reason", ""))
                lines.append(f"     [{sev}] {name}: {reason}")
        elif run.status in ("fresh", "bypassed", "skipped", "parse_failure"):
            lines.append("   No findings recorded.")

    stale_matches_repo = repo_key is None or state.last_stale_repo_key in ("", repo_key)
    if state.last_stale_from_edit_ts and stale_matches_repo:
        lines.append(f"\n⚠️ Advisory marked stale after worktree edit at {state.last_stale_from_edit_ts}.")  # full ts — no [:16]
        if state.last_stale_reason:
            lines.append(f"   Reason: {state.last_stale_reason}")
        lines.append("   Run advisory_pre_review again before repo_commit.")

    # No [-3:] cap — include ALL attempts so nothing is silently dropped.
    recent_attempts = attempts
    if recent_attempts:
        lines.append("\n### Recent reviewed attempts")
        for item in recent_attempts:
            tool = item.tool_name or _DEFAULT_TOOL_NAME
            num = int(item.attempt or 0)
            label = f"{tool}#{num}" if num else tool
            phase = item.phase or "review"
            facts = [f"status={item.status}", f"phase={phase}", f"blocked={'yes' if item.blocked else 'no'}"]
            if item.late_result_pending:
                facts.append("late_result_pending=yes")
            if item.readiness_warnings:
                facts.append(f"warnings={len(item.readiness_warnings)}")
            lines.append(f"- {label}: {', '.join(facts)}")

    ca = last_attempt
    if ca and ca.status in ("blocked", "failed"):
        icon = "🚫" if ca.status == "blocked" else "❌"
        ts_display = ca.ts  # full timestamp — no [:16] truncation
        commit_display = ca.commit_message  # full message — no [:60] truncation
        lines.append(f"\n{icon} **Last commit {ca.status.upper()}** | {ts_display}")
        lines.append(f"   Commit: {commit_display}")
        lines.append(f"   Tool: {ca.tool_name or _DEFAULT_TOOL_NAME}")
        if ca.attempt:
            lines.append(f"   Attempt: {ca.attempt}")
        if ca.block_reason:
            lines.append(f"   Reason: {ca.block_reason}")
        if ca.block_details:
            preview = _truncate_review_artifact(ca.block_details, limit=200).replace("\n", " ")
            lines.append(f"   Details: {preview}")
        if ca.duration_sec > 0:
            lines.append(f"   Duration: {ca.duration_sec:.1f}s")
        if ca.readiness_warnings:
            lines.append(f"   Readiness warnings ({len(ca.readiness_warnings)}):")
            # No [:N] cap — all warnings shown; review outputs must not truncate.
            for warning in ca.readiness_warnings:
                lines.append(f"     - {_truncate_review_reason(warning, limit=160)}")
        critical_findings = list(ca.critical_findings or [])
        advisory_findings = list(ca.advisory_findings or [])
        if critical_findings:
            lines.append(f"   Critical findings ({len(critical_findings)}):")
            # No [:N] cap — all findings preserved for complete carry-over.
            for finding in critical_findings:
                label = str(finding.get("item") or finding.get("reason") or "?")
                reason = _truncate_review_reason(finding.get("reason", ""), limit=160)
                lines.append(f"     - {label}: {reason}")
        elif advisory_findings:
            lines.append(f"   Advisory findings ({len(advisory_findings)}):")
            # No [:N] cap — all findings preserved.
            for finding in advisory_findings:
                label = str(finding.get("item") or finding.get("reason") or "?")
                reason = _truncate_review_reason(finding.get("reason", ""), limit=160)
                lines.append(f"     - {label}: {reason}")

    if open_obs:
        lines.append(f"\n📋 **Open obligations from previous blocking rounds ({len(open_obs)}):**")
        # No [:N] cap — all obligations shown; advisory must address each one.
        for ob in open_obs:
            reason = _truncate_review_reason(ob.reason)
            source = ob.source_attempt_msg  # full message — no [:60] truncation
            lines.append(f"   [{ob.obligation_id}] [{ob.severity.upper()}] {ob.item}: {reason}")
            lines.append(f"      Source: {ob.source_attempt_ts} — \"{source}\"")  # full ts — no [:16]
        lines.append("   Advisory MUST verify each obligation is resolved before PASS.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attempt_identity_tuple(attempt: CommitAttemptRecord) -> tuple[str, str, str, str]:
    attempt_number = int(attempt.attempt or 0)
    identity_token = (
        f"attempt:{attempt_number}"
        if attempt_number > 0
        else f"ts:{attempt.started_ts or attempt.ts or ''}"
    )
    return (
        str(attempt.repo_key or _LEGACY_CURRENT_REPO_KEY),
        str(attempt.tool_name or _DEFAULT_TOOL_NAME),
        str(attempt.task_id or ""),
        identity_token,
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_findings(items: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        elif item:
            normalized.append({"reason": str(item), "severity": "advisory"})
    return normalized


def _merge_attempt(existing: CommitAttemptRecord, incoming: CommitAttemptRecord) -> CommitAttemptRecord:
    merged = CommitAttemptRecord(
        ts=incoming.ts or existing.ts,
        commit_message=incoming.commit_message or existing.commit_message,
        status=incoming.status or existing.status,
        snapshot_hash=incoming.snapshot_hash or existing.snapshot_hash,
        block_reason=incoming.block_reason or existing.block_reason,
        block_details=incoming.block_details or existing.block_details,
        duration_sec=incoming.duration_sec or existing.duration_sec,
        task_id=incoming.task_id or existing.task_id,
        critical_findings=list(incoming.critical_findings),
        repo_key=incoming.repo_key or existing.repo_key,
        tool_name=incoming.tool_name or existing.tool_name,
        attempt=int(incoming.attempt or existing.attempt or 0),
        phase=incoming.phase or existing.phase,
        blocked=bool(incoming.blocked or incoming.status == "blocked"),
        advisory_findings=list(incoming.advisory_findings),
        obligation_ids=list(incoming.obligation_ids),
        readiness_warnings=list(incoming.readiness_warnings),
        late_result_pending=bool(incoming.late_result_pending),
        pre_review_fingerprint=incoming.pre_review_fingerprint or existing.pre_review_fingerprint,
        post_review_fingerprint=incoming.post_review_fingerprint or existing.post_review_fingerprint,
        fingerprint_status=incoming.fingerprint_status or existing.fingerprint_status,
        degraded_reasons=list(incoming.degraded_reasons or existing.degraded_reasons),
        started_ts=existing.started_ts or incoming.started_ts or existing.ts,
        updated_ts=incoming.updated_ts or existing.updated_ts or _utc_now(),
        finished_ts=incoming.finished_ts or existing.finished_ts,
    )
    return merged


def _infer_phase(status: str, block_reason: str = "") -> str:
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


def _parse_iso_ts(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _dedupe_strings(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _sync_compat_views(state: AdvisoryReviewState) -> None:
    """Keep compatibility views and ledger in sync before persistence."""
    if state.last_commit_attempt is not None:
        state._upsert_attempt(state.last_commit_attempt)
    elif state.attempts:
        state.last_commit_attempt = state.attempts[-1]


def _resolve_mutation_repo_keys(
    mutation_root: pathlib.Path | None,
    changed_paths: List[str],
) -> List[str]:
    base = mutation_root.resolve() if mutation_root is not None else None
    repo_keys: List[str] = []

    def _record(candidate: pathlib.Path) -> None:
        key = make_repo_key(candidate)
        if key and key not in repo_keys:
            repo_keys.append(key)

    if base is not None:
        _record(base)
    for rel_path in changed_paths:
        candidate = pathlib.Path(rel_path)
        if not candidate.is_absolute() and base is not None:
            candidate = (base / rel_path).resolve()
        elif not candidate.is_absolute():
            continue
        _record(candidate if candidate.exists() else candidate.parent)
    return repo_keys


def _build_invalidation_reason(
    source_tool: str,
    mutation_root: pathlib.Path | None,
    changed_paths: List[str],
    repo_keys: List[str],
) -> str:
    tool = source_tool or "mutation"
    repo_hint = ""
    if len(repo_keys) == 1:
        repo_hint = f" repo={repo_keys[0]}"
    elif len(repo_keys) > 1:
        repo_hint = " repo=multiple"
    path_hint = ""
    if changed_paths:
        preview = ", ".join(changed_paths[:3])
        if len(changed_paths) > 3:
            preview += f", +{len(changed_paths) - 3} more"
        path_hint = f" paths={preview}"
    elif mutation_root is not None:
        path_hint = f" root={mutation_root}"
    return f"{tool} mutated the worktree; advisory freshness invalidated.{repo_hint}{path_hint}"


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
