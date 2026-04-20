"""Regression tests: per-reviewer timeout enforcement in review pipelines.

Root cause fix for the late_result_pending hang:

In v4.18.4 and earlier, `_query_model` in `ouroboros/tools/review.py` caught
`asyncio.TimeoutError` but had no `asyncio.wait_for()` wrapping the actual
LLM call — the timeout branch was unreachable. A hung provider (e.g. a
`anthropic/claude-opus-4.7` endpoint returning 404 on some parameters while
others stalled on TLS) would freeze the entire triad review. The outer
soft tool timeout (600s) would then fire, shunting the commit tool into
`late_result_pending` for the full 1800s hard ceiling — effectively an
8+ minute silent hang before recovery.

The same bug existed on the `RuntimeError` fallback branch of
`_call_scope_llm` in `scope_review.py`.

These tests assert that:
  1. `_query_model` returns an "Error: Timeout after Ns" result when the
     underlying chat_async hangs longer than `_QUERY_MODEL_TIMEOUT_SEC`.
  2. Both scope-review LLM-call branches apply `asyncio.wait_for()` with
     an explicit deadline (verified structurally — the runtime fallback
     branch cannot be easily exercised in a test without a nested loop).
"""
from __future__ import annotations

import asyncio
import inspect

import pytest


# ---------------------------------------------------------------------------
# _query_model timeout behaviour
# ---------------------------------------------------------------------------

class _HangingClient:
    """Fake LLMClient whose chat_async hangs forever until cancelled."""

    async def chat_async(self, **kwargs):  # noqa: D401
        # Sleep long enough that any sensible deadline fires first, but
        # respond to cancellation promptly (asyncio.sleep is cancel-aware).
        await asyncio.sleep(3600)
        raise AssertionError("chat_async should have been cancelled by wait_for")


def test_query_model_has_hard_deadline():
    """Core fix: asyncio.wait_for must wrap chat_async in _query_model."""
    from ouroboros.tools import review as review_mod

    src = inspect.getsource(review_mod._query_model)
    assert "asyncio.wait_for" in src, (
        "_query_model must wrap chat_async in asyncio.wait_for — "
        "without it, a hung provider freezes the whole review."
    )
    # The dead 'except asyncio.TimeoutError' branch that existed before must
    # now actually be reachable. Sanity check: the constant is exported.
    assert hasattr(review_mod, "_QUERY_MODEL_TIMEOUT_SEC")
    assert 30 <= review_mod._QUERY_MODEL_TIMEOUT_SEC <= 600, (
        "Per-reviewer timeout must be substantial but not infinite."
    )


def test_query_model_returns_timeout_error_when_chat_hangs(monkeypatch):
    """Integration: a hanging client yields a structured timeout error, not a hang."""
    from ouroboros.tools import review as review_mod

    # Reduce the timeout for the test so we don't wait 180s.
    monkeypatch.setattr(review_mod, "_QUERY_MODEL_TIMEOUT_SEC", 0.2)

    async def _run():
        semaphore = asyncio.Semaphore(1)
        return await review_mod._query_model(
            _HangingClient(), "any/model-id", [{"role": "user", "content": "x"}], semaphore
        )

    # Guard: if wait_for is absent the test itself would hang — enforce a
    # generous outer deadline so failures produce a readable error.
    async def _guarded():
        return await asyncio.wait_for(_run(), timeout=5.0)

    model, payload, headers = asyncio.run(_guarded())
    assert model == "any/model-id"
    assert isinstance(payload, str), "Timeout/error path returns a string, not a dict."
    assert "Timeout" in payload


def test_query_model_handles_provider_exception_cleanly():
    """Non-timeout exceptions (e.g. a 404 BadRequestError) still become Error strings."""
    from ouroboros.tools import review as review_mod

    class _Failing:
        async def chat_async(self, **kwargs):
            raise RuntimeError("404: No endpoints found for model XYZ")

    async def _run():
        semaphore = asyncio.Semaphore(1)
        return await review_mod._query_model(
            _Failing(), "bad/model-id", [{"role": "user", "content": "x"}], semaphore
        )

    model, payload, headers = asyncio.run(asyncio.wait_for(_run(), timeout=5.0))
    assert model == "bad/model-id"
    assert isinstance(payload, str)
    assert "Error" in payload
    assert "No endpoints" in payload


# ---------------------------------------------------------------------------
# Scope review fallback deadline
# ---------------------------------------------------------------------------

def test_scope_review_async_call_has_deadline():
    """Source-level guard: both branches of _call_scope_llm enforce wait_for."""
    from ouroboros.tools import scope_review as scope_mod

    src = inspect.getsource(scope_mod._call_scope_llm)
    # Both the running-loop branch AND the RuntimeError fallback must wrap
    # the chat_async call in asyncio.wait_for. Expect >=2 occurrences because
    # the two branches are not code-shared.
    occurrences = src.count("asyncio.wait_for")
    assert occurrences >= 2, (
        "Both scope-review LLM branches must wrap chat_async in asyncio.wait_for. "
        f"Found {occurrences} occurrence(s)."
    )
    # Outer thread-pool ceiling must be >= inner wait_for so the inner deadline
    # wins the race and produces a TimeoutError inside the task (not a stale
    # thread).
    assert ".result(timeout=195)" in src or ".result(timeout=180" in src
