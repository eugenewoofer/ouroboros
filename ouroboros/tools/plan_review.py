"""plan_review.py — Pre-implementation design review tool.

Runs 3 parallel full-codebase reviews of a proposed implementation plan
BEFORE any code is written. Each reviewer sees the entire repository (same as
scope review) plus the plan description and the files to be touched.

Purpose: surface forgotten touchpoints, implicit contract violations, and
simpler alternatives *before* the first edit — preventing the iterative
micro-fix spiral that makes commit-gate expensive.

Usage:
    plan_task(
        plan="I want to add X by changing Y and Z...",
        goal="What should be achieved",
        files_to_touch=["ouroboros/foo.py", "tests/test_foo.py"]  # optional
    )
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from pathlib import Path

from ouroboros.llm import LLMClient
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.review_helpers import (
    build_full_repo_pack,
    build_head_snapshot_section,
    load_checklist_section,
)
from ouroboros.utils import estimate_tokens

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

_PLAN_REVIEW_MAX_TOKENS = 65536
_PLAN_REVIEW_EFFORT = "high"

# Budget gate: skip with advisory warning if prompt exceeds this token estimate.
# 800K gives comfortable margin under most 1M-token models (same as scope review).
_PLAN_BUDGET_TOKEN_LIMIT = 800_000


# ------------------------------------------------------------------ #
# Tool registration
# ------------------------------------------------------------------ #

def get_tools():
    return [
        ToolEntry(
            name="plan_task",
            schema={
                "name": "plan_task",
                "description": (
                    "Run a pre-implementation design review of a proposed plan using 3 parallel "
                    "full-codebase reviewers. Call this BEFORE writing any code for non-trivial tasks "
                    "(>2 files or >50 lines of changes). Each reviewer sees the entire repository "
                    "plus your plan description and the files you plan to touch. They will identify "
                    "forgotten touchpoints, implicit contract violations, simpler alternatives, and "
                    "Bible/architecture compliance issues — before you've written a single line. "
                    "Uses the 3 models configured in OUROBOROS_REVIEW_MODELS (same as commit triad). "
                    "Returns structured feedback from all 3 reviewers with detailed explanations and "
                    "alternative approaches. Non-blocking: you decide what to do with the feedback."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "string",
                            "description": (
                                "Describe what you plan to implement: which files you will change, "
                                "what the key design decisions are, and what you will NOT change."
                            ),
                        },
                        "goal": {
                            "type": "string",
                            "description": "The high-level goal of the task (what problem is being solved).",
                        },
                        "files_to_touch": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of repo-relative file paths you plan to modify. "
                                "Their current content (HEAD snapshot) will be injected so reviewers "
                                "can reason about concrete code, not just abstract plans."
                            ),
                        },
                    },
                    "required": ["plan", "goal"],
                },
            },
            handler=_handle_plan_task,
            timeout_sec=600,
        )
    ]


# ------------------------------------------------------------------ #
# Handler
# ------------------------------------------------------------------ #

def _handle_plan_task(
    ctx: ToolContext,
    plan: str = "",
    goal: str = "",
    files_to_touch: list | None = None,
) -> str:
    if not plan.strip():
        return "ERROR: plan parameter is required and must not be empty."
    if not goal.strip():
        return "ERROR: goal parameter is required and must not be empty."

    files_to_touch = files_to_touch or []

    try:
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    _run_plan_review_async(ctx, plan, goal, files_to_touch),
                ).result(timeout=590)
        except RuntimeError:
            result = asyncio.run(_run_plan_review_async(ctx, plan, goal, files_to_touch))
        return result
    except concurrent.futures.TimeoutError:
        return "ERROR: Plan review timed out after 590s."
    except Exception as e:
        log.error("plan_task failed: %s", e, exc_info=True)
        return f"ERROR: Plan review failed: {e}"


# ------------------------------------------------------------------ #
# Async orchestration
# ------------------------------------------------------------------ #

async def _run_plan_review_async(
    ctx: ToolContext,
    plan: str,
    goal: str,
    files_to_touch: list,
) -> str:
    repo_dir = ctx.repo_dir

    # --- Load review models (same 3 as commit triad) ---
    models = _get_review_models()
    if len(models) < 1:
        return "ERROR: No review models configured. Set OUROBOROS_REVIEW_MODELS in settings."

    # --- Build prompt components ---
    checklist = _load_plan_checklist()
    bible_text = _load_bible(repo_dir)
    dev_md = _load_doc(repo_dir, "docs/DEVELOPMENT.md")
    arch_md = _load_doc(repo_dir, "docs/ARCHITECTURE.md")

    # Full repo pack (same as scope review — reviewers see everything)
    ctx.emit_progress_fn("📐 plan_task: building full repo pack…")
    try:
        repo_pack, omitted = build_full_repo_pack(repo_dir, exclude_paths=set(files_to_touch))
    except Exception as e:
        return f"ERROR: Failed to build repo pack: {e}"

    omitted_note = ""
    if omitted:
        omitted_note = f"\n\n## OMITTED FILES\n" + "\n".join(f"- {p}" for p in omitted)

    # HEAD snapshots for files the agent plans to touch
    ctx.emit_progress_fn(f"📐 plan_task: reading {len(files_to_touch)} planned-touch file(s)…")
    head_snapshots = ""
    if files_to_touch:
        head_snapshots = build_head_snapshot_section(repo_dir, files_to_touch)

    # Assemble the full prompt
    system_prompt = _build_system_prompt(checklist, bible_text, dev_md, arch_md)
    user_content = _build_user_content(plan, goal, files_to_touch, head_snapshots, repo_pack, omitted_note)

    # Budget gate
    estimated_tokens = estimate_tokens(system_prompt + user_content)
    if estimated_tokens > _PLAN_BUDGET_TOKEN_LIMIT:
        return (
            f"⚠️ PLAN_REVIEW_SKIPPED: assembled prompt too large "
            f"({estimated_tokens:,} estimated tokens, limit {_PLAN_BUDGET_TOKEN_LIMIT:,}). "
            f"Consider reducing files_to_touch or splitting the plan into smaller scopes."
        )

    ctx.emit_progress_fn(
        f"📐 plan_task: running {len(models)} parallel reviewers "
        f"(~{estimated_tokens:,} tokens each)…"
    )

    # Run all models in parallel
    llm_client = LLMClient()
    semaphore = asyncio.Semaphore(3)
    tasks = [
        _query_reviewer(llm_client, model, system_prompt, user_content, semaphore)
        for model in models
    ]
    raw_results = await asyncio.gather(*tasks)

    # Format output
    return _format_output(raw_results, models, goal, estimated_tokens)


# ------------------------------------------------------------------ #
# Single-reviewer query
# ------------------------------------------------------------------ #

async def _query_reviewer(
    llm_client: LLMClient,
    model: str,
    system_prompt: str,
    user_content: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        try:
            msg, usage = await llm_client.chat_async(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                reasoning_effort=_PLAN_REVIEW_EFFORT,
                max_tokens=_PLAN_REVIEW_MAX_TOKENS,
                temperature=0.2,
            )
            content = msg.get("content") or "(empty response)"
            resolved_model = str((usage or {}).get("resolved_model") or model)
            prompt_tokens = (usage or {}).get("prompt_tokens", 0)
            completion_tokens = (usage or {}).get("completion_tokens", 0)
            return {
                "model": resolved_model,
                "request_model": model,
                "text": content,
                "error": None,
                "tokens_in": prompt_tokens,
                "tokens_out": completion_tokens,
            }
        except asyncio.TimeoutError:
            return {
                "model": model, "request_model": model,
                "text": "", "error": "Timeout after 120s",
                "tokens_in": 0, "tokens_out": 0,
            }
        except Exception as e:
            # Preserve full exception text — truncation here loses diagnostic provenance.
            return {
                "model": model, "request_model": model,
                "text": "", "error": str(e),
                "tokens_in": 0, "tokens_out": 0,
            }


# ------------------------------------------------------------------ #
# Output formatting
# ------------------------------------------------------------------ #

def _format_output(raw_results: list, models: list, goal: str, estimated_tokens: int) -> str:
    lines = [
        "## Plan Review Results",
        "",
        f"**Goal:** {goal}",
        f"**Models:** {len(models)} parallel reviewers",
        f"**Prompt size:** ~{estimated_tokens:,} tokens per reviewer",
        "",
        "---",
        "",
    ]

    aggregate_signal = "GREEN"  # GREEN | REVIEW_REQUIRED | REVISE_PLAN

    for i, result in enumerate(raw_results):
        model_label = result.get("model") or result.get("request_model") or f"Model {i+1}"
        lines.append(f"### Reviewer {i+1}: {model_label}")
        lines.append("")

        if result.get("error"):
            lines.append(f"⚠️ **ERROR:** {result['error']}")
            lines.append("")
            # Only downgrade to REVIEW_REQUIRED if we haven't already seen a REVISE_PLAN
            if aggregate_signal == "GREEN":
                aggregate_signal = "REVIEW_REQUIRED"
            continue

        text = result.get("text", "").strip()
        if not text:
            lines.append("⚠️ **ERROR:** Empty response from reviewer.")
            lines.append("")
            if aggregate_signal == "GREEN":
                aggregate_signal = "REVIEW_REQUIRED"
            continue

        lines.append(text)
        lines.append("")

        # Update aggregate signal based on reviewer's explicit AGGREGATE: line.
        # Uses the LAST matching line in the response (in case the reviewer self-corrects
        # or includes an earlier example line before their final verdict).
        # If no AGGREGATE: line is found, downgrade to REVIEW_REQUIRED — absence of an
        # explicit aggregate line is a signal of uncertainty, not confidence.
        reviewer_signal = _parse_aggregate_signal(text)
        if not reviewer_signal:
            # No parseable aggregate line → treat as uncertain, at least REVIEW_REQUIRED
            if aggregate_signal == "GREEN":
                aggregate_signal = "REVIEW_REQUIRED"
        elif reviewer_signal == "REVISE_PLAN":
            aggregate_signal = "REVISE_PLAN"
        elif reviewer_signal == "REVIEW_REQUIRED" and aggregate_signal == "GREEN":
            aggregate_signal = "REVIEW_REQUIRED"

        lines.append("---")
        lines.append("")

    # Aggregate signal block
    signal_emoji = {
        "GREEN": "✅",
        "REVIEW_REQUIRED": "⚠️",
        "REVISE_PLAN": "❌",
    }.get(aggregate_signal, "❓")

    lines.append("## Aggregate Signal")
    lines.append("")
    lines.append(f"{signal_emoji} **{aggregate_signal}**")
    lines.append("")
    if aggregate_signal == "GREEN":
        lines.append("No critical issues found. Proceed with implementation.")
    elif aggregate_signal == "REVIEW_REQUIRED":
        lines.append(
            "At least one reviewer found RISKs or had errors. "
            "Read the findings carefully and decide whether to adjust your plan before coding."
        )
    else:
        lines.append(
            "At least one reviewer found FAILs. "
            "Revise the plan to address the flagged issues before writing any code."
        )

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Prompt construction
# ------------------------------------------------------------------ #

def _build_system_prompt(
    checklist: str,
    bible_text: str,
    dev_md: str,
    arch_md: str,
) -> str:
    parts = [
        "You are a senior design reviewer for Ouroboros, a self-creating AI agent.",
        "Your job is to review a proposed implementation plan BEFORE any code is written.",
        "You have full access to the entire codebase to find issues that the implementer may have missed.",
        "",
        "## Your Output Format",
        "",
        "For each checklist item, provide:",
        "  - **verdict**: PASS | RISK | FAIL",
        "  - **explanation**: 2-5 sentences describing what you found (or why it's fine)",
        "  - **concrete fix** (if RISK or FAIL): exact file, function, or line to address",
        "  - **alternative approaches** (if applicable): 1-2 more elegant solutions to the same problem",
        "",
        "End your review with one of three aggregate verdicts (on its own line):",
        "  - `AGGREGATE: GREEN` — no critical issues, implementer can proceed",
        "  - `AGGREGATE: REVIEW_REQUIRED` — risks found, implementer should consider adjustments",
        "  - `AGGREGATE: REVISE_PLAN` — critical issues found, plan must be revised before coding",
        "",
        "Be specific. Name exact files, functions, constants, or call sites.",
        "Vague concerns without a concrete pointer are advisory at most.",
        "If you see a simpler solution, say so directly — don't just hint.",
        "",
        "---",
        "",
    ]

    if checklist:
        parts += [
            "## Plan Review Checklist",
            "",
            checklist,
            "",
            "---",
            "",
        ]

    if bible_text:
        parts += [
            "## BIBLE.md (Constitution — highest priority)",
            "",
            bible_text,
            "",
            "---",
            "",
        ]

    if dev_md:
        parts += [
            "## DEVELOPMENT.md (Engineering handbook)",
            "",
            dev_md,
            "",
            "---",
            "",
        ]

    if arch_md:
        parts += [
            "## ARCHITECTURE.md (Current system structure)",
            "",
            arch_md,
            "",
            "---",
            "",
        ]

    return "\n".join(parts)


def _build_user_content(
    plan: str,
    goal: str,
    files_to_touch: list,
    head_snapshots: str,
    repo_pack: str,
    omitted_note: str,
) -> str:
    parts = [
        "## Implementation Plan Under Review",
        "",
        f"**Goal:** {goal}",
        "",
        "**Proposed Plan:**",
        plan,
        "",
    ]

    if files_to_touch:
        parts += [
            f"**Files planned to touch:** {', '.join(files_to_touch)}",
            "",
        ]

    if head_snapshots:
        parts += [
            "## Current State of Planned-Touch Files (HEAD)",
            "",
            head_snapshots,
            "",
        ]

    if repo_pack:
        parts += [
            "## Full Repository Code (for cross-module analysis)",
            "",
            repo_pack,
        ]

    if omitted_note:
        parts.append(omitted_note)

    return "\n".join(parts)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _parse_aggregate_signal(text: str) -> str:
    """Extract the aggregate signal from a reviewer's response.

    Parses lines matching ``AGGREGATE: <SIGNAL>`` (case-insensitive, optional
    leading whitespace) and returns the LAST valid match.  Using the last match
    means self-corrections or earlier example lines do not override the final
    verdict the reviewer actually intended.

    Returns one of "GREEN", "REVIEW_REQUIRED", "REVISE_PLAN", or "" if no
    valid aggregate line is found.

    Narrow regex prevents false positives when a reviewer discusses signal
    names in the explanatory body of their response.
    """
    import re
    pattern = re.compile(
        r"^\s*AGGREGATE\s*:\s*(GREEN|REVIEW_REQUIRED|REVISE_PLAN)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = pattern.findall(text)
    if matches:
        return matches[-1].upper()  # use the last match — final reviewer verdict
    return ""


def _get_review_models() -> list[str]:
    """Return exactly 3 reviewer models for the plan review.

    Sources models from ``OUROBOROS_REVIEW_MODELS`` (same as commit triad).
    Normalizes to exactly 3 reviewers in all cases so the docs' promise of
    '3 parallel reviewers' is always honoured:

    - If 3+ models are configured: use the first 3.
    - If 1 or 2 models are configured: pad by repeating the last model.
    - If no models are configured: use the main model repeated 3 times.
    """
    raw = os.environ.get("OUROBOROS_REVIEW_MODELS", "")
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-opus-4.6")
        models = [main]

    # Pad to exactly 3 by repeating the last model if needed
    while len(models) < 3:
        models.append(models[-1])

    return models[:3]  # cap at 3


def _load_plan_checklist() -> str:
    """Load the Plan Review Checklist section from CHECKLISTS.md."""
    try:
        return load_checklist_section("Plan Review Checklist")
    except Exception as e:
        log.warning("Could not load Plan Review Checklist: %s", e)
        return ""


def _load_bible(repo_dir: Path) -> str:
    """Load BIBLE.md.

    Returns file contents on success. On failure, returns an explicit omission
    note so the reviewer knows that the Constitution is missing from context
    rather than silently receiving an empty string.
    """
    p = repo_dir / "BIBLE.md"
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")
        return f"[⚠️ OMISSION: BIBLE.md not found at {p}]"
    except Exception as e:
        return f"[⚠️ OMISSION: BIBLE.md could not be loaded ({p}): {e}]"


def _load_doc(repo_dir: Path, rel_path: str) -> str:
    """Load a documentation file relative to the repo root.

    Returns file contents on success. On failure, returns an explicit omission
    note so the reviewer knows that context is missing rather than silently
    receiving an empty string.
    """
    p = repo_dir / rel_path
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")
        return f"[⚠️ OMISSION: {rel_path} not found at {p}]"
    except Exception as e:
        return f"[⚠️ OMISSION: {rel_path} could not be loaded ({p}): {e}]"
