"""Multi-model review — sends code/text to multiple LLMs for consensus review.

Also contains the unified pre-commit review gate: three models review staged
diffs against docs/CHECKLISTS.md before any repo_commit. Review always runs
before commit; enforcement is configurable between blocking and advisory.

BIBLE.md is automatically injected as constitutional context with top priority.
"""

import os
import json
import asyncio
import logging
import pathlib
from typing import Any, List, Optional

from ouroboros.llm import LLMClient
from ouroboros.pricing import infer_api_key_type, infer_model_category
from ouroboros.utils import utc_now_iso, run_cmd, append_jsonl
from ouroboros import config as _cfg
from ouroboros.tools.registry import ToolEntry, ToolContext

log = logging.getLogger(__name__)

MAX_MODELS = 10
CONCURRENCY_LIMIT = 5

_CONSTITUTIONAL_PREAMBLE = """\
## CONSTITUTIONAL CONTEXT — TOP PRIORITY

You are reviewing code for Ouroboros, a self-creating AI agent. The following
is its Constitution (BIBLE.md). **All recommendations you make MUST comply
with these principles.** If a potential improvement would violate any
principle, do NOT recommend it — flag the conflict instead.

Key principles to watch for:
- **P3 (LLM-First):** All decisions through the LLM. No hardcoded logic
  for behavior selection. If behavior can be described in a prompt, it goes
  in a prompt, not in code.
- **P5 (Minimalism):** Complexity is the enemy. Every line justifies its existence.
- **P0 (Agency):** Ouroboros is not a tool but a becoming personality.
- **P1 (Continuity):** BIBLE.md and identity.md are protected.
- **P2 (Self-Creation):** Fix the class, not the instance.

If you are unsure whether a recommendation complies with the Constitution,
err on the side of NOT recommending it and explain the tension.

---

"""


_CHECKLISTS_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "docs" / "CHECKLISTS.md"

from ouroboros.tools.review_helpers import (
    load_checklist_section as _load_checklist_section_precise,
    build_touched_file_pack,
    build_goal_section,
    CRITICAL_FINDING_CALIBRATION,
)


def _load_bible() -> str:
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent.parent / "BIBLE.md",
        pathlib.Path.cwd() / "BIBLE.md",
        pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", "")) / "BIBLE.md",
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    log.warning("BIBLE.md not found for review context")
    return ""


# ---------------------------------------------------------------------------
# Tool: multi_model_review (agent-callable)
# ---------------------------------------------------------------------------

def get_tools():
    return [
        ToolEntry(
            name="multi_model_review",
            schema={
                "name": "multi_model_review",
                "description": (
                    "Send code or text to multiple LLM models for review/consensus. "
                    "Each model reviews independently. Returns structured verdicts. "
                    "Choose diverse models yourself. Budget is tracked automatically. "
                    "BIBLE.md (Constitution) is automatically included as top-priority context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The code or text to review"},
                        "prompt": {"type": "string", "description": "Review instructions — what to check for."},
                        "models": {
                            "type": "array", "items": {"type": "string"},
                            "description": "OpenRouter model identifiers (e.g. 3 diverse models)",
                        },
                    },
                    "required": ["content", "prompt", "models"],
                },
            },
            handler=_handle_multi_model_review,
        )
    ]


def _handle_multi_model_review(ctx: ToolContext, content: str = "",
                                prompt: str = "", models: list = None) -> str:
    if models is None:
        models = []
    try:
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    _multi_model_review_async(content, prompt, models, ctx),
                ).result()
        except RuntimeError:
            result = asyncio.run(_multi_model_review_async(content, prompt, models, ctx))
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        log.error("Multi-model review failed: %s", e, exc_info=True)
        return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)


async def _query_model(llm_client: LLMClient, model: str, messages: list, semaphore):
    async with semaphore:
        try:
            msg, usage = await llm_client.chat_async(
                messages=messages,
                model=model,
                reasoning_effort="medium",
                max_tokens=32768,
                temperature=0.2,
            )
            payload = {
                "choices": [{"message": {"content": msg.get("content") or ""}}],
                "usage": usage or {},
            }
            return model, payload, None
        except asyncio.TimeoutError:
            return model, "Error: Timeout after 120s", None
        except Exception as e:
            error_msg = str(e)[:200]
            return model, f"Error: {error_msg}", None


async def _multi_model_review_async(content: str, prompt: str,
                                     models: list, ctx: ToolContext):
    if not content:
        return {"error": "content is required"}
    if not prompt:
        return {"error": "prompt is required"}
    if not models:
        return {"error": "models list is required"}
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return {"error": "models must be a list of strings"}
    if len(models) > MAX_MODELS:
        return {"error": f"Too many models ({len(models)}). Maximum is {MAX_MODELS}."}

    bible_text = _load_bible()
    if bible_text:
        system_content = (
            _CONSTITUTIONAL_PREAMBLE
            + "### BIBLE.md (Full Text)\n\n" + bible_text
            + "\n\n---\n\n## REVIEW INSTRUCTIONS\n\n" + prompt
        )
    else:
        log.warning("Proceeding without BIBLE.md — constitutional compliance cannot be guaranteed")
        system_content = (
            _CONSTITUTIONAL_PREAMBLE
            + "(BIBLE.md could not be loaded)\n\n## REVIEW INSTRUCTIONS\n\n" + prompt
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content},
    ]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    llm_client = LLMClient()
    tasks = [_query_model(llm_client, m, messages, semaphore) for m in models]
    results = await asyncio.gather(*tasks)

    review_results = []
    for model, result, headers_dict in results:
        review_result = _parse_model_response(model, result, headers_dict)
        _emit_usage_event(review_result, ctx)
        review_results.append(review_result)

    return {
        "model_count": len(models),
        "constitutional_context": bool(bible_text),
        "results": review_results,
    }


def _parse_model_response(model: str, result, headers_dict) -> dict:
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    resolved_model = str(usage.get("resolved_model") or model)
    provider = str(usage.get("provider") or "openrouter")
    if isinstance(result, str):
        return {
            "model": resolved_model, "request_model": model,
            "provider": provider, "verdict": "ERROR", "text": result,
            "tokens_in": 0, "tokens_out": 0, "cost_estimate": 0.0,
        }
    try:
        choices = result.get("choices", [])
        if not choices:
            text = f"(no choices in response: {json.dumps(result)[:200]})"
            verdict = "ERROR"
        else:
            text = choices[0]["message"]["content"]
            verdict = "UNKNOWN"
            for line in text.split("\n")[:3]:
                line_upper = line.upper()
                if "PASS" in line_upper:
                    verdict = "PASS"
                    break
                elif "CONCERNS" in line_upper:
                    verdict = "CONCERNS"
                    break
                elif "FAIL" in line_upper:
                    verdict = "FAIL"
                    break
    except (KeyError, IndexError, TypeError):
        text = f"(unexpected response format: {json.dumps(result)[:200]})"
        verdict = "ERROR"

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    cache_write_tokens = usage.get("cache_write_tokens", 0)

    cost = 0.0
    try:
        if "cost" in usage:
            cost = float(usage["cost"])
        elif "total_cost" in usage:
            cost = float(usage["total_cost"])
        elif headers_dict:
            for key, value in headers_dict.items():
                if key.lower() == "x-openrouter-cost":
                    cost = float(value)
                    break
    except (ValueError, TypeError, KeyError):
        pass

    return {
        "model": resolved_model, "request_model": model,
        "provider": provider, "verdict": verdict, "text": text,
        "tokens_in": prompt_tokens, "tokens_out": completion_tokens,
        "cached_tokens": cached_tokens, "cache_write_tokens": cache_write_tokens,
        "cost_estimate": cost,
    }


def _emit_usage_event(review_result: dict, ctx: ToolContext) -> None:
    if ctx is None:
        return
    usage_event = {
        "type": "llm_usage", "ts": utc_now_iso(),
        "task_id": ctx.task_id if ctx.task_id else "",
        "model": review_result.get("model", ""),
        "api_key_type": infer_api_key_type(
            review_result.get("model", ""),
            review_result.get("provider", ""),
        ),
        "model_category": infer_model_category(review_result.get("model", "")),
        "usage": {
            "prompt_tokens": review_result["tokens_in"],
            "completion_tokens": review_result["tokens_out"],
            "cached_tokens": review_result.get("cached_tokens", 0),
            "cache_write_tokens": review_result.get("cache_write_tokens", 0),
            "cost": review_result["cost_estimate"],
        },
        "provider": review_result.get("provider", "openrouter"),
        "source": "review",
        "category": "review",
    }
    if ctx.event_queue is not None:
        try:
            ctx.event_queue.put_nowait(usage_event)
        except Exception:
            if hasattr(ctx, "pending_events"):
                ctx.pending_events.append(usage_event)
    elif hasattr(ctx, "pending_events"):
        ctx.pending_events.append(usage_event)


# ---------------------------------------------------------------------------
# Unified pre-commit review gate — used by git.py commit tools
# ---------------------------------------------------------------------------

def _load_checklist_section() -> str:
    """Load the Repo Commit Checklist from docs/CHECKLISTS.md (DRY, Bible P5).

    Raises FileNotFoundError or ValueError if missing or malformed — fail-closed.
    Uses the precise section loader from review_helpers.
    """
    try:
        return _load_checklist_section_precise("Repo Commit Checklist")
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"docs/CHECKLISTS.md not found or malformed: {e}"
        ) from e


_REVIEW_PREAMBLE = (
    "You are a pre-commit reviewer for Ouroboros, a self-modifying AI agent.\n"
    "Its Constitution is BIBLE.md. Its engineering handbook is DEVELOPMENT.md.\n"
)

_REVIEW_PROMPT_TEMPLATE = """\
{preamble}

## Review instructions — READ CAREFULLY

- Read the ENTIRE staged diff carefully, line by line. Do NOT skim.
- Use BOTH the staged diff AND the full current text of every changed file provided below.
  Do NOT review from the diff alone — the full file context is essential for correctness.
- Look for ALL bugs, logic errors, off-by-one mistakes, missing error handling,
  race conditions, resource leaks, and regressions.
- Report ALL problems you find — not just the single most critical one.
  If there are 5 bugs, list all 5.
- Do NOT stop after finding the first issue.
  Do NOT summarize multiple distinct problems into one finding.
  Each distinct problem gets its own entry in the output array.
- PASS reasons may be brief (one sentence). FAIL reasons must be detailed and actionable:
  include the file, the line or symbol, what is wrong, and a concrete suggestion for how to fix it.
- For every FAIL, include a concrete how-to-fix suggestion so the developer knows exactly
  what change is needed.

{critical_calibration}

You must produce a JSON array. Each element has:
- "item"
- "verdict": "PASS" or "FAIL"
- "severity": "critical" or "advisory"
- "reason": for FAIL — specific file/line, what is wrong, how to fix it

{checklist_section}

- Output ONLY a valid JSON array.  No markdown fences, no text outside the JSON.

{goal_section}

## DEVELOPMENT.md

{dev_guide_text}

## Current touched files (full content)

{current_files_section}

## Staged diff

{diff_text}

## Changed files

{changed_files}

{rebuttal_section}{review_history_section}
"""


def _parse_review_json(raw: str) -> Optional[list]:
    """Best-effort extraction of a JSON array from model output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _git_show_staged(repo_dir, path: str) -> str:
    """Return the staged (index) content of *path* via `git show :PATH`.

    Returns empty string on any error (file not staged, git unavailable, etc.).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else ""
    except Exception:
        return ""


def _preflight_check(commit_message: str, staged_files: str,
                     repo_dir) -> Optional[str]:
    """Deterministic pre-review sanity check — catches common mismatches
    before calling expensive LLM reviewers.

    Checks (in order):
      1. VERSION staged but README.md not staged
      2. Commit message references a version but VERSION file not staged
      3. Python code in ouroboros/ or supervisor/ changed but no tests/ files staged
      4. New files added in ouroboros/ or supervisor/ but ARCHITECTURE.md not staged
      5. VERSION staged: all version carriers (pyproject.toml, README badge,
         ARCHITECTURE.md header) in the staged index must match VERSION value
      6. VERSION staged: staged README.md changelog must have a row for the new version
      7. (reserved — not implemented)
      8. conftest.py staged: block if it contains test_ functions (should be in test_*.py)
    """
    import re

    # Parse the staged_files string. We accept two deterministic formats:
    #
    # 1. "name-status"-style (produced by _run_unified_review after conversion):
    #    "A  path/to/file.py"  (status char + 2 spaces + path)
    #    "M  path/to/file.py"
    #
    # 2. Plain filename (produced as fallback or from unit-test callers):
    #    "path/to/file.py"
    #
    # We detect format 1 by checking that:
    #   - The line is at least 4 chars
    #   - Character at index 0 is a letter (git status char: A/M/D/R/C/T/?)
    #   - Characters at index 1 and 2 are spaces ("  ")
    # This avoids the filename-with-space ambiguity of the old raw[2]==' ' check.
    import string as _string
    raw_lines = staged_files.strip().splitlines()
    file_status: list[tuple[str, str]] = []  # (status_char, filepath)
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        # Format 1: "X  path" — status char + exactly two spaces
        if (len(raw) >= 4
                and raw[0] in _string.ascii_uppercase
                and raw[1:3] == "  "):
            status = raw[0].upper()
            path = raw[3:].strip()
            # Handle renames: "R  old -> new"
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            file_status.append((status, path))
        else:
            # Format 2: plain filename — treat as modified
            file_status.append(("M", raw))

    # staged_set: all paths that appear in the diff (used for existence/coupling checks).
    # active_staged: exclude Deleted (D) entries — a deleted file cannot satisfy a
    # "companion file must be present" requirement.
    staged_set = {path for _, path in file_status}
    active_staged = {path for status, path in file_status if status != "D"}
    # Treat both Added (A) and Copied (C) as "new" files for preflight check 4.
    # Renamed (R) files are not new-module additions — the old path disappears.
    new_files = {path for status, path in file_status if status in ("A", "C")}
    msg_lower = commit_message.lower()

    has_version_ref = bool(re.search(r'v?\d+\.\d+\.\d+', commit_message)) or "version" in msg_lower
    # Use active_staged (excludes deleted files) for companion-file presence checks.
    # staged_set includes all paths (for "currently staged" display and couping checks).
    version_staged = "VERSION" in active_staged

    missing = []

    # Check 1: VERSION staged (and not deleted) but README missing
    if version_staged and "README.md" not in active_staged:
        missing.append("README.md (badge + changelog)")

    # Check 2: Version reference in message but VERSION not staged
    if has_version_ref and not version_staged:
        if any(f.endswith(('.py', '.md')) and f != 'VERSION' for f in active_staged):
            missing.append("VERSION")

    if missing:
        return (
            f"⚠️ PREFLIGHT_BLOCKED: Staged diff is incomplete — fix before review.\n"
            f"  Missing from staged: {', '.join(missing)}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}\n\n"
            "Stage all related files together. Use repo_write for all files first,\n"
            "then repo_commit to stage and commit everything in one diff."
        )

    # Check 3: Python logic touched (added, modified, or deleted) in ouroboros/ or
    # supervisor/ but no tests/ files are staged (active, non-deleted).
    # We include deleted .py files because deleting a module is a behaviour change
    # that must be reflected in tests (e.g. removing a call site or deleting
    # a test that covered the deleted module).
    _LOGIC_DIRS = ("ouroboros/", "supervisor/")
    logic_changed = any(
        f.startswith(_LOGIC_DIRS) and f.endswith(".py")
        for f in staged_set  # all statuses including D
    )
    tests_staged = any(f.startswith("tests/") for f in active_staged)
    if logic_changed and not tests_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: Python logic changed in ouroboros/ or supervisor/ "
            "but no tests/ files are staged.\n"
            "  Add or update tests to cover the changed behaviour, then re-stage.\n"
            "  If this is a docs/config-only change that triggered a false positive, "
            "check that no .py files from ouroboros/ or supervisor/ are in your staged set.\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # Check 4: New files added/copied in ouroboros/ or supervisor/ but
    # ARCHITECTURE.md is not in active_staged (must not be deleted).
    new_logic_files = [
        f for f in new_files
        if f.startswith(_LOGIC_DIRS) and f.endswith(".py")
    ]
    if new_logic_files and "docs/ARCHITECTURE.md" not in active_staged:
        return (
            "⚠️ PREFLIGHT_BLOCKED: New files added in ouroboros/ or supervisor/ "
            "but docs/ARCHITECTURE.md is not staged.\n"
            "  New structural additions must be documented in ARCHITECTURE.md "
            "(Bible P4: authenticity / architectural mirror).\n"
            f"  New files: {new_logic_files[:5]}\n"
            f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
        )

    # Check 5: If VERSION is staged (non-deleted), verify that pyproject.toml,
    # README.md badge, and ARCHITECTURE.md header in the staged index all carry
    # the same version string. Uses `git show :PATH` to read staged content
    # rather than the worktree, so partially staged changes are handled correctly.
    if version_staged:
        try:
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if version_str and re.match(r'^\d+\.\d+\.\d+$', version_str):
                desync = []
                pyproject_text = _git_show_staged(repo_dir, "pyproject.toml")
                if pyproject_text and f'version = "{version_str}"' not in pyproject_text:
                    desync.append(f"pyproject.toml (expected version = \"{version_str}\")")
                readme_text = _git_show_staged(repo_dir, "README.md")
                if readme_text and f"version-{version_str}-" not in readme_text:
                    desync.append(f"README.md badge (expected version-{version_str}-)")
                arch_text = _git_show_staged(repo_dir, "docs/ARCHITECTURE.md")
                if arch_text and f"# Ouroboros v{version_str}" not in arch_text:
                    desync.append(f"docs/ARCHITECTURE.md header (expected # Ouroboros v{version_str})")
                if desync:
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION file says {version_str} but "
                        "the following staged files have a different version value:\n"
                        + "".join(f"  - {d}\n" for d in desync)
                        + "Update all version references to match VERSION before committing.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal: LLM reviewers handle version sync

    # Check 6: If VERSION is staged, verify the staged README.md changelog
    # contains a row for the new version (structural presence check only).
    if version_staged:
        try:
            version_str = _git_show_staged(repo_dir, "VERSION").strip()
            if version_str and re.match(r'^\d+\.\d+\.\d+$', version_str):
                readme_text = _git_show_staged(repo_dir, "README.md")
                if readme_text and not re.search(r'\|\s*' + re.escape(version_str) + r'\s*\|', readme_text):
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: VERSION is {version_str} but README.md "
                        "changelog has no table row for this version.\n"
                        "  Add a changelog entry in the Version History table in README.md.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
        except Exception:
            pass  # Non-fatal

    # Check 7 (not implemented — advisory check reserved for future use)

    # Check 8: if any conftest.py in active_staged contains collectable test functions,
    # block with an explicit message to move them to test_*.py files.
    # Reads staged content via git show to validate what will actually be committed.
    conftest_files = [f for f in active_staged if pathlib.Path(f).name == "conftest.py"]
    if conftest_files:
        import ast as _ast
        for cf in conftest_files:
            try:
                cf_text = _git_show_staged(repo_dir, cf)
                if not cf_text:
                    continue
                tree = _ast.parse(cf_text, filename=cf)
                # Only scan module-level functions — nested helpers inside fixtures
                # are not collected by pytest and must not trigger this check.
                test_fns = [
                    node.name for node in tree.body
                    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                    and node.name.startswith("test_")
                ]
                if test_fns:
                    shown = test_fns[:5]
                    omission = f" (⚠️ showing first 5 of {len(test_fns)})" if len(test_fns) > 5 else ""
                    return (
                        f"⚠️ PREFLIGHT_BLOCKED: {cf} contains test functions: "
                        f"{shown}{omission}.\n"
                        "  conftest.py is for fixtures/hooks only. Move test_ functions "
                        "to a test_*.py file so pytest can discover them properly.\n"
                        f"  Currently staged: {', '.join(sorted(staged_set)) or '(none)'}"
                    )
            except Exception:
                pass  # Non-fatal: AST parse failure or git error, skip this file

    return None


def _build_review_history_section(history: list) -> str:
    if not history:
        return ""
    lines = ["## Previous review rounds\n"]
    for entry in history:
        lines.append(f"### Round {entry['attempt']}")
        lines.append(f"Commit message: \"{entry['commit_message']}\"")
        if entry.get("critical"):
            lines.append("CRITICAL findings:")
            for f in entry["critical"]:
                lines.append(f"- {f}")
        if entry.get("advisory"):
            lines.append("Advisory findings:")
            for f in entry["advisory"]:
                lines.append(f"- {f}")
        lines.append("")
    lines.append(
        "IMPORTANT: Focus on verifying whether previous CRITICAL findings "
        "were addressed. Do NOT rephrase previous findings as new ones. "
        "If a previous CRITICAL was fixed, verdict it PASS.\n"
    )
    return "\n".join(lines)


def _single_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _review_entry(
    *,
    severity: str,
    item: str,
    reason: str,
    model: str = "",
    tag: str = "triad",
    verdict: str = "FAIL",
) -> dict:
    entry = {
        "severity": severity,
        "item": item,
        "reason": reason,
        "tag": tag,
        "verdict": verdict,
    }
    if model:
        entry["model"] = model
    return entry


def _format_review_entry(entry: Any, *, default_severity: str = "advisory") -> str:
    if isinstance(entry, dict):
        severity = str(entry.get("severity", default_severity) or default_severity).upper()
        tags = []
        if entry.get("tag"):
            tags.append(str(entry.get("tag")))
        if entry.get("model"):
            tags.append(f"model={entry.get('model')}")
        if entry.get("obligation_id"):
            tags.append(f"obligation={entry.get('obligation_id')}")
        label = str(entry.get("item") or entry.get("reason") or "?")
        reason = _single_line(str(entry.get("reason", "") or ""))
        tag_prefix = " ".join(f"[{tag}]" for tag in tags)
        return f"[{severity}] {tag_prefix} {label}: {reason}".strip()
    return _single_line(str(entry))


def _append_review_warning(ctx: ToolContext, text: Any) -> None:
    if isinstance(text, dict):
        ctx._review_advisory.append(text)
        return
    warning = _single_line(str(text))
    if warning:
        ctx._review_advisory.append(warning)


def _handle_review_block_or_warning(
    ctx: ToolContext,
    blocking_review: bool,
    blocked_msg: str,
    advisory_prefix: str,
) -> Optional[str]:
    """Either block immediately or downgrade to advisory warning."""
    if blocking_review:
        return blocked_msg
    _append_review_warning(ctx, advisory_prefix + blocked_msg)
    ctx._review_iteration_count = 0
    ctx._review_history = []
    return None


def _build_rebuttal_section(review_rebuttal: str) -> str:
    if not review_rebuttal:
        return ""
    return (
        "\n## Developer's rebuttal to previous review feedback\n\n"
        f"{review_rebuttal}\n\n"
        "Reconsider previous FAIL verdict(s) in light of this argument. "
        "If the argument is valid, change your verdict to PASS. "
        "If not, maintain FAIL and explain why.\n"
    )


def _load_dev_guide_text(repo_dir: pathlib.Path) -> str:
    dev_guide_path = repo_dir / "docs" / "DEVELOPMENT.md"
    try:
        if dev_guide_path.exists():
            return dev_guide_path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _collect_review_findings(ctx: ToolContext, model_results: list) -> tuple[list[str], list[str], list[str]]:
    critical_fails: List[str] = []
    advisory_warns: List[str] = []
    errored_models: List[str] = []
    # Structured critical findings for obligation tracking (list of dicts)
    structured_critical: List[dict] = []
    structured_advisory: List[dict] = []

    for mr in model_results:
        model_name = mr.get("model", "?")
        raw_text = str(mr.get("text", ""))
        verdict_upper = str(mr.get("verdict", "")).upper()

        if verdict_upper == "ERROR":
            errored_models.append(model_name)
            advisory_warns.append(
                f"[{model_name}] Model unavailable this round: {raw_text[:200]}"
            )
            structured_advisory.append(_review_entry(
                severity="advisory",
                item="review_model_unavailable",
                reason=f"Model unavailable this round: {raw_text[:200]}",
                model=model_name,
            ))
            try:
                append_jsonl(ctx.drive_logs() / "events.jsonl", {
                    "ts": utc_now_iso(), "type": "review_model_error",
                    "model": model_name, "error_preview": raw_text[:200],
                })
            except Exception:
                pass
            continue

        items = _parse_review_json(raw_text)
        if items is None:
            critical_fails.append(
                f"[{model_name}] Could not parse structured review output. "
                f"Raw preview: {raw_text[:300]}"
            )
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            item_verdict = str(item.get("verdict", "")).upper()
            severity = str(item.get("severity", "advisory")).lower()
            item_name = item.get("item", "?")
            reason = item.get("reason", "")
            if item_verdict != "FAIL":
                continue
            desc = f"[{model_name}] {item_name}: {reason}"
            if severity == "critical":
                critical_fails.append(desc)
                structured_critical.append(_review_entry(
                    severity="critical",
                    item=str(item_name),
                    reason=str(reason),
                    model=model_name,
                ))
            else:
                advisory_warns.append(desc)
                structured_advisory.append(_review_entry(
                    severity="advisory",
                    item=str(item_name),
                    reason=str(reason),
                    model=model_name,
                ))

    # Store structured findings on ctx for obligation tracking
    ctx._last_review_critical_findings = structured_critical
    ctx._last_review_advisory_findings = structured_advisory

    return critical_fails, advisory_warns, errored_models


def _build_critical_block_message(
    ctx: ToolContext,
    commit_message: str,
    critical_fails: List[str],
    advisory_warns: List[str],
    errored_note: str,
) -> str:
    ctx._review_history.append({
        "attempt": ctx._review_iteration_count,
        "commit_message": commit_message,  # full — no [:200] truncation
        "critical": list(critical_fails),
        "advisory": list(advisory_warns),
    })

    iteration_note = f" (attempt {ctx._review_iteration_count})"

    # Structured self-verification template — appears from attempt 2 onwards.
    # Forces the agent to explicitly map each finding to evidence before retrying.
    self_verify_hint = ""
    if ctx._review_iteration_count >= 2:
        all_findings = list(getattr(ctx, '_last_review_critical_findings', []) or []) or list(critical_fails)
        finding_lines = "\n".join(
            f"  - Finding: {f.get('item', '?') if isinstance(f, dict) else f}"
            for f in all_findings
        )
        if not finding_lines:
            finding_lines = "  (no findings captured — check review output above)"
        self_verify_hint = (
            "\n\n⚠️ Self-verification required before next repo_commit:\n"
            "For EACH finding listed above, explicitly state:\n"
            "  Finding: [item name]\n"
            "  Status: addressed / rebutted / pending\n"
            "  Evidence: [file:line or symbol or test name]\n"
            "  Note: [one sentence]\n\n"
            "Do NOT call repo_commit until this table is filled in your response.\n"
            f"Open findings:\n{finding_lines}"
        )

    soft_hint = ""
    if ctx._review_iteration_count >= 5:
        soft_hint = (
            "\n\nHint: You have attempted this commit 5+ times. Consider:\n"
            "- Breaking the change into smaller, independently reviewable commits\n"
            "- If the same critical repeats: implement what the reviewer asks, or split the change, or report the blockage to the user instead of retrying"
        )

    critical_entries = list(getattr(ctx, "_last_review_critical_findings", []) or critical_fails)
    advisory_entries = list(getattr(ctx, "_last_review_advisory_findings", []) or advisory_warns)

    return (
        f"⚠️ REVIEW_BLOCKED{iteration_note}: Critical issues found by reviewers.\n"
        "Commit has NOT been created. Fix the issues and try again. Use review_rebuttal\n"
        "ONLY if a finding is factually incorrect — not to argue against requested tests\n"
        "or artifacts. If the same finding repeats after a rebuttal, implement the fix\n"
        "instead of re-arguing.\n\n"
        + "Critical findings:\n"
        + "\n".join(f"  - {_format_review_entry(f, default_severity='critical')}" for f in critical_entries)
        + (
            "\n\nAdvisory warnings:\n"
            + "\n".join(f"  - {_format_review_entry(w)}" for w in advisory_entries)
            if advisory_entries else ""
        )
        + errored_note
        + self_verify_hint
        + soft_hint
    )


def _run_unified_review(ctx: ToolContext, commit_message: str,
                        review_rebuttal: str = "",
                        repo_dir=None,
                        goal: str = "",
                        scope: str = "") -> Optional[str]:
    """Unified pre-commit review: 3 models, structured JSON, consistent severity.

    Returns None if commit may proceed. In blocking mode returns a blocking
    error string when review rejects the commit.
    """
    target_repo = repo_dir or ctx.repo_dir
    ctx._review_iteration_count += 1
    ctx._last_review_block_reason = ""  # reset per attempt
    ctx._last_review_critical_findings = []  # reset to avoid stale findings from previous attempts
    review_enforcement = _cfg.get_review_enforcement()
    blocking_review = review_enforcement == "blocking"

    try:
        diff_text = run_cmd(["git", "diff", "--cached"], cwd=target_repo)
    except Exception:
        diff_text = "(failed to get staged diff)"

    if not diff_text.strip():
        return None

    try:
        changed = run_cmd(["git", "diff", "--cached", "--name-only"], cwd=target_repo)
    except Exception:
        changed = ""

    # Build a status-bearing string for preflight (uses porcelain "XY path" format).
    # We use --name-status (tab-separated "STATUS\tpath") and convert to
    # a two-char porcelain-like prefix so _preflight_check can detect added files.
    try:
        name_status = run_cmd(
            ["git", "diff", "--cached", "--name-status"], cwd=target_repo
        )
        # Convert git --name-status tab-separated lines to "X  path" format.
        # Formats emitted by git:
        #   "A\tpath"          → added
        #   "M\tpath"          → modified
        #   "D\tpath"          → deleted
        #   "R100\told\tnew"   → renamed (similarity score prefix, two paths)
        #   "C100\told\tnew"   → copied  (similarity score prefix, two paths)
        preflight_input_lines = []
        for line in name_status.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if not parts:
                continue
            status_char = parts[0][0].upper()  # first char of status code (strips similarity %)
            if status_char in ("R", "C") and len(parts) >= 3:
                src_path, dst_path = parts[1], parts[-1]
                if status_char == "R":
                    # Rename: source was deleted, destination is a new file.
                    #   "D src" — triggers check 3 if src was in a guarded dir
                    #   "A dst" — triggers check 3 and check 4 if dst is in a guarded dir
                    preflight_input_lines.append(f"D  {src_path}")
                    preflight_input_lines.append(f"A  {dst_path}")
                else:
                    # Copy (C): source is unchanged — only emit the new destination.
                    # Do NOT emit "D src" here; the source file was NOT deleted or modified.
                    # A copy into a guarded dir still constitutes a new module (check 4).
                    preflight_input_lines.append(f"A  {dst_path}")
            elif len(parts) >= 2:
                path = parts[1]
                preflight_input_lines.append(f"{status_char}  {path}")
            else:
                preflight_input_lines.append(f"M  {parts[0]}")
        preflight_staged = "\n".join(preflight_input_lines) if preflight_input_lines else changed
    except Exception:
        preflight_staged = changed  # fallback to name-only (check 4 may not fire, but checks 1-3 still work)

    preflight_err = _preflight_check(commit_message, preflight_staged, target_repo)
    if preflight_err:
        ctx._last_review_block_reason = "preflight"
        result = _handle_review_block_or_warning(
            ctx, blocking_review, preflight_err,
            "Review enforcement=Advisory: preflight warning did not block commit. ",
        )
        if result is not None:
            return result

    rebuttal_section = _build_rebuttal_section(review_rebuttal)

    try:
        checklist_section = _load_checklist_section()
    except (FileNotFoundError, ValueError) as e:
        log.error("Checklist loading failed (fail-closed): %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Cannot load review checklist — commit cannot proceed.\n"
            f"Error: {e}\n"
            "Ensure docs/CHECKLISTS.md exists and contains the expected section headers."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review checklist failed to load; commit proceeding anyway. ",
        )

    dev_guide_text = _load_dev_guide_text(pathlib.Path(ctx.repo_dir))

    review_history_section = _build_review_history_section(ctx._review_history)

    # Build touched-file pack for full current file context
    try:
        touched_paths = [f.strip() for f in changed.strip().splitlines() if f.strip()]
        current_files_section, _omitted = build_touched_file_pack(
            pathlib.Path(target_repo), touched_paths
        )
        if _omitted:
            current_files_section += (
                f"\n\n⚠️ OMISSION NOTE: {len(_omitted)} file(s) omitted from direct context: "
                f"{', '.join(_omitted)}"
            )
        if not current_files_section.strip():
            current_files_section = "(no touched files could be read)"
    except Exception as e:
        log.warning("Failed to build touched file pack for triad review: %s", e)
        current_files_section = f"(touched file pack unavailable: {e})"

    goal_section = build_goal_section(goal, scope, commit_message)

    prompt = _REVIEW_PROMPT_TEMPLATE.format(
        preamble=_REVIEW_PREAMBLE,
        critical_calibration=CRITICAL_FINDING_CALIBRATION,
        checklist_section=checklist_section,
        goal_section=goal_section,
        dev_guide_text=dev_guide_text or "(DEVELOPMENT.md not found)",
        current_files_section=current_files_section,
        rebuttal_section=rebuttal_section,
        review_history_section=review_history_section,
        diff_text=diff_text,
        changed_files=changed,
    )

    models = _cfg.get_review_models()

    try:
        result_json = _handle_multi_model_review(
            ctx,
            content="Review the staged diff and context provided in the instructions above.",
            prompt=prompt,
            models=models,
        )
        result = json.loads(result_json)
    except Exception as e:
        log.error("Unified review infrastructure failure: %s", e)
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review infrastructure failed — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {e}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review infrastructure failure did not block commit. ",
        )

    if "error" in result:
        log.error("Review returned error: %s", result["error"])
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review service returned an error — commit cannot proceed "
            "without a successful review.\n"
            f"Error: {result['error']}\n"
            "Check OPENROUTER_API_KEY, network connectivity, and retry."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review service error did not block commit. ",
        )

    model_results = result.get("results", [])
    if not model_results:
        ctx._last_review_block_reason = "infra_failure"
        blocked_msg = (
            "⚠️ REVIEW_BLOCKED: Review returned no results from any model — "
            "commit cannot proceed without a successful review."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review returned no model results; commit proceeding anyway. ",
        )

    critical_fails, advisory_warns, errored_models = _collect_review_findings(ctx, model_results)

    models_total = len(model_results)

    # Quorum: at least 2 of N reviewers must succeed
    successful_reviewers = models_total - len(errored_models)
    if successful_reviewers < 2:
        ctx._last_review_block_reason = "review_quorum"
        blocked_msg = (
            f"⚠️ REVIEW_BLOCKED: Only {successful_reviewers} of {models_total} review "
            f"models responded successfully (minimum 2 required). "
            f"Unavailable: {', '.join(errored_models)}.\n"
            "Retry the commit — transient model failures usually resolve quickly."
        )
        return _handle_review_block_or_warning(
            ctx, blocking_review, blocked_msg,
            "Review enforcement=Advisory: review quorum failure did not block commit. ",
        )

    errored_note = ""
    if errored_models:
        errored_note = (
            f"\n\nNote: {len(errored_models)} of {models_total} review models "
            f"were unavailable ({', '.join(errored_models)}). "
            "Target is 3 working reviewers."
        )

    if critical_fails:
        # Classify: if all critical failures are parse issues, mark as parse_failure
        all_parse = all("Could not parse" in f for f in critical_fails)
        ctx._last_review_block_reason = "parse_failure" if all_parse else "critical_findings"
        if blocking_review:
            return _build_critical_block_message(
                ctx, commit_message, critical_fails, advisory_warns, errored_note,
            )

        _append_review_warning(
            ctx,
            "Review enforcement=Advisory: critical review findings did not block commit.",
        )
        for finding in getattr(ctx, "_last_review_critical_findings", []) or []:
            _append_review_warning(ctx, finding)
        for warning in getattr(ctx, "_last_review_advisory_findings", []) or []:
            _append_review_warning(ctx, warning)
        if errored_note:
            _append_review_warning(ctx, errored_note)

    # All clear — reset iteration state
    ctx._review_iteration_count = 0
    ctx._review_history = []

    if errored_note:
        advisory_warns.append(errored_note.strip())
    if advisory_warns or getattr(ctx, "_last_review_advisory_findings", None):
        ctx._review_advisory = list(getattr(ctx, "_last_review_advisory_findings", []) or [])
        if errored_note:
            ctx._review_advisory.append(errored_note.strip())
    return None
