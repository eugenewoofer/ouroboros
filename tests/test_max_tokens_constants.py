"""Regression tests: verify raised max_tokens / max_turns constants."""


def test_review_query_model_max_tokens():
    """review.py _query_model must use ≥32768 max_tokens."""
    import ast, textwrap
    from pathlib import Path

    src = Path("ouroboros/tools/review.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "max_tokens":
            # Find the one inside _query_model (chat_async call)
            if isinstance(node.value, ast.Constant) and node.value.value == 32768:
                return  # found
    raise AssertionError("Expected max_tokens=32768 in review.py _query_model")


def test_scope_review_max_tokens():
    """scope_review.py _SCOPE_MAX_TOKENS must be ≥65536."""
    from ouroboros.tools.scope_review import _SCOPE_MAX_TOKENS
    assert _SCOPE_MAX_TOKENS >= 65536


def test_reflection_generate_max_tokens():
    """reflection.py generate_reflection must use ≥4096 max_tokens."""
    src = open("ouroboros/reflection.py", encoding="utf-8").read()
    # generate_reflection is the first function with max_tokens in the file
    assert "max_tokens=4096" in src


def test_consciousness_max_tokens():
    """consciousness.py _think must use ≥4096 max_tokens."""
    src = open("ouroboros/consciousness.py", encoding="utf-8").read()
    assert "max_tokens=4096" in src


def test_compaction_max_tokens():
    """context_compaction.py _summarize_round_batch must use ≥16384."""
    src = open("ouroboros/context_compaction.py", encoding="utf-8").read()
    assert "max_tokens=16384" in src


def test_vision_query_default_max_tokens():
    """llm.py vision_query default max_tokens must be ≥4096."""
    src = open("ouroboros/llm.py", encoding="utf-8").read()
    assert "max_tokens: int = 4096" in src


def test_claude_code_edit_sdk_max_turns():
    """Edit and advisory paths must share the same default Claude Code turn budget."""
    import inspect
    from ouroboros.gateways import claude_code as gw

    assert gw.DEFAULT_CLAUDE_CODE_MAX_TURNS == 30
    assert inspect.signature(gw.run_edit).parameters["max_turns"].default == 30
    assert inspect.signature(gw.run_readonly).parameters["max_turns"].default == 30

    shell_src = open("ouroboros/tools/shell.py", encoding="utf-8").read()
    advisory_src = open("ouroboros/tools/claude_advisory_review.py", encoding="utf-8").read()
    assert "DEFAULT_CLAUDE_CODE_MAX_TURNS" in shell_src
    assert "DEFAULT_CLAUDE_CODE_MAX_TURNS" in advisory_src
    assert "max_turns=25" not in shell_src
    assert "max_turns=8" not in advisory_src


def test_claude_code_sdk_only_no_cli_fallback():
    """shell.py must not contain legacy CLI subprocess fallback."""
    src = open("ouroboros/tools/shell.py", encoding="utf-8").read()
    assert "_run_claude_cli" not in src, "CLI fallback function should be gone"
    assert "ensure_claude_cli" not in src, "CLI install function should be gone"
