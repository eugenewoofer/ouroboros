"""Tests that _tracked_subprocess_run forces UTF-8 decoding and PYTHONIOENCODING.

Context: on Windows (cp1251 locale), subprocess output containing box-drawing
characters from Vite, checkmarks from npm/git, or emoji from user scripts
crashes the parent with UnicodeEncodeError. The wrapper must:
  1. Force encoding='utf-8', errors='replace' when text=True and no encoding given.
  2. Inject PYTHONIOENCODING=utf-8 into the child env so inline python -c scripts
     can print non-ASCII without crashing their own stdout handlers.
"""
from subprocess import CompletedProcess

import pytest

from ouroboros.tools import shell as shell_mod


class _FakePopen:
    """Minimal Popen stub that records the kwargs it was called with."""
    def __init__(self, cmd, **kwargs):
        self.args = cmd
        self.kwargs = kwargs
        self.returncode = 0
        _FakePopen.last_kwargs = kwargs

    def communicate(self, timeout=None):
        return ("ok", "")

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def captured_popen(monkeypatch):
    """Patch Popen inside shell module and expose the last kwargs."""
    _FakePopen.last_kwargs = None
    monkeypatch.setattr(shell_mod, "Popen", _FakePopen)
    # Prevent the subprocess registry from holding a reference
    monkeypatch.setattr(shell_mod, "_active_subprocesses", set())
    return _FakePopen


def test_text_mode_forces_utf8_encoding(captured_popen):
    """text=True without explicit encoding must be upgraded to utf-8/replace."""
    shell_mod._tracked_subprocess_run(["echo", "hi"], text=True)
    kw = captured_popen.last_kwargs
    assert kw["encoding"] == "utf-8"
    assert kw["errors"] == "replace"


def test_explicit_encoding_is_not_overridden(captured_popen):
    """If caller passes encoding= explicitly, don't override it."""
    shell_mod._tracked_subprocess_run(
        ["echo", "hi"], text=True, encoding="latin-1", errors="strict"
    )
    kw = captured_popen.last_kwargs
    assert kw["encoding"] == "latin-1"
    assert kw["errors"] == "strict"


def test_binary_mode_is_not_modified(captured_popen):
    """If text=False (or not set), don't inject encoding at all."""
    shell_mod._tracked_subprocess_run(["echo", "hi"])
    kw = captured_popen.last_kwargs
    assert "encoding" not in kw
    # errors may still be absent — we don't touch binary mode


def test_pythonioencoding_injected_into_child_env(captured_popen):
    """Child env must contain PYTHONIOENCODING=utf-8 even when no env= passed."""
    shell_mod._tracked_subprocess_run(["python", "-c", "print('ok')"], text=True)
    kw = captured_popen.last_kwargs
    assert "env" in kw
    assert kw["env"].get("PYTHONIOENCODING") == "utf-8"


def test_pythonioencoding_injected_into_custom_env(captured_popen):
    """If caller passes a custom env, PYTHONIOENCODING is still injected."""
    shell_mod._tracked_subprocess_run(
        ["python", "-c", "print('ok')"], text=True, env={"FOO": "bar"}
    )
    kw = captured_popen.last_kwargs
    assert kw["env"].get("PYTHONIOENCODING") == "utf-8"
    assert kw["env"].get("FOO") == "bar"


def test_pythonioencoding_not_overridden_if_caller_sets_it(captured_popen):
    """If caller explicitly sets PYTHONIOENCODING, respect their choice."""
    shell_mod._tracked_subprocess_run(
        ["python", "-c", "print('ok')"],
        text=True,
        env={"PYTHONIOENCODING": "cp1251"},
    )
    kw = captured_popen.last_kwargs
    assert kw["env"].get("PYTHONIOENCODING") == "cp1251"
