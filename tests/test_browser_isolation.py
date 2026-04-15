"""Tests for browser state isolation and infrastructure error detection."""
import pathlib
import sys
import types

import pytest

import ouroboros.tools.browser as browser_mod
from ouroboros.tools.browser import _is_infrastructure_error, cleanup_browser


class TestInfrastructureErrorDetection:
    """_is_infrastructure_error should detect structural Playwright failures."""

    def test_detects_greenlet_switch(self):
        assert _is_infrastructure_error(RuntimeError("cannot switch to a different green thread"))

    def test_detects_different_thread(self):
        assert _is_infrastructure_error(RuntimeError("different thread"))

    def test_detects_browser_closed(self):
        assert _is_infrastructure_error(Exception("browser has been closed"))

    def test_detects_page_closed(self):
        assert _is_infrastructure_error(Exception("page has been closed"))

    def test_detects_connection_closed(self):
        assert _is_infrastructure_error(Exception("Connection closed"))

    def test_ignores_normal_errors(self):
        assert not _is_infrastructure_error(ValueError("invalid selector"))
        assert not _is_infrastructure_error(TimeoutError("navigation timeout"))


class TestBrowserModuleState:
    """Module-level state should be properly initialized."""

    def test_is_infrastructure_error_is_function(self):
        assert callable(_is_infrastructure_error)

    def test_ensure_browser_tolerates_missing_thread_id(self, monkeypatch):
        fake_page = types.SimpleNamespace(set_default_timeout=lambda timeout: None)

        def _new_page(**kwargs):
            return fake_page

        fake_browser = types.SimpleNamespace(
            new_page=_new_page,
            is_connected=lambda: True,
        )
        fake_playwright = types.SimpleNamespace(
            chromium=types.SimpleNamespace(launch=lambda **kwargs: fake_browser)
        )
        fake_sync_api = types.SimpleNamespace(
            sync_playwright=lambda: types.SimpleNamespace(start=lambda: fake_playwright)
        )
        monkeypatch.setattr(browser_mod, "_HAS_STEALTH", False)
        monkeypatch.setattr(browser_mod, "_ensure_playwright_installed", lambda: None)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api)

        ctx = types.SimpleNamespace(
            browser_state=types.SimpleNamespace(
                page=None,
                browser=None,
                pw_instance=None,
                last_screenshot_b64=None,
            )
        )

        page = browser_mod._ensure_browser(ctx)

        assert page is fake_page
        assert getattr(ctx.browser_state, "_thread_id", None) is not None

    def test_aliases_arm64_browser_cache_for_missing_x64_binary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(browser_mod.sys, "platform", "darwin", raising=False)
        root = tmp_path / "playwright" / "chromium_headless_shell-1208"
        arm_dir = root / "chrome-headless-shell-mac-arm64"
        arm_dir.mkdir(parents=True)
        arm_binary = arm_dir / "chrome-headless-shell"
        arm_binary.write_text("stub", encoding="utf-8")

        missing_binary = root / "chrome-headless-shell-mac-x64" / "chrome-headless-shell"
        err = RuntimeError(f"BrowserType.launch: Executable doesn't exist at {missing_binary}")

        assert browser_mod._maybe_alias_playwright_binary(err) is True
        alias_dir = missing_binary.parent
        assert alias_dir.is_symlink()
        assert pathlib.Path(alias_dir.resolve()) == arm_dir.resolve()


class TestCleanupBrowser:
    """cleanup_browser should null out all browser_state references."""

    def test_cleanup_nulls_state(self):
        ctx = types.SimpleNamespace(
            browser_state=types.SimpleNamespace(
                page=None,
                browser=None,
                pw_instance=None,
                last_screenshot_b64=None,
            )
        )
        cleanup_browser(ctx)
        assert ctx.browser_state.page is None
        assert ctx.browser_state.browser is None
        assert ctx.browser_state.pw_instance is None
