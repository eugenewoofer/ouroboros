"""Behavioral tests for Phase 7: modern commit pipeline, operational resilience.

Tests:
- repo_write single-file and multi-file modes
- repo_write + repo_commit workflow
- Unified pre-commit review gate (preflight, parse, quorum)
- Blocked review leaves files on disk but unstaged
- review_rebuttal parameter
- configure_remote failure surfacing
- migrate_remote_credentials no-op on clean origin
- Auto-rescue only reports committed when commit actually happened
- repo_write in CORE_TOOL_NAMES
- Review history building
"""
import importlib
import inspect
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _get_git_module():
    return importlib.import_module("ouroboros.tools.git")


def _get_review_module():
    return importlib.import_module("ouroboros.tools.review")


def _get_registry_module():
    return importlib.import_module("ouroboros.tools.registry")


def _get_git_ops_module():
    return importlib.import_module("supervisor.git_ops")


def _make_ctx(tmp_path):
    """Create a minimal ToolContext with a temporary git repo."""
    from ouroboros.tools.registry import ToolContext
    repo = tmp_path / "repo"
    repo.mkdir()
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "logs").mkdir(parents=True)
    (drive / "locks").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), capture_output=True)
    (repo / "dummy.txt").write_text("init", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "branch", "-M", "ouroboros"], cwd=str(repo), capture_output=True)
    return ToolContext(repo_dir=repo, drive_root=drive)


# --- repo_write tool registration ---

class TestRepoWriteRegistration:
    def test_repo_write_registered(self):
        git_mod = _get_git_module()
        names = [t.name for t in git_mod.get_tools()]
        assert "repo_write" in names

    def test_repo_write_in_core_tool_names(self):
        registry = _get_registry_module()
        assert "repo_write" in registry.CORE_TOOL_NAMES

    def test_repo_write_commit_still_registered(self):
        git_mod = _get_git_module()
        names = [t.name for t in git_mod.get_tools()]
        assert "repo_write_commit" in names

    def test_repo_write_schema_has_files_param(self):
        git_mod = _get_git_module()
        tools = git_mod.get_tools()
        rw = next(t for t in tools if t.name == "repo_write")
        props = rw.schema["parameters"]["properties"]
        assert "files" in props
        assert props["files"]["type"] == "array"

    def test_repo_commit_has_review_rebuttal(self):
        git_mod = _get_git_module()
        tools = git_mod.get_tools()
        rc = next(t for t in tools if t.name == "repo_commit")
        props = rc.schema["parameters"]["properties"]
        assert "review_rebuttal" in props


# --- repo_write behavioral tests ---

class TestRepoWriteSingleFile:
    def test_single_file_write(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, path="hello.py", content="print('hello')")
        assert "Written 1 file" in result
        assert "NOT committed" in result
        assert (ctx.repo_dir / "hello.py").read_text() == "print('hello')"

    def test_single_file_creates_directories(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, path="deep/nested/file.py", content="x = 1")
        assert "Written 1 file" in result
        assert (ctx.repo_dir / "deep" / "nested" / "file.py").exists()

    def test_rejects_empty_args(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx)
        assert "WRITE_ERROR" in result

    def test_rejects_compaction_marker(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, path="x.py", content="<<CONTENT_OMITTED something")
        assert "WRITE_ERROR" in result
        assert "compaction marker" in result


class TestRepoWriteMultiFile:
    def test_multi_file_write(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, files=[
            {"path": "a.py", "content": "# a"},
            {"path": "b.py", "content": "# b"},
        ])
        assert "Written 2 file" in result
        assert (ctx.repo_dir / "a.py").read_text() == "# a"
        assert (ctx.repo_dir / "b.py").read_text() == "# b"

    def test_multi_file_rejects_empty_path(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, files=[{"path": "", "content": "x"}])
        assert "WRITE_ERROR" in result

    def test_multi_file_blocks_safety_critical(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(ctx, files=[
            {"path": "ok.py", "content": "x"},
            {"path": "BIBLE.md", "content": "hacked"},
        ])
        assert "SAFETY_VIOLATION" in result

    def test_files_param_takes_priority(self, tmp_path):
        git_mod = _get_git_module()
        ctx = _make_ctx(tmp_path)
        result = git_mod._repo_write(
            ctx, path="ignored.py", content="ignored",
            files=[{"path": "used.py", "content": "used"}],
        )
        assert "Written 1 file" in result
        assert (ctx.repo_dir / "used.py").exists()
        assert not (ctx.repo_dir / "ignored.py").exists()


# --- Unified review gate ---

class TestPreflightCheck:
    def test_missing_version(self):
        review = _get_review_module()
        # Plain filenames without porcelain prefix also work (fallback to "M")
        result = review._preflight_check(
            "v3.24.0: big change",
            "ouroboros/tools/git.py\nREADME.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "VERSION" in result

    def test_missing_readme(self):
        review = _get_review_module()
        # VERSION + ouroboros .py → needs README (check 1) and tests (check 3)
        # Check 1 fires first, so we get README error
        result = review._preflight_check(
            "some change",
            "M  VERSION\nM  ouroboros/tools/git.py",
            "/tmp",
        )
        assert result is not None
        assert "README.md" in result

    def test_all_present_passes(self):
        # git.py is an ouroboros/ .py change so tests/ must also be present
        review = _get_review_module()
        result = review._preflight_check(
            "v3.24.0: change",
            "M  VERSION\nM  README.md\nM  ouroboros/tools/git.py\nM  tests/test_commit_gate.py",
            "/tmp",
        )
        assert result is None

    def test_no_version_ref_passes(self):
        review = _get_review_module()
        result = review._preflight_check(
            "fix typo in docs",
            "M  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    # --- New preflight check 3: tests_affected ---

    def test_logic_changed_without_tests_blocked(self):
        """Python code in ouroboros/ changed but no tests/ staged → blocked."""
        review = _get_review_module()
        result = review._preflight_check(
            "fix something",
            "M  ouroboros/tools/shell.py\nM  VERSION\nM  README.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "tests/" in result

    def test_logic_changed_with_tests_passes(self):
        """Python code in ouroboros/ AND tests/ staged → passes."""
        review = _get_review_module()
        result = review._preflight_check(
            "fix something",
            "M  ouroboros/tools/shell.py\nM  tests/test_shell_recovery.py\nM  VERSION\nM  README.md",
            "/tmp",
        )
        assert result is None

    def test_supervisor_logic_without_tests_blocked(self):
        """Python code in supervisor/ changed but no tests/ staged → blocked."""
        review = _get_review_module()
        result = review._preflight_check(
            "update supervisor",
            "M  supervisor/workers.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result

    def test_docs_only_change_no_tests_required(self):
        """Docs-only change (no .py in ouroboros/) should not require tests."""
        review = _get_review_module()
        result = review._preflight_check(
            "update docs",
            "M  docs/ARCHITECTURE.md\nM  README.md",
            "/tmp",
        )
        assert result is None

    # --- New preflight check 4: architecture_doc ---

    def test_new_module_without_architecture_blocked(self):
        """New .py file added in ouroboros/ but ARCHITECTURE.md not staged → blocked."""
        review = _get_review_module()
        # Porcelain format with "A " prefix indicates a new (added) file
        result = review._preflight_check(
            "add new module",
            "A  ouroboros/new_module.py\nM  tests/test_new_module.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_new_module_with_architecture_passes(self):
        """New .py file added AND ARCHITECTURE.md staged → passes."""
        review = _get_review_module()
        result = review._preflight_check(
            "add new module",
            "A  ouroboros/new_module.py\nM  tests/test_new_module.py\nM  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    def test_modified_module_without_architecture_passes(self):
        """Modified (not new) .py file without ARCHITECTURE.md → passes (check 4 not triggered)."""
        review = _get_review_module()
        result = review._preflight_check(
            "update existing module",
            "M  ouroboros/tools/shell.py\nM  tests/test_shell_recovery.py",
            "/tmp",
        )
        assert result is None


class TestParseReviewJson:
    def test_plain_json(self):
        review = _get_review_module()
        data = '[{"item":"x","verdict":"PASS","severity":"critical","reason":"ok"}]'
        result = review._parse_review_json(data)
        assert result is not None
        assert len(result) == 1

    def test_markdown_fenced(self):
        review = _get_review_module()
        data = '```json\n[{"item":"x","verdict":"FAIL","severity":"advisory","reason":"bad"}]\n```'
        result = review._parse_review_json(data)
        assert result is not None
        assert result[0]["verdict"] == "FAIL"

    def test_text_around_json(self):
        review = _get_review_module()
        data = 'Here is my review:\n[{"item":"x","verdict":"PASS","severity":"critical","reason":"ok"}]\nDone.'
        result = review._parse_review_json(data)
        assert result is not None

    def test_invalid_json(self):
        review = _get_review_module()
        result = review._parse_review_json("not json at all")
        assert result is None


class TestReviewHistoryBuilding:
    def test_empty_history(self):
        review = _get_review_module()
        result = review._build_review_history_section([])
        assert result == ""

    def test_history_with_entries(self):
        review = _get_review_module()
        history = [{
            "attempt": 1,
            "commit_message": "test commit",
            "critical": ["[model] item: reason"],
            "advisory": [],
        }]
        result = review._build_review_history_section(history)
        assert "Round 1" in result
        assert "test commit" in result
        assert "CRITICAL" in result


class TestReviewQuorumLogic:
    def test_review_models_configured(self):
        from ouroboros.config import get_review_models
        models = get_review_models()
        assert len(models) >= 2  # config.py is single source of truth

    def test_checklist_path_exists(self):
        review = _get_review_module()
        assert review._CHECKLISTS_PATH.exists()

    def test_load_checklist_succeeds(self):
        review = _get_review_module()
        section = review._load_checklist_section()
        assert "bible_compliance" in section
        assert "code_quality" in section


class TestReviewEnforcementModes:
    @staticmethod
    def _fake_result(*review_texts):
        return json.dumps({
            "results": [
                {
                    "model": f"model-{idx}",
                    "verdict": "PASS",
                    "text": text,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_estimate": 0.0,
                }
                for idx, text in enumerate(review_texts, start=1)
            ]
        })

    @staticmethod
    def _mock_staged(monkeypatch, review_mod, changed_files="x.py", diff_text="diff --cached",
                     name_status_files=None):
        """Mock git commands for _run_unified_review.

        name_status_files: if provided, used as the --name-status output.
        Defaults to converting changed_files lines to "M  path" format.
        """
        if name_status_files is None:
            # Convert plain filenames to M\tpath format (what git --name-status emits)
            name_status_files = "\n".join(
                f"M\t{f.strip()}" for f in changed_files.splitlines() if f.strip()
            )

        def _fake_run_cmd(cmd, cwd=None):
            cmd = list(cmd)
            if cmd[:5] == ["git", "diff", "--cached", "--name-status"]:
                return name_status_files
            if cmd[:4] == ["git", "diff", "--cached", "--name-only"]:
                return changed_files
            if cmd[:3] == ["git", "diff", "--cached"]:
                return diff_text
            return ""
        monkeypatch.setattr(review_mod, "run_cmd", _fake_run_cmd)

    def test_blocking_mode_blocks_critical_findings(self, tmp_path, monkeypatch):
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"code_quality","verdict":"FAIL","severity":"critical","reason":"broken"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is not None
        assert "REVIEW_BLOCKED" in result

    def test_advisory_mode_downgrades_critical_findings(self, tmp_path, monkeypatch):
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"code_quality","verdict":"FAIL","severity":"critical","reason":"broken"}]',
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            isinstance(w, str) and "critical review findings did not block commit" in w.lower()
            for w in ctx._review_advisory
        )
        assert any(
            (isinstance(w, dict) and w.get("reason") == "broken")
            or (isinstance(w, str) and "broken" in w)
            for w in ctx._review_advisory
        )
        assert ctx._review_iteration_count == 0

    def test_advisory_mode_downgrades_quorum_failure(self, tmp_path, monkeypatch):
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        self._mock_staged(monkeypatch, review, changed_files="x.py")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                "Error: timeout",
                '[{"item":"code_quality","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "test commit", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            "only 1 of 2 review models responded successfully" in w.lower()
            or "review enforcement=advisory" in w.lower()
            for w in ctx._review_advisory
        )

    def test_advisory_mode_keeps_preflight_as_warning(self, tmp_path, monkeypatch):
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        self._mock_staged(monkeypatch, review, changed_files="VERSION")
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
        monkeypatch.setattr(
            review,
            "_handle_multi_model_review",
            lambda *args, **kwargs: self._fake_result(
                '[{"item":"version_bump","verdict":"PASS","severity":"critical","reason":"ok"}]',
                '[{"item":"readme_changelog","verdict":"PASS","severity":"critical","reason":"ok"}]',
            ),
        )
        result = review._run_unified_review(ctx, "version update", repo_dir=ctx.repo_dir)
        assert result is None
        assert any(
            isinstance(w, str) and "preflight warning did not block commit" in w.lower()
            for w in ctx._review_advisory
        )

    def test_new_module_triggers_architecture_preflight_through_run_unified_review(self, tmp_path, monkeypatch):
        """Check 4 (architecture_doc) fires through the real _run_unified_review caller.

        This proves the name-status conversion in _run_unified_review feeds
        _preflight_check correctly, so added files are detected.
        """
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        # Simulate: new ouroboros module added + tests staged, but ARCHITECTURE.md absent
        # name-status format: git emits "A\tpath" for added files
        self._mock_staged(
            monkeypatch, review,
            changed_files="ouroboros/new_module.py\ntests/test_new_module.py",
            name_status_files="A\touroboros/new_module.py\nA\ttests/test_new_module.py",
        )
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        result = review._run_unified_review(ctx, "add new module", repo_dir=ctx.repo_dir)
        # Should be blocked by preflight because ARCHITECTURE.md is not staged
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_rename_out_of_ouroboros_triggers_check3(self):
        """Renaming a .py file OUT of ouroboros/ is treated as a deletion and triggers check 3."""
        review = _get_review_module()
        # Source side should appear as D ouroboros/old.py in preflight
        result = review._preflight_check(
            "move module out of ouroboros",
            "D  ouroboros/old.py\nR  docs/old.py",  # src deleted, dest not in ouroboros/
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "tests/" in result

    def test_rename_out_of_ouroboros_with_tests_passes(self):
        """Renaming a .py file out of ouroboros/ + staging tests passes check 3."""
        review = _get_review_module()
        result = review._preflight_check(
            "move module out of ouroboros",
            "D  ouroboros/old.py\nR  docs/old.py\nM  tests/test_old.py",
            "/tmp",
        )
        assert result is None

    def test_rename_into_ouroboros_triggers_architecture_check(self):
        """Renaming a .py file INTO ouroboros/ without ARCHITECTURE.md triggers check 4."""
        review = _get_review_module()
        # Destination becomes "A ouroboros/new_module.py" → triggers new-module check
        result = review._preflight_check(
            "move module into ouroboros",
            "D  docs/old_module.py\nA  ouroboros/new_module.py\nM  tests/test_new.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_rename_into_ouroboros_with_architecture_passes(self):
        """Renaming a .py file into ouroboros/ + staging ARCHITECTURE.md passes check 4."""
        review = _get_review_module()
        result = review._preflight_check(
            "move module into ouroboros",
            "D  docs/old_module.py\nA  ouroboros/new_module.py\nM  tests/test_new.py\nM  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    def test_rename_lines_parsed_correctly_by_preflight(self, tmp_path, monkeypatch):
        """Rename entries (R100\told\tnew) use the destination path for preflight checks."""
        review = _get_review_module()
        # Direct unit test of _preflight_check with a rename line
        # Renamed VERSION to VERSIONX — preflight should not care (it's not "VERSION")
        result = review._preflight_check(
            "rename version file",
            "R  VERSIONX",
            "/tmp",
        )
        # No version-ref in commit message, so no preflight block expected
        assert result is None

    def test_rename_of_readme_counts_as_present(self, tmp_path, monkeypatch):
        """If README.md appears as a rename destination, preflight sees it as staged."""
        review = _get_review_module()
        # Simulate: VERSION staged + README.md arrived via rename
        result = review._preflight_check(
            "v1.0.0: rename readme",
            "M  VERSION\nR  README.md",
            "/tmp",
        )
        # Both VERSION and README.md present → no check 1 block
        # No ouroboros .py → no check 3 block
        assert result is None

    def test_copied_module_without_architecture_blocked(self):
        """Copied .py file in ouroboros/ (status C) triggers architecture-doc preflight."""
        review = _get_review_module()
        # C status means a new file that was copied from somewhere else — still a new module
        result = review._preflight_check(
            "add copied module",
            "C  ouroboros/new_copy.py\nM  tests/test_new_copy.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_copied_module_with_architecture_passes(self):
        """Copied .py file in ouroboros/ + ARCHITECTURE.md staged → passes."""
        review = _get_review_module()
        result = review._preflight_check(
            "add copied module",
            "C  ouroboros/new_copy.py\nM  tests/test_new_copy.py\nM  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is None

    def test_deleted_tests_file_does_not_satisfy_check3(self):
        """Deleting a test file (D status) does not count as 'tests staged'."""
        review = _get_review_module()
        # Logic file modified, old test deleted — check 3 should still block
        result = review._preflight_check(
            "refactor module",
            "M  ouroboros/some_module.py\nD  tests/test_old.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "tests/" in result

    def test_deleted_logic_file_without_tests_blocked(self):
        """Deleting a .py file in ouroboros/ without staged tests is blocked (check 3)."""
        review = _get_review_module()
        # Only a deletion — no tests staged
        result = review._preflight_check(
            "remove old module",
            "D  ouroboros/old_module.py",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "tests/" in result

    def test_deleted_logic_file_with_tests_passes(self):
        """Deleting a .py file + staging a test file passes check 3."""
        review = _get_review_module()
        result = review._preflight_check(
            "remove old module",
            "D  ouroboros/old_module.py\nM  tests/test_old_module.py",
            "/tmp",
        )
        assert result is None

    def test_deleted_architecture_does_not_satisfy_check4(self):
        """Deleting ARCHITECTURE.md does not count as 'architecture doc staged'."""
        review = _get_review_module()
        result = review._preflight_check(
            "add new module",
            "A  ouroboros/new_module.py\nM  tests/test_new.py\nD  docs/ARCHITECTURE.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_deleted_readme_does_not_satisfy_check1(self):
        """Deleting README.md while VERSION is staged triggers check 1."""
        review = _get_review_module()
        result = review._preflight_check(
            "v1.0.0: bump version",
            "M  VERSION\nD  README.md",
            "/tmp",
        )
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "README.md" in result

    def test_copied_module_triggers_via_run_unified_review(self, tmp_path, monkeypatch):
        """Check 4 fires for C-status copy via _run_unified_review, but source NOT treated as deleted."""
        review = _get_review_module()
        ctx = _make_ctx(tmp_path)
        # Copy from ouroboros/base.py to ouroboros/new_copy.py.
        # The source (ouroboros/base.py) is unchanged — only the destination is new.
        # Architecture doc is absent → check 4 should fire.
        self._mock_staged(
            monkeypatch, review,
            changed_files="ouroboros/new_copy.py\ntests/test_new_copy.py",
            name_status_files="C100\touroboros/base.py\touroboros/new_copy.py\nA\ttests/test_new_copy.py",
        )
        monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
        result = review._run_unified_review(ctx, "add copied module", repo_dir=ctx.repo_dir)
        assert result is not None
        assert "PREFLIGHT_BLOCKED" in result
        assert "ARCHITECTURE.md" in result

    def test_copy_source_not_treated_as_deletion(self):
        """Copy source in ouroboros/ does NOT falsely trigger check 3 (source is not deleted)."""
        review = _get_review_module()
        # C100 ouroboros/base.py → docs/base_copy.py
        # The copy source (ouroboros/base.py) was NOT modified or deleted — no logic change.
        # The destination (docs/base_copy.py) is not in ouroboros/ → no new module.
        # Result: preflight should NOT block for missing tests.
        result = review._preflight_check(
            "copy base to docs",
            "A  docs/base_copy.py",  # only the destination; no D entry for C source
            "/tmp",
        )
        # No .py logic change in ouroboros/ → check 3 should not fire
        assert result is None


# --- Unified review wired into commit functions ---

class TestReviewInCommitPipeline:
    def test_repo_commit_calls_unified_review(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        assert "_run_parallel_review" in source
        parallel_source = inspect.getsource(git_mod._run_parallel_review)
        assert "_run_unified_review" in parallel_source

    def test_repo_write_commit_calls_unified_review(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_write_commit)
        assert "_run_parallel_review" in source
        parallel_source = inspect.getsource(git_mod._run_parallel_review)
        assert "_run_unified_review" in parallel_source

    def test_blocked_review_unstages(self):
        """When review blocks, git reset HEAD must be called."""
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        assert 'git", "reset", "HEAD"' in source

    def test_review_rebuttal_forwarded(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._repo_commit_push)
        assert "review_rebuttal" in source


# --- Auto-push and last_push_succeeded ---

class TestAutoPushBehavior:
    def test_auto_push_exists(self):
        git_mod = _get_git_module()
        assert hasattr(git_mod, "_auto_push")
        assert callable(git_mod._auto_push)

    def test_auto_push_is_best_effort(self):
        git_mod = _get_git_module()
        source = inspect.getsource(git_mod._auto_push)
        assert "except Exception" in source
        assert "non-fatal" in source.lower() or "non_fatal" in source.lower()


# --- configure_remote failure surfacing ---

class TestRemoteConfigSurfacing:
    def test_server_logs_remote_failure(self):
        """server.py must check the (ok, msg) return from configure_remote."""
        server_path = pathlib.Path(REPO) / "server.py"
        source = server_path.read_text(encoding="utf-8")
        assert "remote_ok, remote_msg = configure_remote" in source
        assert "Remote configuration failed" in source

    def test_settings_save_returns_warnings(self):
        """api_settings_post must surface remote config failures."""
        server_path = pathlib.Path(REPO) / "server.py"
        source = server_path.read_text(encoding="utf-8")
        assert '"warnings"' in source

    def test_migrate_credentials_wired_at_startup(self):
        """migrate_remote_credentials called at startup after configure_remote."""
        server_path = pathlib.Path(REPO) / "server.py"
        source = server_path.read_text(encoding="utf-8")
        assert "migrate_remote_credentials" in source


# --- migrate_remote_credentials safety ---

class TestMigrateRemoteCredentials:
    def test_exists(self):
        git_ops = _get_git_ops_module()
        assert hasattr(git_ops, "migrate_remote_credentials")
        assert callable(git_ops.migrate_remote_credentials)

    def test_uses_configure_remote(self):
        git_ops = _get_git_ops_module()
        source = inspect.getsource(git_ops.migrate_remote_credentials)
        assert "configure_remote" in source

    def test_noop_on_clean_origin(self):
        """Clean origin URL (no embedded token) returns True with 'already clean'."""
        git_ops = _get_git_ops_module()
        source = inspect.getsource(git_ops.migrate_remote_credentials)
        assert "already clean" in source.lower() or "Already clean" in source


# --- Startup auto-rescue semantics ---

class TestAutoRescueSemantics:
    def test_auto_rescue_checks_commit_result(self):
        """Auto-rescue must verify git commit actually created a commit."""
        agent_mod = importlib.import_module("ouroboros.agent")
        source = inspect.getsource(agent_mod.OuroborosAgent._check_uncommitted_changes)
        assert "nothing to commit" in source
        assert "capture_output=True" in source

    def test_auto_rescue_does_not_claim_committed_on_noop(self):
        """When git commit produces no new commit, auto_committed must be False."""
        agent_mod = importlib.import_module("ouroboros.agent")
        source = inspect.getsource(agent_mod.OuroborosAgent._check_uncommitted_changes)
        assert "returncode == 0" in source
        assert "auto_committed = True" in source
        idx_committed = source.index("auto_committed = True")
        idx_check = source.index("nothing to commit")
        assert idx_check < idx_committed


# --- ToolContext review state ---

class TestToolContextReviewState:
    def test_review_fields_exist(self):
        from ouroboros.tools.registry import ToolContext
        ctx = ToolContext(
            repo_dir=pathlib.Path("/tmp"),
            drive_root=pathlib.Path("/tmp"),
        )
        assert hasattr(ctx, "_review_advisory")
        assert hasattr(ctx, "_review_iteration_count")
        assert hasattr(ctx, "_review_history")
        assert ctx._review_advisory == []
        assert ctx._review_iteration_count == 0
        assert ctx._review_history == []


# --- Registry sandbox covers repo_write ---

class TestSandboxCoversRepoWrite:
    def test_sandbox_mentions_repo_write(self):
        registry = _get_registry_module()
        source = inspect.getsource(registry.ToolRegistry.execute)
        assert "repo_write" in source

    def test_sandbox_checks_files_param(self):
        """Sandbox must check files array for safety-critical paths."""
        registry = _get_registry_module()
        source = inspect.getsource(registry.ToolRegistry.execute)
        assert "files" in source


# --- index-full instruction fix ---

class TestIndexFullInstruction:
    def test_system_md_warns_against_index_full(self):
        system_md = pathlib.Path(REPO) / "prompts" / "SYSTEM.md"
        content = system_md.read_text(encoding="utf-8")
        assert "Do NOT call" in content or "reserved internal name" in content
        assert "knowledge_list" in content
