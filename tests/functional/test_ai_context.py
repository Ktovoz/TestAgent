"""Functional tests for AIContext — no browser/LLM required."""
from __future__ import annotations

import ast
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAIContext:
    """Test AIContext explore/replay/script generation logic."""

    def test_script_path_and_has_replay(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "mycase"
            case_dir.mkdir()
            ctx = AIContext(page=MagicMock(), case_dir=case_dir, step_id="open_page")

            assert ctx.script_path == case_dir / "scripts" / "open_page.py"
            assert ctx.has_replay() is False

            # Create a script file
            ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
            ctx.script_path.write_text("async def run(page, ctx): pass")
            assert ctx.has_replay() is True

    def test_mode_auto_replays_when_script_exists(self):
        from skiritai.core.ai_context import AIContext
        from skiritai.core.ai_context import _compute_script_hash, SCRIPT_HASH_SUFFIX

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "mycase"
            case_dir.mkdir()
            ctx = AIContext(page=MagicMock(), case_dir=case_dir, step_id="step1")

            # Write a valid replay script
            ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
            script_content = "async def run(page, context):\n    pass\n"
            ctx.script_path.write_text(script_content)
            # Write the integrity hash so _verify_script passes
            hash_path = Path(str(ctx.script_path) + SCRIPT_HASH_SUFFIX)
            hash_path.write_text(_compute_script_hash(script_content))

            result = asyncio.run(ctx.action("do something", mode="auto"))
            assert result["success"] is True
            assert result["steps"][0]["action"] == "replay"

    def test_mode_explore_always_explores(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "mycase"
            case_dir.mkdir()

            mock_page = MagicMock()
            mock_page.url = "http://test.com"
            ctx = AIContext(page=mock_page, case_dir=case_dir, step_id="step1")

            # Even if script exists, explore mode should overwrite
            ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
            ctx.script_path.write_text("# existing")

            # Mock the agent run to avoid real LLM calls
            mock_result = {"success": True, "summary": "done", "steps": []}
            with patch.object(ctx, "_explore", AsyncMock(return_value=mock_result)):
                result = asyncio.run(ctx.action("do something", mode="explore"))
                assert result is mock_result

    def test_mode_replay_raises_without_script(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "mycase"
            case_dir.mkdir()
            ctx = AIContext(page=MagicMock(), case_dir=case_dir, step_id="step1")

            with pytest.raises(FileNotFoundError):
                asyncio.run(ctx.action("do something", mode="replay"))

    def test_generate_script_simple_actions(self):
        from skiritai.core.script_generator import generate_replay_script

        agent_steps = [
            {"action": "navigate", "args": {"url": "https://example.com"}},
            {"action": "click", "args": {"selector": "#btn"}},
            {"action": "fill", "args": {"selector": "#input", "text": "hello"}},
        ]

        script = generate_replay_script("test", agent_steps)
        assert "async def run(page, context):" in script
        assert 'await page.goto("https://example.com", wait_until="domcontentloaded")' in script
        assert '_cdp_click(page, _box)' in script
        assert '#btn' in script
        assert 'page.locator("#input").fill("hello", force=True)' in script

    def test_generate_script_empty_steps_produces_pass(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("test", [])
        assert "async def run(page, context):" in script
        assert "    pass" in script

    def test_generate_script_all_action_types(self):
        from skiritai.core.script_generator import generate_replay_script

        agent_steps = [
            {"action": "navigate", "args": {"url": "http://a.com"}},
            {"action": "click", "args": {"selector": "#b"}},
            {"action": "click_force", "args": {"selector": "#c"}},
            {"action": "fill", "args": {"selector": "#d", "text": "t"}},
            {"action": "type_text", "args": {"selector": "#e", "text": "x"}},
            {"action": "focus", "args": {"selector": "#f"}},
            {"action": "wait_for", "args": {"selector": "#g", "timeout": 3000}},
            {"action": "scroll", "args": {"direction": "down", "amount": 500}},
            {"action": "eval_js", "args": {"expression": "1+1"}},
            {"action": "select_option", "args": {"selector": "#h", "value": "v"}},
            {"action": "hover", "args": {"selector": "#i"}},
        ]

        script = generate_replay_script("test", agent_steps)
        assert 'await page.goto("http://a.com", wait_until="domcontentloaded")' in script
        assert '_cdp_click' in script and '#b' in script
        assert 'await page.click("#c", force=True)' in script
        assert 'page.locator("#d").fill("t", force=True)' in script
        assert 'page.locator("#e").press_sequentially("x", delay=50)' in script
        assert 'await page.locator("#f").focus()' in script
        assert 'await page.wait_for_selector("#g", timeout=3000)' in script
        assert "await page.mouse.wheel(0, 500)" in script
        assert 'await page.evaluate("1+1")' in script or "await page.evaluate('1+1')" in script
        assert 'await page.select_option("#h", "v")' in script
        assert 'await page.hover("#i")' in script

    def test_replay_script_with_error_returns_failure(self):
        from skiritai.core.ai_context import AIContext
        from skiritai.core.ai_context import _compute_script_hash, SCRIPT_HASH_SUFFIX

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "mycase"
            case_dir.mkdir()
            ctx = AIContext(page=MagicMock(), case_dir=case_dir, step_id="step1")

            # Write a script that raises
            ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
            script_content = "async def run(page, context):\n    raise RuntimeError('test error')\n"
            ctx.script_path.write_text(script_content)
            # Write the integrity hash so _verify_script passes
            hash_path = Path(str(ctx.script_path) + SCRIPT_HASH_SUFFIX)
            hash_path.write_text(_compute_script_hash(script_content))

            result = asyncio.run(ctx._replay())
            assert result["success"] is False
            assert "test error" in result["summary"]


class TestASTSandbox:
    """Unit tests for replay script AST validation — blocking sandbox bypass."""

    def test_block_getattr_called_as_forbidden_builtin(self):
        """getattr() is a forbidden builtin — calling it directly must raise."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "getattr(__builtins__, 'eval')"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden builtin"):
            _validate_replay_ast(tree)

    def test_block_setattr_called_as_forbidden_builtin(self):
        """setattr() is a forbidden builtin — calling it directly must raise."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "setattr(obj, '__class__', x)"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden builtin"):
            _validate_replay_ast(tree)

    def test_block_delattr_called_as_forbidden_builtin(self):
        """delattr() is a forbidden builtin — calling it directly must raise."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "delattr(obj, 'attr')"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden builtin"):
            _validate_replay_ast(tree)

    def test_block_dunder_builtins_attribute_access(self):
        """Accessing .__builtins__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "x.__builtins__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_class_attribute_access(self):
        """Accessing .__class__ attribute is blocked (sandbox escape vector)."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "obj.__class__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_subclasses_attribute_access(self):
        """Accessing .__subclasses__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "type.__subclasses__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_dict_attribute_access(self):
        """Accessing .__dict__ attribute is blocked (obj introspection)."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "obj.__dict__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_globals_attribute_access(self):
        """Accessing .__globals__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "func.__globals__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_code_attribute_access(self):
        """Accessing .__code__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "func.__code__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_bases_attribute_access(self):
        """Accessing .__bases__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "cls.__bases__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_mro_attribute_access(self):
        """Accessing .__mro__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "cls.__mro__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_func_attribute_access(self):
        """Accessing .__func__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "method.__func__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_self_attribute_access(self):
        """Accessing .__self__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "method.__self__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_block_dunder_module_attribute_access(self):
        """Accessing .__module__ attribute is blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "func.__module__"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)

    def test_allow_safe_attribute_access(self):
        """Normal attribute access (e.g., page.click()) must be allowed."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "page.click('#btn')\npage.fill('#input', 'hello')"
        tree = ast.parse(code)
        # Should not raise
        _validate_replay_ast(tree)

    def test_allow_basic_function_calls(self):
        """Standard function calls like print(), len() must be allowed."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "print('hello')\nx = len([1, 2, 3])"
        tree = ast.parse(code)
        # Should not raise
        _validate_replay_ast(tree)

    def test_block_getattr_bypass_pattern(self):
        """getattr(__builtins__, 'ev'+'al') — indirect eval access must be blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "getattr(__builtins__, 'eval')"
        tree = ast.parse(code)
        with pytest.raises(ValueError, match="Forbidden builtin"):
            _validate_replay_ast(tree)

    def test_block_subclasses_escape_chain(self):
        """__class__.__subclasses__() chain — common sandbox escape must be blocked."""
        from skiritai.core.ai_context import _validate_replay_ast

        code = "x.__class__.__subclasses__()"
        tree = ast.parse(code)
        # The first forbidden attr found should trigger the error
        with pytest.raises(ValueError, match="Forbidden attribute access"):
            _validate_replay_ast(tree)


class TestSafeImport:
    """Unit tests for _safe_import — the __import__ wrapper used in exec() sandbox."""

    def test_allows_whitelisted_module_asyncio(self):
        from skiritai.core.ai_context import _safe_import

        mod = _safe_import("asyncio")
        assert mod is not None

    def test_allows_whitelisted_submodule_playwright_async_api(self):
        from skiritai.core.ai_context import _safe_import

        mod = _safe_import("playwright.async_api", fromlist=["async_playwright"])
        assert mod is not None

    def test_rejects_os_module(self):
        from skiritai.core.ai_context import _safe_import

        with pytest.raises(ImportError, match="not allowed"):
            _safe_import("os")

    def test_rejects_unknown_module(self):
        from skiritai.core.ai_context import _safe_import

        with pytest.raises(ImportError, match="not allowed"):
            _safe_import("subprocess")

    def test_exec_with_restricted_builtins_rejects_import_os(self):
        from skiritai.core.ai_context import (
            _FORBIDDEN_BUILTINS,
            _safe_import,
            _validate_replay_ast,
        )
        import builtins as _builtins_mod

        safe_builtins = {
            k: v for k, v in vars(_builtins_mod).items()
            if k not in _FORBIDDEN_BUILTINS
        }
        safe_builtins["__import__"] = _safe_import

        code = "import os\nos.getenv('HOME')\n"
        tree = ast.parse(code)
        _validate_replay_ast(tree)  # AST check passes (import is allowed)

        exec_globals = {"__builtins__": safe_builtins}
        with pytest.raises(ImportError, match="not allowed"):
            exec(code, exec_globals)

    def test_exec_with_restricted_builtins_allows_asyncio_import(self):
        from skiritai.core.ai_context import (
            _FORBIDDEN_BUILTINS,
            _safe_import,
            _validate_replay_ast,
        )
        import builtins as _builtins_mod

        safe_builtins = {
            k: v for k, v in vars(_builtins_mod).items()
            if k not in _FORBIDDEN_BUILTINS
        }
        safe_builtins["__import__"] = _safe_import

        code = "import asyncio\nx = asyncio\n"
        tree = ast.parse(code)
        _validate_replay_ast(tree)

        exec_globals = {"__builtins__": safe_builtins}
        exec(code, exec_globals)  # should not raise
