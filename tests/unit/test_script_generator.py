"""Unit tests for script_generator — replay script generation from agent steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


class TestGenerateReplayScript:
    """Test generate_replay_script() core behavior."""

    def test_navigate_generates_goto_and_wait(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("nav", [
            {"action": "navigate", "args": {"url": "https://example.com"}},
        ])
        assert 'await page.goto("https://example.com", wait_until="domcontentloaded")' in script
        assert 'await page.wait_for_load_state("networkidle")' in script

    def test_perception_tools_are_filtered_out(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("perception", [
            {"action": "page_perceive", "args": {}},
            {"action": "find_element", "args": {"description": "search"}},
            {"action": "get_page_info", "args": {}},
            {"action": "get_text", "args": {"selector": ".result"}},
            {"action": "response", "content": "done"},
            {"action": "click", "args": {"selector": "#btn"}},
        ])
        assert "page_perceive" not in script
        assert "find_element" not in script
        assert "get_page_info" not in script
        assert "get_text" not in script
        assert "response" not in script
        assert '_cdp_click(page, _box)' in script
        assert '#btn' in script

    def test_empty_steps_produces_pass(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("empty", [])
        assert "async def run(page, context):" in script
        assert "    pass" in script

    def test_all_readonly_steps_produces_pass(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("readonly", [
            {"action": "page_perceive"},
            {"action": "get_page_info"},
            {"action": "response", "content": "nothing"},
        ])
        assert "    pass" in script

    def test_script_includes_standalone_block(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("standalone", [
            {"action": "click", "args": {"selector": "#x"}},
        ])
        assert 'if __name__ == "__main__":' in script
        assert "async def main():" in script
        assert "asyncio.run(main())" in script

    def test_script_has_imports(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("imports", [
            {"action": "click", "args": {"selector": "#x"}},
        ])
        assert "import asyncio" in script
        assert "from playwright.async_api import async_playwright" in script

    def test_script_is_valid_python(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("syntax", [
            {"action": "navigate", "args": {"url": "https://example.com"}},
            {"action": "click", "args": {"selector": "#btn"}},
            {"action": "eval_js", "args": {"expression": "document.title"}},
        ])
        # Verify syntactically valid
        compile(script, "<replay_script>", "exec")

    def test_script_run_function_is_importable(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("executable", [
            {"action": "click", "args": {"selector": "#btn"}},
        ])
        exec_globals: dict = {"__builtins__": __builtins__}
        exec(script, exec_globals)
        assert "run" in exec_globals
        assert callable(exec_globals["run"])

    def test_step_id_in_script_docstring(self):
        from skiritai.core.script_generator import generate_replay_script

        script = generate_replay_script("my_step_v2", [
            {"action": "click", "args": {"selector": "#btn"}},
        ])
        assert '"""Auto-generated replay script' in script


class TestActionToLine:
    """Test individual action type code generation."""

    def test_navigate_with_url(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("navigate", {"url": "https://x.com"})
        assert 'await page.goto("https://x.com", wait_until="domcontentloaded")' in line
        assert 'wait_for_load_state("networkidle")' in line

    def test_click_with_selector(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("click", {"selector": "#submit"})
        assert '_cdp_click(page, _box)' in line
        assert '#submit' in line

    def test_click_no_networkidle(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("click", {"selector": "#submit"})
        assert "networkidle" not in line

    def test_click_text(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("click_text", {"text": "百度一下"})
        assert 'page.get_by_text("百度一下")' in line
        assert "_cdp_click" in line
        assert "networkidle" not in line

    def test_click_force(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("click_force", {"selector": "#overlay-btn"})
        assert "force=True" in line
        assert "networkidle" not in line

    def test_fill_with_text(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("fill", {"selector": "#input", "text": "hello"})
        assert 'page.locator("#input").fill("hello", force=True)' in line

    def test_type_text(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("type_text", {"selector": "#slow", "text": "world"})
        assert 'page.locator("#slow").press_sequentially("world", delay=50)' in line

    def test_focus(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("focus", {"selector": "#email"})
        assert 'page.locator("#email").focus()' in line

    def test_wait_for_with_default_timeout(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("wait_for", {"selector": ".loading"})
        assert "timeout=5000" in line

    def test_wait_for_with_custom_timeout(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("wait_for", {"selector": ".slow", "timeout": 10000})
        assert "timeout=10000" in line

    def test_scroll_down(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("scroll", {"direction": "down", "amount": 500})
        assert "await page.mouse.wheel(0, 500)" in line

    def test_scroll_up(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("scroll", {"direction": "up", "amount": 300})
        assert "await page.mouse.wheel(0, -300)" in line

    def test_scroll_default_direction(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("scroll", {"amount": 200})
        assert "await page.mouse.wheel(0, 200)" in line  # default down

    def test_eval_js(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("eval_js", {"expression": "1 + 1"})
        assert "result = await page.evaluate(" in line
        assert "1 + 1" in line

    def test_select_option(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("select_option", {"selector": "#country", "value": "CN"})
        assert 'page.select_option("#country", "CN")' in line

    def test_hover(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("hover", {"selector": "#menu"})
        assert 'page.hover("#menu")' in line

    def test_screenshot_filtered_out(self):
        """screenshot is a perception tool — _action_to_line returns None."""
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("screenshot", {"name": "step1"})
        assert line is None

    def test_unknown_action_returns_none(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("non_existent_action", {})
        assert line is None

    def test_wait_with_default_seconds(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("wait", {})
        assert "await asyncio.sleep(1.0)" in line

    def test_wait_with_custom_seconds(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("wait", {"seconds": 2.5})
        assert "await asyncio.sleep(2.5)" in line

    def test_press_key(self):
        from skiritai.core.script_generator import _action_to_line
        line = _action_to_line("press_key", {"key": "Enter"})
        assert "await page.keyboard.press('Enter')" in line

    def test_missing_args_defaults_to_empty(self):
        from skiritai.core.script_generator import _action_to_line
        # fill without text should use empty string
        line = _action_to_line("fill", {"selector": "#x"})
        assert 'page.locator("#x").fill("", force=True)' in line


class TestEscaping:
    """Test character escaping in generated scripts."""

    def test_escapes_double_quotes(self):
        from skiritai.core.script_generator import _esc
        result = _esc('he said "hello"')
        assert result == 'he said \\"hello\\"'

    def test_escapes_backslashes(self):
        from skiritai.core.script_generator import _esc
        result = _esc(r"C:\path\to\file")
        assert result == "C:\\\\path\\\\to\\\\file"

    def test_escapes_newlines(self):
        from skiritai.core.script_generator import _esc
        result = _esc("line1\nline2")
        assert result == "line1\\nline2"

    def test_escapes_combined_special_chars(self):
        from skiritai.core.script_generator import _esc
        result = _esc('say "hi"\nnew\\end')
        assert '\\"' in result
        assert '\\n' in result
        assert '\\\\' in result

    def test_eval_js_escapes_in_script(self):
        from skiritai.core.script_generator import generate_replay_script
        script = generate_replay_script("esc", [
            {"action": "eval_js", "args": {"expression": 'document.querySelector("div").click()'}},
        ])
        # repr() should produce a valid Python string literal containing the expression
        assert "document.querySelector" in script
        compile(script, "<replay_script>", "exec")  # must be valid Python

    def test_url_with_special_chars(self):
        from skiritai.core.script_generator import generate_replay_script
        script = generate_replay_script("url", [
            {"action": "navigate", "args": {"url": 'https://example.com/search?q="test"&page=1'}},
        ])
        assert '\\"' in script
        compile(script, "<replay_script>", "exec")  # must be valid Python

    def test_all_action_types_produce_valid_python(self):
        """Every supported action type should produce syntactically valid Python."""
        from skiritai.core.script_generator import generate_replay_script

        all_actions = []
        for action, args in [
            ("navigate", {"url": "https://example.com"}),
            ("click", {"selector": "#btn"}),
            ("click_text", {"text": "百度一下"}),
            ("click_force", {"selector": "#forced"}),
            ("fill", {"selector": "#input", "text": "hello"}),
            ("type_text", {"selector": "#slow", "text": "world"}),
            ("focus", {"selector": "#focus"}),
            ("wait_for", {"selector": "#wait", "timeout": 3000}),
            ("scroll", {"direction": "down", "amount": 500}),
            ("scroll", {"direction": "up", "amount": 200}),
            ("eval_js", {"expression": "document.title"}),
            ("select_option", {"selector": "#sel", "value": "opt"}),
            ("hover", {"selector": "#hover"}),
            ("wait", {"seconds": 0.5}),
            ("press_key", {"key": "Enter"}),
        ]:
            all_actions.append({"action": action, "args": args})

        script = generate_replay_script("all_actions", all_actions)
        compile(script, "<replay_script>", "exec")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "--no-header"],
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    sys.exit(result.returncode)
