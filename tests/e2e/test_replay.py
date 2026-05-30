"""E2E tests — replay execution and case runner with real browser."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

os.environ["SKIRITAI_HEADLESS"] = "true"

from playwright.async_api import async_playwright

from skiritai.core.ai_context import AIContext
from skiritai.core.browser import get_launch_args
from skiritai.core.runner import run_case


class TestReplayE2E:
    """Run replay scripts against a real browser and real local page."""

    @pytest.mark.asyncio
    async def test_replay_executes_successfully(self, simple_case):
        case_dir, url = simple_case

        pw = await async_playwright().start()
        br = await pw.chromium.launch(**get_launch_args())
        ctx = await br.new_context()
        pg = await ctx.new_page()

        try:
            ai = AIContext(page=pg, case_dir=case_dir, step_id="fill_and_submit")
            assert ai.has_replay() is True

            result = await ai.action("fill and submit", mode="replay")

            assert result["success"] is True
            assert result["steps"][0]["action"] == "replay"

            input_val = await pg.input_value("#text-input")
            assert input_val == "Hello E2E"

            result_text = await pg.text_content("#result")
            assert "Hello E2E" in result_text
        finally:
            await br.close()
            await pw.stop()

    @pytest.mark.asyncio
    async def test_replay_full_case_run(self, simple_case):
        case_dir, url = simple_case

        report = await run_case(
            case_dir=case_dir,
            execution_id="e2e_replay_test",
        )

        assert report["case_name"] == "E2ETestCase"
        assert report["status"] == "completed"
        assert report["total_steps"] == 1
        assert report["success_count"] == 1
        assert report["failed_count"] == 0
        assert len(report["steps"]) == 1
        assert report["steps"][0]["step_id"] == "fill_and_submit"
        assert report["steps"][0]["status"] == "success"
        assert report["steps"][0]["mode"] == "replay"


class TestCaseRunner:
    """Test run_case() with a real browser."""

    @pytest.mark.asyncio
    async def test_run_case_returns_valid_report(self, simple_case):
        case_dir, url = simple_case
        results_dir = case_dir / "results" / "test_run"

        report = await run_case(
            case_dir=case_dir,
            execution_id="runner_test",
            results_dir=results_dir,
        )

        assert "case_name" in report
        assert "status" in report
        assert "total_steps" in report
        assert "success_count" in report
        assert "failed_count" in report
        assert "steps" in report
        assert report["status"] == "completed"
        assert report["total_steps"] == 1
        assert report["success_count"] == 1

        step = report["steps"][0]
        assert "step_id" in step
        assert "status" in step
        assert "mode" in step
        assert step["step_id"] == "fill_and_submit"
        assert step["mode"] == "replay"

    @pytest.mark.asyncio
    async def test_run_case_with_multi_steps(self, multi_step_case):
        case_dir, case_id, url = multi_step_case

        report = await run_case(
            case_dir=case_dir,
            execution_id="multi_step_test",
        )

        assert report["status"] == "completed"
        assert report["total_steps"] == 2
        assert report["success_count"] == 2
        assert len(report["steps"]) == 2

        step_ids = [s["step_id"] for s in report["steps"]]
        assert "fill_form" in step_ids
        assert "navigate_page" in step_ids


class TestScriptRoundtrip:
    """Generate scripts and replay them with a real browser."""

    @pytest.mark.asyncio
    async def test_generate_script_then_replay(self, case_url):
        url, _ = case_url
        tmpdir = tempfile.mkdtemp(prefix="e2e_roundtrip_")
        case_dir = Path(tmpdir) / "roundtrip"
        case_dir.mkdir()

        pw = await async_playwright().start()
        try:
            br = await pw.chromium.launch(**get_launch_args())
            ctx_obj = await br.new_context()
            page = await ctx_obj.new_page()

            ai = AIContext(page=page, case_dir=case_dir, step_id="gen_test")

            agent_steps = [
                {"action": "navigate", "args": {"url": url}},
                {"action": "fill", "args": {"selector": "#text-input", "text": "Generated"}},
                {"action": "click", "args": {"selector": "#submit-btn"}},
            ]
            from skiritai.core.script_generator import generate_replay_script

            script = generate_replay_script("gen_test", agent_steps)

            assert "async def run(page, context):" in script
            assert f'await page.goto("{url}", wait_until="domcontentloaded")' in script
            assert 'page.locator("#text-input").fill("Generated", force=True)' in script
            assert '_cdp_click' in script

            ai.scripts_dir.mkdir(parents=True, exist_ok=True)
            ai.script_path.write_text(script, encoding="utf-8")
            from skiritai.core.ai_context import _save_script_hash
            _save_script_hash(ai.script_path, script)

            assert ai.has_replay() is True
            result = await ai.action("test", mode="replay")
            assert result["success"] is True

            input_val = await page.input_value("#text-input")
            assert input_val == "Generated"

            await br.close()
        finally:
            await pw.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_replay_broken_script_returns_failure(self):
        tmpdir = tempfile.mkdtemp(prefix="e2e_broken_")
        case_dir = Path(tmpdir) / "broken"
        case_dir.mkdir()

        pw = await async_playwright().start()
        try:
            br = await pw.chromium.launch(**get_launch_args())
            ctx_obj = await br.new_context()
            page = await ctx_obj.new_page()

            ai = AIContext(page=page, case_dir=case_dir, step_id="broken")
            ai.scripts_dir.mkdir(parents=True, exist_ok=True)
            broken_content = (
                "async def run(page, context):\n"
                "    raise ValueError('script is broken')\n"
            )
            ai.script_path.write_text(broken_content, encoding="utf-8")
            from skiritai.core.ai_context import _save_script_hash
            _save_script_hash(ai.script_path, broken_content)

            result = await ai._replay()
            assert result["success"] is False
            assert "script is broken" in result["summary"]

            await br.close()
        finally:
            await pw.stop()
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_generate_script_all_action_types(self):
        from skiritai.core.script_generator import generate_replay_script

        agent_steps = [
            {"action": "navigate", "args": {"url": "http://example.com"}},
            {"action": "click", "args": {"selector": "#btn"}},
            {"action": "click_force", "args": {"selector": "#forced"}},
            {"action": "fill", "args": {"selector": "#input", "text": "hello"}},
            {"action": "type_text", "args": {"selector": "#slow", "text": "world"}},
            {"action": "focus", "args": {"selector": "#focus"}},
            {"action": "wait_for", "args": {"selector": "#wait", "timeout": 3000}},
            {"action": "scroll", "args": {"direction": "down", "amount": 500}},
            {"action": "scroll", "args": {"direction": "up", "amount": 200}},
            {"action": "eval_js", "args": {"expression": "document.title"}},
            {"action": "select_option", "args": {"selector": "#sel", "value": "opt"}},
            {"action": "hover", "args": {"selector": "#hover"}},
        ]

        script = generate_replay_script("all_actions", agent_steps)

        assert 'await page.goto("http://example.com", wait_until="domcontentloaded")' in script
        assert '_cdp_click' in script
        assert 'await page.click("#forced", force=True)' in script
        assert 'page.locator("#input").fill("hello", force=True)' in script
        assert 'await page.keyboard.press("Control+a")' in script
        assert 'await page.keyboard.press("Backspace")' in script
        assert 'await page.locator("#slow").press_sequentially("world", delay=50)' in script
        assert 'await page.locator("#focus").focus()' in script
        assert 'await page.wait_for_selector("#wait", timeout=3000)' in script
        assert "await page.mouse.wheel(0, 500)" in script
        assert "await page.mouse.wheel(0, -200)" in script
        assert "await page.evaluate('document.title')" in script or \
               'await page.evaluate("document.title")' in script
        assert 'await page.select_option("#sel", "opt")' in script
        assert 'await page.hover("#hover")' in script


if __name__ == "__main__":
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "--no-header"],
        cwd=Path(__file__).resolve().parent.parent.parent,
        env={**os.environ, "HEADLESS": "true"},
    )
    sys.exit(result.returncode)
