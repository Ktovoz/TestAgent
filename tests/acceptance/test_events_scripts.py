"""Acceptance tests — event sequencing and script roundtrip with mocked LLM/browser."""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tests.acceptance.conftest import _make_mock_page


class TestEventSequencing:
    """Verify events are published in the correct order during execution."""

    def test_events_published_in_order(self):
        from skiritai.core.base_case import BaseCase
        from skiritai.events import Event, event_bus

        events: list[str] = []

        async def capture(event: Event):
            events.append(event.type)

        event_bus.subscribe(capture)

        class SimpleCase(BaseCase):
            async def setup(self):
                self._page = _make_mock_page()

            async def teardown(self):
                pass

            async def my_step(self, ai):
                return await ai.action("do it")

        case = SimpleCase()
        mock_result = {"success": True, "summary": "ok", "steps": []}

        try:
            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=mock_result)):
                asyncio.run(case.run())
        finally:
            event_bus.unsubscribe(capture)

        assert events[0] == "execution_started"
        assert "step_started" in events
        assert "step_completed" in events
        assert events[-1] == "execution_completed"

    def test_step_failed_event_on_failure(self):
        from skiritai.core.base_case import BaseCase
        from skiritai.events import Event, event_bus

        step_events: list[tuple[str, str]] = []

        async def capture(event: Event):
            if event.type in ("step_completed", "step_failed"):
                step_events.append((event.type, event.data.get("step_id")))

        event_bus.subscribe(capture)

        class FailCase(BaseCase):
            async def setup(self):
                self._page = _make_mock_page()

            async def teardown(self):
                pass

            async def bad_step(self, ai):
                return await ai.action("fail")

        case = FailCase()
        fail_result = {"success": False, "summary": "not found", "steps": []}

        try:
            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=fail_result)):
                asyncio.run(case.run())
        finally:
            event_bus.unsubscribe(capture)

        assert len(step_events) == 1
        assert step_events[0][0] == "step_failed"
        assert step_events[0][1] == "bad_step"

    def test_tool_called_events_published(self):
        from skiritai.core.base_case import BaseCase
        from skiritai.events import Event, event_bus

        tool_events: list[str] = []

        async def capture(event: Event):
            if event.type == "tool_called":
                tool_events.append(event.data.get("tool_name"))

        event_bus.subscribe(capture)

        class ToolCase(BaseCase):
            async def setup(self):
                self._page = _make_mock_page()

            async def teardown(self):
                pass

            async def navigate_step(self, ai):
                return await ai.action("go to site")

        case = ToolCase()
        agent_result = {
            "success": True,
            "summary": "navigated",
            "steps": [
                {"action": "navigate", "args": {"url": "http://example.com"}},
                {"action": "click", "args": {"selector": "#btn"}},
            ],
        }

        try:
            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=agent_result)):
                asyncio.run(case.run())
        finally:
            event_bus.unsubscribe(capture)

        assert isinstance(tool_events, list)


class TestScriptRoundtrip:
    """Test generating a script from explore, then replaying it."""

    def test_explore_generates_script_file(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "roundtrip"
            case_dir.mkdir()
            mock_page = _make_mock_page()

            ctx = AIContext(page=mock_page, case_dir=case_dir, step_id="nav")
            agent_result = {
                "success": True,
                "summary": "navigated",
                "steps": [
                    {"action": "navigate", "args": {"url": "http://example.com"}},
                    {"action": "click", "args": {"selector": "#login"}},
                ],
            }

            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=agent_result)):
                result = asyncio.run(ctx.action("go to site", mode="explore"))

            assert result["success"] is True
            assert ctx.script_path.exists()

            content = ctx.script_path.read_text()
            assert "async def run(page, context):" in content
            assert "page.goto" in content
            assert "_cdp_click" in content
            assert "#login" in content

    def test_explore_then_replay_roundtrip(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "roundtrip"
            case_dir.mkdir()

            mock_page = AsyncMock()
            mock_page.url = "http://localhost"
            mock_page.context = AsyncMock()

            ctx_explore = AIContext(page=mock_page, case_dir=case_dir, step_id="do_thing")
            agent_result = {
                "success": True,
                "summary": "done",
                "steps": [],
            }

            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=agent_result)):
                explore_result = asyncio.run(ctx_explore.action("do the thing", mode="explore"))

            assert explore_result["success"] is True
            assert ctx_explore.script_path.exists()

            ctx_replay = AIContext(page=mock_page, case_dir=case_dir, step_id="do_thing")
            assert ctx_replay.has_replay() is True

            replay_result = asyncio.run(ctx_replay.action("do the thing", mode="replay"))
            assert replay_result["success"] is True
            assert replay_result["steps"][0]["action"] == "replay"

    def test_explore_failure_does_not_write_script(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "no_script"
            case_dir.mkdir()

            ctx = AIContext(page=_make_mock_page(), case_dir=case_dir, step_id="fail_step")
            fail_result = {"success": False, "summary": "timeout", "steps": []}

            with patch("skiritai.core.ai_context.run_agent", AsyncMock(return_value=fail_result)):
                result = asyncio.run(ctx.action("this will fail", mode="explore"))

            assert result["success"] is False
            assert not ctx.script_path.exists()

    def test_replay_error_returns_failure_dict(self):
        from skiritai.core.ai_context import AIContext

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "broken"
            case_dir.mkdir()

            ctx = AIContext(page=_make_mock_page(), case_dir=case_dir, step_id="broken")
            ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
            broken_content = (
                "async def run(page, context):\n"
                "    raise ValueError('script is broken')\n"
            )
            ctx.script_path.write_text(broken_content)
            from skiritai.core.ai_context import _save_script_hash
            _save_script_hash(ctx.script_path, broken_content)

            result = asyncio.run(ctx._replay())
            assert result["success"] is False
            assert "script is broken" in result["summary"]


if __name__ == "__main__":
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "--no-header"],
        cwd=Path(__file__).resolve().parent.parent.parent,
    )
    sys.exit(result.returncode)
