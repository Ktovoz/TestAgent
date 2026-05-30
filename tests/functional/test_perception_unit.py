"""Test suite for perception layer, CaseContext, and replay script generation.

Run: python -m pytest tests/functional/test_perception_unit.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest


# ─── Test 1: CaseContext state machine ────────────────────────────────

def test_case_context():
    from skiritai.core.case_context import CaseContext, CasePhase

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = CaseContext(case_dir=Path(tmpdir))

        # Initial state
        assert ctx.phase == CasePhase.IDLE
        assert ctx.started_at is None
        assert ctx.elapsed_seconds is None

        # Valid transitions
        asyncio.run(ctx.transition("setup"))
        assert ctx.phase == CasePhase.SETUP
        assert ctx.started_at is not None

        asyncio.run(ctx.transition("running"))
        assert ctx.phase == CasePhase.RUNNING

        asyncio.run(ctx.transition("teardown"))
        asyncio.run(ctx.transition("done"))
        assert ctx.phase == CasePhase.DONE
        assert ctx.finished_at is not None
        assert ctx.elapsed_seconds is not None

        print("[PASS] test_case_context: state machine transitions")


def test_case_context_invalid_transition():
    from skiritai.core.case_context import CaseContext, StateError

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = CaseContext(case_dir=Path(tmpdir))

        # Can't go directly from IDLE to RUNNING
        try:
            asyncio.run(ctx.transition("running"))
            assert False, "Should have raised StateError"
        except StateError:
            pass

        print("[PASS] test_case_context_invalid_transition")


def test_case_context_global_store():
    from skiritai.core.case_context import CaseContext

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = CaseContext(case_dir=Path(tmpdir))

        ctx.store.set("cdp_port", 9222)
        ctx.store.set("login_token", "abc123")
        ctx.store.set("nested", {"key": "value"})

        assert ctx.store.get("cdp_port") == 9222
        assert ctx.store.get("login_token") == "abc123"
        assert ctx.store.get("nonexistent", "default") == "default"
        assert ctx.store.has("nested")
        assert not ctx.store.has("missing")

        ctx.store.remove("login_token")
        assert not ctx.store.has("login_token")

        print("[PASS] test_case_context_global_store")


def test_case_context_snapshot_restore():
    from skiritai.core.case_context import CaseContext

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = CaseContext(case_dir=Path(tmpdir))
        asyncio.run(ctx.transition("setup"))
        ctx.store.set("cdp_port", 9222)
        ctx.browser.cdp_port = 9222
        ctx.browser.mode = "persistent"
        ctx.completed_steps.append("step1")

        # Save snapshot
        path = ctx.save_snapshot()
        assert path.exists()

        # Load snapshot
        loaded = CaseContext.load_snapshot(Path(tmpdir))
        assert loaded is not None
        assert loaded.store.get("cdp_port") == 9222
        assert loaded.browser.cdp_port == 9222
        assert loaded.browser.mode == "persistent"
        assert loaded.completed_steps == ["step1"]

        print("[PASS] test_case_context_snapshot_restore")


# ─── Test 2: Tool registration ────────────────────────────────────────

def test_tool_registration():
    """Check that both Playwright and perception tools are registered.

    Note: tools are registered on first import via @register_tool decorator.
    We check the current registry state (may include tools from other tests).
    """
    from skiritai.core.tool_registry import ToolRegistry

    # Ensure modules are imported (triggers registration)
    from skiritai.core.agent_loop import register_all_tools
    register_all_tools()

    registry = ToolRegistry()
    all_tools = registry.get_all()
    tool_names = [t.name for t in all_tools]

    # Check original tools
    for name in ["navigate", "click", "fill", "get_page_info", "screenshot"]:
        assert name in tool_names, f"Missing tool: {name}"

    # Check perception tools
    assert "page_perceive" in tool_names, "page_perceive not registered!"
    assert "find_element" in tool_names, "find_element not registered!"

    print(f"[PASS] test_tool_registration: {len(tool_names)} tools registered")
    print(f"       Tools: {tool_names}")


# ─── Test 3: Perception tools are in PERCEPTION_TOOLS set ─────────────

def test_perception_tools_set():
    from skiritai.core.agent_loop import PERCEPTION_TOOLS

    assert "page_perceive" in PERCEPTION_TOOLS
    assert "find_element" in PERCEPTION_TOOLS
    assert "navigate" not in PERCEPTION_TOOLS
    assert "click" not in PERCEPTION_TOOLS

    print(f"[PASS] test_perception_tools_set: {PERCEPTION_TOOLS}")


# ─── Test 4: Replay script generation ─────────────────────────────────

def test_replay_script_generation():
    from skiritai.core.script_generator import generate_replay_script

    # Simulate agent steps including perception tools
    agent_steps = [
        {"action": "page_perceive", "args": {}},  # should be filtered
        {"action": "find_element", "args": {"description": "search"}},  # filtered
        {"action": "navigate", "args": {"url": "https://www.baidu.com"}},
        {"action": "get_page_info", "args": {}},  # filtered
        {"action": "fill", "args": {"selector": "#kw", "text": "hello world"}},
        {"action": "click", "args": {"selector": "#su"}},
        {"action": "page_perceive", "args": {}},  # filtered
        {"action": "get_text", "args": {"selector": ".result"}},  # filtered
        {"action": "response", "content": "done"},  # filtered
    ]

    script = generate_replay_script("test_step", agent_steps)

    # Verify header
    assert "async def run(page, context):" in script
    assert "import asyncio" in script
    assert "from playwright.async_api import async_playwright" in script
    assert '__name__ == "__main__"' in script

    # Verify action steps are present
    assert 'page.goto("https://www.baidu.com"' in script
    assert 'page.locator("#kw").fill("hello world", force=True)' in script
    assert '_cdp_click' in script and '#su' in script

    # Verify perception steps are NOT present
    assert "page_perceive" not in script
    assert "find_element" not in script
    assert "get_page_info" not in script

    # Verify standalone execution block
    assert "asyncio.run(main())" in script

    print("[PASS] test_replay_script_generation")
    print("--- Generated script ---")
    print(script)
    print("--- End ---")


def test_replay_script_escape():
    """Test that special characters are properly escaped."""
    from skiritai.core.script_generator import generate_replay_script

    agent_steps = [
        {"action": "navigate", "args": {"url": "https://example.com/search?q=\"hello\"&page=1"}},
        {"action": "fill", "args": {"selector": "#input", "text": "line1\nline2"}},
        {"action": "eval_js", "args": {"expression": "document.querySelector(\"#btn\")"}},
    ]

    script = generate_replay_script("escape_test", agent_steps)

    # Verify the script is syntactically valid Python
    try:
        compile(script, "<replay_script>", "exec")
    except SyntaxError as e:
        print(f"[FAIL] Script has syntax error: {e}")
        print(script)
        raise

    print("[PASS] test_replay_script_escape: special chars handled correctly")


def test_replay_script_empty_steps():
    """Test that empty steps produce a valid pass script."""
    from skiritai.core.script_generator import generate_replay_script

    # Only perception/read-only steps → no action steps
    agent_steps = [
        {"action": "page_perceive", "args": {}},
        {"action": "get_page_info", "args": {}},
        {"action": "response", "content": "nothing to do"},
    ]

    script = generate_replay_script("empty_test", agent_steps)
    assert "pass" in script

    # Should still be valid Python
    compile(script, "<replay_script>", "exec")

    print("[PASS] test_replay_script_empty_steps")


# ─── Test 5: Replay script can actually run independently ──────────────

def test_replay_script_standalone():
    """Verify the generated script is importable and the run() function exists."""
    from skiritai.core.script_generator import generate_replay_script

    agent_steps = [
        {"action": "navigate", "args": {"url": "https://example.com"}},
        {"action": "click", "args": {"selector": "#btn"}},
    ]

    script = generate_replay_script("standalone_test", agent_steps)

    # Execute the script and check that `run` function is defined
    exec_globals: dict = {"__builtins__": __builtins__}
    exec(script, exec_globals)
    assert "run" in exec_globals
    assert callable(exec_globals["run"])
    assert "asyncio" in exec_globals  # imports are present
    assert "async_playwright" in exec_globals

    print("[PASS] test_replay_script_standalone: run() function importable")


# ─── Test 6: LLM API connectivity ─────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set (required for live API test)"
)
def test_llm_api():
    """Test that the configured LLM API is reachable (reads from .env)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,
        max_tokens=100,
    )

    response = llm.invoke("What is 2 plus 3? Reply with just the number.")
    content = response.content or ""
    assert content.strip(), f"Empty response from {model} at {base_url}"
    print(f"[PASS] test_llm_api: model={model}, response={content[:80]}")


# ─── Test 7: Agent loop builds correctly with perception tools ────────

@pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="No LLM API key configured (required for agent build)"
)
def test_agent_builds():
    """Verify agent can be built with all tools including perception."""
    from skiritai.core.agent_loop import build_agent

    agent = build_agent()
    assert agent is not None
    print("[PASS] test_agent_builds: agent created with perception tools")


# ─── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load .env if available
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
    else:
        # Fallback defaults for CI / bare environments
        os.environ.setdefault("OPENAI_API_KEY", "sk-gDh8e9b476b685f3a205804a830bbca09b1232379d8qL81r")
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
        os.environ.setdefault("LLM_MODEL", "gpt-5")
        os.environ.setdefault("LLM_PROVIDER", "openai")

    tests = [
        # Unit tests (no network needed)
        ("CaseContext state machine", test_case_context),
        ("CaseContext invalid transition", test_case_context_invalid_transition),
        ("CaseContext global store", test_case_context_global_store),
        ("CaseContext snapshot/restore", test_case_context_snapshot_restore),
        ("Tool registration", test_tool_registration),
        ("Perception tools set", test_perception_tools_set),
        ("Replay script generation", test_replay_script_generation),
        ("Replay script escape", test_replay_script_escape),
        ("Replay script empty steps", test_replay_script_empty_steps),
        ("Replay script standalone", test_replay_script_standalone),
        # Integration tests (needs network)
        ("LLM API connectivity", test_llm_api),
        ("Agent builds", test_agent_builds),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
