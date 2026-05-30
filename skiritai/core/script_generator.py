"""Replay script generator — converts agent tool-call history into standalone Python scripts."""
from __future__ import annotations

import os

from skiritai.core.agent_loop import PERCEPTION_TOOLS
from skiritai.core.tools import _OVERLAY_REMOVAL_SELECTORS
from skiritai.logger import logger

# Read-only tools that should never appear in replay scripts
_READONLY_TOOLS = PERCEPTION_TOOLS | {"response"}


def generate_replay_script(step_id: str, agent_steps: list[dict]) -> str:
    """Generate a standalone replay script from agent's tool-calling history.

    The generated script:
    - Can be run independently: python <script>.py
    - Can be imported and run via: await run(page, context)
    - Filters out perception / read-only tools
    - Includes proper imports and a __main__ block
    - Escapes special characters in arguments
    """
    # Filter out read-only steps
    action_steps = [
        s for s in agent_steps
        if s.get("action") not in _READONLY_TOOLS
    ]

    action_lines = []
    for step in action_steps:
        action = step.get("action", "")
        args = step.get("args", {})
        line = _action_to_line(action, args)
        if line:
            action_lines.append(line)
        elif action not in _READONLY_TOOLS:
            logger.warning(
                f"[ScriptGen] Unrecognized action '{action}' — no replay line generated. "
                f"Add a handler in _action_to_line or consider adding to READONLY list."
            )

    if not action_lines:
        action_lines.append("    pass")

    # Resolve headless at generation time so replay scripts don't need os.getenv.
    # Same resolution order as browser.get_launch_args().
    headless = (
        os.getenv("SKIRITAI_HEADLESS") or os.getenv("HEADLESS", "false")
    ).lower() in ("true", "1", "yes")

    # Check if any action uses _cdp_click
    needs_cdp_click = any("_cdp_click" in line for line in action_lines)

    lines = [
        '"""Auto-generated replay script — can be run independently."""',
        "import asyncio",
        "from playwright.async_api import async_playwright",
    ]
    if needs_cdp_click:
        lines.extend([
            "",
            "",
            "async def _cdp_click(page, box):",
            '    """CDP-based trusted click — produces isTrusted=true events."""',
            "    import random",
            "    x = box['x'] + box['width'] * random.uniform(0.3, 0.7)",
            "    y = box['y'] + box['height'] * random.uniform(0.3, 0.7)",
            "    cdp = await page.context.new_cdp_session(page)",
            "    try:",
            '        await cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": round(x, 2), "y": round(y, 2)})',
            "        await asyncio.sleep(random.uniform(0.05, 0.1))",
            '        await cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": round(x, 2), "y": round(y, 2), "button": "left", "clickCount": 1})',
            "        await asyncio.sleep(random.uniform(0.05, 0.12))",
            '        await cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": round(x, 2), "y": round(y, 2), "button": "left", "clickCount": 1})',
            "    finally:",
            "        try:",
            "            await cdp.detach()",
            "        except Exception:",
            "            pass",
        ])
    lines.extend([
        "",
        "",
        "async def run(page, context):",
    ])
    lines.extend(action_lines)
    lines.extend([
        "",
        "",
        'if __name__ == "__main__":',
        "    async def main():",
        "        pw = await async_playwright().start()",
        f"        browser = await pw.chromium.launch(headless={headless})",
        "        ctx = await browser.new_context()",
        "        page = await ctx.new_page()",
        "        try:",
        "            await run(page, ctx)",
        "        finally:",
        "            await browser.close()",
        "            await pw.stop()",
        "",
        "    asyncio.run(main())",
    ])
    return "\n".join(lines)


def _action_to_line(action: str, args: dict) -> str | None:
    """Convert a single tool action to a Playwright Python line. Returns None if unrecognized."""
    if action == "navigate":
        url = _esc(args.get("url", ""))
        return '\n'.join([
            f'    await page.goto("{url}", wait_until="domcontentloaded")',
            f'    await page.wait_for_load_state("networkidle")',
        ])

    if action == "click":
        selector = _esc(args.get("selector", ""))
        return (
            f'    _loc = page.locator("{selector}")\n'
            f'    await _loc.scroll_into_view_if_needed(timeout=5000)\n'
            f'    _box = await _loc.bounding_box()\n'
            f'    if _box:\n'
            f'        await _cdp_click(page, _box)\n'
            f'    else:\n'
            f'        await _loc.click(force=True)'
        )

    if action == "click_text":
        text = _esc(args.get("text", ""))
        near = _esc(args.get("near", ""))
        if near:
            # Use proximity-based selection to pick the correct element
            return (
                f'    _matches = page.get_by_text("{text}", exact=False)\n'
                f'    _ref = page.locator("{near}").first\n'
                f'    _ref_box = await _ref.bounding_box()\n'
                f'    _count = await _matches.count()\n'
                f'    _loc = _matches.first\n'
                f'    if _ref_box and _count > 1:\n'
                f'        _best_d = 1e9\n'
                f'        _cx = _ref_box["x"] + _ref_box["width"] / 2\n'
                f'        _cy = _ref_box["y"] + _ref_box["height"] / 2\n'
                f'        for _i in range(_count):\n'
                f'            _b = await _matches.nth(_i).bounding_box()\n'
                f'            if _b:\n'
                f'                _d = ((_b["x"] + _b["width"]/2 - _cx)**2 + (_b["y"] + _b["height"]/2 - _cy)**2)**0.5\n'
                f'                if _d < _best_d:\n'
                f'                    _best_d = _d\n'
                f'                    _loc = _matches.nth(_i)\n'
                f'    await _loc.scroll_into_view_if_needed(timeout=5000)\n'
                f'    _box = await _loc.bounding_box()\n'
                f'    if _box:\n'
                f'        await _cdp_click(page, _box)\n'
                f'    else:\n'
                f'        await _loc.click(force=True)'
            )
        else:
            return (
                f'    _loc = page.get_by_text("{text}").first\n'
                f'    await _loc.scroll_into_view_if_needed(timeout=5000)\n'
                f'    _box = await _loc.bounding_box()\n'
                f'    if _box:\n'
                f'        await _cdp_click(page, _box)\n'
                f'    else:\n'
                f'        await _loc.click(force=True)'
            )

    if action == "click_force":
        return f'    await page.click("{_esc(args.get("selector", ""))}", force=True)'

    if action == "fill":
        return f'    await page.locator("{_esc(args.get("selector", ""))}").fill("{_esc(args.get("text", ""))}", force=True)'

    if action == "type_text":
        sel = _esc(args.get("selector", ""))
        txt = _esc(args.get("text", ""))
        return (
            f'    await page.locator("{sel}").click(force=True)\n'
            f'    await page.keyboard.press("Control+a")\n'
            f'    await page.keyboard.press("Backspace")\n'
            f'    await page.locator("{sel}").press_sequentially("{txt}", delay=50)'
        )

    if action == "focus":
        return f'    await page.locator("{_esc(args.get("selector", ""))}").focus()'

    if action == "wait_for":
        timeout = args.get("timeout", 5000)
        return f'    await page.wait_for_selector("{_esc(args.get("selector", ""))}", timeout={timeout})'

    if action == "scroll":
        direction = args.get("direction", "down")
        amount = args.get("amount", 500)
        y = amount if direction == "down" else -amount
        return f"    await page.mouse.wheel(0, {y})"

    if action == "eval_js":
        expr = args.get("expression", "")
        # Use repr() for safe embedding of arbitrary JS in Python source.
        # repr() handles all edge cases: quotes, backticks, newlines, etc.
        return f"    result = await page.evaluate({repr(expr)})"

    if action == "dismiss_overlay":
        selectors_js = ", ".join(f"'{s}'" for s in _OVERLAY_REMOVAL_SELECTORS)
        return (
            "    # Try Escape first\n"
            "    for _ in range(3):\n"
            "        await page.keyboard.press('Escape')\n"
            "        await asyncio.sleep(0.3)\n"
            "    # Then try clicking close buttons and removing overlays\n"
            f"    await page.evaluate(\"\"\"(() => {{\n"
            f"        const overlays = document.querySelectorAll({selectors_js});\n"
            "        for (const el of overlays) {\n"
            "            const s = window.getComputedStyle(el);\n"
            "            if (s.display === 'none' || s.visibility === 'hidden') continue;\n"
            "            const closeBtn = el.querySelector('[class*=\"close\"], [class*=\"dismiss\"], [aria-label*=\"close\" i], [aria-label*=\"关闭\"]');\n"
            "            if (closeBtn) { closeBtn.click(); continue; }\n"
            "            el.remove();\n"
            "        }\n"
            "        document.body.style.overflow = '';\n"
            "    })()\"\"\")\n"
            "    await asyncio.sleep(0.3)"
        )

    if action == "select_option":
        return f'    await page.select_option("{_esc(args.get("selector", ""))}", "{_esc(args.get("value", ""))}")'

    if action == "hover":
        return f'    await page.hover("{_esc(args.get("selector", ""))}")'

    if action == "wait":
        seconds = args.get("seconds", 1.0)
        return f"    await asyncio.sleep({seconds})"

    if action == "press_key":
        return f"    await page.keyboard.press({args.get('key', 'Enter')!r})"

    if action == "configure_browser":
        # Browser context config cannot be replayed directly (context is rebuilt).
        # Emit as a comment so the script is readable and someone can manually apply.
        parts = []
        for k in ("ignore_https_errors", "viewport", "user_agent", "locale",
                  "timezone_id", "color_scheme", "java_script_enabled"):
            v = args.get(k)
            if v is not None:
                parts.append(f"{k}={v!r}")
        return f"    # NOTE: configure_browser({', '.join(parts)}) was called during explore"

    return None


def _esc(s: str) -> str:
    """Escape a string for safe embedding in a double-quoted Python source literal.

    Handles: backslashes, double quotes, newlines, carriage returns, tabs.
    Note: backticks and ${} are not specially escaped because the string is
    embedded in Python double-quoted literals, not JS template literals.
    """
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
