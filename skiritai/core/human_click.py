"""Human-like click using CDP Input.dispatchMouseEvent.

Produces events with isTrusted=true that are indistinguishable from real user
input at the browser level.  Combines:

- Bezier-curve mouse movement with acceleration / deceleration and micro-jitter
- Randomised target position within the element bounds
- Realistic down→up timing (50-150 ms)
- Slight coordinate drift between down and up events

Usage::

    from skiritai.core.human_click import human_click

    await human_click(page, "button#submit")
    await human_click(page, page.get_by_text("搜索"))
"""
from __future__ import annotations

import asyncio
import math
import random
from typing import Any

from skiritai.logger import logger

__all__ = ["human_click"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bezier_point(t: float, p0: float, p1: float, p2: float, p3: float) -> float:
    """Cubic Bezier interpolation at *t* ∈ [0, 1]."""
    u = 1 - t
    return u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3


async def _get_bounding_box(page: Any, locator: Any) -> dict[str, float] | None:
    """Return bounding box dict or None."""
    try:
        return await locator.bounding_box()
    except Exception:
        return None


async def _ensure_cdp_session(page: Any) -> Any:
    """Create (or retrieve) a CDP session for *page*."""
    return await page.context.new_cdp_session(page)


# ---------------------------------------------------------------------------
# Mouse movement
# ---------------------------------------------------------------------------

async def _move_mouse_bezier(
    cdp: Any,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> None:
    """Move the mouse from (start_x, start_y) to (end_x, end_y) along a
    randomised cubic Bezier curve with ease-in-out timing and micro-jitter.
    """
    distance = math.hypot(end_x - start_x, end_y - start_y)

    # Number of steps proportional to distance (≈ 1 step per 8 px)
    steps = max(12, int(distance / 8))

    # Random control-point offsets (30 % of distance)
    offset_x = (end_x - start_x) * random.uniform(-0.3, 0.3)
    offset_y = (end_y - start_y) * random.uniform(-0.3, 0.3)
    cp1_x = start_x + (end_x - start_x) * 0.25 + offset_x
    cp1_y = start_y + (end_y - start_y) * 0.25 + offset_y
    cp2_x = start_x + (end_x - start_x) * 0.75 - offset_x
    cp2_y = start_y + (end_y - start_y) * 0.75 - offset_y

    for i in range(steps + 1):
        t = i / steps
        # Ease-in-out: slow start, fast middle, slow end
        t_ease = t * t * (3 - 2 * t)

        x = _bezier_point(t_ease, start_x, cp1_x, cp2_x, end_x)
        y = _bezier_point(t_ease, start_y, cp1_y, cp2_y, end_y)

        # Add sub-pixel jitter except at endpoints
        if 0 < i < steps:
            x += random.gauss(0, 0.5)
            y += random.gauss(0, 0.5)

        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": round(x, 2),
            "y": round(y, 2),
        })

        # Variable delay: faster in the middle, slower at edges
        progress = abs(t - 0.5) * 2  # 0 at middle, 1 at edges
        base_delay = 1.5 + progress * 3.0  # 1.5–4.5 ms
        jitter = random.uniform(0.6, 1.4)
        await asyncio.sleep(base_delay * jitter / 1000)


# ---------------------------------------------------------------------------
# Human-like click
# ---------------------------------------------------------------------------

async def human_click(
    page: Any,
    locator_or_selector: Any,
    *,
    pre_move: bool = True,
) -> bool:
    """Perform a human-like click using CDP ``Input.dispatchMouseEvent``.

    Args:
        page: Playwright Page object.
        locator_or_selector: A Playwright Locator, or a CSS-selector string.
        pre_move: Whether to move the mouse along a Bezier curve before
            clicking (recommended; disable only for debugging).

    Returns:
        True if the click was dispatched, False if the element could not be
        found or had no bounding box.
    """
    # Resolve locator --------------------------------------------------------
    if isinstance(locator_or_selector, str):
        locator = page.locator(locator_or_selector)
    else:
        locator = locator_or_selector

    # Scroll into view first (without using Playwright's click)
    try:
        await locator.scroll_into_view_if_needed(timeout=5000)
    except Exception as e:
        logger.debug(f"[human_click] scroll_into_view failed: {e}")

    box = await _get_bounding_box(page, locator)
    if not box:
        logger.warning("[human_click] element has no bounding box, falling back to Playwright click")
        try:
            await locator.click(timeout=3000)
            return True
        except Exception:
            return False

    # Random target within element (avoid dead-centre; pick a random interior point)
    target_x = box["x"] + box["width"] * random.uniform(0.25, 0.75)
    target_y = box["y"] + box["height"] * random.uniform(0.25, 0.75)

    cdp = await _ensure_cdp_session(page)

    try:
        # 1. Move mouse along Bezier curve ----------------------------------
        if pre_move:
            # Start from a random-ish position (simulate current cursor)
            start_x = random.uniform(max(0, target_x - 300), target_x + 300)
            start_y = random.uniform(max(0, target_y - 200), target_y + 200)
            await _move_mouse_bezier(cdp, start_x, start_y, target_x, target_y)
        else:
            # Just jump
            await cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": round(target_x, 2),
                "y": round(target_y, 2),
            })

        # 2. Hesitate (human reaction time)
        await asyncio.sleep(random.uniform(0.05, 0.18))

        # 3. mousePressed (mousedown)
        down_x = target_x + random.gauss(0, 0.3)
        down_y = target_y + random.gauss(0, 0.3)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": round(down_x, 2),
            "y": round(down_y, 2),
            "button": "left",
            "clickCount": 1,
        })

        # 4. Human hold duration (50–150 ms)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # 5. mouseReleased (mouseup) — slight drift from down position
        up_x = down_x + random.gauss(0, 0.5)
        up_y = down_y + random.gauss(0, 0.5)
        await cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": round(up_x, 2),
            "y": round(up_y, 2),
            "button": "left",
            "clickCount": 1,
        })

        logger.info(
            f"[human_click] dispatched at ({target_x:.0f}, {target_y:.0f}) "
            f"element size {box['width']:.0f}×{box['height']:.0f}"
        )
        return True

    except Exception as e:
        logger.warning(f"[human_click] CDP click failed: {e}, falling back to Playwright click")
        try:
            await locator.click(timeout=3000)
            return True
        except Exception:
            return False
    finally:
        # Detach CDP session to avoid resource leak
        try:
            await cdp.detach()
        except Exception:
            pass
