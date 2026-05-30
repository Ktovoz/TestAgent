"""Auto-generated replay script — can be run independently."""
import asyncio
from playwright.async_api import async_playwright


async def _cdp_click(page, box):
    """CDP-based trusted click — produces isTrusted=true events."""
    import random
    x = box['x'] + box['width'] * random.uniform(0.3, 0.7)
    y = box['y'] + box['height'] * random.uniform(0.3, 0.7)
    cdp = await page.context.new_cdp_session(page)
    try:
        await cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": round(x, 2), "y": round(y, 2)})
        await asyncio.sleep(random.uniform(0.05, 0.1))
        await cdp.send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": round(x, 2), "y": round(y, 2), "button": "left", "clickCount": 1})
        await asyncio.sleep(random.uniform(0.05, 0.12))
        await cdp.send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": round(x, 2), "y": round(y, 2), "button": "left", "clickCount": 1})
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass


async def run(page, context):
    await page.locator("#chat-textarea").fill("Playwright 自动化测试", force=True)
    _loc = page.get_by_text("百度一下").first
    await _loc.scroll_into_view_if_needed(timeout=5000)
    _box = await _loc.bounding_box()
    if _box:
        await _cdp_click(page, _box)
    else:
        await _loc.click(force=True)


if __name__ == "__main__":
    async def main():
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        try:
            await run(page, ctx)
        finally:
            await browser.close()
            await pw.stop()

    asyncio.run(main())
