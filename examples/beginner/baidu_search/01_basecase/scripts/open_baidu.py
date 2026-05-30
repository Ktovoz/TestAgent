"""Auto-generated replay script — can be run independently."""
import asyncio
from playwright.async_api import async_playwright


async def run(page, context):
    await page.goto("https://www.baidu.com", wait_until="domcontentloaded")
    await page.wait_for_load_state("networkidle")


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
