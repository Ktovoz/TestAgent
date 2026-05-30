"""百度搜索 — flow() 函数式 API。

展示如何用 flow() 函数式 API 编写测试。
适合不需要 BaseCase 生命周期管理的场景。
LLM 从 .env / 环境变量自动加载。

运行：
    python examples/beginner/baidu_search/02_flow/demo.py
"""
import asyncio
from pathlib import Path

from skiritai import flow


async def main():
    async with flow(results_dir=Path("results/baidu_flow")) as ai:
        await ai.action("打开百度首页 https://www.baidu.com")
        await ai.screenshot("homepage")

        await ai.action("在搜索框中输入'Playwright 自动化测试'，然后点击搜索按钮")
        await ai.screenshot("search_result")

        result = await ai.verify("搜索结果页面包含与 Playwright 相关的内容")
        print(f"验证结果: {'PASS' if result['passed'] else 'FAIL'} — {result['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
