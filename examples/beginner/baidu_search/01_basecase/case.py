"""百度搜索 — BaseCase 类方式 + 环境变量 LLM 配置。

最简单的入门方式：
- 继承 BaseCase，用 self.ai.action() 描述操作
- LLM 从 .env / 环境变量自动加载，零配置

运行：
    skiritai run examples/beginner/baidu_search/01_basecase
    python examples/beginner/baidu_search/01_basecase/case.py
"""
import asyncio
from pathlib import Path

from skiritai.core.base_case import BaseCase


class BaiduSearchCase(BaseCase):
    """百度搜索测试用例。"""

    async def setup(self):
        await self.launch_browser()

    async def teardown(self):
        await self.close_browser()

    async def open_baidu(self):
        """打开百度首页。"""
        await self.ai.action("打开 https://www.baidu.com，确认页面标题包含'百度'")

    async def search_keyword(self):
        """搜索关键词。"""
        await self.ai.action("在搜索框中输入'Playwright 自动化测试'，然后点击搜索按钮")

    async def verify_results(self):
        """验证搜索结果。"""
        await self.ai.verify("搜索结果页面包含与 Playwright 相关的内容")


if __name__ == "__main__":
    asyncio.run(BaiduSearchCase(case_dir=Path(__file__).parent).run())
