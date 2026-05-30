"""抖音精选搜索 — BaseCase 类方式 + 环境变量 LLM 配置。

进阶示例，展示在复杂 SPA 页面上的测试能力：
- 抖音精选页面动态内容加载
- 登录弹窗处理
- 搜索框交互
- 搜索结果验证
- 生命周期钩子

运行：
    skiritai run examples/advanced/douyin_search/01_basecase
    python examples/advanced/douyin_search/01_basecase/case.py
"""
import asyncio
from pathlib import Path

from skiritai.core.base_case import BaseCase


class DouyinSearchCase(BaseCase):
    """抖音精选搜索测试用例。"""

    async def setup(self):
        await self.launch_browser()

    async def teardown(self):
        await self.close_browser()

    # ---- 步骤 1: 打开首页 ----

    async def open_homepage(self):
        """打开抖音精选首页。"""
        await self.ai.action("打开 https://www.douyin.com/jingxuan 首页")
        await self.ai.screenshot("01_homepage_with_popup")

    # ---- 步骤 2: 关闭弹窗 ----

    async def dismiss_popup(self):
        """关闭登录弹窗。"""
        await self.ai.action("关闭页面上的登录弹窗")
        await self.ai.screenshot("02_homepage_clean")

    # ---- 步骤 3: 输入搜索词 ----

    async def input_keyword(self):
        """在搜索框中输入搜索关键词。"""
        await self.ai.action("在页面顶部的搜索框中输入'陈伯全能王'")
        await self.ai.screenshot("03_keyword_typing")

    # ---- 步骤 4: 确认搜索词已输入 ----

    async def verify_keyword_input(self):
        """确认搜索框中已正确输入关键词。"""
        await self.ai.verify("页面顶部搜索框中显示文字'陈伯全能王'")
        await self.ai.screenshot("04_keyword_entered")

    # ---- 步骤 5: 点击搜索按钮 ----

    async def click_search(self):
        """点击搜索框旁边的搜索按钮。"""
        await self.ai.action(
            "先仔细观察页面结构，找到顶部导航区搜索框旁边的搜索按钮"
            "（不是左侧导航栏的搜索链接），然后点击它"
        )
        await self.ai.screenshot("05_after_click_search")

    # ---- 步骤 6: 验证登录弹窗 ----

    async def verify_login_popup(self):
        """验证点击搜索后弹出了登录浮窗。"""
        await self.ai.verify("页面上弹出了登录弹窗或登录浮窗")

    # ---- 钩子 ----

    async def before_step(self, step_name: str):
        print(f"\n{'='*40}")
        print(f"[DouyinSearch] >>> {step_name}")

    async def after_step(self, step_name: str, result: dict):
        status = "OK" if result.get("success") else "FAIL"
        elapsed = result.get("elapsed", 0)
        print(f"[DouyinSearch] <<< {step_name} [{status}] ({elapsed:.1f}s)")


if __name__ == "__main__":
    asyncio.run(DouyinSearchCase(case_dir=Path(__file__).parent).run())
