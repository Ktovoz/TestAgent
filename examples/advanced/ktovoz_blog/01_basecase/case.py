"""ktovoz.com 博客测试 — BaseCase 类方式 + 环境变量 LLM 配置。

进阶示例，展示 BaseCase 的完整能力：
- ai.action() 自然语言操作
- ai.screenshot() 关键截图
- ai.verify() AI 断言
- 生命周期钩子

运行：
    skiritai run examples/advanced/ktovoz_blog/01_basecase
    python examples/advanced/ktovoz_blog/01_basecase/case.py
"""
import asyncio
from pathlib import Path

from skiritai.core.base_case import BaseCase, FailurePolicy, on_failure


class KtovozBlogCase(BaseCase):
    """ktovoz.com 博客全链路测试。"""

    async def setup(self):
        await self.launch_browser()

    async def teardown(self):
        await self.close_browser()

    # ---- 首页发现 ----

    async def open_homepage(self):
        """打开博客首页。"""
        await self.ai.action("打开 https://ktovoz.com 首页")
        await self.ai.screenshot("homepage")

    async def verify_homepage(self):
        """验证首页内容。"""
        await self.ai.verify("页面标题包含博客相关文字")
        await self.ai.verify("导航栏可见，包含 Blog、About 等链接")

    async def discover_navigation(self):
        """探索页面导航结构。"""
        await self.ai.action(
            "分析当前页面的所有导航链接和页面入口，"
            "列出主导航有哪些页面、首页显示了几篇文章、有哪些标签或分类"
        )

    # ---- 文章详情 ----

    async def open_first_article(self):
        """打开第一篇文章。"""
        await self.ai.action("点击第一篇文章，进入文章详情页")
        await self.ai.screenshot("article_detail")

    async def verify_article(self):
        """验证文章详情页。"""
        await self.ai.verify("页面显示了文章标题、发布日期和正文内容")

    # ---- 导航与分类 ----

    async def visit_about_page(self):
        """访问 About 页面。"""
        await self.ai.action("返回首页，点击导航栏的 About 链接")
        await self.ai.verify("页面包含博主个人信息，如头像或简介")
        await self.ai.screenshot("about_page")

    async def explore_tags(self):
        """探索标签分类。"""
        await self.ai.action("返回首页")
        await self.ai.action(
            "点击任意一个标签或分类链接，进入标签页，"
            "报告选择的标签名称和该标签下的文章数量"
        )
        await self.ai.screenshot("tag_page")

    # ---- 搜索 ----

    async def use_search(self):
        """使用搜索功能。"""
        await self.ai.action("返回首页")
        await self.ai.action(
            "找到页面上的搜索按钮，点击后输入 'GCC' 并回车，"
            "确认搜索结果返回了与 GCC 相关的文章"
        )
        await self.ai.screenshot("search_result")

    # ---- 页脚与总结 ----

    @on_failure(FailurePolicy.SKIP)
    async def check_rss(self):
        """检查 RSS 订阅。"""
        await self.ai.action(
            "检查页面上是否有 RSS 订阅链接或邮件订阅入口，"
            "如果没有找到，直接说明并跳过"
        )

    async def verify_footer(self):
        """验证页脚。"""
        await self.ai.action("返回首页，滚动到页面最底部")
        await self.ai.verify("页脚区域包含版权信息")
        await self.ai.screenshot("footer")

    async def final_review(self):
        """总结报告。"""
        await self.ai.action(
            "基于当前页面和之前步骤的上下文，给出一份简洁的总结报告："
            "网站主题、主要页面、文章数量估算、标签分类覆盖、搜索功能状态"
        )

    # ---- 钩子 ----

    async def before_step(self, step_name: str):
        print(f"\n{'='*40}")
        print(f"[KtovozBlog] >>> {step_name}")

    async def after_step(self, step_name: str, result: dict):
        status = "OK" if result.get("success") else "FAIL"
        elapsed = result.get("elapsed", 0)
        print(f"[KtovozBlog] <<< {step_name} [{status}] ({elapsed:.1f}s)")


if __name__ == "__main__":
    asyncio.run(KtovozBlogCase(case_dir=Path(__file__).parent).run())
