"""ktovoz.com 博客测试 — flow() 函数式 API。

用 flow() 函数式风格完成博客测试。
适合不需要 BaseCase 生命周期管理的场景。

运行：
    python examples/advanced/ktovoz_blog/02_flow/demo.py
"""
import asyncio
from pathlib import Path

from skiritai import flow


async def main():
    async with flow(results_dir=Path("results/ktovoz_flow")) as ai:
        # 首页
        await ai.action("打开 https://ktovoz.com 首页")
        await ai.screenshot("homepage")
        await ai.verify("页面标题包含博客相关文字，导航栏可见")

        # 探索页面结构
        await ai.action(
            "列出当前页面的主导航链接、首页文章数量和标签分类"
        )

        # 文章详情
        await ai.action("点击第一篇文章进入详情页")
        await ai.screenshot("article_detail")
        await ai.verify("页面显示了文章标题、发布日期和正文内容")

        # About 页面
        await ai.action("返回首页，点击导航栏的 About 链接")
        await ai.screenshot("about_page")
        await ai.verify("页面包含博主个人信息")

        # 标签分类
        await ai.action("返回首页，点击任意标签或分类链接，报告标签名称和文章数量")
        await ai.screenshot("tag_page")

        # 搜索
        await ai.action("返回首页，找到搜索功能，搜索 'GCC'")
        await ai.screenshot("search_result")

        # 页脚
        await ai.action("返回首页，滚动到页面最底部")
        await ai.verify("页脚区域包含版权信息")
        await ai.screenshot("footer")


if __name__ == "__main__":
    asyncio.run(main())
