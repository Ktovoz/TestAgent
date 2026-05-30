"""抖音精选搜索 — flow() 函数式 API。

用 flow() 函数式风格完成抖音搜索测试。
适合不需要 BaseCase 生命周期管理的场景。

运行：
    python examples/advanced/douyin_search/02_flow/demo.py
"""
import asyncio
from pathlib import Path

from skiritai import flow


async def main():
    async with flow(results_dir=Path("results/douyin_flow")) as ai:
        # 打开首页
        await ai.action("打开 https://www.douyin.com/jingxuan 首页")

        # 关闭登录弹窗
        await ai.action("关闭页面上的登录弹窗")
        await ai.screenshot("homepage")

        # 搜索
        await ai.action(
            "先仔细观察页面结构，找到顶部导航区搜索框旁边的搜索按钮"
            "（不是左侧导航栏的搜索链接），然后在搜索框中输入'陈伯全能王'，"
            "点击搜索按钮"
        )
        await ai.screenshot("search_result")

        # 验证登录弹窗
        result = await ai.verify("页面上弹出了登录弹窗或登录浮窗")
        print(f"登录弹窗验证: {'PASS' if result['passed'] else 'FAIL'} — {result['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
