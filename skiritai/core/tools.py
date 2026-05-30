"""Playwright tools for LangGraph agent — text only, no vision."""
from __future__ import annotations

import asyncio
import contextvars
import json as _json
import tempfile
from pathlib import Path
from typing import Any

from skiritai.core.human_click import human_click
from skiritai.core.llm_retry import _is_navigation_error
from skiritai.core.tool_registry import register_tool
from skiritai.logger import logger

_page_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("_page_ctx", default=None)
_browser_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("_browser_ctx", default=None)
_context_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("_context_ctx", default=None)

# Callback invoked when configure_browser replaces the context/page.
_on_context_replaced: Any = None

# Track last interacted element for proximity-based disambiguation
_last_interacted_selector: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_last_interacted_selector", default=""
)

# ── Shared popup/overlay selector lists ──────────────────────────────────
# Used by _check_post_click_popup, click(), click_text(), type_text(),
# dismiss_overlay, and ai_context.py verify().  Single source of truth.

_POPUP_DETECT_SELECTORS = [
    '[id*="login"]', '[class*="login-popup"]', '[class*="login-modal"]',
    '[id*="modal"]', '[id*="popup"]', '[class*="overlay"]',
    '[class*="mask"]', '[class*="modal"]', '[class*="popup"]',
    '[class*="dialog"]', '[role="dialog"]',
    '[class*="Login"]', '[class*="semi-modal"]', '[class*="semi-dialog"]',
    '[class*="semi-popup"]', '[data-testid*="login"]', '[data-testid*="modal"]',
    '.login-guide', '.login-panel', '#login-guide', '#login-panel',
]

_OVERLAY_REMOVAL_SELECTORS = [
    '[id*="login"]', '[id*="modal"]', '[id*="popup"]',
    '[class*="overlay"]', '[class*="mask"]', '[class*="modal"]',
    '[class*="popup"]', '[class*="dialog"]', '[class*="interstitial"]',
    '[role="dialog"]',
]


def _popup_detect_js() -> str:
    """Return JS IIFE that returns descriptions of visible popup elements, or null."""
    selectors_js = ", ".join(f'\'{s}\'' for s in _POPUP_DETECT_SELECTORS)
    return (
        "(() => {"
        f"  const popupSelectors = [{selectors_js}];"
        "    const found = [];"
        "    for (const sel of popupSelectors) {"
        "        document.querySelectorAll(sel).forEach(el => {"
        "            const s = window.getComputedStyle(el);"
        "            if (s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0') {"
        "                const rect = el.getBoundingClientRect();"
        "                if (rect.width > 0 && rect.height > 0) {"
        "                    found.push('detected:' + (el.id || el.className?.toString().substring(0, 30)));"
        "                }"
        "            }"
        "        });"
        "    }"
        "    return found.length ? found.join('; ') : null;"
        "})()"
    )


def _overlay_removal_js() -> str:
    """Return JS IIFE that removes visible overlay/popup elements and restores body scroll."""
    selectors_js = ", ".join(f'\'{s}\'' for s in _OVERLAY_REMOVAL_SELECTORS)
    return (
        "(() => {"
        f"    const selectors = [{selectors_js}];"
        "    for (const sel of selectors) {"
        "        document.querySelectorAll(sel).forEach(el => {"
        "            const style = window.getComputedStyle(el);"
        "            if (style.display !== 'none' && style.visibility !== 'hidden') {"
        "                el.remove();"
        "            }"
        "        });"
        "    }"
        "    document.body.style.overflow = '';"
        "})()"
    )


async def _remove_overlays(page: Any) -> None:
    """Remove visible overlay/popup elements from the page."""
    await safe_evaluate(page, _overlay_removal_js())
    await asyncio.sleep(0.3)


def _record_interaction(selector: str) -> None:
    if selector:
        _last_interacted_selector.set(selector)


async def _check_post_click_popup(page: Any) -> str | None:
    """After a click, check if a popup (login, modal) appeared.

    Returns a description of what was found, or None.
    Only restores body scroll-lock; does NOT auto-close/remove popups so that
    subsequent verify() steps can assert their presence.
    """
    # Wait for popup to appear — some SPAs render popups asynchronously.
    # Early exit: stop checking once a popup is detected.
    for wait_ms in (500, 1000, 1500):
        await asyncio.sleep(wait_ms / 1000)
        try:
            result = await safe_evaluate(page, _popup_detect_js())
            if result:
                logger.info(f"[click] Post-click popup detected: {result}")
                return result
            # No popup detected after this round — no need to keep waiting
            # if the first check (500ms) found nothing.
            if wait_ms == 500:
                return None
        except Exception as e:
            logger.debug(f"[click] Post-click popup check failed: {e}")
    return None


def set_page(page: Any):
    """Set the active Playwright page for tools to use."""
    _page_ctx.set(page)


def set_browser(browser: Any, context: Any):
    """Set the Playwright Browser and BrowserContext references."""
    _browser_ctx.set(browser)
    _context_ctx.set(context)


def on_context_replaced(callback):
    """Register a callback to be called when configure_browser replaces context/page.

    The callback receives (new_context, new_page). Overwrites any previous callback.
    """
    global _on_context_replaced
    _on_context_replaced = callback


def get_page() -> Any:
    """Get the active Playwright page."""
    page = _page_ctx.get()
    if page is None:
        raise RuntimeError("Page not initialized. Call set_page() first.")
    return page


async def safe_evaluate(page: Any, expression: str, max_retries: int = 2, delay: float = 1.0) -> Any:
    """page.evaluate() with automatic retry on SPA navigation context destruction.

    Args:
        max_retries: 额外重试次数（不含首次尝试），默认 2 表示最多 3 次总尝试。
        delay: 基础重试延迟秒数，每次重试延迟 = delay * (attempt + 1)。
    """
    for attempt in range(max_retries + 1):
        try:
            return await page.evaluate(expression)
        except Exception as e:
            if attempt >= max_retries:
                raise
            if _is_navigation_error(e):
                logger.warning(f"[Tools] page.evaluate() failed due to navigation, "
                               f"retrying ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(delay * (attempt + 1))
            else:
                raise


@register_tool
async def navigate(url: str) -> str:
    """导航到指定 URL。

    Args:
        url: 目标 URL
    """
    page = get_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass
    # SPA 二次导航稳定：再等一轮 networkidle（较短超时）
    try:
        await page.wait_for_load_state("networkidle", timeout=3000)
    except Exception:
        pass
    return f"已导航到 {page.url}"


@register_tool
async def click(selector: str) -> str:
    """点击页面元素。使用 CSS 选择器定位元素。

    Args:
        selector: CSS 选择器，如 'button#submit', '.login-btn'
    """
    page = get_page()
    locator = page.locator(selector)
    try:
        ok = await human_click(page, locator)
        if not ok:
            raise RuntimeError(f"human_click returned False for '{selector}'")
    except Exception as e:
        err = str(e)
        if "intercept" in err.lower() or "timed out" in err.lower() or "no bounding box" in err.lower():
            logger.warning(f"[click] Click failed for '{selector}': removing overlays...")
            try:
                await _remove_overlays(page)
                ok = await human_click(page, locator)
                if not ok:
                    raise RuntimeError("human_click returned False after overlay removal")
                _record_interaction(selector)
                return f"已点击元素: {selector}（已自动移除遮挡元素）"
            except Exception:
                logger.warning(f"[click] Overlay removal didn't resolve, trying force click...")
                await locator.click(force=True, timeout=3000)
                _record_interaction(selector)
                return f"已点击元素: {selector}（已自动使用 force 模式）"
        raise
    _record_interaction(selector)
    # Check if click triggered a popup (e.g. login wall) — that means click succeeded
    popup = await _check_post_click_popup(page)
    if popup:
        return f"已点击元素: {selector}（点击触发了弹窗: {popup}）"
    return f"已点击元素: {selector}"


async def _resolve_text_locator(page, text: str, timeout: int, near: str = ""):
    """Find the best locator for a text match, using proximity when *near* is given."""
    if not near:
        near = _last_interacted_selector.get("")
    logger.info(f"[click_text] _resolve_text_locator: text='{text}', near='{near}', last_selector='{_last_interacted_selector.get('')}'")
    if near:
        matches = page.get_by_text(text, exact=False)
        try:
            await matches.first.wait_for(timeout=timeout or 5000)
        except Exception:
            raise
        count = await matches.count()
        if count <= 1:
            return matches.first
        ref_box = await page.locator(near).first.bounding_box()
        if not ref_box:
            logger.warning(f"[click_text] near element '{near}' has no bounding box, using first match")
            return matches.first
        ref_cx = ref_box["x"] + ref_box["width"] / 2
        ref_cy = ref_box["y"] + ref_box["height"] / 2
        best_idx = 0
        best_dist = float("inf")
        for i in range(count):
            box = await matches.nth(i).bounding_box()
            if box:
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                dist = ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        logger.info(f"[click_text] near='{near}': picked index {best_idx}/{count} "
                     f"(distance={best_dist:.0f}px) for text='{text}'")
        return matches.nth(best_idx)
    # No near — prefer button elements over links when multiple matches exist
    exact_matches = page.get_by_text(text, exact=True)
    partial_matches = page.get_by_text(text, exact=False)
    matches = exact_matches
    try:
        await matches.first.wait_for(timeout=min(timeout or 5000, 2000))
    except Exception:
        matches = partial_matches
        if timeout:
            await matches.first.wait_for(timeout=timeout)

    count = await matches.count()
    if count <= 1:
        return matches.first

    # Multiple matches: prefer <button> > [role="button"] > <a> > other
    # Also check ancestor elements since get_by_text often matches inner spans
    # When priority ties, prefer elements closer to visible input fields (spatial context)
    best_idx = 0
    best_priority = 99
    best_score = float("inf")
    for i in range(count):
        el = matches.nth(i)
        result = await el.evaluate("""el => {
            // Walk up to find the closest interactive ancestor (or self)
            let node = el;
            let priority = 3;
            for (let depth = 0; depth < 3 && node; depth++) {
                const tag = node.tagName.toLowerCase();
                const role = (node.getAttribute('role') || '').toLowerCase();
                const cursor = window.getComputedStyle(node).cursor;
                if (tag === 'button') { priority = 0; break; }
                if (role === 'button') { priority = 1; break; }
                if (tag === 'a') { priority = 2; break; }
                if (cursor === 'pointer') { priority = 1.5; break; }
                node = node.parentElement;
            }
            // Spatial context: distance to nearest visible input field
            const elRect = el.getBoundingClientRect();
            const elCx = elRect.x + elRect.width / 2;
            const elCy = elRect.y + elRect.height / 2;
            let minDist = 99999;
            document.querySelectorAll('input, textarea').forEach(inp => {
                const s = window.getComputedStyle(inp);
                if (s.display === 'none' || s.visibility === 'hidden') return;
                const r = inp.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) return;
                const dx = (r.x + r.width / 2) - elCx;
                const dy = (r.y + r.height / 2) - elCy;
                const d = Math.sqrt(dx * dx + dy * dy);
                if (d < minDist) minDist = d;
            });
            return JSON.stringify({priority, dist: minDist});
        }""")
        try:
            info = _json.loads(result)
            priority = info["priority"]
            dist = info.get("dist", 99999)
        except Exception:
            priority = 3
            dist = 99999

        # Better = lower priority; on ties, prefer closer to an input field
        if priority < best_priority or (priority == best_priority and dist < best_score):
            best_priority = priority
            best_score = dist
            best_idx = i
    logger.info(f"[click_text] no near: picked index {best_idx}/{count} "
                 f"(priority={best_priority}, input_dist={best_score:.0f}px) for text='{text}'")
    return matches.nth(best_idx)


@register_tool
async def click_text(text: str, timeout: int = 5000, near: str = "") -> str:
    """通过可见文本点击元素。不需要知道 CSS 选择器，直接根据页面上显示的文字来点击。

    适用场景：点击按钮、链接、菜单项等。会匹配包含该文本的第一个可见元素。
    当页面上有多个同名元素时，使用 near 参数指定参考元素的 selector，
    工具会自动选择距离参考元素最近的匹配项。

    Args:
        text: 页面上可见的文字内容，如 '登录'、'GCC Installation'
        timeout: 等待元素出现的超时时间（毫秒），默认 5000。设为 0 表示不超时。
        near: 参考元素的 CSS 选择器，用于就近定位。当有多个同名元素时选择距离此元素最近的。
    """
    page = get_page()
    locator = await _resolve_text_locator(page, text, timeout, near)
    try:
        ok = await human_click(page, locator)
        if not ok:
            raise RuntimeError(f"human_click returned False for text '{text}'")
    except Exception as e:
        err = str(e)
        if "intercept" in err.lower() or "timed out" in err.lower() or "no bounding box" in err.lower():
            logger.warning(f"[click_text] Click failed for '{text}': removing overlays...")
            try:
                await _remove_overlays(page)
                ok = await human_click(page, locator)
                if not ok:
                    raise RuntimeError("human_click returned False after overlay removal")
                return f"已点击文本为 '{text}' 的元素（已自动移除遮挡元素）"
            except Exception:
                logger.warning(f"[click_text] Overlay removal didn't resolve, trying force click...")
                await locator.click(force=True, timeout=3000)
                return f"已点击文本为 '{text}' 的元素（已自动使用 force 模式）"
        raise
    popup = await _check_post_click_popup(page)
    if popup:
        return f"已点击文本为 '{text}' 的元素（点击触发了弹窗: {popup}）"
    return f"已点击文本为 '{text}' 的元素"


@register_tool
async def click_force(selector: str) -> str:
    """强制点击元素（即使元素不可见）。适用于被遮挡或隐藏的元素。

    Args:
        selector: CSS 选择器
    """
    page = get_page()
    await page.locator(selector).click(force=True)
    return f"已强制点击元素: {selector}"


@register_tool
async def fill(selector: str, text: str) -> str:
    """在输入框中填写文本。要求元素可见。

    Args:
        selector: 输入框的 CSS 选择器
        text: 要填写的文本内容
    """
    page = get_page()
    await page.locator(selector).fill(text, force=True)
    _record_interaction(selector)
    return f"已在 {selector} 中填写: {text}"


@register_tool
async def type_text(selector: str, text: str) -> str:
    """逐字符输入文本到元素。适用于 fill 无效的动态输入框（如 React 受控组件）。

    通过模拟真实键盘输入逐字符触发事件，对 React/Vue 等框架的受控组件兼容性更好。
    注意：会先全选并删除已有内容（Ctrl+A → Backspace），再逐字符输入新文本。

    Args:
        selector: 输入框的 CSS 选择器
        text: 要输入的文本内容
    """
    page = get_page()
    locator = page.locator(selector)
    # Click to focus — handle popups that might block the click
    try:
        await locator.click(timeout=5000)
    except Exception as e:
        err = str(e).lower()
        if "intercept" in err or "timed out" in err:
            logger.warning(f"[type_text] Click blocked, removing overlays...")
            await _remove_overlays(page)
            await locator.click(force=True, timeout=3000)
        else:
            raise
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Backspace")
    await locator.press_sequentially(text, delay=50)
    _record_interaction(selector)
    return f"已在 {selector} 中输入: {text}"


@register_tool
async def focus(selector: str) -> str:
    """聚焦到指定元素。

    Args:
        selector: 元素的 CSS 选择器
    """
    page = get_page()
    await page.locator(selector).focus()
    return f"已聚焦到 {selector}"


@register_tool
async def get_text(selector: str) -> str:
    """获取指定元素的文本内容。

    Args:
        selector: 元素的 CSS 选择器
    """
    page = get_page()
    text = await page.locator(selector).text_content()
    return f"元素文本: {text}"


@register_tool
async def wait_for(selector: str, timeout: int = 5000) -> str:
    """等待指定元素出现。

    Args:
        selector: 等待出现的元素 CSS 选择器
        timeout: 超时时间（毫秒），默认 5000
    """
    page = get_page()
    await page.locator(selector).wait_for(timeout=timeout)
    return f"元素已出现: {selector}"


@register_tool
async def wait(seconds: float = 1.0) -> str:
    """等待指定的秒数。用于页面加载、动画完成、AJAX 请求等场景。

    Args:
        seconds: 等待的秒数，默认 1.0
    """
    import asyncio
    await asyncio.sleep(seconds)
    return f"已等待 {seconds} 秒"


@register_tool
async def scroll(direction: str, amount: int = 500) -> str:
    """滚动页面。

    Args:
        direction: 滚动方向，'up' 或 'down'
        amount: 滚动像素数，默认 500
    """
    page = get_page()
    y = amount if direction == "down" else -amount
    await page.mouse.wheel(0, y)
    return f"已向{direction}滚动 {amount}px"


@register_tool
async def get_page_info() -> str:
    """获取当前页面的标题、URL 和页面文本摘要。"""
    page = get_page()
    title = await page.title()
    url = page.url
    body_text = await safe_evaluate(page, "document.body?.innerText?.substring(0, 2000) || ''")
    return f"标题: {title}\nURL: {url}\n页面文本:\n{body_text}"


@register_tool
async def eval_js(expression: str) -> str:
    """在页面中执行 JavaScript 表达式并返回结果。

    Args:
        expression: 要执行的 JS 表达式
    """
    page = get_page()
    result = await safe_evaluate(page, expression)
    return f"JS 执行结果: {result}"


@register_tool
async def select_option(selector: str, value: str) -> str:
    """在下拉选择框中选择选项。

    Args:
        selector: select 元素的 CSS 选择器
        value: 要选择的选项值
    """
    page = get_page()
    await page.locator(selector).select_option(value)
    return f"已选择 {value} in {selector}"


@register_tool
async def hover(selector: str) -> str:
    """鼠标悬停在元素上。

    Args:
        selector: 元素的 CSS 选择器
    """
    page = get_page()
    await page.locator(selector).hover()
    return f"已悬停在 {selector}"


@register_tool
async def analyze_page() -> str:
    """分析当前页面的 DOM 结构，返回所有可交互元素的详细信息。

    会穿透 Shadow DOM 查找所有嵌套元素。当 AI 无法找到目标元素时，
    必须使用此工具获取页面的真实 DOM 结构，而不是依赖训练数据中已知的选择器。

    返回值是 JSON 格式，按页面区域（layout）分组展示：
    - layout: 页面布局概览，按区域列出各类元素数量
    - areas: 按区域分组的详细元素列表，每个元素含 area、nearby、position 等上下文
    """
    page = get_page()
    return await safe_evaluate(page, """
        (() => {
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            const isVisible = (el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' &&
                       style.visibility !== 'hidden' &&
                       style.opacity !== '0' &&
                       rect.width > 0 && rect.height > 0;
            };

            // Build a unique selector: id > name > nth-child path
            const mkSelector = (el) => {
                if (el.id) return '#' + CSS.escape(el.id);
                if (el.name) return '[name="' + el.name.replace(/"/g, '\\\\"') + '"]';
                const parts = [];
                let cur = el;
                while (cur && cur !== document.body && cur !== document.documentElement) {
                    const parent = cur.parentElement;
                    if (!parent) break;
                    if (cur.id) {
                        parts.unshift('#' + CSS.escape(cur.id));
                        break;
                    }
                    const tag = cur.tagName.toLowerCase();
                    const idx = [...parent.children].indexOf(cur) + 1;
                    let seg = tag;
                    if (cur.className && typeof cur.className === 'string') {
                        const cls = cur.className.trim().split(/\\s+/)[0];
                        if (cls && !cls.match(/^[0-9]/)) {
                            seg += '.' + CSS.escape(cls);
                        }
                    }
                    if (parent.children.length > 1) {
                        seg += ':nth-child(' + idx + ')';
                    }
                    parts.unshift(seg);
                    cur = parent;
                }
                return parts.join(' > ') || el.tagName.toLowerCase();
            };

            // Position description: "顶部居中", "左侧", etc.
            const describePosition = (rect) => {
                const cx = rect.x + rect.width / 2;
                const cy = rect.y + rect.height / 2;
                const hPos = cx < vw * 0.33 ? '左侧' : cx > vw * 0.66 ? '右侧' : '居中';
                const vPos = cy < vh * 0.33 ? '顶部' : cy > vh * 0.66 ? '底部' : '中部';
                return vPos + hPos;
            };

            // Semantic area: determine which page region the element belongs to
            const getArea = (el) => {
                let p = el.parentElement;
                while (p && p !== document.body && p !== document.documentElement) {
                    const tag = p.tagName.toLowerCase();
                    if (tag === 'header' || tag === 'nav') return '顶部导航区';
                    if (tag === 'aside') return '侧边栏';
                    if (tag === 'footer') return '底部区域';
                    if (tag === 'main') return '主内容区';
                    if (tag === 'form') return '表单区';
                    const role = (p.getAttribute('role') || '').toLowerCase();
                    if (role === 'navigation') return '顶部导航区';
                    if (role === 'sidebar' || role === 'complementary') return '侧边栏';
                    if (role === 'banner') return '顶部导航区';
                    if (role === 'main') return '主内容区';
                    if (role === 'search') return '搜索区';
                    const cls = (typeof p.className === 'string' ? p.className.toLowerCase() : '');
                    const eid = (p.id || '').toLowerCase();
                    const combined = cls + ' ' + eid;
                    if (/sidebar|side-bar|aside|left-panel|left-nav/.test(combined)) return '侧边栏';
                    if (/header|topbar|top-bar|navbar|nav-bar/.test(combined)) return '顶部导航区';
                    if (/footer|bottom-bar/.test(combined)) return '底部区域';
                    if (/searchbar|search-bar|searchbox|search-box|search-form/.test(combined)) return '搜索区';
                    if (/modal|dialog|popup|overlay/.test(combined)) return '弹窗';
                    p = p.parentElement;
                }
                return '';
            };

            // Describe neighboring elements for spatial context
            const getNearby = (el) => {
                const parts = [];
                // Previous sibling
                let prev = el.previousElementSibling;
                while (prev && parts.length < 2) {
                    const t = (prev.textContent || '').trim().substring(0, 30);
                    if (t && prev.tagName.toLowerCase() !== 'script') {
                        parts.push('前邻:' + prev.tagName.toLowerCase() + '("' + t + '")');
                        break;
                    }
                    prev = prev.previousElementSibling;
                }
                // Next sibling
                let next = el.nextElementSibling;
                while (next && parts.length < 2) {
                    const t = (next.textContent || '').trim().substring(0, 30);
                    if (t && next.tagName.toLowerCase() !== 'script') {
                        parts.push('后邻:' + next.tagName.toLowerCase() + '("' + t + '")');
                        break;
                    }
                    next = next.nextElementSibling;
                }
                // Same form inputs
                const form = el.closest('form');
                if (form) {
                    const formInputs = form.querySelectorAll('input, textarea, select');
                    const inputDescs = [];
                    formInputs.forEach(inp => {
                        if (inp !== el) {
                            const ph = (inp.placeholder || inp.name || inp.type || '').substring(0, 30);
                            if (ph) inputDescs.push(inp.tagName.toLowerCase() + '("' + ph + '")');
                        }
                    });
                    if (inputDescs.length > 0) parts.push('同表单: ' + inputDescs.slice(0, 3).join(', '));
                }
                // Parent-level input neighbors
                const parent = el.parentElement;
                if (parent) {
                    const siblingInputs = parent.querySelectorAll('input, textarea');
                    siblingInputs.forEach(inp => {
                        if (inp !== el) {
                            const ph = (inp.placeholder || inp.name || inp.type || '').substring(0, 30);
                            if (ph) parts.push('紧邻输入框: input("' + ph + '")');
                        }
                    });
                }
                return parts.join('; ');
            };

            // Collect from both light DOM and shadow DOM
            const collectAll = (root, selector) => {
                const results = [...root.querySelectorAll(selector)];
                const allElements = root.querySelectorAll('*');
                for (const el of allElements) {
                    if (el.shadowRoot) {
                        results.push(...collectAll(el.shadowRoot, selector));
                    }
                }
                return results;
            };

            const inputs = collectAll(document, 'input, textarea, select')
                .filter(isVisible)
                .slice(0, 30)
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        type: el.type || '',
                        name: el.name || '',
                        id: el.id || '',
                        placeholder: (el.placeholder || '').substring(0, 60),
                        selector: mkSelector(el),
                        position: describePosition(rect),
                        area: getArea(el),
                        nearby: getNearby(el),
                        in_shadow_dom: el.getRootNode() !== document
                    };
                });

            const buttons = collectAll(document, 'button, input[type="submit"], input[type="button"], [role="button"]')
                .filter(isVisible)
                .map(el => ({el, isButton: true}));

            // Also find clickable non-button elements (span, div, etc.) that look like buttons
            // These are elements with cursor:pointer, tabindex, onclick, or button-like classes
            const clickableCandidates = collectAll(document,
                'span, div, li, [tabindex], [onclick], [role="link"]'
            ).filter(isVisible).filter(el => {
                // Skip if already in the buttons collection
                if (/^(button|input|select|textarea|a)$/i.test(el.tagName)) return false;
                // Skip tiny/invisible text
                const t = (el.textContent || '').trim();
                if (!t || t.length > 80) return false;
                // Skip if it's just a wrapper with many children (layout div)
                if (el.children.length > 5) return false;
                // Detect clickable signals
                const style = window.getComputedStyle(el);
                const cursor = style.cursor;
                const tabindex = el.getAttribute('tabindex');
                const onclick = el.getAttribute('onclick');
                const role = (el.getAttribute('role') || '').toLowerCase();
                const cls = (typeof el.className === 'string' ? el.className.toLowerCase() : '');
                // Is clickable if any signal matches
                return cursor === 'pointer'
                    || tabindex !== null
                    || onclick !== null
                    || role === 'button' || role === 'link'
                    || /btn|button|clickable|action|tab|item|menu|nav/.test(cls);
            }).map(el => ({el, isButton: false}));

            // Merge and deduplicate (by DOM element reference)
            const allClickable = [...buttons];
            const seenEls = new Set(buttons.map(b => b.el));
            clickableCandidates.forEach(c => {
                // Avoid adding an element whose ancestor is already in the list
                if (!seenEls.has(c.el)) {
                    let dominated = false;
                    for (const s of seenEls) {
                        if (s.contains(c.el) || c.el.contains(s)) { dominated = true; break; }
                    }
                    if (!dominated) {
                        allClickable.push(c);
                        seenEls.add(c.el);
                    }
                }
            });

            const mergedButtons = allClickable
                .slice(0, 30)
                .map(({el}) => {
                    const rect = el.getBoundingClientRect();
                    const hasSvg = el.querySelector('svg') !== null || (el.shadowRoot && el.shadowRoot.querySelector('svg') !== null);
                    const btnText = (el.textContent || el.value || '').trim().substring(0, 60);
                    return {
                        tag: el.tagName.toLowerCase(),
                        text: btnText,
                        selector: mkSelector(el),
                        has_svg: hasSvg,
                        aria_label: (el.getAttribute('aria-label') || '').substring(0, 80),
                        title: (el.getAttribute('title') || '').substring(0, 80),
                        position: describePosition(rect),
                        area: getArea(el),
                        nearby: getNearby(el),
                        in_shadow_dom: el.getRootNode() !== document
                    };
                });

            const links = collectAll(document, 'a[href]')
                .filter(isVisible)
                .slice(0, 20)
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        text: (el.textContent || '').trim().substring(0, 60),
                        href: el.href,
                        selector: mkSelector(el),
                        position: describePosition(rect),
                        area: getArea(el),
                        in_shadow_dom: el.getRootNode() !== document
                    };
                });

            // Disambiguation: describe context for duplicate-text elements
            const addDisambiguation = (items, textField = 'text') => {
                const textCounts = {};
                items.forEach(item => {
                    const key = (item[textField] || '').trim();
                    textCounts[key] = (textCounts[key] || 0) + 1;
                });
                items.forEach(item => {
                    const key = (item[textField] || '').trim();
                    if (textCounts[key] > 1) {
                        const parts = ['页面中有' + textCounts[key] + '个"' + key + '"'];
                        if (item.area) parts.push('此元素在' + item.area);
                        if (item.nearby) parts.push(item.nearby);
                        if (!item.area && item.position) parts.push('位于' + item.position);
                        item.note = parts.join('，');
                    }
                });
            };
            addDisambiguation(mergedButtons, 'text');
            addDisambiguation(links, 'text');

            // Build area-grouped structure for clearer spatial reasoning
            const areaMap = {};
            const addToArea = (area, category, items) => {
                if (!items.length) return;
                if (!areaMap[area]) areaMap[area] = [];
                items.forEach(item => {
                    areaMap[area].push(Object.assign({category}, item));
                });
            };
            const seenAreas = new Set();
            inputs.forEach(el => { if (el.area) seenAreas.add(el.area); });
            mergedButtons.forEach(el => { if (el.area) seenAreas.add(el.area); });
            links.forEach(el => { if (el.area) seenAreas.add(el.area); });

            const layout = {};
            seenAreas.forEach(area => {
                const areaInputs = inputs.filter(el => el.area === area);
                const areaButtons = mergedButtons.filter(el => el.area === area);
                const areaLinks = links.filter(el => el.area === area);
                const parts = [];
                if (areaInputs.length) parts.push(areaInputs.length + '个输入框');
                if (areaButtons.length) parts.push(areaButtons.length + '个按钮');
                if (areaLinks.length) parts.push(areaLinks.length + '个链接');
                layout[area] = parts.join('，');
                addToArea(area, 'input', areaInputs);
                addToArea(area, 'button', areaButtons);
                addToArea(area, 'link', areaLinks);
            });
            // Elements without area
            const noAreaInputs = inputs.filter(el => !el.area);
            const noAreaButtons = mergedButtons.filter(el => !el.area);
            const noAreaLinks = links.filter(el => !el.area);
            if (noAreaInputs.length || noAreaButtons.length || noAreaLinks.length) {
                addToArea('其他区域', 'input', noAreaInputs);
                addToArea('其他区域', 'button', noAreaButtons);
                addToArea('其他区域', 'link', noAreaLinks);
                const parts = [];
                if (noAreaInputs.length) parts.push(noAreaInputs.length + '个输入框');
                if (noAreaButtons.length) parts.push(noAreaButtons.length + '个按钮');
                if (noAreaLinks.length) parts.push(noAreaLinks.length + '个链接');
                layout['其他区域'] = parts.join('，');
            }

            return JSON.stringify({
                url: location.href,
                title: document.title,
                layout,
                areas: areaMap,
                counts: {inputs: inputs.length, buttons: mergedButtons.length, links: links.length}
            }, null, 2);
        })()
    """)


@register_tool
async def dismiss_overlay() -> str:
    """检测并关闭页面上的遮挡元素（登录弹窗、对话框、广告浮层等）。

    按优先级依次尝试：
    1. 查找弹窗内的关闭按钮并点击
    2. 按 Escape 键关闭
    3. 移除常见遮罩层 DOM 元素

    适用于：登录弹窗、广告弹窗、Cookie 提示、引导遮罩等。
    """
    page = get_page()

    js = """
    (() => {
        const overlays = document.querySelectorAll(
            '[role="dialog"], [id*="login"], [id*="modal"], [class*="overlay"], ' +
            '[class*="mask"], [class*="popup"], [class*="modal"], [class*="dialog"], ' +
            '[class*="interstitial"], [id*="popup"]'
        );
        let closed = [];
        let removed = [];

        for (const el of overlays) {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden') continue;
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) continue;

            // Try close button first
            const closeBtn = el.querySelector(
                '[class*="close"], [class*="dismiss"], [aria-label*="close" i], ' +
                '[aria-label*="关闭"], button'
            );
            if (closeBtn) {
                const text = (closeBtn.textContent || '').trim();
                if (text === '×' || text === '✕' || text === 'X' ||
                    /close|关闭|取消|skip|跳过/i.test(text) ||
                    /close|关闭/i.test(closeBtn.getAttribute('aria-label') || '')) {
                    closeBtn.click();
                    closed.push(el.id || el.className.substring(0, 40));
                    continue;
                }
            }
            // No close button found, remove from DOM
            el.remove();
            removed.push(el.id || el.className.substring(0, 40));
        }

        // Clean up body scroll lock
        if (document.body.style.overflow === 'hidden') {
            document.body.style.overflow = '';
        }

        return JSON.stringify({closed, removed});
    })()
    """

    try:
        result = await safe_evaluate(page, js)
        info = _json.loads(result)
        parts = []
        if info["closed"]:
            parts.append(f"已点击关闭按钮: {info['closed']}")
        if info["removed"]:
            parts.append(f"已移除遮罩元素: {info['removed']}")
        if not parts:
            return "未检测到遮挡元素"
        return "；".join(parts)
    except Exception as e:
        logger.warning(f"[dismiss_overlay] JS detection failed: {e}, falling back to Escape")
        await page.keyboard.press("Escape")
        return "JS 检测失败，已按 Escape 键尝试关闭"


@register_tool
async def press_key(key: str) -> str:
    """按下键盘按键。用于提交表单、关闭弹窗等。

    常用按键：'Enter'（提交搜索/表单）、'Escape'（关闭弹窗）、'Tab'（切换焦点）、'Backspace'、'Delete' 等。
    也可以通过 selector + focus 先聚焦到某个元素，再 press_key('Enter') 来触发该元素的键盘事件。

    Args:
        key: 按键名称，如 'Enter', 'Escape', 'Tab', 'Backspace', 'ArrowDown', 'a', 'Control+A' 等
    """
    if len(key) > 50:
        raise ValueError(f"按键名称过长: {len(key)} 字符，超过 50 字符上限")
    page = get_page()
    await page.keyboard.press(key)
    return f"已按下按键: {key}"


@register_tool
async def screenshot(name: str = "screenshot") -> str:
    """截取当前页面的截图并保存。

    Args:
        name: 截图文件名（不含扩展名），默认 'screenshot'
    """
    page = get_page()
    path = str(Path(tempfile.gettempdir()) / f"testagent_{name}.png")
    # full_page=False: only capture visible viewport — avoids timeout on pages
    # with lazy-loaded content below the fold and font-loading waits.
    await page.screenshot(path=path, full_page=False, timeout=15000)
    return f"截图已保存到 {path}"


@register_tool
async def configure_browser(
    ignore_https_errors: bool | None = None,
    viewport: str | None = None,
    user_agent: str | None = None,
    locale: str | None = None,
    timezone_id: str | None = None,
    color_scheme: str | None = None,
    java_script_enabled: bool | None = None,
) -> str:
    """修改浏览器配置。会重建浏览器上下文并保留当前页面状态（cookies、URL）。

    遇到 SSL/证书错误时，调用 configure_browser(ignore_https_errors=True) 后重试导航。
    需要切换语言、时区、User-Agent 等时也可以使用此工具。

    Args:
        ignore_https_errors: 是否忽略 SSL 证书错误
        viewport: 视口大小，格式为 '宽x高'，如 '1920x1080'
        user_agent: 自定义 User-Agent 字符串
        locale: 浏览器语言，如 'zh-CN', 'en-US'
        timezone_id: 时区，如 'Asia/Shanghai'
        color_scheme: 配色方案 'light' 或 'dark'
        java_script_enabled: 是否启用 JavaScript
    """
    browser = _browser_ctx.get()
    old_context = _context_ctx.get()
    old_page = _page_ctx.get()

    if browser is None or old_context is None:
        return "错误：浏览器尚未初始化"

    # Build new context options
    ctx_options: dict[str, Any] = {}
    if ignore_https_errors is not None:
        ctx_options["ignore_https_errors"] = ignore_https_errors
    if viewport:
        parts = viewport.lower().split("x")
        if len(parts) == 2:
            try:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    ctx_options["viewport"] = {"width": w, "height": h}
                else:
                    return f"错误：视口尺寸必须为正数，收到 {w}x{h}"
            except ValueError:
                return f"错误：视口格式无效 '{viewport}'，应为 '宽x高' 如 '1920x1080'"
    if user_agent:
        ctx_options["user_agent"] = user_agent
    if locale:
        ctx_options["locale"] = locale
    if timezone_id:
        ctx_options["timezone_id"] = timezone_id
    if color_scheme:
        if color_scheme not in ("light", "dark", "no-preference"):
            return f"错误：color_scheme 必须为 'light'、'dark' 或 'no-preference'，收到 '{color_scheme}'"
        ctx_options["color_scheme"] = color_scheme
    if java_script_enabled is not None:
        ctx_options["java_script_enabled"] = java_script_enabled

    if not ctx_options:
        return "未指定任何配置项，无需修改"

    # Save current state
    current_url = old_page.url if old_page else "about:blank"
    try:
        storage = await old_context.storage_state()
        cookies = storage.get("cookies", [])
    except Exception as e:
        logger.warning(f"[configure_browser] Failed to save storage state: {e}")
        cookies = []

    # Create new context with updated settings
    new_context = await browser.new_context(**ctx_options)

    # Restore cookies
    if cookies:
        try:
            await new_context.add_cookies(cookies)
        except Exception as e:
            logger.warning(f"[configure_browser] Failed to restore cookies: {e}")

    # Create new page and navigate
    new_page = await new_context.new_page()
    nav_failed = False
    if current_url and current_url != "about:blank":
        try:
            await new_page.goto(current_url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            logger.warning(f"[configure_browser] Failed to navigate to {current_url}: {e}")
            nav_failed = True

    # Close old context
    try:
        await old_context.close()
    except Exception as e:
        logger.warning(f"[configure_browser] Failed to close old context: {e}")

    # Update global references
    _context_ctx.set(new_context)
    _page_ctx.set(new_page)

    # Notify BrowserSession to sync internal refs
    if _on_context_replaced is not None:
        try:
            _on_context_replaced(new_context, new_page)
        except Exception as e:
            logger.warning(f"[configure_browser] context_replaced callback failed: {e}")

    changed = ", ".join(f"{k}={v}" for k, v in ctx_options.items())
    result = f"浏览器配置已更新: {changed}。当前页面: {new_page.url}"
    if nav_failed:
        result += "（警告：页面恢复失败，当前为空白页，请重新 navigate）"
    return result

