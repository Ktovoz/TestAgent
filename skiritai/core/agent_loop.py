"""Agent loop powered by LangGraph ReAct agent."""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from langgraph.prebuilt import create_react_agent

from skiritai.core.llm_retry import retry_async
from skiritai.core.tool_registry import ToolRegistry
from skiritai.core.tools import set_page, set_browser
from skiritai.events import Event, event_bus
from skiritai.llm import get_provider
from skiritai.logger import logger

# Tools that are read-only perception — excluded from replay scripts
PERCEPTION_TOOLS = {
    "page_perceive", "find_element", "analyze_page",
    "get_page_info", "get_text", "screenshot",
}

# ---------------------------------------------------------------------------
# Default system prompt (used when no custom prompt is provided)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """你是一个浏览器自动化测试 Agent。你通过调用工具来操作浏览器完成测试任务。

## 最重要规则：只做任务要求的事
- 严格只执行任务描述中明确要求的操作
- 不自行添加额外步骤（如任务只说打开页面，就不要去搜索或点击其他元素）

## 核心工作流：
1. 进入新页面或页面状态变化后，先调用 analyze_page 了解页面结构
2. 根据 analyze_page 返回的信息操作：
   - 输入框：从 areas 中找到输入框的 selector，用 fill(selector, text)
   - 按钮/链接：根据 text 字段用 click_text(text)，没有 text 的按钮用 click(selector)
   - 注意查看 area 和 nearby 字段区分同名元素
3. 关键操作完成后，可用 get_page_info 确认页面变化（这是验证，不是额外操作）

## 硬性规则：
- 必须使用 analyze_page 返回的 selector，禁止自造选择器
- 每次进入新页面后先调用 analyze_page
- 如果 fill 后输入框内容未生效或操作无效，改用 type_text
- 任务完成后用自然语言总结结果

## 重试策略：
- 填写无效 → 改用 type_text 逐字符输入
- 点击无效 → 用 click 切换到精确 selector，或用 press_key('Enter') 替代
- 操作后页面未变化 → 分析原因，按任务要求决定是否重试

## 错误处理：
- 遇到 SSL/证书错误导致页面无法打开时，调用 configure_browser(ignore_https_errors=True) 后重新 navigate

## 弹窗/遮罩：
- 页面弹窗遮挡时用 dismiss_overlay 关闭，未生效则 press_key('Escape')
"""


def load_system_prompt(case_dir: Path | None = None) -> str:
    """Resolve the system prompt with the following priority:

    1. Case-level prompt file: ``<case_dir>/prompt.md`` or ``<case_dir>/prompt.txt``
    2. Environment variable ``SYSTEM_PROMPT_FILE`` pointing to a file
    3. Environment variable ``SYSTEM_PROMPT`` containing the prompt text
    4. Built-in ``DEFAULT_SYSTEM_PROMPT``
    """
    # 1. Case-level prompt file
    if case_dir is not None:
        for name in ("prompt.md", "prompt.txt"):
            p = case_dir / name
            if p.is_file():
                prompt = p.read_text(encoding="utf-8").strip()
                if prompt:
                    logger.info(f"[Agent] Using case-level system prompt from {p}")
                    return prompt

    # 2. SYSTEM_PROMPT_FILE env var
    prompt_file = os.getenv("SYSTEM_PROMPT_FILE")
    if prompt_file:
        p = Path(prompt_file)
        if p.is_file():
            prompt = p.read_text(encoding="utf-8").strip()
            if prompt:
                logger.info(f"[Agent] Using system prompt from env file: {p}")
                return prompt

    # 3. SYSTEM_PROMPT env var (inline text)
    inline = os.getenv("SYSTEM_PROMPT")
    if inline and inline.strip():
        logger.info("[Agent] Using inline system prompt from env var")
        return inline.strip()

    # 4. Default
    return DEFAULT_SYSTEM_PROMPT


# Module-level default prompt for backward compatibility
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def register_all_tools() -> None:
    """Explicitly register all tool modules.

    Must be called once at startup (or before building the agent).
    This replaces relying on import side-effects.
    """
    import skiritai.core.tools  # noqa: F401 — registers action tools
    import skiritai.core.perception  # noqa: F401 — registers perception tools


# ---------------------------------------------------------------------------
# Internal flags — ensure one-time init without user boilerplate
# ---------------------------------------------------------------------------

_env_loaded: bool = False
_tools_registered: bool = False

# Module-level LLM provider override (set via set_llm())
_module_llm: Any = None


def set_llm(llm: Any) -> None:
    """Set a module-level LLM provider override.

    Once set, all subsequent ``_build_llm()`` calls will use this provider
    instead of auto-detecting from environment variables.

    Args:
        llm: An LLMProvider instance.
    """
    global _module_llm
    _module_llm = llm


def _ensure_env() -> None:
    """Load .env once per process. Safe to call multiple times."""
    global _env_loaded
    if not _env_loaded:
        from dotenv import load_dotenv
        load_dotenv(override=False)
        _env_loaded = True


def _ensure_tools() -> None:
    """Register all tools once per process. Safe to call multiple times."""
    global _tools_registered
    if not _tools_registered:
        register_all_tools()
        _tools_registered = True


def _build_llm(llm=None):
    """Build and return an LLM instance.

    Priority: explicit *llm* arg > module-level override > create_llm() auto.

    Automatically loads .env on first call.
    """
    _ensure_env()
    if llm is not None:
        return llm.build()
    if _module_llm is not None:
        return _module_llm.build()
    from skiritai.llm._factory import create_llm
    return create_llm().build()


# Agent cache: bounded LRU dict (max 8 entries) keyed by (prompt_hash, model_id)
_MAX_CACHE_SIZE = 8
_agent_cache: dict[tuple[int, str], Any] = {}
_agent_cache_order: list[tuple[int, str]] = []  # LRU eviction order


def _agent_cache_key(prompt: str, model) -> tuple[int, str]:
    """Build a stable cache key: (hash(prompt), model identifier string).

    Uses ``model_name`` + provider metadata to avoid id(model) which
    churns on every ``build()`` call.
    """
    model_id = (
        getattr(model, "model_name", "")
        or getattr(model, "model", "")
        or type(model).__name__
    )
    # Append provider name if available (distinguishes openai/gpt-4o from anthropic/gpt-4o)
    provider = getattr(model, "provider", "") or getattr(model, "_provider", "")
    if provider:
        model_id = f"{provider}/{model_id}"
    return (hash(prompt), model_id)


def build_agent(system_prompt: str | None = None, llm=None):
    """Build a LangGraph ReAct agent with Playwright tools + perception tools.

    Agent instances are cached (max 8, LRU eviction) and reused when
    system_prompt and model are unchanged.

    Automatically registers tools and loads .env on first call.
    No manual setup required.

    Args:
        system_prompt: Custom system prompt. If None, uses the default prompt.
        llm: Optional LLM provider instance. If None, auto-detects from env.
    """
    global _agent_cache_order
    _ensure_env()
    _ensure_tools()
    model = _build_llm(llm)
    prompt = system_prompt or SYSTEM_PROMPT
    cache_key = _agent_cache_key(prompt, model)

    if cache_key not in _agent_cache:
        # Evict oldest entry if cache is full
        if len(_agent_cache) >= _MAX_CACHE_SIZE and _agent_cache_order:
            oldest = _agent_cache_order.pop(0)
            _agent_cache.pop(oldest, None)
            logger.debug(f"[Agent] Cache evicted entry: {oldest}")

        registry = ToolRegistry()
        tools = registry.get_all()
        _agent_cache[cache_key] = create_react_agent(
            model=model,
            tools=tools,
            prompt=prompt,
        )
        _agent_cache_order.append(cache_key)
        logger.info(f"[Agent] Built new agent (cache size={len(_agent_cache)}/{_MAX_CACHE_SIZE})")
    else:
        # Move to end (most recently used)
        if cache_key in _agent_cache_order:
            _agent_cache_order.remove(cache_key)
        _agent_cache_order.append(cache_key)
        logger.debug(f"[Agent] Reusing cached agent (cache size={len(_agent_cache)})")

    return _agent_cache[cache_key]


async def run_agent(
        page: Any,
        task_description: str,
        url: str = "",
        on_log: Any = None,
        execution_id: str = "default",
        case_dir: Path | None = None,
        max_steps: int = 20,
        llm=None,
) -> dict:
    """
    Run the LangGraph ReAct agent on a task.

    Args:
        page: Playwright Page object
        task_description: What to accomplish
        url: Optional URL to navigate to first
        on_log: Optional callback for real-time log streaming
        execution_id: Execution identifier for event publishing
        case_dir: Optional case directory for loading case-level system prompt
        max_steps: Maximum agent tool-call steps (recursion_limit)
        llm: Optional LLM provider instance. If None, auto-detects from env.

    Returns:
        dict with keys: success, summary, steps, token_usage
    """
    set_page(page)
    # Store browser/context references for configure_browser tool
    try:
        set_browser(page.context.browser, page.context)
    except Exception as e:
        logger.warning(f"[Agent] Failed to set browser refs: {e}")

    # Resolve system prompt (case-level > env > default)
    system_prompt = load_system_prompt(case_dir)
    agent = build_agent(system_prompt=system_prompt, llm=llm)

    # Build user message with current page context
    page_url = page.url or "about:blank"
    page_title = ""
    try:
        page_title = await page.title()
    except Exception:
        pass
    page_context = f"当前页面: {page_title} ({page_url})"

    if url:
        user_msg = f"{page_context}\n请先导航到 {url}，然后执行以下任务：\n{task_description}"
    else:
        user_msg = f"{page_context}\n请执行以下测试任务（你已经在目标页面上，不要导航到其他网站）：\n{task_description}"

    steps: list[dict] = []
    final_summary = ""
    success = False
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def _stream_once() -> None:
        """Consume the agent stream once, populating outer-scope variables."""
        nonlocal steps, final_summary, success
        # Reset per-attempt state but keep token_usage cumulative
        steps = []
        final_summary = ""

        async for event in agent.astream(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"recursion_limit": max_steps},
        ):
            _accumulate_token_usage(event, token_usage)

            if "agent" in event:
                msg = event["agent"].get("messages", [])
                if msg:
                    last = msg[-1]
                    if hasattr(last, "tool_calls") and last.tool_calls:
                        for tc in last.tool_calls:
                            tool_name = tc["name"]
                            tool_args = tc.get("args", {})
                            logger.info(f"[Agent] Tool call: {tool_name}({tool_args})")
                            if on_log:
                                if inspect.iscoroutinefunction(on_log):
                                    await on_log(f"调用工具: {tool_name}({tool_args})")
                                else:
                                    on_log(f"调用工具: {tool_name}({tool_args})")
                            await event_bus.publish(Event(
                                type="tool_called",
                                execution_id=execution_id,
                                data={"tool_name": tool_name, "tool_args": tool_args},
                            ))
                            steps.append({
                                "action": tool_name,
                                "args": tool_args,
                            })
                    elif hasattr(last, "content") and last.content:
                        final_summary = last.content
                        steps.append({"action": "response", "content": last.content})

            if "tools" in event:
                msg = event["tools"].get("messages", [])
                if msg:
                    last = msg[-1]
                    result_text = last.content[:500] if hasattr(last, "content") else ""
                    if steps and steps[-1].get("action") != "response":
                        steps[-1]["result"] = result_text

        # Determine success from the last tool call or summary
        if steps:
            last_step = steps[-1]
            if last_step.get("action") == "task_complete":
                success = last_step.get("args", {}).get("success", False)
                final_summary = last_step.get("args", {}).get("summary", "")
            else:
                success = True
                if not final_summary:
                    final_summary = "任务执行完成"

    try:
        await retry_async(
            _stream_once,
            operation_name=f"agent task: {task_description[:50]}",
        )
    except Exception as e:
        logger.error(f"[Agent] Error after retries: {e}")
        return {
            "success": False,
            "summary": f"Agent 执行失败: {e}",
            "steps": steps,
            "token_usage": token_usage,
        }

    logger.info(f"[Agent] Done: success={success} summary={final_summary[:100]} "
                f"tokens={token_usage['total_tokens']}")
    return {
        "success": success,
        "summary": final_summary,
        "steps": steps,
        "token_usage": token_usage,
    }


def _accumulate_token_usage(event: dict, token_usage: dict) -> None:
    """Extract token usage metadata from LangGraph streaming events."""
    try:
        # LangGraph stores usage metadata on agent messages
        if "agent" not in event:
            return
        messages = event["agent"].get("messages", [])
        for msg in messages:
            usage = getattr(msg, "usage_metadata", None)
            if usage and isinstance(usage, dict):
                token_usage["prompt_tokens"] += usage.get("input_tokens", 0)
                token_usage["completion_tokens"] += usage.get("output_tokens", 0)
                token_usage["total_tokens"] += usage.get("total_tokens", 0)

            # Fallback: response_metadata (some LangChain versions)
            resp_meta = getattr(msg, "response_metadata", None)
            if resp_meta and isinstance(resp_meta, dict):
                token_info = resp_meta.get("token_usage", {})
                if token_info:
                    token_usage["prompt_tokens"] += token_info.get("prompt_tokens", 0)
                    token_usage["completion_tokens"] += token_info.get("completion_tokens", 0)
                    token_usage["total_tokens"] += token_info.get("total_tokens", 0)
    except Exception:
        pass  # token tracking is best-effort
