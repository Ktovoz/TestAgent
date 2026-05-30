"""Browser configuration - use local Chrome installation."""
import atexit
import http.client
import json
import os
import signal
import socket
import subprocess
import threading
from pathlib import Path

from skiritai.logger import logger

# File that stores the browser session info for persistent sessions
SESSION_FILE = ".browser_session"

# Registry of launched PIDs for atexit cleanup
_launched_pids: list[int] = []
_launched_pids_lock = threading.Lock()


def _atexit_cleanup() -> None:
    """Kill any browser subprocesses that are still alive when Python exits."""
    with _launched_pids_lock:
        for pid in _launched_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"[Browser] atexit: killed orphan browser pid={pid}")
            except ProcessLookupError:
                pass  # process already exited
            except OSError:
                pass
        _launched_pids.clear()


# Register cleanup on normal interpreter exit
atexit.register(_atexit_cleanup)


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT: kill tracked browser processes before exiting."""
    logger.info(f"[Browser] Received signal {signum}, cleaning up browser processes...")
    _atexit_cleanup()
    # Re-raise the signal with default handler to terminate the process
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def _register_pid(pid: int) -> None:
    """Track a launched browser PID for cleanup."""
    with _launched_pids_lock:
        _launched_pids.append(pid)


def _unregister_pid(pid: int) -> None:
    """Remove a PID from tracking (e.g., after explicit kill)."""
    with _launched_pids_lock:
        try:
            _launched_pids.remove(pid)
        except ValueError:
            pass


def get_launch_args(headless: bool | None = None) -> dict:
    """Get Playwright launch args.

    Resolution order for headless:
        1. Explicit parameter (per-case override)
        2. SKIRITAI_HEADLESS env var
        3. HEADLESS env var
        4. Default: false (headful / 有头模式)

    Reads CHROME_PATH env var for custom Chrome executable path.
    """
    if headless is None:
        headless = (os.getenv("SKIRITAI_HEADLESS") or os.getenv("HEADLESS", "false")).lower() in ("true", "1", "yes")

    in_ci = os.getenv("CI", "").lower() in ("true", "1")
    chrome_args = ["--disable-blink-features=AutomationControlled"]
    if in_ci:
        chrome_args.append("--no-sandbox")
    args: dict = {
        "headless": headless,
        "args": chrome_args,
    }
    chrome_path = os.getenv("SKIRITAI_CHROME_PATH") or os.getenv("CHROME_PATH")
    if chrome_path:
        args["executable_path"] = chrome_path
    return args


def _session_path(case_dir: Path) -> Path:
    """Path to the persisted session file."""
    return case_dir / SESSION_FILE


def _save_session(case_dir: Path, cdp_port: int, pid: int) -> None:
    """Persist the CDP port and process ID to a file."""
    path = _session_path(case_dir)
    data = {"cdp_port": cdp_port, "pid": pid}
    path.write_text(json.dumps(data), encoding="utf-8")
    logger.info(f"[Browser] Session saved: port={cdp_port}, pid={pid} -> {path}")


def load_session(case_dir: Path) -> dict | None:
    """Load persisted session info. Returns None if not found."""
    path = _session_path(case_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"[Browser] Corrupted session file: {path}")
        return None


def has_persistent_session(case_dir: Path) -> bool:
    """Check if a persistent browser session file exists."""
    return _session_path(case_dir).exists()


def is_browser_alive(case_dir: Path) -> bool:
    """Check if the persisted browser process is still running.

    Returns True only if both the session file exists AND the process is alive.
    """
    session = load_session(case_dir)
    if not session:
        return False
    pid = session.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = check if process exists
        return True
    except OSError:
        return False


def cleanup_session(case_dir: Path) -> None:
    """Remove the persisted session file (does NOT kill the browser)."""
    path = _session_path(case_dir)
    if path.exists():
        path.unlink()
        logger.info(f"[Browser] Session file cleaned up: {path}")


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_chromium_path(pw) -> str:
    """Get the Chromium executable path from Playwright."""
    return pw.chromium.executable_path


def _wait_for_cdp(port: int, timeout: float = 10.0) -> bool:
    """Wait for Chrome's CDP endpoint to become available.

    Uses http.client directly to avoid urllib proxy interference on macOS.
    Returns True if the endpoint is ready, False on timeout.
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/json/version")
            resp = conn.getresponse()
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning(f"[Browser] CDP on port {port} returned non-JSON: {body[:200]}")
                    conn.close()
                    time.sleep(0.3)
                    continue
                if data.get("webSocketDebuggerUrl"):
                    conn.close()
                    return True
            conn.close()
        except (http.client.HTTPException, ConnectionRefusedError, OSError):
            logger.debug(f"[Browser] CDP not ready on port {port}")
        except Exception as e:
            logger.warning(f"[Browser] Unexpected error probing CDP port {port}: {e}")
        time.sleep(0.3)
    return False


async def _apply_stealth(page) -> None:
    """Apply playwright-stealth patches to hide automation fingerprints."""
    try:
        from playwright_stealth import Stealth
        await Stealth().apply_stealth_async(page)
        logger.info("[Browser] Stealth patches applied")
    except ImportError:
        logger.warning("[Browser] playwright-stealth not installed, skipping stealth patches")
    except Exception as e:
        logger.warning(f"[Browser] Stealth patch failed: {e}")


async def launch_browser_server(pw, case_dir: Path, headless: bool | None = None):
    """Launch Chromium as an independent subprocess with CDP enabled.

    The browser runs as a separate process — it survives when the Python
    program exits. The CDP port and PID are persisted to case_dir.

    Args:
        pw: Playwright instance (for getting the Chromium path)
        case_dir: Directory to persist the session file
        headless: Per-case override (None = use env var)

    Returns:
        (cdp_port, browser, context, page) tuple
    """
    launch_args = get_launch_args(headless)
    extra_args = launch_args.get("args", [])
    chrome_path = launch_args.get("executable_path") or _get_chromium_path(pw)
    headless = launch_args["headless"]

    cdp_port = _find_free_port()

    cmd = [chrome_path, f"--remote-debugging-port={cdp_port}"]
    if headless:
        cmd.append("--headless=new")
    cmd.extend(extra_args)
    # Required: a starting URL or Chrome may exit immediately
    cmd.append("about:blank")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for CDP to be ready
    if not _wait_for_cdp(cdp_port):
        proc.kill()
        raise RuntimeError(f"Chrome CDP not ready on port {cdp_port} after timeout")

    # Persist session info
    _save_session(case_dir, cdp_port, proc.pid)

    # Register PID for atexit cleanup (prevents orphan processes on crash)
    _register_pid(proc.pid)

    # Connect via Playwright CDP
    browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    page = context.pages[0] if context.pages else await context.new_page()

    # Apply stealth patches to hide automation fingerprints
    await _apply_stealth(page)

    logger.info(f"[Browser] Launched subprocess pid={proc.pid}, CDP port={cdp_port}")
    return cdp_port, browser, context, page


async def connect_to_browser(pw, case_dir: Path):
    """Reconnect to an existing browser via persisted CDP port.

    Args:
        pw: Playwright instance (from async_playwright().start())
        case_dir: Directory containing the persisted session file

    Returns:
        (browser, context, page) tuple — reuses existing pages

    Raises:
        FileNotFoundError: if no session file exists
        ConnectionError: if the browser process is dead
    """
    session = load_session(case_dir)
    if not session:
        raise FileNotFoundError(
            f"No persisted browser session found in {case_dir}. "
            f"Run a step first to launch the browser."
        )

    cdp_port = session.get("cdp_port")
    pid = session.get("pid")

    # Check if the process is still alive
    try:
        os.kill(pid, 0)
    except OSError:
        cleanup_session(case_dir)
        raise ConnectionError(
            f"Browser process (pid={pid}) is no longer running. "
            f"Session file is stale."
        )

    try:
        browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{cdp_port}")
    except Exception as e:
        cleanup_session(case_dir)
        raise ConnectionError(
            f"Cannot connect to browser CDP on port {cdp_port}. {e}"
        ) from e

    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    page = context.pages[0] if context.pages else await context.new_page()

    # Apply stealth patches on reconnection as well
    await _apply_stealth(page)

    logger.info(f"[Browser] Reconnected to browser on port {cdp_port}")
    return browser, context, page


def kill_browser(case_dir: Path) -> bool:
    """Kill the persisted browser process and clean up the session file.

    Returns True if a process was killed, False if no session was found.
    """
    session = load_session(case_dir)
    if not session:
        return False

    pid = session.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info(f"[Browser] Sent SIGTERM to browser pid={pid}")
        except OSError:
            pass  # already dead
        finally:
            _unregister_pid(pid)

    cleanup_session(case_dir)
    return True
