import os
import sys
import time
import socket
import subprocess
import urllib.request
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """Funciona en modo normal y en PyInstaller onefile (_MEIPASS)."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative_path)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(url: str, timeout_s: int = 60) -> None:
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(url) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Streamlit no inició a tiempo: {url}")
        time.sleep(0.25)


def find_chrome_exe() -> str | None:
    """Intenta ubicaciones típicas de Chrome en Windows."""
    candidates = [
        os.environ.get("CHROME_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / r"Google\Chrome\Application\chrome.exe"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def open_in_chrome(url: str) -> None:
    chrome = find_chrome_exe()
    if chrome:
        subprocess.Popen([chrome, "--new-window", url],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import webbrowser
        webbrowser.open(url)


def main():
    # Streamlit local-only y sin telemetría
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    port = get_free_port()
    url = f"http://127.0.0.1:{port}"

    app_main = resource_path("main.py")
    workdir = os.path.dirname(app_main)

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_main,
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--server.headless", "true",
    ]

    proc = subprocess.Popen(cmd, cwd=workdir)

    try:
        wait_for_server(url, timeout_s=60)
        open_in_chrome(url)
        proc.wait()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
