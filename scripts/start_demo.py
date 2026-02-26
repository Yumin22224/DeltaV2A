#!/usr/bin/env python
"""
DeltaV2A Demo Launcher
----------------------
Starts the FastAPI backend and a Cloudflare Quick Tunnel in a single process.
The public HTTPS URL is printed when the tunnel is ready — paste it into the
Vercel frontend's backend URL field.

Usage:
    python scripts/start_demo.py
    python scripts/start_demo.py --port 8000
    python scripts/start_demo.py --config configs/pipeline.yaml

Requirements:
    cloudflared must be installed. Install with:
        winget install Cloudflare.cloudflared
    or download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Press Ctrl+C to stop both server and tunnel.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cloudflared() -> str | None:
    import shutil

    # Check PATH first
    found = shutil.which("cloudflared")
    if found:
        return found

    # Fallback to known Windows MSI install location
    fallbacks = [
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
    ]
    for path in fallbacks:
        if Path(path).exists():
            return path

    return None


def _print_install_instructions():
    print()
    print("=" * 60)
    print("  cloudflared not found.")
    print()
    print("  Install with one of the following:")
    print()
    print("  [Windows - winget]")
    print("    winget install Cloudflare.cloudflared")
    print()
    print("  [Windows - direct download]")
    print("    https://developers.cloudflare.com/cloudflare-one/")
    print("    connections/connect-networks/downloads/")
    print("    -> Download cloudflared-windows-amd64.exe,")
    print("       rename to cloudflared.exe, put on PATH.")
    print("=" * 60)
    print()


def _tail_tunnel_log(logfile_path: str, url_found: threading.Event, stop: threading.Event):
    """Poll a log file for the tunnel URL (avoids pipe-buffering issues on Windows)."""
    url_pattern = re.compile(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')
    with open(logfile_path, "r", encoding="utf-8", errors="replace") as f:
        while not stop.is_set():
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            line = line.rstrip()
            match = url_pattern.search(line)
            if match:
                url = match.group(0)
                print()
                print("=" * 60)
                print(f"  Tunnel URL: {url}")
                print()
                print("  Paste this URL into the Vercel frontend header")
                print("  (click the URL field in the top-right corner).")
                print("=" * 60)
                print()
                url_found.set()
            elif any(kw in line for kw in ("ERR", "error", "failed", "Unable")):
                print(f"[tunnel] {line}")


def _stream_server_output(proc: subprocess.Popen, prefix: str = "[server]"):
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeltaV2A demo launcher (server + tunnel)")
    p.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    p.add_argument("--config", type=str, default=None, help="Path to pipeline.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    port = args.port

    # Verify cloudflared is available
    cf = _find_cloudflared()
    if cf is None:
        _print_install_instructions()
        sys.exit(1)

    # Build server command
    server_cmd = [
        sys.executable,
        str(Path(__file__).parent / "serve.py"),
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if args.config:
        server_cmd += ["--config", args.config]

    # Build tunnel command
    tunnel_cmd = [cf, "tunnel", "--url", f"http://localhost:{port}"]

    print(f"[demo] Starting backend server on port {port}...")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).parent.parent),
    )

    # Give the server a moment to start binding the port
    time.sleep(2)

    # Write cloudflared output to a temp file to avoid Windows pipe-buffering issues.
    # The tail thread reads from the file with polling instead of blocking on a pipe.
    print("[demo] Starting Cloudflare Quick Tunnel...")
    tunnel_logfile = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".log", delete=False, prefix="cloudflared_"
    )
    tunnel_logfile_path = tunnel_logfile.name
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=tunnel_logfile,
        stderr=subprocess.STDOUT,
    )
    tunnel_logfile.close()  # subprocess holds it open; we open separately for reading

    url_found = threading.Event()
    stop_tail = threading.Event()

    # Thread: stream server logs to stdout
    t_server = threading.Thread(
        target=_stream_server_output,
        args=(server_proc,),
        daemon=True,
    )

    # Thread: tail tunnel log file for the public URL
    t_tunnel = threading.Thread(
        target=_tail_tunnel_log,
        args=(tunnel_logfile_path, url_found, stop_tail),
        daemon=True,
    )

    t_server.start()
    t_tunnel.start()

    print("[demo] Waiting for tunnel URL... (this may take ~10 seconds)")

    try:
        while True:
            if server_proc.poll() is not None:
                print(f"\n[demo] Server exited (code {server_proc.returncode}). Stopping.")
                break
            if tunnel_proc.poll() is not None:
                print(f"\n[demo] Tunnel exited (code {tunnel_proc.returncode}). Stopping.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[demo] Shutting down...")
    finally:
        stop_tail.set()
        tunnel_proc.terminate()
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        try:
            tunnel_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel_proc.kill()
        try:
            os.unlink(tunnel_logfile_path)
        except OSError:
            pass
        print("[demo] Done.")


if __name__ == "__main__":
    main()
