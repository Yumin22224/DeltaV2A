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
import re
import subprocess
import sys
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


def _stream_tunnel_output(proc: subprocess.Popen, url_found: threading.Event):
    """Read cloudflared output line by line, extract and print the tunnel URL."""
    url_pattern = re.compile(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')
    for line in proc.stdout:
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
        else:
            # Only show cloudflared lines that look useful (suppress verbose INF lines)
            if any(kw in line for kw in ("ERR", "error", "failed", "Unable")):
                print(f"[tunnel] {line}")


def _stream_server_output(proc: subprocess.Popen, prefix: str = "[server]"):
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}")


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

    print("[demo] Starting Cloudflare Quick Tunnel...")
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr->stdout so we capture all output
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    url_found = threading.Event()

    # Thread: stream server logs to stdout
    t_server = threading.Thread(
        target=_stream_server_output,
        args=(server_proc,),
        daemon=True,
    )

    # Thread: parse tunnel output for the public URL
    t_tunnel = threading.Thread(
        target=_stream_tunnel_output,
        args=(tunnel_proc, url_found),
        daemon=True,
    )


    t_server.start()
    t_tunnel.start()

    print("[demo] Waiting for tunnel URL... (this may take ~10 seconds)")

    try:
        while True:
            # Check if either subprocess died unexpectedly
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
        print("[demo] Done.")


if __name__ == "__main__":
    main()
