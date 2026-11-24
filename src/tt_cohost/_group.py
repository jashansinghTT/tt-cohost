# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Boot a group with ``tt_cohost.launch`` in a child process, watch it, read its status, stop it.

The child owns the chips while it runs, so the tools that use this (fit check, conformance test) never import
ttnn themselves, and a hung group can be killed without taking the tool down.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from .launch import READY_LINE


def free_port(avoid=()) -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in avoid:
            return port


class GroupProcess:
    def __init__(self, raw_cfg: dict, workdir: Path, launcher: Optional[List[str]] = None,
                 echo: Optional[Callable[[str], None]] = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(raw_cfg))
        if not cfg.get("status_port"):
            cfg["status_port"] = free_port({m["port"] for m in cfg["members"]})
        cfg.setdefault("socket_dir", str(self.workdir / "sockets"))
        self.cfg = cfg
        self.config_path = self.workdir / "config.json"
        self.config_path.write_text(json.dumps(cfg, indent=1))
        self.status_path = self.workdir / "status.json"
        self.status_path.unlink(missing_ok=True)
        self.log_path = self.workdir / "launch.log"
        self.launcher = launcher or [sys.executable, "-m", "tt_cohost.launch"]
        self.echo = echo
        self.proc: Optional[subprocess.Popen] = None
        self.ready = threading.Event()
        self.started = 0.0
        self._pumped = threading.Event()

    def start(self) -> None:
        env = {**os.environ, "TT_COHOST_STATUS_FILE": str(self.status_path)}
        env.pop("TT_COHOST_CONFIG", None)
        self.started = time.monotonic()
        self.proc = subprocess.Popen([*self.launcher, "--config", str(self.config_path)], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     start_new_session=True)
        threading.Thread(target=self._pump, daemon=True, name="tt-cohost-group-log").start()

    def _pump(self) -> None:
        with open(self.log_path, "w") as log:
            for line in iter(self.proc.stdout.readline, ""):
                log.write(line)
                log.flush()
                if line.rstrip("\n") == READY_LINE:
                    self.ready.set()
                if self.echo and ("tt_cohost " in line or line.startswith("cohost:")):   # the host's own lines
                    self.echo(line.rstrip("\n"))
        self._pumped.set()

    def wait_ready(self, timeout_s: float) -> str:
        """``ready``, ``exited`` (before it was ready) or ``timeout``."""
        deadline = time.monotonic() + timeout_s
        while True:
            if self.ready.wait(2):
                return "ready"
            if self.proc.poll() is not None:
                self._pumped.wait(10)
                return "ready" if self.ready.is_set() else "exited"
            if time.monotonic() > deadline:
                return "timeout"

    @property
    def seconds(self) -> float:
        return round(time.monotonic() - self.started, 1)

    def status(self) -> Optional[dict]:
        """The live status endpoint, else the last status the host wrote to its file."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg['status_port']}/", timeout=10) as r:
                    return json.loads(r.read())
            except OSError:
                pass
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return None

    def stop(self, grace_s: float = 180) -> Optional[int]:
        """SIGTERM the launcher (it stops its members and closes the mesh); kill the whole group if it hangs."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        self._pumped.wait(10)
        return self.proc.returncode

    def log_tail(self, n: int = 40) -> List[str]:
        try:
            return self.log_path.read_text(errors="replace").splitlines()[-n:]
        except OSError:
            return []
