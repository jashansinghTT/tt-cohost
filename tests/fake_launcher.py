# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A stand-in for ``python -m tt_cohost.launch`` in the fit and conformance tests: no chips, no vLLM.

FAKE_MODE picks what the group does:
  ok     every member answers /v1/models and /v1/completions; the output depends only on the prompt
  drift  as ok, but with more than one member the second prompt's output changes
  late   as ok, and the status counts programs compiled after capture
  refuse as ok, but the last member answers every completion with HTTP 500 and its status has the host's error
  clash  the last member fails its compile with tt-metal's L1 clash; the others are aborted; exit 1
  hang   the last member never finishes loading
  crash  exit 1 before the host exists (no status at all)
"""

import argparse
import http.server
import json
import os
import signal
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--config")
cfg = json.load(open(ap.parse_args().config))
mode = os.environ.get("FAKE_MODE", "ok")
names = [m["name"] for m in cfg["members"]]
status = {"mesh_device": cfg["mesh_device"],
          "members": {n: {"kind": "vllm", "port": 0, "ready": False} for n in names},
          "compiles_after_capture": {}, "dram_gib_per_chip": {"members": {}, "total": None},
          "l1_kib_per_core": {}, "boot": {n: {"phase": "start", "status": "booting", "error": None} for n in names},
          "failure": None, "serve_errors": {}}
REFUSED = ("RuntimeError: Found 26 device buffer(s) still alive before replay of trace 0 in memory its capture used. "
           "These will be corrupted on replay.\nBuffer 62818 [op: ttnn.to_device]: DRAM [0x2664900, 0x2664b00), "
           "512 B per bank")


def write():
    path = os.environ["TT_COHOST_STATUS_FILE"]
    with open(path + ".tmp", "w") as f:
        json.dump(status, f)
    os.replace(path + ".tmp", path)


if mode == "crash":
    print("ImportError: No module named 'vllm'", flush=True)
    sys.exit(1)

for i, n in enumerate(names):
    status["boot"][n]["phase"] = "load"
    status["dram_gib_per_chip"]["members"][n] = {"load_model": 1.5 + i, "initialize_from_config": 1.0}
    status["l1_kib_per_core"][n] = {"load_model": 370.5}
status["dram_gib_per_chip"]["total"] = 9.0
write()

if mode == "clash":
    last = names[-1]
    err = ("TT_THROW @ /opt/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2601: tt::exception\ninfo:\n"
           "Statically allocated dataflow buffers in program 2950 clash with L1 buffers on core range "
           "[0-0 - 3-0]. L1 buffer allocated at 1172096 and static dataflow buffer region ends at 1258496")
    status["boot"][last].update(phase="compile", status="failed", error="RuntimeError: " + err)
    status["failure"] = {"member": last, "phase": "compile", "error": "RuntimeError: " + err}
    for n in names[:-1]:
        status["boot"][n].update(phase="compile", status="aborted",
                                 error=f"barrier compiled failed: {last}: {err}")
    write()
    print(f"cohost: member(s) {names} exited; stopping the group", flush=True)
    sys.exit(1)

if mode == "hang":
    for n in names[:-1]:
        status["boot"][n]["phase"] = "compile"
    write()
    while True:
        time.sleep(1)

grouped = len(names) > 1


class Member(http.server.BaseHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._json({"data": [{"id": f"org/{self.server.member}"}]})

    def do_POST(self):  # noqa: N802
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if mode == "refuse" and self.server.member == names[-1]:
            status["serve_errors"][names[-1]] = {"method": "execute_model", "error": REFUSED, "count": 1}
            write()
            self.send_error(500, "Internal Server Error")
            return
        text = f" {req['prompt'][::-1]} ({req['max_tokens']})"
        if mode == "drift" and grouped and req["prompt"] == self.server.prompts_seen.get(1):
            text = text.replace(" ", "_", 1)
        self._json({"choices": [{"text": text}], "usage": {"completion_tokens": req["max_tokens"]}})

    def log_message(self, *a):
        pass


class Status(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps(status).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


second_prompt = {1: "Write a haiku about silicon:"}
for m in cfg["members"]:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", m["port"]), Member)
    srv.member, srv.prompts_seen = m["name"], second_prompt
    threading.Thread(target=srv.serve_forever, daemon=True).start()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", cfg["status_port"]), Status)
threading.Thread(target=srv.serve_forever, daemon=True).start()
for n in names:
    status["boot"][n].update(phase="ready", status="ok")
    status["members"][n]["ready"] = True
if mode == "late":
    status["compiles_after_capture"] = {names[0]: 3}
write()
print("cohost: all members ready", flush=True)
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: stop.set())
stop.wait()
sys.exit(0)
