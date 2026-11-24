# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Boot-time fit check: can these members share these chips?

``python -m tt_cohost.fit --config group.json [--json report.json]``

It boots the group exactly as ``tt_cohost.launch`` does (in a child process that owns the chips), sends each vLLM
member one short greedy request, stops the group, and reports per member and phase the DRAM and L1 in use, the
programs compiled after trace capture, and, if a member failed, the member, the phase and the error (an L1 clash,
out of DRAM, opening its own device, a timeout). The verdict is ``shared`` when every member booted and answered,
else ``separate`` with the reason. It exits 0 with a verdict either way, and 2 only when the check itself could not
run (an invalid config, a launcher that died before the group existed), and 3 when a device fault (a chip that needs
a reset) stopped the boot: that report says nothing about whether the members fit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import greedy
from ._group import GroupProcess
from .config import loads

REPORT_VERSION = 1
#: Boot phases in order; a member stuck in an earlier phase than the others is the one a timeout names.
PHASES = ["start", "init", "open", "load", "kv_cache", "compile", "capture", "start-server", "ready", "serve"]
PROBE_PROMPT = "The capital of France is"

#: (kind, pattern) in priority order: the first match in a member's error names the kind and the line shown.
_KINDS: List[Tuple[str, str]] = [
    ("device", r"[^\n]*(?:Try resetting the board|Timed out while waiting for active ethernet core|"
               r"failed to initialize FW|Fabric Router Sync: Timeout|Ethernet handshake likely failed)[^\n]*"),
    ("l1_clash", r"Statically allocated (?:dataflow|circular) buffers[^\n]*clash with L1 buffers[^\n]*"),
    ("l1_clash", r"Statically allocated (?:dataflow|circular) buffers[^\n]*beyond max L1 size[^\n]*"),
    ("dram", r"[^\n]*(?:Out of Memory|Not enough space to allocate)[^\n]*"),
    ("opens_device", r"[^\n]*called ttnn\.\w+ while the group's mesh is open[^\n]*"),
    ("timeout", r"[^\n]*(?:did not finish|never arrived|was not ready within)[^\n]*"),
    ("server_exited", r"[^\n]*(?:its server exited|its HTTP server thread ended)[^\n]*"),
    ("late_compile", r"[^\n]*compiled \d+ program\(s\) after trace capture[^\n]*"),
]
KIND_TEXT = {
    "l1_clash": "L1 clash (its programs need L1 another member holds)",
    "dram": "out of DRAM",
    "opens_device": "it opens its own device",
    "timeout": "timed out",
    "server_exited": "its server exited",
    "late_compile": "programs compiled after trace capture",
    "probe": "it did not answer a request",
    "error": "error",
    "device": "device fault (the chips need a reset; this is not a fit result)",
}
EXIT_DEVICE_FAULT = 3


class ToolError(RuntimeError):
    """The fit check itself could not run."""


def classify(error: str) -> Tuple[str, str]:
    """``(kind, the line that says it)`` for a member's error."""
    for kind, pat in _KINDS:
        m = re.search(pat, error)
        if m:
            return kind, m.group(0).strip()[:400]
    lines = [ln.strip() for ln in error.splitlines()
             if ln.strip() and not re.match(r"^(\w+: )?(TT_THROW|TT_FATAL) @ .*: tt::exception$", ln.strip())
             and ln.strip() != "info:"]
    return "error", (lines[0] if lines else error.strip())[:400]


def build_report(raw_cfg: dict, outcome: str, status: Optional[dict], *, probes: Optional[Dict[str, dict]] = None,
                 strict: bool = False, boot_seconds: Optional[float] = None, timeout_s: Optional[float] = None,
                 log_tail: Optional[List[str]] = None) -> dict:
    """The report from what the group's host recorded (``status``) and how the boot ended (``outcome``:
    ``ready``, ``exited`` or ``timeout``)."""
    status = status or {}
    probes = probes or {}
    boot = status.get("boot") or {}
    dram = (status.get("dram_gib_per_chip") or {}).get("members") or {}
    l1 = status.get("l1_kib_per_core") or {}
    l1_owned = status.get("l1_kib_per_core_owned") or {}
    by_owner = status.get("memory_by_owner") or {}
    late = status.get("compiles_after_capture") or {}
    serve_errors = status.get("serve_errors") or {}
    names = [m["name"] for m in raw_cfg["members"]]
    members: Dict[str, dict] = {}
    for m in raw_cfg["members"]:
        b = boot.get(m["name"]) or {"phase": "start", "status": "booting", "error": None}
        members[m["name"]] = {"kind": m["kind"], "port": m["port"], "status": b["status"], "phase": b["phase"],
                              "error": b.get("error")}

    if outcome == "ready":
        for e in members.values():
            if e["status"] == "booting":
                e["status"], e["phase"] = "ok", "ready"
    elif outcome == "timeout":
        stuck = [n for n, e in members.items() if e["status"] in ("booting", "ok")]
        if stuck and not any(e["status"] in ("failed", "timeout") for e in members.values()):
            first = min(PHASES.index(members[n]["phase"]) if members[n]["phase"] in PHASES else 0 for n in stuck)
            for n in stuck:
                if (PHASES.index(members[n]["phase"]) if members[n]["phase"] in PHASES else 0) == first:
                    members[n].update(status="timeout",
                                      error=f"the group was not ready within {timeout_s:.0f} s; "
                                            f"{n} was still in {members[n]['phase']}")
        for e in members.values():
            if e["status"] in ("booting", "ok"):
                e.update(status="aborted", error=e["error"] or "the group was stopped")
    else:
        for e in members.values():
            if e["status"] in ("booting", "ok"):
                e.update(status="aborted", error=e["error"] or "the group stopped")

    warnings = []
    for n, e in members.items():
        p = probes.get(n)
        if p is not None:
            e["probe"] = p
            if not p["ok"] and e["status"] == "ok":
                # the host's exception says why the request failed; the HTTP error only that it did
                error = (serve_errors.get(n) or {}).get("error") or p["error"]
                e.update(status="failed", phase="serve", error=error, error_kind="probe")
        e["compiles_after_capture"] = int(late.get(n, 0))
        if e["compiles_after_capture"]:
            msg = f"{n} compiled {e['compiles_after_capture']} program(s) after trace capture"
            if strict and e["status"] == "ok":
                e.update(status="failed", phase="serve", error=msg)
            else:
                warnings.append(msg + "; a parked trace may be corrupted (check greedy output with "
                                      "tt_cohost.conformance)")
        e["dram_gib_per_chip"] = dram.get(n, {})
        e["l1_kib_per_core"] = l1.get(n, {})
        if n in l1_owned:
            e["l1_kib_per_core_owned"] = l1_owned[n]
        if e["status"] != "ok" and e.get("error"):
            kind, line = classify(e["error"])
            e["error_kind"] = e.get("error_kind") or kind
            e["error_full"], e["error"] = e["error"][:4000], line
        else:
            e["error_kind"] = None

    order = ([status["failure"]["member"]] if status.get("failure") else []) + names
    culprit = next((n for n in order if n in members and members[n]["status"] in ("failed", "timeout")), None)
    fits = outcome == "ready" and all(e["status"] == "ok" for e in members.values())
    if fits:
        reason = f"all {len(members)} members booted on {raw_cfg['mesh_device']} and answered"
        failure = None
    elif culprit:
        e = members[culprit]
        reason = f"{culprit} failed in {e['phase']}: {KIND_TEXT[e['error_kind']]}: {e['error']}"
        holders = sorted(((o, m.get("l1_kib_per_core", 0)) for o, m in by_owner.items() if m.get("l1_kib_per_core")),
                         key=lambda x: -x[1])
        if e["error_kind"] == "l1_clash" and holders:   # ttnn with per-owner accounting names who holds the L1
            reason += "; L1 held per core: " + ", ".join(f"{o or 'no owner'} {k} KiB" for o, k in holders)
        failure = {"member": culprit, "phase": e["phase"], "kind": e["error_kind"], "error": e["error"]}
    else:
        last = next((ln for ln in reversed(log_tail or []) if ln.strip()), "no output")
        reason = f"the group stopped before it was ready ({outcome}); last output: {last[:300]}"
        failure = {"member": None, "phase": None, "kind": "error", "error": last[:300]}
    return {
        "fit_report_version": REPORT_VERSION,
        "verdict": "shared" if fits else "separate",
        "fits": fits,
        "reason": reason,
        "failure": failure,
        "mesh_device": raw_cfg["mesh_device"],
        "outcome": outcome,
        "boot_seconds": boot_seconds,
        "dram_gib_per_chip_total": (status.get("dram_gib_per_chip") or {}).get("total"),
        "memory_by_owner": by_owner or None,
        "members": members,
        "warnings": warnings,
        "device_fault": bool(failure and failure["kind"] == "device"),
    }


def probe(raw_cfg: dict) -> Dict[str, dict]:
    """One short greedy request per vLLM member (also exercises the serving path after capture)."""
    out = {}
    for m in raw_cfg["members"]:
        if m["kind"] != "vllm":
            continue
        try:
            row = greedy.complete(f"http://127.0.0.1:{m['port']}", greedy.served_model(f"http://127.0.0.1:{m['port']}"),
                                  PROBE_PROMPT, max_tokens=8, timeout=600)
            out[m["name"]] = {"ok": True, "seconds": row["seconds"], "text": row["text"]}
        except Exception as e:  # noqa: BLE001 - reported per member
            out[m["name"]] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out


def run_fit(config_path: str, *, workdir: Optional[str] = None, timeout_s: float = 5400, do_probe: bool = True,
            strict: bool = False, launcher: Optional[List[str]] = None, echo=None) -> dict:
    text = Path(config_path).read_text()
    try:
        loads(text)                 # the runtime's own validation, before any chip is touched
    except (ValueError, KeyError, TypeError) as e:
        raise ToolError(f"invalid group config {config_path}: {e}") from e
    raw = json.loads(text)
    wd = Path(workdir or tempfile.mkdtemp(prefix="tt-cohost-fit-"))
    group = GroupProcess(raw, wd, launcher=launcher, echo=echo)
    group.start()
    try:
        outcome = group.wait_ready(timeout_s)
        boot_seconds = group.seconds
        probes = probe(group.cfg) if outcome == "ready" and do_probe else {}
        status = group.status()
    finally:
        rc = group.stop()
    if status is None:
        raise ToolError(f"the launcher exited (code {rc}) before the group existed; see {group.log_path}:\n"
                        + "\n".join(group.log_tail(15)))
    report = build_report(raw, outcome, status, probes=probes, strict=strict, boot_seconds=boot_seconds,
                          timeout_s=timeout_s, log_tail=group.log_tail())
    report["log"] = str(group.log_path)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tt_cohost.fit", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the group config JSON (as for tt_cohost.launch)")
    ap.add_argument("--json", help="write the report here (default: stdout)")
    ap.add_argument("--timeout", type=float, default=5400, help="seconds the group may take to boot (default 5400)")
    ap.add_argument("--no-probe", action="store_true", help="do not send each vLLM member a request")
    ap.add_argument("--strict", action="store_true", help="a program compiled after capture fails the member")
    ap.add_argument("--workdir", help="keep the group's config, log and status here (default: a temp dir)")
    args = ap.parse_args(argv)

    def echo(line):
        print(line, file=sys.stderr, flush=True)
    try:
        report = run_fit(args.config, workdir=args.workdir, timeout_s=args.timeout, do_probe=not args.no_probe,
                         strict=args.strict, echo=echo)
    except ToolError as e:
        print(f"tt_cohost.fit: {e}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    else:
        print(text)
    print(f"fit: {report['verdict']}: {report['reason']}", file=sys.stderr)
    if report["device_fault"]:
        print("fit: a device fault stopped the boot; reset the chips and run the check again", file=sys.stderr)
    for n, e in report["members"].items():
        print(f"  {n}: {e['status']} ({e['phase']}); DRAM GiB/chip {e['dram_gib_per_chip']}; "
              f"L1 KiB/core {e['l1_kib_per_core']}; compiles after capture {e['compiles_after_capture']}",
              file=sys.stderr)
    for w in report["warnings"]:
        print(f"  warning: {w}", file=sys.stderr)
    return EXIT_DEVICE_FAULT if report["device_fault"] else 0


if __name__ == "__main__":
    sys.exit(main())
