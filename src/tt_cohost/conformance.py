# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Conformance test: does a member give the same greedy output next to a neighbour as alone?

``python -m tt_cohost.conformance --config group.json --member NAME --out DIR [--prompts FILE] [--baseline FILE]``

1. Alone: the member served by itself, as a one-member group on the same mesh (or ``--baseline``, greedy rows
   from an earlier run of the member alone, e.g. its own server).
2. Next to its neighbours: the whole group. The prompts once one at a time (sequential), then again while every
   vLLM neighbour answers its own stream of requests, so the members' steps interleave on the device.
3. Every prompt's output compared with the alone run. Pass = all identical, sequential and interleaved.

Pick a neighbour that holds a typical amount of L1 (GPT-OSS-20B holds about 370 KiB per core on 4 chips). Exits 0
on pass, 1 on a difference or a group that did not boot, 2 when the test itself could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import List, Optional

from . import greedy
from ._group import GroupProcess
from .config import loads


class ToolError(RuntimeError):
    pass


def _serve(raw: dict, wd: Path, launcher, timeout_s: float, echo):
    group = GroupProcess(raw, wd, launcher=launcher, echo=echo)
    group.start()
    outcome = group.wait_ready(timeout_s)
    if outcome != "ready":
        group.stop()
        st = group.status() or {}
        why = (st.get("failure") or {}).get("error") or "\n".join(group.log_tail(5))
        raise RuntimeError(f"the group in {wd} did not boot ({outcome}): {why[:500]} (python -m tt_cohost.fit "
                           "reports why)")
    return group


def run_conformance(config_path: str, member: str, out: str, *, prompts: Optional[List[str]] = None,
                    baseline: Optional[str] = None, max_tokens: int = 64, timeout_s: float = 5400,
                    launcher=None, echo=None) -> dict:
    text = Path(config_path).read_text()
    try:
        cfg = loads(text)
    except (ValueError, KeyError, TypeError) as e:
        raise ToolError(f"invalid group config {config_path}: {e}") from e
    raw = json.loads(text)
    by_name = {m.name: m for m in cfg.members}
    if member not in by_name:
        raise ToolError(f"no member {member!r} in {config_path} (members: {sorted(by_name)})")
    if by_name[member].kind != "vllm":
        raise ToolError(f"{member} is an app member; this test drives OpenAI-compatible (vllm) members")
    neighbours = [m for m in cfg.members if m.name != member]
    if not neighbours:
        raise ToolError("the group has no neighbour for the member")
    prompts = prompts or list(greedy.PROMPTS)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    port = by_name[member].port
    base = f"http://127.0.0.1:{port}"

    if baseline:
        alone = json.loads(Path(baseline).read_text())["rows"]
    else:
        solo = {**raw, "members": [m for m in raw["members"] if m["name"] == member]}
        solo.pop("status_port", None)
        group = _serve(solo, out_dir / "alone", launcher, timeout_s, echo)
        try:
            alone = greedy.run(base, prompts, max_tokens)
        finally:
            group.stop()
    (out_dir / "alone.json").write_text(json.dumps({"rows": alone}, indent=1))

    group = _serve(raw, out_dir / "group", launcher, timeout_s, echo)
    busy = [n for n in neighbours if n.kind == "vllm"]
    try:
        sequential = greedy.run(base, prompts, max_tokens)
        interleaved, served = None, {}
        if busy:
            stop = threading.Event()
            threads = []
            for n in busy:
                def drive(n=n):
                    served[n.name] = greedy.keep_busy(f"http://127.0.0.1:{n.port}", prompts, stop, max_tokens)
                t = threading.Thread(target=drive, daemon=True)
                t.start()
                threads.append(t)
            try:
                interleaved = greedy.run(base, prompts, max_tokens)
            finally:
                stop.set()
                for t in threads:
                    t.join(900)
        status = group.status() or {}
    finally:
        group.stop()
    (out_dir / "sequential.json").write_text(json.dumps({"rows": sequential}, indent=1))
    if interleaved is not None:
        (out_dir / "interleaved.json").write_text(json.dumps({"rows": interleaved}, indent=1))

    seq = greedy.compare(alone, sequential)
    inter = greedy.compare(alone, interleaved) if interleaved is not None else None
    ok = seq["identical"] == seq["total"] and not seq["diffs"] and \
        (inter is None or (inter["identical"] == inter["total"] and not inter["diffs"]))
    report = {
        "member": member,
        "neighbours": [n.name for n in neighbours],
        "prompts": len(prompts),
        "max_tokens": max_tokens,
        "alone": "baseline file " + baseline if baseline else "one-member group on the same mesh",
        "sequential": seq,
        "interleaved": inter if inter is not None else "skipped: no vLLM neighbour to keep busy",
        "neighbour_requests_while_interleaved": served,
        "compiles_after_capture": status.get("compiles_after_capture"),
        "pass": ok,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tt_cohost.conformance", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the group config JSON: the member and its neighbour(s)")
    ap.add_argument("--member", required=True, help="the member under test (a vllm member)")
    ap.add_argument("--out", required=True, help="directory for the outputs and report.json")
    ap.add_argument("--prompts", help="prompt file: a JSON list or one prompt per line (default: 16 built in)")
    ap.add_argument("--baseline", help="greedy rows of the member alone ({'rows': [...]}) instead of a solo boot")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=5400, help="seconds each boot may take (default 5400)")
    args = ap.parse_args(argv)
    try:
        report = run_conformance(args.config, args.member, args.out, prompts=greedy.load_prompts(args.prompts),
                                 baseline=args.baseline, max_tokens=args.max_tokens, timeout_s=args.timeout,
                                 echo=lambda line: print(line, file=sys.stderr, flush=True))
    except ToolError as e:
        print(f"tt_cohost.conformance: {e}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as e:      # a group that did not boot, a member that stopped answering
        print(f"tt_cohost.conformance: FAIL: {e}", file=sys.stderr)
        return 1
    inter = report["interleaved"]
    inter_txt = f"{inter['identical']}/{inter['total']} identical" if isinstance(inter, dict) else inter
    print(f"{report['member']} next to {', '.join(report['neighbours'])}: sequential "
          f"{report['sequential']['identical']}/{report['sequential']['total']} identical, interleaved {inter_txt}: "
          f"{'PASS' if report['pass'] else 'FAIL'}")
    for mode in ("sequential", "interleaved"):
        r = report[mode]
        for d in (r["diffs"] if isinstance(r, dict) else []):
            print(f"  {mode} {d['prompt']!r}: differs at char {d['at_char']}: {d['reference']!r} vs {d['got']!r}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
