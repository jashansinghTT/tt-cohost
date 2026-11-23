# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Memory per member, when ttnn accounts memory per owner (ttnn.memory_owner / get_memory_usage_by_owner)."""

import contextlib
import json
import sys
import threading
import types

import pytest

from tt_cohost import config as cfgmod
from tt_cohost import fit
from tt_cohost import host as hostmod
from tt_cohost.device import Device


@pytest.fixture
def ttnn_owners(monkeypatch):
    """A fake ttnn whose memory_owner scope is thread-local, like the real one."""
    local = threading.local()
    usage = []

    @contextlib.contextmanager
    def memory_owner(name):
        prev = getattr(local, "owner", "")
        local.owner = name
        try:
            yield
        finally:
            local.owner = prev

    ttnn = types.ModuleType("ttnn")
    ttnn.memory_owner = memory_owner
    ttnn.current_memory_owner = lambda: getattr(local, "owner", "")
    ttnn.get_memory_usage_by_owner = lambda mesh: list(usage)
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    return types.SimpleNamespace(ttnn=ttnn, usage=usage)


def _entry(owner, kind, total, per_bank):
    return types.SimpleNamespace(owner=owner, buffer_type=f"BufferType.{kind}", num_buffers=1, total_bytes=total,
                                 bytes_per_bank=per_bank)


def test_each_piece_of_device_work_runs_as_its_member_and_nested_host_work_as_the_host(ttnn_owners):
    dev = Device()
    dev.scope = ttnn_owners.ttnn.memory_owner
    seen = dev.run("gptoss", lambda: (ttnn_owners.ttnn.current_memory_owner(),
                                      dev.run("host", ttnn_owners.ttnn.current_memory_owner),
                                      ttnn_owners.ttnn.current_memory_owner()))
    assert seen == ("gptoss", "host", "gptoss")
    assert dev.run("llama8b", ttnn_owners.ttnn.current_memory_owner) == "llama8b"


def test_the_host_records_the_l1_each_member_holds_itself(ttnn_owners, monkeypatch):
    cfg = cfgmod.loads(json.dumps({"cohost_config_version": 1, "mesh_device": "P150x4", "members": [
        {"name": "gptoss", "kind": "vllm", "port": 1}, {"name": "llama8b", "kind": "vllm", "port": 2}]}))
    host = hostmod.Host(cfg)
    host.mesh = object()
    host.track_owners()
    assert host.device.scope is ttnn_owners.ttnn.memory_owner
    monkeypatch.setattr(host, "dram_gb_per_chip", lambda: 1.0)
    monkeypatch.setattr(host, "l1_kib_per_core", lambda: 380.8)
    ttnn_owners.usage[:] = [_entry("gptoss", "L1", 370 * 1024 * 140, 370 * 1024),
                            _entry("gptoss", "DRAM", 7 * 2**30, 2**28), _entry("", "L1", 4096, 4096)]
    host.account("gptoss", "load_model", 0.0)
    ttnn_owners.usage.append(_entry("llama8b", "L1", 10 * 1024 * 140, 10 * 1024))
    host.account("llama8b", "compile", 0.0)
    st = host.status()
    assert st["l1_kib_per_core_owned"] == {"gptoss": {"load_model": 370.0}, "llama8b": {"compile": 10.0}}
    assert st["memory_by_owner"]["gptoss"] == {"l1_kib_per_core": 370.0, "dram_gib_per_chip": 7.0}
    assert st["memory_by_owner"][""] == {"l1_kib_per_core": 4.0}


def test_without_per_owner_accounting_nothing_changes(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    cfg = cfgmod.loads(json.dumps({"cohost_config_version": 1, "mesh_device": "P150", "members": [
        {"name": "a", "kind": "vllm", "port": 1}]}))
    host = hostmod.Host(cfg)
    host.mesh = object()
    host.track_owners()
    assert host.device.scope is None and host.memory_by_owner() is None


def test_an_l1_clash_names_who_holds_the_l1():
    err = ("RuntimeError: Statically allocated dataflow buffers in program 2950 clash with L1 buffers on core range "
           "[0-0 - 3-0]. L1 buffer allocated at 1172096 and static dataflow buffer region ends at 1258496")
    r = fit.build_report(
        {"mesh_device": "P150x4", "members": [{"name": "gptoss", "kind": "vllm", "port": 1},
                                              {"name": "llama8b", "kind": "vllm", "port": 2}]}, "exited",
        {"boot": {"llama8b": {"phase": "compile", "status": "failed", "error": err},
                  "gptoss": {"phase": "compile", "status": "aborted", "error": "barrier compiled failed"}},
         "failure": {"member": "llama8b", "phase": "compile", "error": err},
         "l1_kib_per_core_owned": {"gptoss": {"load_model": 370.5}},
         "memory_by_owner": {"gptoss": {"l1_kib_per_core": 370.5}, "llama8b": {"l1_kib_per_core": 2.0}}})
    assert r["failure"]["kind"] == "l1_clash"
    assert r["reason"].endswith("; L1 held per core: gptoss 370.5 KiB, llama8b 2.0 KiB")
    assert r["members"]["gptoss"]["l1_kib_per_core_owned"] == {"load_model": 370.5}
