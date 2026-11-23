# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The plugin's co-host hooks when it has them (mesh provider, split compile/capture, vLLM's KV block override),
and the run-time patches for plugins that do not."""

import importlib.machinery
import json
import os
import sys
import threading
import types

import pytest

from tt_cohost import config as cfgmod
from tt_cohost import host as hostmod
from tt_cohost import launch

GROUP_TT = {"fabric_config": "FABRIC_1D_RING", "trace_region_size": 950000000}


def _cfg(**over):
    raw = {"cohost_config_version": 1, "mesh_device": "P150x4", "tt_config": GROUP_TT,
           "members": [{"name": "a", "kind": "vllm", "port": 21000, "model": "m/a", "env": {"COHOST_TEST_ENV": "a"}},
                       {"name": "b", "kind": "vllm", "port": 21001, "model": "m/b", "kv_tokens": 131072}]}
    raw.update(over)
    return cfgmod.loads(json.dumps(raw))


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def plugin(monkeypatch):
    """A fake vllm-tt-plugin; ``new=True`` gives it the hooks, ``new=False`` makes it an older plugin."""
    def make(new):
        state = types.SimpleNamespace(opened=[], closed=[], provider=None)
        mesh = types.SimpleNamespace(get_num_devices=lambda: 4)

        def open_mesh_device(tt_config, trace_mode, local_dp_rank=0, **kw):
            state.opened.append((tt_config, trace_mode, kw))
            return mesh

        class TTWorker:
            pass
        if new:
            TTWorker.compile_model = lambda self: None
            TTWorker.capture_model = lambda self: None
        worker = _module("vllm_tt_plugin.worker", open_mesh_device=open_mesh_device, TTWorker=TTWorker,
                         close_mesh_device=lambda m, tc: state.closed.append((m, tc)),
                         get_num_available_blocks_tt=lambda cfg, n=1: 1000)
        pkg = _module("vllm_tt_plugin", worker=worker)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "vllm_tt_plugin", pkg)
        monkeypatch.setitem(sys.modules, "vllm_tt_plugin.worker", worker)
        if new:
            mp = _module("vllm_tt_plugin.mesh_provider",
                         set_mesh_provider=lambda p: setattr(state, "provider", p))
            pkg.mesh_provider = mp
            monkeypatch.setitem(sys.modules, "vllm_tt_plugin.mesh_provider", mp)
        else:
            monkeypatch.delitem(sys.modules, "vllm_tt_plugin.mesh_provider", raising=False)
        state.mesh, state.worker = mesh, worker
        return state
    return make


def test_the_hooks_are_detected_only_when_the_plugin_has_them(plugin):
    plugin(new=True)
    assert hostmod.plugin_hooks() == {"mesh_provider": True, "split_warmup": True}
    plugin(new=False)
    assert hostmod.plugin_hooks() == {"mesh_provider": False, "split_warmup": False}


def test_with_a_mesh_provider_every_worker_gets_the_group_mesh_and_nothing_is_patched(plugin):
    p = plugin(new=True)
    real_open = p.worker.open_mesh_device
    host = hostmod.Host(_cfg())
    host.hooks = hostmod.plugin_hooks()
    host.install_mesh_hooks()
    assert p.provider is not None
    assert p.worker.open_mesh_device is real_open and p.worker.get_num_available_blocks_tt(None) == 1000
    got = [host.device.run(who, lambda: p.provider.acquire({"trace_mode": "decode_only"}, "decode_only",
                                                           model_fabric_config={"x": 1}))
           for who in ("b", "a")]
    assert got == [p.mesh, p.mesh]
    assert p.opened == [({"trace_mode": "decode_only", **GROUP_TT}, "decode_only", {"model_fabric_config": {"x": 1}})]
    p.provider.release(p.mesh)
    assert p.closed == []
    host.close()
    assert p.closed == [(p.mesh, {"trace_mode": "decode_only", **GROUP_TT})]   # closed as it was opened
    assert host.status()["plugin_hooks"] == {"mesh_provider": True, "split_warmup": True}


def test_an_older_plugin_is_patched_as_before(plugin):
    p = plugin(new=False)
    host = hostmod.Host(_cfg())
    host.hooks = hostmod.plugin_hooks()
    host.install_mesh_hooks()
    assert p.provider is None
    assert host.device.run("b", lambda: p.worker.open_mesh_device({}, "all")) is p.mesh
    vcfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=64))
    assert host.device.run("b", lambda: p.worker.get_num_available_blocks_tt(vcfg, 4)) == 1000  # 2048 > 1000
    assert host.device.run("a", lambda: p.worker.get_num_available_blocks_tt(vcfg, 4)) == 1000


@pytest.mark.parametrize("args, kv, want", [
    ([], 131072, "2048"),                         # TT's default block size, 64
    (["--block-size", "32"], 131072, "4096"),
    (["--block-size=64"], 100, "2"),              # rounded up to whole blocks
])
def test_kv_tokens_become_vllms_block_override_with_the_hooks(args, kv, want):
    m = cfgmod.Member(name="b", kind="vllm", port=1, model="m", args=args, kv_tokens=kv)
    argv = launch.frontend_argv(m, {"mesh_provider": True})
    assert argv[argv.index("--num-gpu-blocks-override") + 1] == want
    assert "--num-gpu-blocks-override" not in launch.frontend_argv(m, {"mesh_provider": False})
    assert "--num-gpu-blocks-override" not in launch.frontend_argv(m)


def test_a_members_own_block_override_is_left_alone():
    m = cfgmod.Member(name="b", kind="vllm", port=1, model="m", args=["--num-gpu-blocks-override", "7"],
                      kv_tokens=131072)
    argv = launch.frontend_argv(m, {"mesh_provider": True})
    assert argv.count("--num-gpu-blocks-override") == 1 and argv[argv.index("--num-gpu-blocks-override") + 1] == "7"


@pytest.fixture
def worker_base(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", types.ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base",
                        types.SimpleNamespace(CompilationTimes=lambda **k: k))


def _warm(host, vm):
    """Run vm's phased warmup while a second member 'b' passes the gates and takes its capture turn after it."""
    def other():
        host.loaded.wait("b")
        host.compiled.wait("b")
        with host.capture_turn("b"):
            pass
    t = threading.Thread(target=other, daemon=True)
    t.start()
    out = vm._phased_warmup()
    t.join(5)
    return out


def test_with_split_warmup_the_host_runs_the_plugins_own_compile_and_capture(worker_base):
    host = hostmod.Host(_cfg())
    vm = hostmod.VllmMember(host, host.cfg.members[0])
    calls = []

    class Worker:
        model_runner = property(lambda self: pytest.fail("the runner's model must not be driven directly"))

        def compile_model(self):
            calls.append(("compile", threading.current_thread() is host.device._thread, os.environ.get("COHOST_TEST_ENV")))

        def capture_model(self):
            calls.append(("capture", host._capture_turn, os.environ.get("COHOST_TEST_ENV")))
    vm.wrapper = types.SimpleNamespace(worker=Worker())
    out = _warm(host, vm)
    assert calls == [("compile", True, "a"), ("capture", 0, "a")]
    assert set(out) == {"language_model", "encoder"} and host._capture_turn == 2
    assert "COHOST_TEST_ENV" not in os.environ


def test_an_older_plugin_is_warmed_through_the_runners_model(worker_base, monkeypatch):
    calls = []
    tt_platform = types.SimpleNamespace(check_and_update_config=lambda cfg: None, sample_on_device_mode="all")
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.ModuleType("vllm_tt_plugin"))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.platform", types.SimpleNamespace(TTPlatform=tt_platform))
    model = types.SimpleNamespace(
        warmup_model_prefill=lambda enable_trace, **k: calls.append(("prefill", enable_trace)),
        warmup_model_decode=lambda enable_trace, **k: calls.append(("decode", enable_trace)))
    runner = types.SimpleNamespace(model=model, kv_caches=[], tt_max_batch_size=32, max_num_blocks_per_req=8,
                                   trace_mode="all")
    host = hostmod.Host(_cfg())
    vm = hostmod.VllmMember(host, host.cfg.members[0])
    vm.wrapper = types.SimpleNamespace(worker=types.SimpleNamespace(model_runner=runner))
    _warm(host, vm)
    assert calls == [("prefill", False), ("decode", False), ("prefill", True), ("decode", True)]
