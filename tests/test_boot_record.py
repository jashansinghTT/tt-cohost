# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The boot record: which member failed, in which phase, and why; what the fit report is built from."""

import json
import sys
import threading
import types

import pytest

from tt_cohost import config as cfgmod
from tt_cohost import host as hostmod
from tt_cohost import launch


def _cfg(members=None, **over):
    raw = {"cohost_config_version": 1, "mesh_device": "P150x4",
           "members": members or [{"name": "a", "kind": "vllm", "port": 21000, "model": "m/a"},
                                  {"name": "b", "kind": "vllm", "port": 21001, "model": "m/b"}]}
    raw.update(over)
    return cfgmod.loads(json.dumps(raw))


def test_the_member_that_failed_is_the_cause_and_the_ones_that_followed_are_aborted():
    host = hostmod.Host(_cfg())
    host.enter_phase("a", "compile")
    host.member_failed("a", RuntimeError("Statically allocated dataflow buffers ... clash with L1 buffers"))
    host.compiled.fail("a: clash")
    with pytest.raises(hostmod.BarrierFailed) as err:
        host.compiled.wait("b")
    host.member_failed("b", err.value)
    assert host.boot["a"] == {"phase": "compile", "status": "failed",
                              "error": "RuntimeError: Statically allocated dataflow buffers ... clash with L1 buffers"}
    assert host.boot["b"]["status"] == "aborted"
    assert host.failure == {"member": "a", "phase": "compile", "error": host.boot["a"]["error"]}


def test_a_failed_gate_keeps_the_first_cause():
    b = hostmod.Barrier("loaded", ["a", "b"])
    b.fail("a: out of DRAM")
    b.fail("b: barrier loaded failed: a: out of DRAM")
    with pytest.raises(hostmod.BarrierFailed, match="failed: a: out of DRAM"):
        b.wait("b")


def test_a_timed_out_gate_names_the_member_that_never_arrived(monkeypatch):
    monkeypatch.setattr(hostmod, "BARRIER_TIMEOUT_S", 0.1)
    host = hostmod.Host(_cfg())
    host.enter_phase("b", "load")
    with pytest.raises(hostmod.BarrierTimeout) as err:
        host.loaded.wait("a")
    assert err.value.missing == ["b"]
    host.member_failed("a", err.value)
    assert host.boot["b"]["status"] == "timeout" and "did not finish load" in host.boot["b"]["error"]
    assert host.boot["a"]["status"] == "aborted"
    assert host.failure["member"] == "b"


def test_a_later_error_does_not_replace_a_members_first_one():
    host = hostmod.Host(_cfg())
    host.member_failed("a", RuntimeError("out of DRAM"), phase="load")
    host.member_failed("a", RuntimeError("its server exited (code 1)"))
    assert host.boot["a"]["error"] == "RuntimeError: out of DRAM"


def test_a_ready_member_is_ok_and_the_status_file_follows_every_change(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    monkeypatch.setenv("TT_COHOST_STATUS_FILE", str(path))
    host = hostmod.Host(_cfg())
    host.enter_phase("a", "load")
    assert json.loads(path.read_text())["boot"]["a"] == {"phase": "load", "status": "booting", "error": None}
    host.member_ready("a")
    on_disk = json.loads(path.read_text())
    assert on_disk["boot"]["a"]["status"] == "ok" and on_disk["members"]["a"]["ready"]
    assert on_disk["failure"] is None


def test_l1_is_recorded_for_every_phase_of_a_member(monkeypatch):
    host = hostmod.Host(_cfg())
    readings = iter([1.0, 2.0, 2.0, 3.0])
    monkeypatch.setattr(host, "dram_gb_per_chip", lambda: next(readings))
    l1 = iter([100.0, 380.8])
    monkeypatch.setattr(host, "l1_kib_per_core", lambda: next(l1))
    host.account("a", "load_model", host.dram_gb_per_chip())
    host.account("a", "compile", host.dram_gb_per_chip())
    assert host.l1_phases == {"a": {"load_model": 100.0, "compile": 380.8}}
    assert host.status()["l1_kib_per_core"] == {"a": {"load_model": 100.0, "compile": 380.8}}
    assert host.dram == {"a": {"load_model": 1.0, "compile": 1.0}}


def test_a_member_that_opens_its_own_device_fails_by_name_and_the_guard_goes_with_the_mesh(monkeypatch):
    opened = []
    ttnn = types.ModuleType("ttnn")
    ttnn.open_mesh_device = lambda *a, **k: opened.append("mesh") or types.SimpleNamespace(get_num_devices=lambda: 4)
    ttnn.open_device = lambda *a, **k: opened.append("device")
    real_open = ttnn.open_mesh_device
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    ttw = types.SimpleNamespace(open_mesh_device=lambda *a, **k: ttnn.open_mesh_device(),
                                close_mesh_device=lambda *a: None, get_num_available_blocks_tt=lambda cfg, n=1: 1)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.SimpleNamespace(worker=ttw))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.worker", ttw)
    host = hostmod.Host(_cfg())
    host.install_mesh_hooks()
    host.device.run("a", lambda: ttw.open_mesh_device({}, "all"))
    host.device.run("b", lambda: ttw.open_mesh_device({}, "all"))      # the shared mesh, not a second open
    assert opened == ["mesh"]
    with pytest.raises(hostmod.MemberOpenedDevice, match="b called ttnn.open_device"):
        host.device.run("b", lambda: ttnn.open_device(device_id=0))
    host.close()
    assert ttnn.open_mesh_device is real_open


def test_an_app_member_that_fails_its_warmup_is_recorded_in_that_phase(monkeypatch):
    from tt_cohost import apps

    class Fake(apps.AppMember):
        def __init__(self, host, member):
            self.n = member.name

        def load(self, mesh):
            pass

        def warmup(self):
            if self.n == "tts":
                raise RuntimeError("Statically allocated circular buffers in program 2090 clash with L1 buffers")

        def release_traces(self):
            pass

    members = [{"name": n, "kind": "app", "port": p, "app": "x:y"} for n, p in (("stt", 1), ("tts", 2))]
    host = hostmod.Host(_cfg(members))
    monkeypatch.setattr(host, "open_mesh_direct", lambda: "MESH")
    monkeypatch.setattr(apps, "load_factory", lambda spec: Fake)
    with pytest.raises(RuntimeError, match="clash"):
        apps.boot_apps(host, host.cfg.members)
    assert host.boot["tts"]["status"] == "failed" and host.boot["tts"]["phase"] == "compile"
    assert host.failure["member"] == "tts"


def test_a_frontend_that_exits_on_its_own_is_recorded_with_its_exit_code(monkeypatch):
    """No host-side error to name: the member whose server process died is the cause."""
    monkeypatch.setattr(launch, "frontend_argv", lambda m: [sys.executable, "-c", "import sys; sys.exit(3)"])
    host = hostmod.Host(_cfg([{"name": "a", "kind": "vllm", "port": 1, "model": "m"}]))
    stopping, rc = threading.Event(), []
    t = threading.Thread(target=lambda: rc.append(launch._supervise(host.cfg, host, stopping)), daemon=True)
    t.start()
    t.join(30)
    stopping.set()
    t.join(10)
    assert rc == [1]
    assert host.boot["a"]["status"] == "failed"
    assert host.boot["a"]["error"] == "RuntimeError: its server exited (code 3)"


def test_a_vllm_member_whose_init_worker_fails_fails_the_gates(tmp_path, monkeypatch):
    """A preload hook that cannot import must stop the group at once, not leave the others at a gate."""
    host = hostmod.Host(_cfg(socket_dir=str(tmp_path / "s")))
    vm = hostmod.VllmMember(host, host.cfg.members[0])

    def broken(kwargs):
        raise ImportError("No module named 'models.cohost_llm'")
    monkeypatch.setattr(vm, "_init_worker", broken)
    vm.serve()
    from tt_cohost import rpc
    c = rpc.Client(host.cfg.socket_path("a"))
    with pytest.raises(rpc.RemoteError, match="cohost_llm"):
        c.call("init_worker", kwargs={})
    assert host.boot["a"]["status"] == "failed" and host.boot["a"]["phase"] == "init"
    with pytest.raises(hostmod.BarrierFailed, match="cohost_llm"):
        host.loaded.wait("b")


def test_an_unwritable_status_file_never_fails_a_boot_step(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_COHOST_STATUS_FILE", str(tmp_path / "missing-dir" / "status.json"))
    host = hostmod.Host(_cfg())
    host.enter_phase("a", "load")
    host.member_failed("a", RuntimeError("out of DRAM"))
    assert host.boot["a"]["status"] == "failed"
