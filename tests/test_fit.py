# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The fit check, end to end against a fake launcher (no chips)."""

import json

import pytest
from conftest import FAKE_LAUNCHER

from tt_cohost import fit


@pytest.fixture
def run(tmp_path, monkeypatch, group_config):
    def go(mode, **kw):
        monkeypatch.setenv("FAKE_MODE", mode)
        return fit.run_fit(group_config(), workdir=str(tmp_path / "fit"), launcher=FAKE_LAUNCHER, **kw)
    return go


def test_a_group_that_boots_and_answers_is_shared(run):
    r = run("ok")
    assert r["verdict"] == "shared" and r["fits"] and r["failure"] is None
    g = r["members"]["gptoss"]
    assert g["status"] == "ok" and g["probe"]["ok"] and g["probe"]["text"]
    assert g["dram_gib_per_chip"] == {"load_model": 1.5, "initialize_from_config": 1.0}
    assert g["l1_kib_per_core"] == {"load_model": 370.5}
    assert r["dram_gib_per_chip_total"] == 9.0 and r["boot_seconds"] is not None


def test_an_l1_clash_is_separate_and_names_the_member_and_the_phase(run):
    r = run("clash")
    assert r["verdict"] == "separate" and not r["fits"]
    assert r["failure"]["member"] == "llama8b" and r["failure"]["phase"] == "compile"
    assert r["failure"]["kind"] == "l1_clash"
    assert r["reason"].startswith("llama8b failed in compile: L1 clash")
    assert "clash with L1 buffers on core range [0-0 - 3-0]" in r["reason"]
    assert r["members"]["gptoss"]["status"] == "aborted"


def test_a_member_stuck_in_load_is_named_when_the_boot_times_out(run):
    r = run("hang", timeout_s=4)
    assert r["verdict"] == "separate" and r["outcome"] == "timeout"
    assert r["failure"]["member"] == "llama8b" and r["failure"]["kind"] == "timeout"
    assert "still in load" in r["reason"]
    assert r["members"]["gptoss"]["status"] == "aborted"


def test_late_compiles_are_a_warning_unless_strict(run):
    r = run("late")
    assert r["verdict"] == "shared" and r["members"]["gptoss"]["compiles_after_capture"] == 3
    assert any("gptoss compiled 3 program(s)" in w for w in r["warnings"])
    r = run("late", strict=True)
    assert r["verdict"] == "separate" and r["failure"]["kind"] == "late_compile"


def test_a_launcher_that_dies_before_the_group_exists_is_a_tool_error(run):
    with pytest.raises(fit.ToolError, match="No module named 'vllm'"):
        run("crash")


def test_the_cli_exits_0_with_a_separate_verdict_and_2_on_a_bad_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fit, "run_fit", lambda *a, **k: fit.build_report(
        {"mesh_device": "P150", "members": [{"name": "a", "kind": "vllm", "port": 1}]}, "exited",
        {"boot": {"a": {"phase": "load", "status": "failed", "error": "RuntimeError: Out of Memory: Not enough "
                        "space to allocate 2147483648 B DRAM buffer across 8 banks"}},
         "failure": {"member": "a", "phase": "load", "error": "x"}}))
    out = tmp_path / "r.json"
    assert fit.main(["--config", "unused", "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["verdict"] == "separate" and r["failure"]["kind"] == "dram"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cohost_config_version": 2, "mesh_device": "P150", "members": []}))
    monkeypatch.undo()
    assert fit.main(["--config", str(bad)]) == 2
    assert "invalid group config" in capsys.readouterr().err


@pytest.mark.parametrize("error, kind, shown", [
    ("RuntimeError: TT_THROW @ /x/dataflow_buffer.cpp:2601: tt::exception\ninfo:\nStatically allocated dataflow "
     "buffers in program 2950 clash with L1 buffers on core range [0-0 - 3-0]. L1 buffer allocated at 1172096",
     "l1_clash", "Statically allocated dataflow buffers in program 2950 clash"),
    ("MemberOpenedDevice: llama8b called ttnn.open_mesh_device while the group's mesh is open; a co-host member "
     "must use the mesh it is given", "opens_device", "llama8b called ttnn.open_mesh_device"),
    ("did not finish load within 7200 s (barrier loaded: ['b'] never arrived)", "timeout", "did not finish load"),
    ("RuntimeError: its server exited (code 1)", "server_exited", "its server exited (code 1)"),
    ("RuntimeError: TT_THROW @ /x/y.cpp:1: tt::exception\ninfo:\nsomething else broke", "error",
     "something else broke"),
])
def test_errors_are_classified_by_the_line_that_says_what_happened(error, kind, shown):
    got_kind, line = fit.classify(error)
    assert got_kind == kind and shown in line
