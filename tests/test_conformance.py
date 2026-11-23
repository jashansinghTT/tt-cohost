# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The conformance test, end to end against a fake launcher (no chips)."""

import json

import pytest
from conftest import FAKE_LAUNCHER

from tt_cohost import conformance


def test_conformance_passes_when_the_member_answers_the_same_next_to_its_neighbour(tmp_path, monkeypatch, group_config):
    monkeypatch.setenv("FAKE_MODE", "ok")
    r = conformance.run_conformance(group_config(), "llama8b", str(tmp_path / "out"), launcher=FAKE_LAUNCHER)
    assert r["pass"] and r["sequential"]["identical"] == 16 and r["interleaved"]["identical"] == 16
    assert r["neighbour_requests_while_interleaved"]["gptoss"] >= 1
    assert (tmp_path / "out" / "alone.json").is_file() and (tmp_path / "out" / "report.json").is_file()


def test_conformance_names_the_prompt_that_changed_next_to_a_neighbour(tmp_path, monkeypatch, group_config, capsys):
    monkeypatch.setenv("FAKE_MODE", "drift")
    real = conformance.run_conformance
    monkeypatch.setattr(conformance, "run_conformance", lambda *a, **k: real(*a, **{**k, "launcher": FAKE_LAUNCHER}))
    rc = conformance.main(["--config", group_config(), "--member", "llama8b", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 1 and "sequential 15/16 identical" in out and "FAIL" in out
    assert "'Write a haiku about silicon:'" in out


def test_conformance_with_a_baseline_file_skips_the_solo_boot(tmp_path, monkeypatch, group_config):
    monkeypatch.setenv("FAKE_MODE", "ok")
    rows = [{"prompt": p, "text": f" {p[::-1]} (64)"} for p in ["Water boils at", "def fibonacci(n):"]]
    base = tmp_path / "alone.json"
    base.write_text(json.dumps({"rows": rows}))
    r = conformance.run_conformance(group_config(), "llama8b", str(tmp_path / "out"), launcher=FAKE_LAUNCHER,
                                    prompts=[r["prompt"] for r in rows], baseline=str(base))
    assert r["pass"] and r["sequential"]["total"] == 2
    assert not (tmp_path / "out" / "alone").exists()


def test_conformance_refuses_an_app_member_or_a_member_without_neighbours(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"cohost_config_version": 1, "mesh_device": "P150", "members": [
        {"name": "a", "kind": "vllm", "port": 1, "model": "m"}]}))
    with pytest.raises(conformance.ToolError, match="no neighbour"):
        conformance.run_conformance(str(path), "a", str(tmp_path / "o"))
    with pytest.raises(conformance.ToolError, match="no member"):
        conformance.run_conformance(str(path), "b", str(tmp_path / "o"))
