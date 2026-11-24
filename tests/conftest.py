# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import json
import sys
from pathlib import Path

import pytest

from tt_cohost._group import free_port

#: Stands in for ``python -m tt_cohost.launch``: no chips, no vLLM (see fake_launcher.py).
FAKE_LAUNCHER = [sys.executable, str(Path(__file__).with_name("fake_launcher.py"))]


@pytest.fixture
def group_config(tmp_path):
    """Write a group config of vLLM members on free ports; returns its path."""
    def make(names=("gptoss", "llama8b")):
        ports, members = set(), []
        for n in names:
            p = free_port(ports)
            ports.add(p)
            members.append({"name": n, "kind": "vllm", "port": p, "model": f"org/{n}"})
        path = tmp_path / "group.json"
        path.write_text(json.dumps({"cohost_config_version": 1, "mesh_device": "P150x4", "members": members}))
        return str(path)
    return make
