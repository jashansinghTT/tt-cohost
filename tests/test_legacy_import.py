# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Members written for tt-model's runtime (``from tt_kernel.cohost.apps import AppMember``) load under tt_cohost."""

import os
import subprocess
import sys
import textwrap

import tt_cohost

MEMBER = """
from tt_kernel.cohost.apps import AppMember

class Speech(AppMember):
    pass

def factory(host, member):
    return Speech()
"""


def _run(tmp_path, code, extra_path=()):
    """Run ``code`` in a fresh interpreter (the alias changes sys.modules) with ``tmp_path`` importable."""
    (tmp_path / "old_member.py").write_text(MEMBER)
    pkg_root = os.path.dirname(os.path.dirname(tt_cohost.__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), *map(str, extra_path), pkg_root]))
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=str(tmp_path),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_old_app_member_loads_without_tt_kernel(tmp_path):
    out = _run(tmp_path, """
        import importlib, sys
        import tt_cohost.launch                       # registered at import time, before launch.main() runs
        import tt_cohost.apps
        app = tt_cohost.apps.load_factory("old_member:factory")(None, None)
        assert isinstance(app, tt_cohost.apps.AppMember)
        import tt_kernel.cohost, tt_kernel.cohost.host, tt_cohost.host
        assert tt_kernel.cohost is tt_cohost and tt_kernel.cohost.host is tt_cohost.host
        assert tt_cohost.apps.__spec__.name == "tt_cohost.apps" and tt_cohost.__spec__.name == "tt_cohost"
        assert not hasattr(tt_cohost.apps, "__path__")
        loaded = {id(m): getattr(m, "__file__", None) or "" for m in list(sys.modules.values())}
        assert sum(f.endswith(("tt_cohost/apps.py", "tt_cohost/host.py")) for f in loaded.values()) == 2  # no copies
        for missing in ("tt_kernel.manifest", "tt_kernel.cohost.nope"):
            try:
                importlib.import_module(missing)
                raise SystemExit(f"{missing} imported")
            except ModuleNotFoundError:
                pass
        print("ok")
    """)
    assert out.strip() == "ok"


def test_tt_model_managers_own_tt_kernel_is_kept(tmp_path):
    # tt-model-manager after the runtime moved out: a real tt_kernel package without cohost
    root = tmp_path / "tmm"
    (root / "tt_kernel").mkdir(parents=True)
    (root / "tt_kernel" / "__init__.py").write_text("REAL = True\n")
    (root / "tt_kernel" / "manifest.py").write_text("X = 1\n")
    out = _run(tmp_path, """
        import tt_cohost.apps, tt_cohost.apps as new
        from old_member import Speech
        import tt_kernel, tt_kernel.manifest
        assert tt_kernel.REAL and tt_kernel.manifest.X == 1
        assert Speech.__mro__[1] is new.AppMember
        print("ok")
    """, extra_path=[root])
    assert out.strip() == "ok"


def test_installed_old_runtime_still_wins(tmp_path):
    # an image that still ships tt-model's runtime: its own modules are imported, not aliased
    root = tmp_path / "old"
    (root / "tt_kernel" / "cohost").mkdir(parents=True)
    (root / "tt_kernel" / "__init__.py").write_text("")
    (root / "tt_kernel" / "cohost" / "__init__.py").write_text("")
    (root / "tt_kernel" / "cohost" / "apps.py").write_text("class AppMember:\n    OLD = True\n")
    out = _run(tmp_path, """
        import tt_cohost.apps
        from old_member import Speech
        assert Speech.OLD and Speech.__mro__[1] is not tt_cohost.apps.AppMember
        print("ok")
    """, extra_path=[root])
    assert out.strip() == "ok"
