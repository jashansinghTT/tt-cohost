# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt_kernel.cohost`` as an alias of ``tt_cohost``, for members written for tt-model's runtime.

App members published before this package existed import ``from tt_kernel.cohost.apps import AppMember``.
``install()`` puts a finder first on ``sys.meta_path`` that resolves ``tt_kernel.cohost`` and its submodules to
the ``tt_cohost`` modules themselves (same objects, so one ``host`` module, one ``AppMember`` class), but only
when nothing else on the path provides ``tt_kernel.cohost``: where tt-model's old runtime is still installed,
it is imported as before. ``tt_kernel`` (tt-model-manager's package name) is never shadowed: an empty
placeholder is used only when no ``tt_kernel`` exists, and it holds nothing but ``cohost``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys

OLD = "tt_kernel.cohost"
NEW = "tt_cohost"
log = logging.getLogger("tt_cohost")


def _find_elsewhere(finder, name, path, target):
    for f in sys.meta_path:
        if f is finder or not hasattr(f, "find_spec"):
            continue
        spec = f.find_spec(name, path, target)
        if spec is not None:
            return spec
    return None


class _Alias(importlib.abc.Loader):
    """Loads ``name`` as the already importable module ``target`` (the same object)."""

    def __init__(self, target: str):
        self.target = target

    def create_module(self, spec):
        mod = importlib.import_module(self.target)
        self._spec = mod.__spec__       # the import system overwrites __spec__ with the alias's spec
        return mod

    def exec_module(self, module):
        module.__spec__ = self._spec    # keep relative imports and reloads on the real name


class _Placeholder(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__doc__ = f"Placeholder so {OLD} resolves to {NEW}; tt-model-manager is not installed."


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "tt_kernel":
            return _find_elsewhere(self, name, path, target) or importlib.machinery.ModuleSpec(
                name, _Placeholder(), is_package=True)
        if name == OLD:
            spec = _find_elsewhere(self, name, path, target)
            if spec is not None:
                return spec
            log.info("cohost: %s is provided by %s (members written for tt-model's runtime)", OLD, NEW)
            return importlib.machinery.ModuleSpec(name, _Alias(NEW), is_package=True)
        old = sys.modules.get(OLD)
        if name.startswith(OLD + ".") and old is not None and old is sys.modules.get(NEW):
            new = NEW + name[len(OLD):]
            spec = importlib.util.find_spec(new)
            if spec is None:
                return None
            return importlib.machinery.ModuleSpec(name, _Alias(new),
                                                  is_package=spec.submodule_search_locations is not None)
        return None


def install() -> None:
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())

