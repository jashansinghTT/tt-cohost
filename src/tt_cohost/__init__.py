# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Co-host runtime: several models served from ONE process that owns the chips.

Only one process can open a Tenstorrent chip, so models that share chips must share a
process. This package is that process. It runs next to vLLM and ttnn (whichever build the
image or environment provides), so it imports nothing beyond the standard library at module
level and declares no hard dependencies.

- ``host``: the device host. It opens one mesh, runs every member through a phased boot,
  and serializes their device work on one device thread.
- ``vllm_remote``: a vLLM executor whose worker lives in the device host.
- ``launch``: the container entrypoint. It starts the host, then one vLLM server per member.
"""

#: Version of the JSON group config this runtime reads (whoever writes it: a deploy tool or a person).
CONFIG_VERSION = 1
