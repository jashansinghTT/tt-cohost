# tt-cohost

Serve several models from one process on the same Tenstorrent chips.

A Tenstorrent chip can be opened by one process at a time: UMD takes a per-chip lock when the device
starts and holds it until the process exits. Two models that must share chips therefore have to share a
process. `tt_cohost` is that process: a **device host** that opens one mesh, boots every model (a
**member**) through a phased boot, and runs every member's device work on one thread, one member at a
time. Each vLLM member keeps its own stock vLLM OpenAI server on its own port; app members (speech,
music, embedders: models with their own HTTP API) run their ASGI app on their own port.

The runtime moved here from tt-model-manager (`tt_kernel.cohost`) with its history. Measurements and the
reasoning behind the design are in [docs/design.md](docs/design.md); the rules a member must follow are in
[docs/contract.md](docs/contract.md).

## Shared chips or separate chips?

Separate chips (one server process per model, each on its own chips) always work and keep every
model's fastest path. Share chips only when you have more models than chip sets and they fit:

- **Members take turns.** Nothing runs at the same time; two busy members each get roughly half the
  device. Measured on a QuietBox 2 (4 Blackhole chips), GPT-OSS-20B + Llama-3.1-8B in one group, 32 users
  each: 825 + 790 = 1,615 tok/s with both busy, against 1,444 and 1,150 tok/s when each was busy
  alone. Greedy output was identical to each model served alone (16 of 16 prompts each).
- **DRAM adds up** across members (weights + KV caches; cap each KV pool with `kv_tokens`).
- **L1 is the binding budget.** A member that keeps buffers in L1 between steps takes that L1 from every
  other member's programs; the next member then fails to compile with `Statically allocated dataflow
  buffers ... clash with L1 buffers`. The boot fails loudly, never silently.
- **One environment.** Every member runs on the same ttnn, vLLM and plugin builds.

## Install

`tt_cohost` has no hard dependencies and imports only the standard library at module level. Install it
into an environment that already holds tt-metal (ttnn), the TT vLLM build and vllm-tt-plugin:

```bash
pip install "tt-cohost[vllm] @ git+https://github.com/jashansinghTT/tt-cohost@<sha>"
```

vLLM is deliberately not a declared dependency (as in vllm-tt-plugin): it must be the TT build, never the
CUDA wheel from PyPI. The `vllm` extra only lists `cloudpickle` and `uvicorn`, which the TT vLLM build
already brings. A group of app members alone needs ttnn and `uvicorn`.

## Run a group

```bash
python -m tt_cohost.launch --config group.json      # or: tt-cohost --config group.json
# or pass the config inline: TT_COHOST_CONFIG='{...}' python -m tt_cohost.launch
```

The launcher prints `cohost: all members ready` once every vLLM member lists its model on `/v1/models`
and every app member's port accepts connections. If any member fails to boot or exits later, the whole
group stops with a non-zero exit code: the members share the device, so a half-alive group is not
served.

Each vLLM member's server is a stock `vllm.entrypoints.openai.api_server` started with
`--distributed-executor-backend tt_cohost.vllm_remote.RemoteExecutor`: its scheduler, parsers and API
stay vLLM's own, and only its worker runs in the device host (over a Unix socket private to the host's
user).

### vllm-tt-plugin versions

The host uses the plugin's co-host hooks when the installed plugin has them, and says which path ran in its log
and in the status (`plugin_hooks`):

| hook | with it | without it (older plugins) |
|---|---|---|
| mesh provider (`vllm_tt_plugin.mesh_provider`) | the host registers a provider that hands every worker the group's mesh | the host patches the plugin's `open_mesh_device` / `close_mesh_device` |
| KV block override (vLLM's `--num-gpu-blocks-override`) | a member's `kv_tokens` is passed to its vLLM server | the host patches the plugin's `get_num_available_blocks_tt` |
| split warmup (`TTWorker.compile_model` / `capture_model`) | the host calls the worker's own two steps | the host calls the runner's `warmup_model_prefill` / `warmup_model_decode` itself |

The KV override goes with the mesh provider: a plugin without the provider gets the patched block count.

## Group config (version 1)

```json
{
  "cohost_config_version": 1,
  "mesh_device": "P150x4",
  "tt_config": {"fabric_config": "FABRIC_1D_RING", "trace_region_size": 950000000},
  "status_port": 21099,
  "members": [
    {"name": "gptoss", "kind": "vllm", "port": 21000, "model": "openai/gpt-oss-20b",
     "args": ["--max-num-seqs", "32"], "env": {"GPT_OSS_20B_FUSED_B1": "0"}},
    {"name": "llama8b", "kind": "vllm", "port": 21001, "model": "meta-llama/Llama-3.1-8B-Instruct",
     "kv_tokens": 65536, "preload": "models.cohost_llm.fit:tt_transformers"}
  ]
}
```

| key | meaning |
|---|---|
| `cohost_config_version` | must be `1` |
| `mesh_device` | the one mesh every member shares (`MESH_DEVICE`), e.g. `P150`, `P300`, `P150x4` |
| `tt_config` | the plugin's `tt` additional config the mesh is opened with when no vLLM worker opens it (app-only groups): `fabric_config`, `trace_region_size`, `l1_small_size`, `num_command_queues` |
| `status_port` | optional port of the status endpoint |
| `socket_dir` | optional; where the frontends' Unix sockets live (default `/tmp/tt-cohost`) |
| `members` | the members, in boot and capture order |

| member key | meaning |
|---|---|
| `name` | unique member name |
| `kind` | `vllm` or `app` (see below). A group may mix kinds; it then lists its vLLM members first. |
| `port` | the member's HTTP port (unique in the group) |
| `model`, `args` | vllm: what `vllm serve` gets (`--model <model> <args...>`) |
| `env` | set around this member's load-time calls (and in its vLLM frontend), restored afterwards |
| `kv_tokens` | vllm: this member's KV pool in tokens (all users), so the members' DRAM adds up. With a plugin that has the co-host hooks it is passed as vLLM's `--num-gpu-blocks-override` (rounded up to whole blocks) and sets the pool; with an older plugin it caps the plugin's own count |
| `preload` | `module:function` called in the member's env before it is built (a fit patch, e.g. a smaller-L1 program choice) |
| `app` | app: `module:factory` returning an `AppMember` |

The config is refused before any chip is touched if the version is wrong, names or ports repeat, a kind is
unknown, an app member has no `module:factory`, or a mixed group lists an app member before a vLLM member.

## Member kinds

- **`vllm`**: any model vllm-tt-plugin serves. The host runs the plugin's normal `TTWorker` for it; the
  worker gets the group's shared mesh instead of opening its own and never closes it. The mesh opens with the
  first worker's own arguments, with every key of the group's `tt_config` on top.
- **`app`**: a model with its own HTTP API, wrapped in an adapter that implements the app-member
  contract below. Its ASGI app is served from the host process on the member's port.

A model whose server starts its own process (and so opens the chips itself) cannot be a member; serve it
on separate chips.

### The app-member contract

```python
from tt_cohost.apps import AppMember

class MyModel(AppMember):          # every method runs on the device thread
    def load(self, mesh): ...      # weights, caches, model objects on the shared mesh; never open a device
    def warmup(self): ...          # compile every program, allocate every lazy buffer; may capture traces
    def release_traces(self): ...  # release every trace warmup captured
    def asgi_app(self): ...        # the FastAPI/Starlette app that serves this member

def factory(host, member):         # named in the config as "app": "my_pkg.cohost:factory"
    return MyModel(host, member)
```

- Request handlers run their device work through `host.step(member.name, fn)`. That makes each piece of
  device work exclusive and counts any program it compiles after capture.
- `warmup()` is called twice; the second call must compile and allocate nothing new.
- Keep nothing resident in L1 between steps that other members' programs need.
- Do not change process-wide state while loading, or restore it (one interpreter: a model that points
  `HF_MODEL` at its own weights changes it for every member loaded after it).

## The boot phases, and why

A trace replays the exact addresses it used at capture. A buffer or program binary allocated after some
trace was captured may sit in that trace's scratch memory and be overwritten when it replays. So the host
boots ALL members together, phase by phase:

1. **Load**: every member's weights and KV cache (app members first in a mixed group, so their weights are
   in DRAM before any vLLM member sizes its KV pool).
2. **Compile**: every member warms up with tracing off, so every program is compiled and every lazily
   allocated buffer exists. App members warm up once and release their traces.
3. **Capture**: traces are captured one member at a time, in config order. Nothing new is compiled or
   allocated after this.

While serving, a member owns the device for a whole step (vLLM's `execute_model` through its
`sample_tokens`; an app's `host.step`), so another member's replay never lands between a step's forward
and its readback. Owners are served first come, first served. A program compiled after capture (a new
batch size or prompt length the warmup did not build) is counted per member and logged, because it can
corrupt a parked trace.

## Status endpoint

With `status_port` set, `GET http://<host>:<status_port>/` returns:

```json
{
  "mesh_device": "P150x4",
  "members": {"gptoss": {"kind": "vllm", "port": 20000, "ready": true},
              "llama8b": {"kind": "vllm", "port": 20001, "ready": true}},
  "device_owner": null,
  "compiles_after_capture": {"gptoss": 3, "llama8b": 21},
  "dram_gib_per_chip": {"members": {
      "gptoss": {"load_model": 7.101, "initialize_from_config": 3.287, "compile": 0.081, "capture": 0.02},
      "llama8b": {"load_model": 1.848, "initialize_from_config": 2.125, "compile": 0.081, "capture": 0.008}},
    "total": 14.47},
  "l1_kib_per_core_after_phase": {"gptoss": 380.8, "llama8b": 380.8}
}
```

(GPT-OSS-20B + Llama-3.1-8B on a QuietBox 2, after one request each.)

`dram_gib_per_chip` is what each member's phases added to per-chip DRAM; `l1_kib_per_core_after_phase` is
the L1 held by live buffers after that member's last measured phase, and `l1_kib_per_core` the same after each of
its phases. `boot` holds each member's phase, status (`booting`, `ok`, `failed`, `timeout`, `aborted`) and error,
and `failure` the first member that failed on its own. With a ttnn that accounts memory per owner (`ttnn.memory_owner`,
`ttnn.get_memory_usage_by_owner`), every member's device work runs as that member: `l1_kib_per_core_owned` is the L1
each member's own buffers hold after each of its phases, `memory_by_owner` the latest L1 and DRAM per owner, and a fit
check that fails on an L1 clash names who holds the L1. `TT_COHOST_L1_REPORT=1` also logs
every live L1 buffer after each phase.

If a group stops answering, `device_owner` names the member whose call holds the device, and
`kill -USR1 <launcher pid>` writes every thread's Python stack to the launcher's stderr. The device thread
(the main thread) inside a ttnn call means the device itself is stuck. Member threads waiting in
`Device.acquire` behind an idle device thread would be a deadlock in the host.

## Will they fit? The fit check

```bash
python -m tt_cohost.fit --config group.json --json fit.json
```

The fit check boots the group exactly as the launcher does (in a child process that owns the chips), sends each
vLLM member one short greedy request, stops the group and writes a report. The verdict is `shared` when every member
booted and answered, else `separate` with the member, the phase and the reason. Measured on a QuietBox 2
(GPT-OSS-20B + Llama-3.1-8B on P150x4, Llama without its fit hook):

```text
fit: separate: llama8b-nohook failed in compile: L1 clash (its programs need L1 another member holds): Statically
allocated dataflow buffers in program 2950 clash with L1 buffers on core range [0-0 - 3-0]. L1 buffer allocated at
1172096 and static dataflow buffer region ends at 1258496
```

```json
{
  "fit_report_version": 1,
  "verdict": "separate",
  "fits": false,
  "reason": "llama8b-nohook failed in compile: L1 clash (its programs need L1 another member holds): Statically ...",
  "failure": {"member": "llama8b-nohook", "phase": "compile", "kind": "l1_clash", "error": "Statically allocated ..."},
  "mesh_device": "P150x4",
  "outcome": "exited",
  "boot_seconds": 156.0,
  "dram_gib_per_chip_total": 14.441,
  "members": {
    "gptoss": {"kind": "vllm", "port": 20000, "status": "aborted", "phase": "compile", "error_kind": "l1_clash",
               "dram_gib_per_chip": {"load_model": 7.101, "initialize_from_config": 3.287, "compile": 0.08},
               "l1_kib_per_core": {"load_model": 370.5, "initialize_from_config": 370.5, "compile": 380.8},
               "compiles_after_capture": 0, "error": "Statically allocated ...", "error_full": "..."},
    "llama8b-nohook": {"kind": "vllm", "port": 20002, "status": "failed", "phase": "compile", "error_kind": "l1_clash",
                       "dram_gib_per_chip": {"load_model": 1.848, "initialize_from_config": 2.125},
                       "l1_kib_per_core": {"load_model": 4.2, "initialize_from_config": 4.2},
                       "compiles_after_capture": 0, "error": "Statically allocated ...", "error_full": "..."}
  },
  "warnings": []
}
```

With Llama's fit hook the same group gives `fit: shared: all 2 members booted on P150x4 and answered` (boot 150.6 s,
14.47 GiB/chip DRAM).

- Member `status`: `ok`, `failed` (this member is the cause), `timeout` (it never finished its phase), `aborted`
  (it stopped because another member failed).
- Phases: `init`, `open`, `load`, `kv_cache`, `compile`, `capture`, `start-server`, `ready`, then `serve` for the
  probe request.
- Failure kinds: `l1_clash`, `dram`, `opens_device`, `timeout`, `server_exited`, `probe`, `late_compile` (only with
  `--strict`), `device` (a chip that needs a reset: not a fit result; `device_fault` is true), `error`.
- `l1_kib_per_core` is the L1 held on each core after each of the member's phases, counting every member's buffers.
- Programs compiled after capture are listed under `warnings`; `--strict` makes them fail the member.
- Exit code 0 with a verdict either way; 2 when the check itself could not run (an invalid config, a launcher that
  died before the group existed); 3 when a device fault stopped the boot (reset the chips and run it again).
  `--timeout` (default 5400 s) bounds the boot; `--workdir` keeps the group's config, log and status.

## Does a model behave next to others? The conformance test

```bash
python -m tt_cohost.conformance --config group.json --member llama8b --out conformance/
# llama8b next to gptoss: sequential 16/16 identical, interleaved 16/16 identical: PASS
```

It serves the member alone (a one-member group on the same mesh, or `--baseline` greedy rows from its own server),
then with the whole group: the prompts once one at a time and once while every vLLM neighbour answers its own stream
of requests, so the members' steps interleave on the device. Each prompt's output is compared exactly with the alone
run. Pass = all identical in both modes. `--prompts` takes a JSON list or one prompt per line (16 built in). Pick a
neighbour that holds a typical amount of L1. Exit 0 on pass, 1 on a difference, 2 when the test could not run.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `TT_COHOST_CONFIG` | | the group config as inline JSON, when `--config` is not given |
| `TT_COHOST_BARRIER_TIMEOUT_S` | `7200` | how long a phase waits for the slowest member |
| `TT_COHOST_TIMING_S` | `30` | log per-call wall time every N seconds of activity (`0` = off) |
| `TT_COHOST_L1_REPORT` | | set to log every live L1 buffer after each phase |
| `TT_COHOST_CONNECT_TIMEOUT_S` | `600` | connect timeout of a vLLM frontend's socket to the host |
| `TT_COHOST_STATUS_FILE` | | also write the status JSON (with the boot record) to this file on every change |

## Limits

- **No overlap.** Members take turns on the device; busy members share its throughput.
- **One ttnn per process.** Every member runs on the same tt-metal, vLLM and plugin builds.
- **The fastest single-model paths may be off.** A model that keeps state in L1 between steps must turn
  that off in a group (GPT-OSS-20B: `GPT_OSS_20B_FUSED_B1=0`, 1,444 vs. 1,753 tok/s at 32 users alone).
- **On-device sampling may not survive a neighbour.** In tt-inference-server's 0.17.0 image, Llama-3.1-8B with
  `sample_on_device_mode: decode_only` next to Qwen3-32B (P300x2) produced garbage from about its third token once
  their decode steps alternated, and a burst of requests ended in a device timeout. Without that setting the same
  pair answered identically to the members served one at a time. Run the conformance test before sharing chips.
- **Late compiles are counted, not prevented.** A program compiled after capture can corrupt a parked
  trace. None was seen in the measured groups (output stayed identical after full benchmark runs), but the
  risk is not ruled out.
- **The group is fixed at launch.** Adding or removing a member means restarting the group.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
python -m pytest -q
```

The tests need no chips, ttnn or vLLM: those are faked where the runtime imports them.

## License

Apache-2.0. See [LICENSE](LICENSE).
