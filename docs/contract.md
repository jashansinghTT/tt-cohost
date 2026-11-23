# The co-host contract

What a model must do to share chips with other models in one `tt_cohost` group, why, and what happens
when it does not. A model that breaks a rule is not broken: serve it on its own chips.

| rule | why | how it shows today |
|---|---|---|
| Uses the mesh it is given; never opens or closes a device itself. | A chip is opened by one process at a time (UMD's per-chip lock), and every member lives in the host's process. | vLLM members: the host hands the plugin's `TTWorker` the group's mesh and ignores its close. App members get the mesh in `load(mesh)`. A member that opens a device itself fails at boot. |
| Keeps nothing in L1 between steps that the other members' programs need. | Each member's programs were sized for the whole L1. Persistent L1 in one member makes the next member's program fail with `Statically allocated dataflow buffers ... clash with L1 buffers`. | A hard compile error at boot, naming the program. The host records the L1 held after each member's phase (status `l1_kib_per_core_after_phase`); `TT_COHOST_L1_REPORT=1` lists every live L1 buffer. |
| Compiles and allocates everything during warmup; nothing after trace capture. | A trace replays the addresses it captured. A buffer or program allocated after a capture can sit in that trace's scratch and be overwritten when it replays. | The boot is phased across all members (load, compile, capture). Programs compiled after capture are counted per member (status `compiles_after_capture`) and logged. |
| Changes no process-wide state while loading, or restores it. | One interpreter: an environment variable one model sets (e.g. `HF_MODEL` pointed at its own weights) changes it for every member loaded after it. | The host sets each member's configured `env` around its load-time calls and restores it. Keys a model writes itself are not restored; the member's adapter must. |
| Runs on the same ttnn (and vLLM and plugin) build as the other members. | One process imports one ttnn. | A member built against another tt-metal fails at import or load. Check the builds before grouping members. |

## Checking a model

A member that follows the contract gives the same greedy output next to a neighbour as it gives alone. To
check: serve it next to a reference neighbour that holds a typical amount of L1, send a fixed prompt set
with greedy sampling, and compare each answer with the same prompt served alone. Pass = every prompt
identical, both with the prompts sent one at a time and with the neighbour answering at the same time
(so the two members' steps interleave).
