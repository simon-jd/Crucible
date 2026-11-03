# Task 1: 节点基线与调度

**Real four-node acceptance: UNVERIFIED / pending external hardware.** Local CPU
implementation tests pass; no paid resources were created, node software installed,
production cluster contacted, or changes pushed. Captures dated 2026-10-24 UTC
(2026-10-23 America/Los_Angeles).

## Required behavior and implementation

`cluster/provision/check_baseline.sh` provides Slurm and Kubernetes profiles. Slurm
executes exactly `srun -N4 --ntasks-per-node=8 --gpus-per-node=8 nvidia-smi -L`,
requires 32 unique physical GPU UUIDs, then independently measures four hosts with
eight H100 GPUs each. Deduplication matters because Slurm launches eight tasks per
node and multiple tasks may list the same GPUs. This follows the documented task
and GPU allocation options in [SchedMD srun](https://slurm.schedmd.com/srun.html).

Kubernetes requires 32 Running/Ready one-GPU Pods, eight on each of four nodes,
plus four host probes, actual imageIDs, unique device visibility, and matching
versions. The templates use extended GPU resources as documented by
[Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/).
Kueue admits the suspended Job; an NFS RWX PV/PVC and a fresh controller-written
nonce verify shared storage availability on every worker. Both profiles archive
command argv/stdout/stderr/returncodes and captured topology/ibstat; unavailable
commands produce failures, never successful empty observations.

The lock fixes **candidate** driver 570.133.20, CUDA toolkit 12.8.1, CUDA runtime API
12080, NCCL API 22501 (2.25.1), MLNX_OFED_LINUX-24.10-3.2.5.0, Slurm 25.05.4,
Enroot 4.2.1, Pyxis package 0.21.0-1, Kubernetes v1.33.5, GPU Operator v25.3.2,
Network Operator 25.4.0 and Kueue 0.13.4. These are not site observations or a tested
compatibility matrix. NVIDIA's [PyTorch 25.03 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html)
list CUDA 12.8.1 and NCCL 2.25.1 and describe driver requirements. The
[OFED LTS release table](https://networking-docs.nvidia.com/software/lts-releases)
identifies 24.10-3.2.5.0. CUDA toolkit `version.json`, `cudaRuntimeGetVersion` and
`ncclGetVersion` are read in the selected workload image; `nvidia-smi`'s advertised
CUDA compatibility is not treated as an installed toolkit version.

Slurm/Pyxis execution uses a real SquashFS path, a documented Pyxis image mode in
[NVIDIA Pyxis](https://github.com/NVIDIA/pyxis). Worker probes verify its SHA256,
actual driver/OFED, scheduler/runtime package versions and script checksum. The
upstream [Pyxis v0.21.0 Debian package definition](https://raw.githubusercontent.com/NVIDIA/pyxis/v0.21.0/debian/control)
names `nvslurm-plugin-pyxis`; its [build rules](https://raw.githubusercontent.com/NVIDIA/pyxis/v0.21.0/debian/rules)
confirm the multiarch plugin directory. [Enroot v4.2.1's importer](https://raw.githubusercontent.com/NVIDIA/enroot/v4.2.1/src/docker.sh)
accepts digest references. Slurm 25.05.4 is available in the
[official release archive](https://download.schedmd.com/slurm/).

K8s uses fixed Helm candidates, preinstalled host drivers/toolkit and host OFED;
GPU/Network Operators therefore do not replace these host components. Deployment
settings follow [GPU Operator installation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3.0/getting-started.html),
[GPU Operator 25.3.2 platform support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3.2/platform-support.html),
and [Network Operator 25.4.0](https://docs.nvidia.com/networking/display/kubernetes2540/index.html).
Kueue's fixed candidate is documented in its [v0.13.4 release](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.13.4)
and [Helm chart metadata](https://raw.githubusercontent.com/kubernetes-sigs/kueue/v0.13.4/charts/kueue/Chart.yaml).
These references support syntax and candidate availability; they do not establish
this site's driver/kernel/fabric compatibility or future security suitability.

## RED → GREEN and genuine local failures

- RED: `.venv/bin/python -m pytest cluster/provision/tests/test_baseline.py -q`
  produced **3 failed** before the implementation. Full capture:
  [tdd-red.txt](cluster-baseline/tdd-red.txt).
- Initial real RED: `cluster/provision/check_baseline.sh` returned **1** with
  `FAIL: missing required scheduler: srun`.
  [local-red.txt](cluster-baseline/local-red.txt).
- GREEN: focused CPU tests cover the real shell/Python entrypoint, subprocess
  capture, nonzero exits/timeouts, synthetic Slurm and K8s success paths, wrong
  cards/UUIDs/distribution, driver/CUDA/NCCL/OFED and scheduler version rejection,
  unready Pods, image/script changes, malformed locks and configuration rendering.
  All command shims are explicitly **SYNTHETIC**, never cluster observations.
  [tdd-green.txt](cluster-baseline/tdd-green.txt).
- Final native Slurm command: `BASELINE_PYTHON=.venv/bin/python
  cluster/provision/check_baseline.sh --output
  docs/evidence/cluster-baseline/local-slurm` returned **1**, capturing `srun` missing
  (127) and unfinished image digest. [summary](cluster-baseline/local-slurm/summary.json),
  [raw commands](cluster-baseline/local-slurm/commands.json).
- Final native K8s command adds `--scheduler k8s` and uses `local-k8s` output;
  returned **1**, capturing `kubectl` missing (127) and unfinished image digest.
  [summary](cluster-baseline/local-k8s/summary.json),
  [raw commands](cluster-baseline/local-k8s/commands.json).
- Additional native hardware audit, using `baseline.Runner` on the unmodified
  local PATH: `nvidia-smi -L`, driver query and `nvidia-smi topo -m` return **9**
  (cannot communicate with NVIDIA driver); OFED/ibstat commands are missing (127).
  No physical GPU UUID was enumerated, and the count assertion rejects 0 vs 32.
  This is **not proof the machine physically has zero GPUs**.
  [raw commands](cluster-baseline/local-hardware/commands.json).
- Real local `nvcc --version` reports V13.2.86; `/usr/local/cuda/version.json` reports
  toolkit **13.2.2**, and `node_probe.library_version('cudart',
  'cudaRuntimeGetVersion')` returns **13020**. Both reject the candidate lock's
  12.8.1/12080. The actual NCCL library is unavailable. The audit did not invent
  node versions or a runtime library result. [hardware summary](cluster-baseline/local-hardware/summary.json).

## External prerequisites and limits

The workload OCI digest, node image ID/SHA256, Slurm SquashFS checksum and complete
K8s managed-image inventory are **not supplied**. Lock status remains
`candidate-unverified`; both renderer and checker reject it. No successful version,
digest, topology or `ibstat` cluster result is claimed. Real topology/ibstat files
can only be produced after real workers are available.

`cluster/provision/README.md` documents pre-rental owner/cost/time/termination
conditions, minimal base OS and pinned node-bootstrap requirements, the two rendering/deploy
entrypoints, credentials/storage/fabric inputs and cleanup. Empty-node end-to-end
reproducibility remains pending provider images/build provenance, actual package
artifacts, four 8×H100 nodes, IB/RoCE routing/firmware, NFS endpoint, munge or cluster
credentials, operator image overrides and real platform digests. Node image ID
checking is inventory matching, not cryptographic runtime attestation. Active
fabric is checked here; throughput, link-rate burn-in and silent downclock
thresholds remain Task 2. No model runs in Task 1; later tasks must pin model
revision before rental.

## Review fix round 1 (base a85d4eb)

Two implementation gaps were corrected without deploying a host or cluster:

1. **Empty-node installation and deployment automation.** `bootstrap.py` now
   consumes an operator-supplied complete pinned Debian artifact closure and
   private secret/configuration files. Default invocation is a pure offline plan;
   null hashes or missing inputs fail readiness. Explicit `--apply` on a target
   verifies package SHA256 and Debian metadata, installs only missing exact
   versions, verifies installed state, atomically installs configuration/secret
   permissions, configures NFS, Slurm directories/ownership and services, or
   kubeadm init/join. `deploy.py` defaults to an offline inventory plan and has an
   explicit SSH apply path for one controller/four workers, private staging,
   credential-safe output, cleanup and K8s local pinned chart/manifest application.
   A supplied static containerd configuration keeps repeated K8s applies stable.
   No preinstalled GPU/scheduler image is required, only a minimal Ubuntu base OS.
   Real package/image/chart hashes remain unprovided; no build/install success is
   claimed. Actual commands were checked against [Debian dpkg documentation](https://manpages.debian.org/bookworm/dpkg/dpkg.1.en.html),
   [NVIDIA container-runtime configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   and [Kubernetes kubeadm join](https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm-join/).
2. **Diagnostics survive later failures.** Each node-probe field is captured
   independently with structured command/error records. A failing OFED/library/
   storage/scheduler check no longer discards earlier topology/ibstat. The probe
   emits a partial report and exits nonzero. Controllers parse and persist
   available reports before evaluating aggregate Slurm or K8s exec exit codes;
   each K8s host is archived as soon as received. Partial output and healthy
   peers survive failures in another node.

TDD RED: five failing bootstrap/diagnostic regressions plus a separately recorded
missing deploy-entrypoint regression, before their respective implementations:
[fix1-red.txt](cluster-baseline/fix1-red.txt) (trailing whitespace normalized only).
GREEN: **49 Task 1 focused tests passed**, including actual `node_probe.py`
subprocesses whose synthetic topo/ibstat commands succeed and later OFED fails;
Slurm aggregated nonzero with healthy and partial reports; K8s third-host failure;
Slurm/K8s repeated bootstrap; complete empty-PATH offline planning; checksum/secret
permission rejection; and injected SSH transport/staging with no real network or
installer calls. [fix1-green.txt](cluster-baseline/fix1-green.txt).

Native default plans, without any SSH/installer/cluster command, both correctly
returned 1 for missing/unlocked inputs:
[bootstrap plan](cluster-baseline/fix1-bootstrap-plan.json),
[deployment plan](cluster-baseline/fix1-deploy-plan.json).
Full-suite result for this fix is in [fix1-full-tests.txt](cluster-baseline/fix1-full-tests.txt).
No actual install/apply, cloud resource, credential fabrication or real multi-node
acceptance occurred. Current resource policy is **$0 / multi-node work paused**;
local RTX 4090 work does not certify this four-node H100 baseline. The installer
is reviewable code for a future explicitly authorized lab execution.

## Review fix round 2 (base 75962cf): verify the live shared mount

The bootstrap previously accepted an already-active mount by checking only that
`findmnt` succeeded. A requested `storage-b:/crucible` could therefore write into
an existing `storage-a:/crucible`. This is now rejected before any `/shared` payload
write. The offline plan parses the supplied `shared.mount` What/Where/Type, and
requires that definition when a manifest includes shared files.

On an explicit apply, the bootstrap reads the kernel mount table as JSON with
explicit TARGET/SOURCE/FSTYPE columns and compares the exact live source/type to
the requested values before **each** shared-file write. An existing mismatch,
missing identity or ambiguous stacked mount returns `applied=false` with pending
shared-mount verification. It does not unmount, restart or replace an unknown
mount. A genuinely absent mount is activated, then its actual source/type must be
verified before writing. A successful activation command alone is insufficient.
Source aliases and filesystem types are not silently normalized into matches.
The machine-readable options follow the [util-linux findmnt manual](https://man7.org/linux/man-pages/man8/findmnt.8.html).

RED: four focused regressions failed against the prior implementation: wrong
source, wrong FSTYPE, existing correct mount behavior and first-activation
verification. [fix2-red.txt](cluster-baseline/fix2-red.txt) (trailing whitespace
normalized). GREEN: **58 Task 1 tests passed**, including those cases plus wrong
source after activation, malformed/incomplete/stacked mount identities and an
explicit pending CLI result. [fix2-green.txt](cluster-baseline/fix2-green.txt).
Fixtures use injected commands and temporary roots; the wrong-source/type cases
assert no shared probe/image path is created, existing data is preserved and no
activation is attempted for an already mismatched mount. No actual mount,
installation, SSH, remote GPU access or spend occurred. Real multi-node storage
acceptance remains **未在真实多机上验证** under the $0 budget policy.
Full-suite validation for this fix ran once: **139 passed in 10.03s**;
[fix2-full-tests.txt](cluster-baseline/fix2-full-tests.txt). Task 1 ruff checks and
format verification also pass.
