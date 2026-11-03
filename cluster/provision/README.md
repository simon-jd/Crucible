# Four-node H100 baseline

Current budget is **$0**; real multi-node work is paused. The local RTX 4090 does
not replace the four-node H100 acceptance requirement. Deployment commands below
are future operator entrypoints, not permission to deploy or spend.

Real four-node acceptance remains **UNVERIFIED**. No node or cluster changes have
been executed here. All checked-in artifact hashes remain null; examples cannot
pass readiness checks. Candidate versions are proposals, not measured versions.

## Rental and inputs

Before renting, record the owner, provider/region, four 8×H100 node IDs, hourly cost
ceiling, maximum two-hour allocation and automatic termination deadline. Stop and
release all four nodes on any baseline failure, unexpected cost, unavailable shared
storage or inactive fabric. Performance/downclock burn-in belongs to Task 2.

The bootstrap starts from a minimal approved Ubuntu 24.04 x86_64 image with
Python3, SSH, sudo/systemd and dpkg. GPU drivers, OFED and schedulers **need not be
preinstalled**. The operator supplies these reviewed inputs as a local site bundle:

- A site `versions.lock` with real platform OCI digest, node base-image ID/SHA256,
  Slurm SquashFS SHA256 or complete K8s managed-image digest inventory. Review the
  actual OS/kernel/NIC/GPU compatibility and security before setting `status=locked`.
- A complete local `.deb` dependency closure, including NVIDIA driver, OFED/NFS/
  RDMA tools, kernel headers/DKMS build dependencies, Python/util-linux, munge and
  Slurm/Pyxis/Enroot, or containerd/NVIDIA toolkit/kubeadm/kubelet/kubectl. Each file
  requires an exact **Debian package version and SHA256**. The bootstrap never uses
  an unpinned apt repository or resolves dependencies from the network. Example
  package names/revisions are site-build candidates and must match real `.deb`
  metadata. Missing dependencies fail installation; they are not silently fetched.
- Real NFS export, four worker names, controller name, `slurmd -C` CPU/RealMemory
  values or K8s RDMA PF name; site network/firewall settings. Ubuntu kernel/network
  prerequisites and containerd SystemdCgroup configuration must be included in the
  reviewed base image or supplied configuration files in the bundle.
- A private munge key file, or private kubeadm init/join configurations using a
  preselected bootstrap token and verified discovery CA, plus operator kubeconfig.
  Secret sources must be mode 0600/0400. Include only file paths in manifests;
  never put credential bytes on the command line or in the repository.

No real package, image build, digest or credential has been invented. Supplying
artifacts is distinct from implementing installation: the installer and deployment
transport are included and exercised offline with explicitly synthetic fixtures.

## Offline render, install plan and one-command deployment

Render site configuration first. This only writes a new local directory:

```bash
.venv/bin/python cluster/provision/render.py --scheduler slurm \
  --lock /path/site/site.lock --output /path/site/rendered \
  --nfs-export storage.example:/crucible --controller head.example \
  --nodes gpu1,gpu2,gpu3,gpu4 --cpus 192 --memory-mb 1000000
```

CPU/memory numbers are examples, not hardware observations. K8s replaces the
controller/node/CPU options with `--scheduler k8s --rdma-interface ib0`. Both
profiles produce `shared.mount`; K8s also produces namespace/RuntimeClass manifests.

Use `bootstrap.example.json` to define one controller manifest and worker
manifests. Set roles to `slurm-controller`/`slurm-worker` or
`k8s-controller`/`k8s-worker`. Source paths are relative to each manifest's directory
and cannot escape it. `files` maps rendered configuration to final node paths;
the supplied Slurm example covers `/etc/slurm`, `/etc/enroot`, munge and the NFS
mount unit. For the controller, also map the checked-in baseline Python/shell
files and site lock into `/shared/crucible/cluster/provision/`, and the locked
SquashFS into its `/shared/images/` path. The bootstrap copies these files only
after verifying the live mount's SOURCE and FSTYPE against What/Type in the supplied `shared.mount`. It checks again before every shared
payload write. An existing mismatched or ambiguous mount is left unchanged and
returns `applied=false`, pending shared-mount verification; the bootstrap never
unmounts or replaces it. If the mount is absent, activation is followed by the same
verification before writing. Source aliases are not silently treated as equal.
K8s roles require secret `/etc/kubernetes/init.yaml` or `/etc/kubernetes/join.yaml`
inputs; the required rendered `containerd.toml` maps into `/etc/containerd/config.toml`.
That template targets pinned containerd 1.7.x; other major versions need a reviewed
site configuration. It includes SystemdCgroup and the NVIDIA runtime handler.

Use `deploy.example.json` as the inventory: exactly one controller and four
workers, each referencing its role manifest. All hosts use preapproved SSH host
keys and `user@hostname`; the operator account needs noninteractive sudo. The
controller may be an existing CPU host; the four workers are the rented GPU nodes.

```bash
# Default is offline: no subprocess, network or system writes.
.venv/bin/python cluster/provision/bootstrap.py --manifest /path/site/worker.json
.venv/bin/python cluster/provision/deploy.py --inventory /path/site/inventory.json

# Explicit operator action, run only against the intended lab after plan is ready.
.venv/bin/python cluster/provision/deploy.py --inventory /path/site/inventory.json --apply
```

`--apply` stages only each node's bundle in a private temporary directory, installs
SHA-verified packages whose real Debian metadata matches, verifies installed
versions, atomically installs configuration/secret modes, sets munge ownership,
creates Slurm spool/log directories, mounts NFS and enables/starts services. It
restarts services when package/configuration changes require it. Repeated identical
runs skip package installation and preserve unchanged files. Credentials and
installer/join output are withheld from console output; local and remote staging
directories are removed. A required OS reboot returns a failure/pending result;
perform the maintenance reboot explicitly and rerun. No automatic reboot is issued.
A single target can instead run `sudo python3 bootstrap.py --manifest manifest.json
--apply` from its staged bundle. No SSH/dpkg/systemctl/kubeadm apply was run here.

For K8s, the node bootstrap installs the supplied NVIDIA containerd configuration and runs kubeadm
init/join only when the corresponding node marker does not already exist. Inventory
entries supply `node_name`. Deployment labels the four workers, applies pinned CNI,
namespace and RuntimeClass YAML, installs the three Helm releases from local
SHA-locked chart archives and values files, then applies the probe/baseline
manifests. Fill the K8s fields already shown in `deploy.example.json`; the
`manifests_before_helm` list must include the rendered `runtimeclass.yaml` as well
as the pinned site CNI and namespace. Chart versions match the site lock; site
values must override all managed images to the recorded platform digests. Use
preselected certificate/token configuration and a matching private kubeconfig;
these are external credentials, not values this tool fabricates.

`cloud-init.yaml` is an optional rendered base-OS/NFS helper. It does not install
software; `bootstrap.py` is the actual package/configuration/service installer.

## Acceptance and cleanup

```bash
BASELINE_PYTHON=.venv/bin/python cluster/provision/check_baseline.sh \
  --lock /path/site/site.lock --output docs/evidence/cluster-baseline/UNIQUE-RUN
# Add --scheduler k8s for that profile.
```

Slurm executes exactly `srun -N4 --ntasks-per-node=8 --gpus-per-node=8 nvidia-smi -L`.
Listings are deduplicated by UUID and independently matched to four hosts with
eight H100s each. Runtime probes use the locked SquashFS through Pyxis. K8s requires
32 Ready one-GPU/RDMA Pods spread 8×4, correct imageIDs and four host diagnostics.
Both validate actual driver/OFED/CUDA toolkit and CUDA/NCCL library API versions,
shared-storage nonce, script identity and active fabric. Missing/unlocked inputs,
versions, commands or any probe error fail closed.

Each probe captures independent diagnostic fields and command errors. Even when a
later field/node fails, `partial-reports.json`, `node-*-report.json`, available
`node-*-topology.txt` and `node-*-ibstat.txt` are saved **before** interpreting the
aggregate command failure. `commands.json` retains raw scheduler command streams.
Existing evidence directories are never overwritten. Only `summary.json` says
whether acceptance passed; a successful install is not a successful baseline.

After collection, remove the K8s baseline Job/privileged host DaemonSet/namespace,
review the Retain PV separately, and terminate all paid nodes before the deadline.
Real hardware, actual artifacts, credentials, fabric and NFS verification remain
external inputs; local tests do not certify any of them. No model runs in Task 1;
later experiments also require model revision pins.
