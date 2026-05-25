# Remote GPU Access Runbook

This machine is intended to act as a GPU experiment host that can be reached from
other machines for DHB-XR development and unattended runs.

## Current Host Snapshot

- Hostname: `andyparkp16`
- User: `andypark`
- LAN address observed during setup: `192.168.4.26`
- SSH daemon: installed and active on port `22`
- Repo path: `/home/andypark/Projects/repos/dhb-xr`
- Working branch: `dev/harness-and-rotation-fix`

The LAN address can change after reboot or network changes. Re-check it on the
host with:

```bash
hostname -I
ip -brief addr
```

## Connect From Another Machine

From another machine on the same LAN:

```bash
ssh andypark@192.168.4.26
cd /home/andypark/Projects/repos/dhb-xr
git status --short --branch
```

For a stable local alias, add this to the client machine's `~/.ssh/config`:

```sshconfig
Host dhbxr-gpu
    HostName 192.168.4.26
    User andypark
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then connect with:

```bash
ssh dhbxr-gpu
```

## Add Another Client SSH Key

On the client machine, create a key if needed:

```bash
ssh-keygen -t ed25519 -C "dhbxr-client"
```

Copy the public key to this host:

```bash
ssh-copy-id andypark@192.168.4.26
```

If `ssh-copy-id` is unavailable, append the client's public key manually:

```bash
cat ~/.ssh/id_ed25519.pub | ssh andypark@192.168.4.26 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
```

Do not commit private keys, `.ssh/`, `.env`, API tokens, or machine-specific
credentials to this repository.

## Access From Outside The LAN

Prefer a private mesh VPN such as Tailscale, WireGuard, or your existing VPN.
Avoid exposing port `22` directly to the public internet unless there is a clear
reason and SSH is locked down.

Tailscale example on Ubuntu:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4
```

Then connect from a signed-in client:

```bash
ssh andypark@<tailscale-ip>
```

## GPU Pre-Flight

Before starting experiments, verify that the driver is active:

```bash
scripts/remote_gpu_preflight.sh
```

Minimum expected result:

- `nvidia-smi` prints GPU information.
- `torch.cuda.is_available()` is `True` inside the project environment.

During setup on this host, the NVIDIA PCI device was visible but `nvidia-smi`
could not communicate with the NVIDIA driver. The observed cause was a kernel
module mismatch: the machine was booted into `6.8.0-111-generic`, but the
matching NVIDIA 580 module packages were not installed for that kernel. A later
attempt to boot `6.8.0-117-generic` caused local input/network issues, so this
host is intentionally pinned back to `6.8.0-111-generic`.

```bash
uname -r
lspci | grep -i nvidia
lsmod | grep -E 'nvidia|nouveau'
modinfo nvidia
nvidia-smi
apt-cache policy nvidia-dkms-580 linux-modules-nvidia-580-$(uname -r)
```

The current repair path is to build the `580.159.03` NVIDIA kernel modules for
`6.8.0-111-generic` using DKMS:

```bash
sudo apt update
sudo apt install nvidia-dkms-580
sudo reboot
```

GRUB is configured to boot the stable kernel by default:

```bash
grep -E '^(GRUB_DEFAULT|GRUB_TIMEOUT|GRUB_TIMEOUT_STYLE)' /etc/default/grub
# Expected:
# GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-111-generic"
# GRUB_TIMEOUT_STYLE=menu
# GRUB_TIMEOUT=5
```

After reboot, verify that the running kernel and NVIDIA module line up:

```bash
cd /home/andypark/Projects/repos/dhb-xr
uname -r
modinfo nvidia | head
scripts/remote_gpu_preflight.sh
```

If future package updates change the default kernel, restore the `111` boot
default with:

```bash
sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-111-generic"/' /etc/default/grub
sudo sed -i 's/^GRUB_TIMEOUT_STYLE=.*/GRUB_TIMEOUT_STYLE=menu/' /etc/default/grub
sudo sed -i 's/^GRUB_TIMEOUT=.*/GRUB_TIMEOUT=5/' /etc/default/grub
sudo update-grub
```

If Secure Boot is enabled, Ubuntu may require module signing or Secure Boot
changes before the NVIDIA kernel module can load.

## Keep A Remote Session Alive

If `tmux` is installed:

```bash
tmux new -s dhbxr
cd /home/andypark/Projects/repos/dhb-xr
```

Detach with `Ctrl-b d`, then reattach later:

```bash
tmux attach -t dhbxr
```

If `tmux` is not installed, use `nohup` for unattended runs:

```bash
mkdir -p _workspace/remote_logs
nohup pixi run python <script.py> <args> > _workspace/remote_logs/<run-name>.log 2>&1 &
echo $! > _workspace/remote_logs/<run-name>.pid
tail -f _workspace/remote_logs/<run-name>.log
```

Stop a `nohup` job with:

```bash
kill "$(cat _workspace/remote_logs/<run-name>.pid)"
```

## Resume DHB-XR Work On This Host

On the GPU host:

```bash
cd /home/andypark/Projects/repos/dhb-xr
git fetch origin
git switch dev/harness-and-rotation-fix
git status --short --branch
pixi install
scripts/remote_gpu_preflight.sh
```

If continuing from another machine, push the branch first:

```bash
git push origin dev/harness-and-rotation-fix
```

Then on the GPU host:

```bash
git pull --ff-only origin dev/harness-and-rotation-fix
```

Keep large local artifacts out of Git. The repository already ignores common
local outputs such as `.pixi/`, `_workspace/*.mp4`, `_workspace/*.log`,
`_workspace/paper_eval/`, and `results/`.

## Useful Monitoring Commands

```bash
nvidia-smi
nvidia-smi dmon
watch -n 2 nvidia-smi
ps -fu "$USER"
df -h
du -sh _workspace results 2>/dev/null
```

## Minimal Security Checklist

- Use SSH keys, not passwords, where practical.
- Keep this host behind a trusted LAN or private VPN.
- Do not expose SSH directly to the public internet without hardening.
- Do not commit secrets or private keys.
- Review `git status --short --branch --untracked-files=all` before pushing.
- Store large checkpoints, datasets, and videos outside Git or in an artifact
  store.
