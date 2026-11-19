# kaya-tools

A small command line tool for working on UWA's Kaya HPC
cluster from your own machine. You keep editing code locally, and `kaya` handles the
rest: syncing the project up, running things on the login node, submitting SLURM jobs,
watching them, pulling results back, and showing you what the GPUs are up to.

```bash
kaya status                      # GPU nodes, the queue, your recent jobs
kaya submit train.py --time 02:00:00 -- --epochs 3
kaya jobs                        # what you have queued or running
kaya logs -f                     # follow the last job's output
kaya pull                        # bring results back
```

It's plain Python 3.11+ with no dependencies. You need `ssh` and `rsync` locally.

## Install

```bash
git clone git@github.com:LeweiXu/kaya-tools.git ~/kaya
~/kaya/install.sh
```

`install.sh` symlinks `~/kaya/bin/kaya` into `~/.local/bin`, so `kaya` works from any
folder with no alias or bashrc edit, and copies `config.example.toml` to `config.toml`.
Everything kaya uses lives in this folder: the code, your `config.toml`, an optional
`.env` for secrets, and `.state/` (ssh sockets and the last job id). The one exception is
your ssh key, which stays wherever you keep it; the config just points at it.

If `python3` is older than 3.11, set `KAYA_PYTHON=/path/to/python3.12` (for both the
installer and day-to-day use).

## Setup

Open `~/kaya/config.toml` and fill in at least:

```toml
[ssh]
user = "your-uwa-username"
key = "~/.ssh/id_ed25519"

[cluster]
home = "/group/your-group/your-username"
```

Your key needs to be in `~/.ssh/authorized_keys` on Kaya already (`ssh-copy-id` does
that). Then add a project. From inside the project folder:

```bash
cd ~/code/myproject
kaya init --project myproject --env envs/main
kaya doctor
```

That adds a `[projects.myproject]` section mapping this folder to
`<home>/myproject` on Kaya. `kaya doctor` walks through every link (key file, DNS, ssh
login, your home folder, SLURM, each module, the project folder, the env) and tells you
which one is broken.

You have to be on the UWA network or VPN. If `kaya.hpc.uwa.edu.au` won't resolve or ssh
times out, that's almost always the VPN.

### How kaya picks a project

Commands that need a project (`push`, `pull`, `submit`, `run` on a project file) use the
project whose `local` folder holds your current directory. Override with
`kaya --project NAME ...` or `KAYA_PROJECT=NAME`. Commands like `status`, `jobs`, `logs`,
`cancel`, `shell` and `interactive` work from anywhere.

## Commands

Run `kaya COMMAND --help` for every flag.

| Command | What it does |
|---|---|
| `init` | Create `config.toml`, or `--project NAME` to add the current folder as a project |
| `doctor` | Check the connection, folders, modules and env step by step |
| `config` | Print the effective config (with the project's overrides applied) as JSON |
| `push` | rsync the project to Kaya. The local copy wins: remote files you deleted go too (`push_delete`) |
| `pull [PATHS]` | rsync the project's `pull` folders (or the given ones) back. `--filter NAME` keeps only matching files |
| `run PROGRAM` | Run a project `.py` file or any command on the login node, in the project env |
| `submit PROGRAM` | Submit a project `.py` file (kaya writes the sbatch) or your own `.sbatch`/`.slurm`/`.sh` |
| `watch [ID]` | Wait for a job to end, pull results, show its logs |
| `jobs` | Your queued and running jobs |
| `info [ID]` | sacct accounting plus scontrol details for one job |
| `logs [ID]` | Read a job's log files on Kaya directly. `-f` follows, `--stream err` for stderr only |
| `cancel [IDS]` | Cancel jobs by id, `--name`, or `--all` (asks first). No arguments cancels the last job |
| `status` | GPU nodes by model, who can use them, the queue, your recent jobs. `--json` for scripts |
| `shell` | A login shell on Kaya, starting in the project folder |
| `interactive` | An `srun --pty` shell on a compute node, with the usual `--gres`/`--time`/`--mem` flags |

A job id left out means the last job you submitted through kaya.

### Passing arguments to your script

Everything after a bare `--` goes to your script as is. Everything before it is for
kaya, wherever it sits:

```bash
kaya submit train.py --gres gpu:v100:2 --mem 64G -- --lr 3e-4 --epochs 10
kaya run prepare_data.py -- --download
kaya run nvidia-smi              # any command works on the login node too
```

### Login node or GPU job?

The login node has internet but no GPU. Use it for setup, installs and downloads. GPU work
goes through SLURM, and compute nodes have **no internet**, so download weights and
datasets on the login node first.

`kaya run file.py` uses the login node unless the file says otherwise. `kaya submit` always
makes a job. A Python file can set its own defaults in its first 40 lines:

```python
# kaya: target=gpu          # login or gpu (gpu makes `kaya run` submit it)
# kaya: env=true            # activate the project env
# kaya: offline=true        # export [offline_env_vars]
# kaya: job-name=trainer
```

Order of precedence: config, then the header, then command line flags.

### What a run or job gets

Before your command runs, kaya `cd`s into the project's remote folder, loads the
`[cluster] modules`, exports `[env_vars]` (plus `[offline_env_vars]` when offline), and
activates the project env. Login-node runs also get the `[secrets]` variables from your
local `.env`. Secrets never go into job scripts, since those sit on disk on Kaya.

A `.py` submit becomes a generated sbatch file under `<home>/.kaya-tools/` (handy to
look at when a job does something odd), with output in the project's `logs/` folder as
`<job-name>_<id>.out` and `.err`.

## Config

`config.example.toml` has every setting with a comment. The main tables:

- `[ssh]`: host, user, key, port, timeouts, connection sharing, extra `-o` options
- `[cluster]`: your home folder on Kaya, the scratch folder, modules to load
- `[slurm]`: default partition, gres, cpus, mem, time, account, qos
- `[behaviour]`: defaults like push before run, wait for jobs, pull after, the default
  pull filter, log tail length, whether gpu jobs run offline, confirming `cancel --all`
- `[pull_filters]`: named file patterns for `pull --filter`
- `[env_vars]`, `[offline_env_vars]`, `[secrets]`
- `[projects.NAME]`: `local`, `remote`, `env`, `env_kind` (conda or venv), `python`,
  `logs`, `pull`, `excludes`, `pre_submit`

A project can override any of the cluster-wide tables for itself:

```toml
[projects.bigmodel.slurm]
gres = "gpu:v100:2"
mem = "64G"

[projects.bigmodel.behaviour]
pull_filter = "text"     # results are huge; only ever pull text by default
```

`pre_submit` is a command run on the login node before every submit for that project,
and the submit is cancelled if it fails. `{program}` and `{args}` are filled in, so
something like `python check_inputs.py {args}` can catch a broken config before the job
waits hours in the queue. Skip it once with `--no-pre-submit`.

## Things that catch people out on Kaya

- **Can't connect?** Check the VPN first. kaya's errors and `kaya doctor` both point
  there.
- **The host name covers two login nodes.** A `ps` over ssh can land on the other one,
  so check background login-node work through its log, not `pgrep`. If one node goes
  dead, a short `connect_timeout` lets ssh move on to the other quickly.
- **Plain `sinfo` hides partitions you can't submit to**, and `scontrol show node` fails
  on restricted nodes. `kaya status` reads them the way that works and marks access.
- **An empty `squeue` isn't proof a job ended.** The scheduler blips sometimes. kaya waits
  for `sacct` to agree before it calls a job finished.
- **`module: command not found`** means a non-login shell. kaya always uses login shells.
- **SLURM writes no output at all if the log folder is missing.** kaya creates it before
  every submit.
- **Two 16 GB GPUs are not one 32 GB GPU.** A model split across cards still keeps
  attention on one card, so long inputs can run out of memory on two small cards where
  they fit on one big one.
- **Long `--time` values wait longer.** Ask for about what the job needs plus some
  headroom, not the partition maximum, and the scheduler can backfill it sooner. A pending
  job's limit can be lowered with `scontrol update JobId=<id> TimeLimit=<t>`.
- **`$HOME` is small.** The example `[env_vars]` point pip, conda, Hugging Face and torch
  caches into the project folder instead.

## Development

```bash
python -m pytest            # needs pytest; pyproject.toml puts src/ on the path
```

The code is in `src/kaya_tools/`, one file per job:

| File | Role |
|---|---|
| `cli.py` | The single entry point: argument parsing, dispatch, error messages |
| `config.py` | Loading `config.toml`, project detection, per-project overrides |
| `ssh.py` | Building ssh commands, running remote commands and scripts |
| `prelude.py` | The shell setup every remote run starts with |
| `scripts.py` | `# kaya:` headers, mapping local files into the mirror |
| `sync.py` | `push` and `pull` |
| `slurm.py` | sbatch options and wrappers, submitting, waiting, squeue |
| `execute.py` | `run`, `submit`, `shell`, `interactive` |
| `jobs.py` | `watch`, `jobs`, `info`, `logs`, `cancel` |
| `doctor.py` | `init`, `config`, `doctor` |
| `status/` | `kaya status`: `cluster.py` reads and parses, `report.py` prints |
| `table.py` | Plain text tables |
