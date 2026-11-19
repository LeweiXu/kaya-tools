"""The `kaya` command: parse arguments, load the config, and hand off to the command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

from . import __version__
from .config import Config, ConfigError, load_config
from .doctor import handle_config, handle_doctor, handle_init
from .execute import handle_interactive, handle_run, handle_shell, handle_submit
from .jobs import handle_cancel, handle_info, handle_jobs, handle_logs, handle_watch
from .ssh import RemoteError
from .status import handle_status
from .sync import pull, push

EPILOG = textwrap.dedent("""\
    Everything after a bare -- goes to your script untouched, so kaya's own flags can
    sit before or after the program name:

      kaya submit train.py --time 02:00:00 -- --epochs 3
      kaya run setup.sh -- --fast

    Python files can set their own defaults in the first 40 lines:
      # kaya: target=gpu          (login or gpu)
      # kaya: env=true            (activate the project env)
      # kaya: offline=true        (export [offline_env_vars])
      # kaya: job-name=my-job

    Config lives in config.toml next to this tool (or $KAYA_CONFIG). See README.md.
""")


def add_slurm_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("slurm (defaults from [slurm])")
    g.add_argument("--job-name")
    g.add_argument("--partition")
    g.add_argument("--gres", help="e.g. gpu:1 or gpu:v100:2")
    g.add_argument("--account")
    g.add_argument("--qos")
    g.add_argument("--cpus-per-task", type=int)
    g.add_argument("--mem", help="e.g. 64G")
    g.add_argument("--time", help="wall time, e.g. 02:00:00 or 1-00:00:00")
    g.add_argument("--dependency", help="e.g. afterany:12345")


def add_run_flags(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("environment and lifecycle (defaults from [behaviour])")
    g.add_argument("--env", action=argparse.BooleanOptionalAction, default=None, help="activate the project env")
    g.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None,
                   help="export [offline_env_vars] and skip secrets")
    g.add_argument("--push", action=argparse.BooleanOptionalAction, default=None, help="push the project first")
    g.add_argument("--wait", action=argparse.BooleanOptionalAction, default=None, help="wait for a submitted job")
    g.add_argument("--pull", dest="pull_after", action=argparse.BooleanOptionalAction, default=None,
                   help="pull results once a waited job ends")
    g.add_argument("--filter", help="pull filter name from [pull_filters]")
    g.add_argument("--tail-lines", type=int, help="log lines to show when a waited job ends")
    g.add_argument("--export", action="append", default=[], metavar="NAME=VALUE",
                   help="extra env var for a generated job (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kaya", description="Run things on UWA's Kaya cluster.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    parser.add_argument("--config", type=Path, help="config file (default: config.toml beside the tool)")
    parser.add_argument("--project", help="project name from config (default: the one holding this folder)")
    parser.add_argument("--version", action="version", version=f"kaya-tools {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("init", help="create config.toml, or add this folder as a project")
    p.add_argument("--project", dest="init_project", metavar="NAME", help="add the current folder as this project")
    p.add_argument("--remote", help="remote folder, absolute or relative to [cluster] home (default: NAME)")
    p.add_argument("--env", help="env path, relative to the remote folder")

    sub.add_parser("doctor", help="check ssh, VPN, modules, folders and env")
    sub.add_parser("config", help="print the effective config as JSON")

    p = sub.add_parser("push", help="rsync the project up to Kaya")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("pull", help="rsync results back (the project's `pull` folders, or the paths given)")
    p.add_argument("paths", nargs="*", help="project-relative folders (default: the project's pull list)")
    p.add_argument("--filter", help="only files matching this [pull_filters] entry")
    p.add_argument("--all", dest="everything", action="store_true", help="ignore the default filter")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("run", help="run a project .py file or any command on the login node",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    p.add_argument("--target", choices=["auto", "login", "gpu"], default="auto",
                   help="auto reads the file's header; gpu submits it as a job")
    add_run_flags(p)
    add_slurm_flags(p)
    p.add_argument("program", help="a .py file in the project, or a command")

    p = sub.add_parser("submit", help="submit a project .py file or sbatch script to SLURM",
                       formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    add_run_flags(p)
    add_slurm_flags(p)
    p.add_argument("--no-pre-submit", action="store_true", help="skip the project's pre_submit check")
    p.add_argument("program", help=".py, .sbatch, .slurm or .sh file in the project")

    p = sub.add_parser("watch", help="wait for a job, pull results, show its logs")
    p.add_argument("job_id", nargs="?", help="default: the last job submitted through kaya")
    p.add_argument("--pull", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--filter", help="pull filter name")
    p.add_argument("--tail-lines", type=int)

    sub.add_parser("jobs", help="your queued and running jobs")

    p = sub.add_parser("info", help="sacct and scontrol details for one job")
    p.add_argument("job_id", nargs="?")

    p = sub.add_parser("logs", help="show a job's log files straight from Kaya")
    p.add_argument("job_id", nargs="?")
    p.add_argument("-n", "--lines", type=int)
    p.add_argument("-f", "--follow", action="store_true", help="keep following (Ctrl-C to stop)")
    p.add_argument("--stream", choices=["both", "out", "err"], default="both")

    p = sub.add_parser("cancel", help="cancel jobs: ids, --name, --all, or the last job")
    p.add_argument("job_id", nargs="*")
    p.add_argument("--name", help="every job of yours with this name")
    p.add_argument("--all", action="store_true", help="every job of yours")
    p.add_argument("--state", help="only jobs in this state, e.g. PENDING")
    p.add_argument("-y", "--yes", action="store_true", help="no confirmation for --all")

    p = sub.add_parser("status", help="GPU nodes, the queue and your recent jobs")
    p.add_argument("--partition", default="auto", help="comma-separated, or auto for every GPU partition")
    p.add_argument("--user", help="whose jobs go under 'Your jobs' (default: you)")
    p.add_argument("--limit", type=int, default=50, help="max rows of other users' jobs")
    p.add_argument("--json", action="store_true")

    sub.add_parser("shell", help="open a login shell on Kaya in the project folder")

    p = sub.add_parser("interactive", help="get an interactive session on a compute node via srun")
    add_slurm_flags(p)
    return parser


def split_forwarded(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split on the first bare `--`: kaya's arguments, then the script's."""

    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return list(argv), []


def dispatch(cfg: Config, args: argparse.Namespace) -> int | None:
    command = args.command
    if command == "push":
        push(cfg, dry_run=args.dry_run)
    elif command == "pull":
        pull(cfg, args.paths, filter_name=args.filter, everything=args.everything, dry_run=args.dry_run)
    else:
        handler = {
            "config": handle_config,
            "doctor": handle_doctor,
            "run": handle_run,
            "submit": handle_submit,
            "watch": handle_watch,
            "jobs": handle_jobs,
            "info": handle_info,
            "logs": handle_logs,
            "cancel": handle_cancel,
            "status": handle_status,
            "shell": handle_shell,
            "interactive": handle_interactive,
        }[command]
        return handler(cfg, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    own, forwarded = split_forwarded(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(own)
    args.forwarded = forwarded
    try:
        if args.command == "init":
            args.project = args.init_project
            return handle_init(args.config, args)
        cfg = load_config(args.config, project=args.project)
        return dispatch(cfg, args) or 0
    except ConfigError as exc:
        print(f"kaya: {exc}", file=sys.stderr)
        return 2
    except RemoteError as exc:
        print(f"kaya: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 255 and exc.cmd and exc.cmd[0] in ("ssh", "rsync"):
            print("kaya: could not reach Kaya over ssh. If the host does not resolve or times out, "
                  "the VPN is probably down; `kaya doctor` checks each step.", file=sys.stderr)
        else:
            print(f"kaya: remote command exited {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except KeyboardInterrupt:
        print("\nkaya: interrupted", file=sys.stderr)
        return 130
