"""Commands that start work on Kaya: `run` on the login node, `submit` to SLURM, and the
interactive `shell` and `interactive` (a GPU session through srun)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from .config import Config, Project
from .jobs import follow_job
from .prelude import prelude
from .scripts import RunSettings, local_file, python_command, remote_file, resolve_settings
from .slurm import OVERRIDES, explicit_options, sbatch_options, submit, wrapper_script
from .ssh import quote, remote_tmp_path, run_command, run_script, shell_join, write_remote_file
from .sync import push

SBATCH_SUFFIXES = {".sbatch", ".slurm", ".sh"}


def overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {attr: getattr(args, attr, None) for attr, _ in OVERRIDES}


def maybe_push(cfg: Config, args: argparse.Namespace) -> None:
    wanted = cfg.behaviour["push_before_run"] if args.push is None else args.push
    if wanted and cfg.project:
        push(cfg)


def after_submit(cfg: Config, args: argparse.Namespace, job_id: str) -> None:
    wait = cfg.behaviour["wait"] if args.wait is None else args.wait
    if wait:
        follow_job(cfg, job_id, pull_after=args.pull_after, filter_name=args.filter, lines=args.tail_lines)


def submit_python(cfg: Config, args: argparse.Namespace, local: Path, forwarded: list[str], settings: RunSettings) -> str:
    """Wrap a project .py file in a generated sbatch file and submit it."""

    project = cfg.require_project()
    job_name = args.job_name or settings.job_name or local.stem.replace("_", "-")
    script = wrapper_script(
        cfg,
        python_command(project, local, forwarded),
        settings,
        overrides(args),
        job_name=job_name,
        exports=args.export,
    )
    path = remote_tmp_path(cfg, f"job_{job_name}", ".sbatch")
    write_remote_file(cfg, path, script)
    return submit(cfg, path, [], [])


def handle_run(cfg: Config, args: argparse.Namespace) -> None:
    """Run a project .py file or any command on the login node (or a gpu-target file via SLURM)."""

    project = cfg.project
    local = local_file(project, args.program) if project else None
    settings = resolve_settings(cfg, local, target=args.target, activate=args.env, offline=args.offline)
    maybe_push(cfg, args)

    if settings.target == "gpu":
        if local is None or local.suffix != ".py":
            raise SystemExit("run --target gpu needs a .py file inside the project")
        after_submit(cfg, args, submit_python(cfg, args, local, args.forwarded, settings))
        return

    if local is not None and local.suffix == ".py":
        command = python_command(project, local, args.forwarded)
    else:
        command = [args.program, *args.forwarded]
    script = "\n".join([
        prelude(cfg, activate=settings.activate_env, offline=settings.offline,
                secrets=cfg.behaviour["forward_secrets"]),
        shell_join(command),
    ])
    run_script(cfg, script, interactive=True)


def run_pre_submit(cfg: Config, project: Project, local: Path, forwarded: list[str]) -> None:
    """Run the project's `pre_submit` check on the login node; a failure stops the submit."""

    hook = project.pre_submit
    if not hook:
        return
    rel = local.relative_to(project.local).as_posix()
    command = hook.replace("{program}", quote(rel)).replace("{args}", shell_join(forwarded))
    print(f"[pre_submit] {command}")
    script = prelude(cfg, activate=True, offline=cfg.behaviour["gpu_offline"]) + "\n" + command
    try:
        run_script(cfg, script)
    except subprocess.CalledProcessError:
        raise SystemExit("[pre_submit] failed, not submitting (skip it with --no-pre-submit)") from None


def handle_submit(cfg: Config, args: argparse.Namespace) -> None:
    """Submit a project .py file through a generated wrapper, or an existing sbatch script."""

    project = cfg.require_project()
    local = local_file(project, args.program)
    if local is None:
        raise SystemExit(f"{args.program} is not a file inside project {project.name} ({project.local})")
    if local.suffix != ".py" and local.suffix not in SBATCH_SUFFIXES:
        raise SystemExit(f"submit takes a .py file or an sbatch script ({', '.join(sorted(SBATCH_SUFFIXES))})")
    maybe_push(cfg, args)
    if not args.no_pre_submit:
        run_pre_submit(cfg, project, local, args.forwarded)

    if local.suffix == ".py":
        settings = resolve_settings(cfg, local, target="gpu", activate=args.env, offline=args.offline,
                                    default_target="gpu")
        job_id = submit_python(cfg, args, local, args.forwarded, settings)
    else:
        job_id = submit(cfg, remote_file(project, local), explicit_options(overrides(args)), args.forwarded)
    after_submit(cfg, args, job_id)


def handle_shell(cfg: Config, args: argparse.Namespace) -> None:
    """An interactive login shell on Kaya, starting in the project mirror."""

    start = f"cd {quote(cfg.workdir)} 2>/dev/null; " if cfg.workdir else ""
    run_command(cfg, f"{start}exec bash --login", tty=True, check=False)


def handle_interactive(cfg: Config, args: argparse.Namespace) -> None:
    """Ask SLURM for an interactive session (a GPU node by default) and open a shell on it."""

    values = overrides(args)
    options = sbatch_options(cfg, values, job_name=args.job_name or "interactive")
    options = [o for o in options if not o.startswith("--dependency")]
    start = f"cd {quote(cfg.workdir)} 2>/dev/null; " if cfg.workdir else ""
    command = f"{start}srun {shell_join(options)} --pty bash --login"
    print(f"[interactive] {command}  (waits in the queue like any job)")
    run_command(cfg, command, tty=True, check=False)
