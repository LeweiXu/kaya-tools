"""Copy the project between this machine and Kaya: `push` mirrors source up with rsync,
`pull` brings result folders back, optionally through a named include filter."""

from __future__ import annotations

import subprocess

from .config import Config, ConfigError, Project
from .ssh import destination, quote, rsync_shell, run_local, run_script


def check_safe_mirror(cfg: Config, project: Project) -> None:
    """Refuse a `--delete` push that could wipe something other than the mirror."""

    if not project.local.is_dir():
        raise ConfigError(f"project folder {project.local} does not exist")
    remote = project.remote.rstrip("/")
    if remote in {"", "/"} or remote == cfg.home or remote.count("/") < 2:
        raise ConfigError(f"refusing to mirror into {remote!r}; point the project at its own folder")


def ensure_remote_dirs(cfg: Config, project: Project) -> None:
    dirs = [project.remote, project.remote_logs, cfg.tmp_dir]
    run_script(cfg, "mkdir -p " + " ".join(quote(d) for d in dirs))


def push(cfg: Config, *, dry_run: bool = False) -> None:
    """Mirror the local project to Kaya. The local tree is the source of truth."""

    project = cfg.require_project()
    check_safe_mirror(cfg, project)
    if not dry_run:
        ensure_remote_dirs(cfg, project)
    command = ["rsync", "-az", "-e", rsync_shell(cfg)]
    if cfg.behaviour["push_delete"]:
        command.append("--delete")
    if dry_run:
        command += ["--dry-run", "--itemize-changes"]
    for pattern in project.excludes:
        command += ["--exclude", pattern]
    command += [f"{project.local}/", f"{destination(cfg)}:{project.remote}/"]
    print(f"[push] {project.local} -> {cfg.ssh['host']}:{project.remote}")
    run_local(command)


def filter_args(patterns: list[str]) -> list[str]:
    """rsync args that keep only files matching `patterns`, without empty dir husks.

    Directories have to be included so rsync can walk into them; pruning stops that
    leaving an empty local copy of every directory that held nothing wanted.
    """

    if not patterns:
        return []
    return ["--prune-empty-dirs", "--include=*/", *(f"--include={p}" for p in patterns), "--exclude=*"]


def pull_patterns(cfg: Config, filter_name: str | None, everything: bool) -> list[str]:
    if everything:
        return []
    name = filter_name if filter_name is not None else cfg.behaviour["pull_filter"]
    if not name:
        return []
    if name not in cfg.pull_filters:
        known = ", ".join(sorted(cfg.pull_filters)) or "none"
        raise ConfigError(f"no pull filter {name!r} (configured: {known})")
    return list(cfg.pull_filters[name])


def pull(
    cfg: Config,
    paths: list[str] | None = None,
    *,
    filter_name: str | None = None,
    everything: bool = False,
    dry_run: bool = False,
) -> None:
    """Bring `paths` (default: the project's `pull` list) back from Kaya.

    A missing remote folder is skipped rather than treated as a failure, since a
    fresh project has no results yet.
    """

    project = cfg.require_project()
    patterns = pull_patterns(cfg, filter_name, everything)
    for rel in paths or project.pull:
        rel = rel.strip("/")
        local_dir = project.local / rel
        local_dir.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-az", "-e", rsync_shell(cfg), *filter_args(patterns)]
        if dry_run:
            command += ["--dry-run", "--itemize-changes"]
        command += [f"{destination(cfg)}:{project.remote_path(rel)}/", f"{local_dir}/"]
        label = f" (filter: {', '.join(patterns)})" if patterns else ""
        print(f"[pull] {rel}/{label}")
        try:
            run_local(command)
        except subprocess.CalledProcessError as exc:
            print(f"[pull] {rel}/ skipped (rsync exited {exc.returncode})")
