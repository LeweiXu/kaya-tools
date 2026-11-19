"""Commands for jobs already submitted: watch one to the end, list yours, show one's
accounting, read its logs on Kaya, and cancel."""

from __future__ import annotations

import argparse

from .config import Config
from .slurm import describe, job_or_last, my_jobs, wait_for
from .ssh import RemoteError, query, quote, run_command
from .sync import pull
from .table import print_table


def remote_logs(cfg: Config, job_id: str) -> list[str]:
    """The job's stdout/stderr files on Kaya.

    Looks in the project's logs dir first (`<name>_<id>.out` is what kaya-generated
    jobs write), then asks scontrol, which knows the paths while the job is queued.
    """

    parts = []
    if cfg.project:
        logs = quote(cfg.project.remote_logs)
        parts.append(f"ls -1 {logs}/*_{job_id}.out {logs}/*_{job_id}.err 2>/dev/null")
    parts.append(f"scontrol show job {quote(job_id)} 2>/dev/null | grep -oE 'Std(Out|Err)=[^ ]+' | cut -d= -f2-")
    output = query(cfg, "; ".join(parts) + "; true")
    paths: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line and line not in paths and line != "/dev/null":
            paths.append(line)
    return paths


def show_logs(cfg: Config, job_id: str, *, lines: int, follow: bool = False, stream: str = "both") -> None:
    paths = remote_logs(cfg, job_id)
    if stream != "both":
        paths = [p for p in paths if p.endswith(f".{stream}")] or paths
    if not paths:
        print(f"[logs] no log files found for job {job_id}")
        return
    if follow:
        run_command(cfg, f"tail -n {lines} -F " + " ".join(quote(p) for p in paths), check=False, tty=True)
        return
    for path in paths:
        print(f"\n===== {path} =====", flush=True)
        run_command(cfg, f"tail -n {lines} {quote(path)} 2>/dev/null || echo '(not readable yet)'", check=False)


def follow_job(cfg: Config, job_id: str, *, pull_after: bool | None, filter_name: str | None, lines: int | None) -> None:
    """Wait for a job, pull the project's results (if configured to), then show its logs."""

    wait_for(cfg, job_id)
    do_pull = cfg.behaviour["pull_after_job"] if pull_after is None else pull_after
    if do_pull and cfg.project:
        pull(cfg, filter_name=filter_name)
    show_logs(cfg, job_id, lines=lines or cfg.behaviour["tail_lines"])


def handle_watch(cfg: Config, args: argparse.Namespace) -> None:
    follow_job(cfg, job_or_last(args.job_id), pull_after=args.pull, filter_name=args.filter, lines=args.tail_lines)


def handle_jobs(cfg: Config, args: argparse.Namespace) -> None:
    jobs = my_jobs(cfg)
    print_table(
        ["job_id", "name", "partition", "state", "elapsed", "limit", "nodes", "reason/nodelist"],
        [[j.job_id, j.name, j.partition, j.state, j.elapsed, j.time_limit, j.nodes, j.reason] for j in jobs],
        empty="(no queued or running jobs)",
    )


def handle_info(cfg: Config, args: argparse.Namespace) -> None:
    job_id = job_or_last(args.job_id)
    fields = "JobID,JobName%30,Partition,State,ExitCode,Elapsed,Timelimit,Submit,Start,End,NodeList,AllocTRES%50,MaxRSS"
    print(query(cfg, f"sacct -j {quote(job_id)} --format={fields}").rstrip())
    queued = query(cfg, f"scontrol show job {quote(job_id)} 2>/dev/null || true").strip()
    if queued:
        print("\n" + queued)


def handle_logs(cfg: Config, args: argparse.Namespace) -> None:
    show_logs(
        cfg,
        job_or_last(args.job_id),
        lines=args.lines or cfg.behaviour["tail_lines"],
        follow=args.follow,
        stream=args.stream,
    )


def list_mine(cfg: Config, label: str) -> None:
    try:
        jobs = my_jobs(cfg)
    except RemoteError as exc:
        print(f"[cancel] could not list jobs: {exc}")
        return
    print(f"[cancel] {len(jobs)} job(s) {label}")
    for job in jobs:
        print("  " + describe(job))


def handle_cancel(cfg: Config, args: argparse.Namespace) -> None:
    """Cancel ids, every job with a name, or all of yours; no arguments means the last job."""

    state = f" --state={quote(args.state)}" if args.state else ""
    commands = []
    if args.job_id:
        commands.append("scancel " + " ".join(quote(j) for j in args.job_id))
    if args.name:
        commands.append(f'scancel -u "$(whoami)" --name={quote(args.name)}{state}')
    if args.all:
        if cfg.behaviour["confirm_cancel_all"] and not args.yes:
            list_mine(cfg, "would be cancelled")
            if input("[cancel] cancel all of these? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("[cancel] nothing cancelled")
                return
        commands.append(f'scancel -u "$(whoami)"{state}')
    if not commands:
        commands.append(f"scancel {quote(job_or_last(None))}")

    remote = " && ".join(commands)
    print(f"[cancel] {remote}")
    if run_command(cfg, remote, check=False) != 0:
        raise SystemExit("[cancel] scancel failed")
    list_mine(cfg, "still queued or running")
