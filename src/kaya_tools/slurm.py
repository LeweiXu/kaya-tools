"""Talk to SLURM: build sbatch options and wrappers, submit, read squeue, wait for a job
to finish, and remember the last job id."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import STATE_DIR, Config
from .prelude import prelude
from .ssh import RemoteError, query, quote, run_script, shell_join
from .scripts import RunSettings

LAST_JOB = STATE_DIR / "last_job"

#: CLI attribute -> sbatch flag, for the options a user can override per job.
OVERRIDES = (
    ("job_name", "job-name"),
    ("partition", "partition"),
    ("gres", "gres"),
    ("account", "account"),
    ("qos", "qos"),
    ("cpus_per_task", "cpus-per-task"),
    ("mem", "mem"),
    ("time", "time"),
    ("dependency", "dependency"),
)

QUEUE_FORMAT = "%i|%P|%j|%u|%t|%M|%l|%D|%R"
STATE_NAMES = {"PD": "pending", "R": "running", "CG": "completing", "CF": "configuring", "S": "suspended"}


@dataclass(frozen=True)
class QueuedJob:
    """One `squeue` row."""

    job_id: str
    partition: str
    name: str
    user: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    reason: str


def simple_id(job_id: str) -> str:
    """Drop the `;cluster` suffix `sbatch --parsable` can add."""

    return job_id.split(";", 1)[0].strip()


def sbatch_options(cfg: Config, overrides: dict[str, Any], *, job_name: str) -> list[str]:
    """Config defaults with any non-None overrides on top."""

    values = {key.replace("_", "-"): value for key, value in cfg.slurm.items() if key != "poll_seconds"}
    for attr, flag in OVERRIDES:
        if overrides.get(attr) is not None:
            values[flag] = overrides[attr]
    values["job-name"] = job_name
    return [f"--{flag}={value}" for flag, value in values.items() if value not in ("", None)]


def explicit_options(overrides: dict[str, Any]) -> list[str]:
    """Only the options the user passed, for an existing .sbatch that sets its own."""

    return [f"--{flag}={overrides[attr]}" for attr, flag in OVERRIDES if overrides.get(attr) is not None]


def export_lines(items: list[str]) -> list[str]:
    lines = []
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            raise SystemExit(f"--export wants NAME=VALUE, got {item!r}")
        lines.append(f"export {name}={quote(value)}")
    return lines


def wrapper_script(
    cfg: Config,
    command: list[str],
    settings: RunSettings,
    overrides: dict[str, Any],
    *,
    job_name: str,
    exports: list[str],
) -> str:
    """A complete sbatch file that runs `command` inside the project prelude."""

    logs = cfg.require_project().remote_logs
    directives = sbatch_options(cfg, overrides, job_name=job_name)
    return "\n".join([
        "#!/bin/bash --login",
        *[f"#SBATCH {option}" for option in directives],
        f"#SBATCH --output={logs}/%x_%j.out",
        f"#SBATCH --error={logs}/%x_%j.err",
        "",
        prelude(cfg, activate=settings.activate_env, offline=settings.offline),
        *export_lines(exports),
        'echo "[job] host=$(hostname) pwd=$(pwd)"',
        "echo " + quote(f"[job] command: {shell_join(command)}"),
        shell_join(command),
        "",
    ])


def submit(cfg: Config, sbatch_path: str, options: list[str], script_args: list[str]) -> str:
    """Submit a remote sbatch file from the working dir and return the job id.

    The logs dir is created first: SLURM silently writes nothing if it is missing.
    """

    mkdir = f"mkdir -p {quote(cfg.project.remote_logs)}\n" if cfg.project else ""
    command = shell_join(["sbatch", "--parsable", *options, sbatch_path, *script_args])
    result = run_script(cfg, f"cd {quote(cfg.workdir)}\n{mkdir}{command}", capture=True)
    job_id = simple_id(result.stdout.strip().splitlines()[-1])
    remember_job(job_id)
    print(f"Submitted job {job_id}")
    return job_id


def remember_job(job_id: str) -> None:
    LAST_JOB.parent.mkdir(parents=True, exist_ok=True)
    LAST_JOB.write_text(job_id + "\n")


def job_or_last(job_id: str | None) -> str:
    if job_id:
        return simple_id(job_id)
    if not LAST_JOB.is_file():
        raise SystemExit("no job id given and no job submitted through kaya yet")
    return LAST_JOB.read_text().strip()


def parse_queue_row(line: str) -> QueuedJob | None:
    parts = line.rstrip("\n").split("|", 8)
    return QueuedJob(*parts) if len(parts) == 9 else None


def squeue(cfg: Config, selector: str) -> list[QueuedJob]:
    output = query(cfg, f"squeue -h {selector} -o {quote(QUEUE_FORMAT)}")
    return [row for row in map(parse_queue_row, output.splitlines()) if row]


def my_jobs(cfg: Config) -> list[QueuedJob]:
    return squeue(cfg, '-u "$(whoami)"')


def describe(job: QueuedJob) -> str:
    state = STATE_NAMES.get(job.state, job.state)
    if job.state == "PD":
        return f"Job {job.job_id} ({job.name}) pending {job.elapsed}/{job.time_limit} on {job.partition}: {job.reason}"
    if job.state in {"R", "CG"}:
        return f"Job {job.job_id} ({job.name}) {state} {job.elapsed}/{job.time_limit} on {job.reason}"
    return f"Job {job.job_id} ({job.name}) {state} {job.elapsed}/{job.time_limit} [{job.reason}]"


def final_state(cfg: Config, job_id: str) -> str:
    try:
        out = query(cfg, f"sacct -X -n -P -j {quote(job_id)} --format=State,ExitCode,Elapsed")
    except RemoteError:
        return ""
    return out.strip().splitlines()[0] if out.strip() else ""


def wait_for(cfg: Config, job_id: str) -> str:
    """Poll until the job leaves the queue, then return its sacct State|ExitCode|Elapsed.

    An empty squeue answer can be a scheduler blip rather than a finished job, so
    the job only counts as gone once sacct agrees it has an end state.
    """

    poll = int(cfg.slurm["poll_seconds"])
    start = time.monotonic()
    print(f"Waiting for job {job_id} (poll every {poll}s, Ctrl-C stops waiting, not the job)")
    unconfirmed = 0
    while True:
        waited = int(time.monotonic() - start)
        try:
            rows = squeue(cfg, f"-j {quote(job_id)}")
        except RemoteError as exc:
            print(f"[wait {waited}s] squeue failed: {exc}; retrying")
            time.sleep(poll)
            continue
        if rows:
            print(f"[wait {waited}s] {describe(rows[0])}", flush=True)
            time.sleep(poll)
            continue
        state = final_state(cfg, job_id)
        if state and not state.split("|")[0].startswith(("PENDING", "RUNNING", "REQUEUED")):
            print(f"Job {job_id} finished after {waited}s: {state.replace('|', '  ')}")
            return state
        unconfirmed += 1
        if unconfirmed >= 3:
            print(f"Job {job_id} left the queue after {waited}s (sacct has no end state for it)")
            return state
        time.sleep(poll)
