"""Read cluster state over ssh and parse it: GPU nodes, partition access lists, the
queue, and your recently completed jobs."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any

from ..config import Config
from ..ssh import RemoteError, query


@dataclass(frozen=True)
class NodeStatus:
    """One node's GPU model, allocation and whether you can submit to it."""

    name: str
    partitions: tuple[str, ...]
    state: str
    gres: str
    gpu_model: str
    gpu_alloc: int
    gpu_total: int
    cpu_alloc: int
    cpu_total: int
    mem_alloc_gib: float
    mem_total_gib: float
    mem_free_gib: float
    access: str


@dataclass(frozen=True)
class PartitionAcl:
    """A partition's submit restrictions. `None` means unrestricted."""

    name: str
    allow_groups: frozenset[str] | None
    allow_accounts: frozenset[str] | None


@dataclass(frozen=True)
class QueueRow:
    """One squeue row, plus how long it has waited."""

    job_id: str
    name: str
    user: str
    partition: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    gres: str
    reason_or_nodelist: str
    submitted: str
    started: str
    queue_wait: str


@dataclass(frozen=True)
class CompletedJob:
    """One completed allocation from sacct."""

    job_id: str
    name: str
    user: str
    state: str
    elapsed: str
    time_limit: str
    started: str
    ended: str
    partition: str
    alloc_tres: str


# sinfo -O fields in the order parse_nodes reads them. scontrol is not used for nodes:
# it says "Node not found" for nodes in restricted partitions, sinfo reports them fine.
NODE_FIELDS = (
    "NodeList:.18,Partition:.24,StateCompact:.14,Gres:.44,GresUsed:.44,"
    "CPUsState:.20,Memory:.14,AllocMem:.14,FreeMem:.14"
)


def parse_gres_gpu_count(text: str) -> int:
    """Sum GPU counts in a Gres/GresUsed string like `gpu:v100:4` or `gpu:2(IDX:0-1)`."""

    total = 0
    for entry in (text or "").split(","):
        entry = entry.split("(", 1)[0].strip()
        if not entry.startswith("gpu:"):
            continue
        try:
            total += int(entry.split(":")[-1])
        except ValueError:
            continue
    return total


def parse_gpu_model(gres: str) -> str:
    """GPU model(s) in a Gres string; `gpu` for a typeless `gpu:N`, empty for CPU nodes."""

    models: list[str] = []
    for entry in (gres or "").split(","):
        entry = entry.split("(", 1)[0].strip()
        if not entry.startswith("gpu:"):
            continue
        parts = entry.split(":")
        model = parts[1] if len(parts) >= 3 else "gpu"
        if model not in models:
            models.append(model)
    return "+".join(models)


def parse_mem_mb(value: str | None) -> int:
    """Parse a SLURM memory string (`64G`, `512000`) into MiB."""

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTP]?)", (value or "").strip())
    if not match:
        return 0
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3}
    return int(float(match.group(1)) * scale[match.group(2)])


def mib_to_gib(value: int) -> float:
    return round(value / 1024, 1)


def parse_acl_field(value: str) -> frozenset[str] | None:
    text = (value or "").strip()
    if not text or text in ("ALL", "(null)"):
        return None
    return frozenset(part for part in text.split(",") if part)


def partition_access(acl: PartitionAcl | None, groups: frozenset[str]) -> str:
    """`yes`, `no`, or `?` for whether these Unix groups may submit to a partition."""

    if acl is None:
        return "?"
    if acl.allow_groups is not None and not (acl.allow_groups & groups):
        return "no"
    # An account allowlist depends on the job's account, which is not known here.
    return "?" if acl.allow_accounts is not None else "yes"


def best_access(values: list[str]) -> str:
    for verdict in ("yes", "?", "no"):
        if verdict in values:
            return verdict
    return "?"


def parse_nodes(output: str, *, acls: dict[str, PartitionAcl], groups: frozenset[str]) -> list[NodeStatus]:
    """Parse `sinfo -N -O NODE_FIELDS` rows, merging a node listed under several partitions."""

    seen: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 9:
            continue
        name, partition, state, gres, gres_used, cpus, memory, alloc_mem, free_mem = fields
        cpu = cpus.split("/")
        entry = seen.setdefault(name, {
            "partitions": [],
            "state": state,
            "gres": "" if gres == "(null)" else gres,
            "gpu_total": parse_gres_gpu_count(gres),
            "gpu_alloc": parse_gres_gpu_count(gres_used),
            "cpu_alloc": int(cpu[0]) if cpu[0].isdigit() else 0,
            "cpu_total": int(cpu[3]) if len(cpu) == 4 and cpu[3].isdigit() else 0,
            "mem_total": parse_mem_mb(memory),
            "mem_alloc": parse_mem_mb(alloc_mem),
            "mem_free": parse_mem_mb(free_mem),
        })
        if partition not in entry["partitions"]:
            entry["partitions"].append(partition)

    nodes = [
        NodeStatus(
            name=name,
            partitions=tuple(sorted(e["partitions"])),
            state=e["state"],
            gres=e["gres"],
            gpu_model=parse_gpu_model(e["gres"]) or "-",
            gpu_alloc=e["gpu_alloc"],
            gpu_total=e["gpu_total"],
            cpu_alloc=e["cpu_alloc"],
            cpu_total=e["cpu_total"],
            mem_alloc_gib=mib_to_gib(e["mem_alloc"]),
            mem_total_gib=mib_to_gib(e["mem_total"]),
            mem_free_gib=mib_to_gib(e["mem_free"]),
            access=best_access([partition_access(acls.get(p), groups) for p in e["partitions"]]),
        )
        for name, e in seen.items()
    ]
    return sorted(nodes, key=lambda n: (n.gpu_model, n.name))


def parse_slurm_time(value: str, scheduler_tz: tzinfo) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=scheduler_tz)
    except ValueError:
        return None


def format_wait(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


def queue_wait(*, submitted: str, started: str, state: str, scheduler_tz: tzinfo, now: datetime) -> str:
    """Time spent queued: up to the start for a running job, up to now for a pending one."""

    submitted_at = parse_slurm_time(submitted, scheduler_tz)
    if submitted_at is None:
        return "unknown"
    started_at = parse_slurm_time(started, scheduler_tz) if state == "R" else None
    return format_wait((started_at or now) - submitted_at)


def parse_queue(output: str, *, scheduler_tz: tzinfo, now: datetime | None = None) -> list[QueueRow]:
    rows: list[QueueRow] = []
    current = now or datetime.now(scheduler_tz)
    for line in output.splitlines():
        if not line.strip() or line.startswith("JOBID|"):
            continue
        parts = line.split("|", 11)
        if len(parts) == 12:
            wait = queue_wait(submitted=parts[10], started=parts[11], state=parts[4],
                              scheduler_tz=scheduler_tz, now=current)
            rows.append(QueueRow(*parts, queue_wait=wait))
    return rows


def parse_completed_jobs(output: str, *, partitions: list[str] | None = None, limit: int = 10) -> list[CompletedJob]:
    """Completed sacct allocations, newest end time first."""

    wanted = set(partitions) if partitions else None
    rows: list[CompletedJob] = []
    for line in output.splitlines():
        parts = line.split("|", 9)
        if len(parts) != 10:
            continue
        state = parts[3].split()
        if state and state[0] == "COMPLETED" and (wanted is None or parts[8] in wanted):
            rows.append(CompletedJob(*parts))
    rows.sort(key=lambda r: r.ended if r.ended not in ("", "Unknown") else "", reverse=True)
    return rows[:limit]


def parse_utc_offset(value: str) -> timezone:
    match = re.fullmatch(r"([+-])(\d{2})(\d{2})", value.strip())
    if not match:
        return datetime.now().astimezone().tzinfo or timezone.utc  # type: ignore[return-value]
    sign = 1 if match.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))


# ---- remote loaders -------------------------------------------------------------


def remote_timezone(cfg: Config) -> tuple[tzinfo, str]:
    offset, _, name = query(cfg, "date '+%z|%Z'").strip().partition("|")
    return parse_utc_offset(offset), name or offset


def remote_user(cfg: Config, override: str | None) -> str | None:
    if override:
        return override
    try:
        return query(cfg, "whoami").strip() or None
    except RemoteError:
        return None


def remote_groups(cfg: Config) -> frozenset[str]:
    try:
        return frozenset(query(cfg, "id -nG").split())
    except RemoteError:
        return frozenset()


def gpu_partitions(cfg: Config) -> list[str]:
    """Every partition with GPU nodes. `sinfo -a` is needed to see restricted ones."""

    names: list[str] = []
    for line in query(cfg, "sinfo -a -h -o '%R|%G'").splitlines():
        name, _, gres = line.partition("|")
        name = name.strip()
        if name and parse_gpu_model(gres) and name not in names:
            names.append(name)
    return sorted(names)


def resolve_partitions(cfg: Config, requested: str) -> list[str]:
    if requested and requested != "auto":
        names = [p.strip() for p in requested.split(",") if p.strip()]
        if not names:
            raise SystemExit(f"no partitions in --partition {requested!r}")
        return names
    names = gpu_partitions(cfg)
    if not names:
        raise SystemExit("no GPU partitions found on the cluster")
    return names


def load_partition_acls(cfg: Config) -> dict[str, PartitionAcl]:
    acls: dict[str, PartitionAcl] = {}
    for block in query(cfg, "scontrol show partition -a").split("\n\n"):
        values = dict(re.findall(r"(\S+?)=(\S+)", block))
        name = values.get("PartitionName")
        if name:
            acls[name] = PartitionAcl(
                name=name,
                allow_groups=parse_acl_field(values.get("AllowGroups", "")),
                allow_accounts=parse_acl_field(values.get("AllowAccounts", "")),
            )
    return acls


def load_nodes(cfg: Config, partitions: list[str], *, acls: dict[str, PartitionAcl],
               groups: frozenset[str]) -> list[NodeStatus]:
    """GPU nodes in these partitions. CPU-only nodes sharing a GPU partition are dropped."""

    # No -a here: sinfo rejects it with -p, and -p already reaches hidden partitions.
    spec = shlex.quote(",".join(partitions))
    output = query(cfg, f"sinfo -h -p {spec} -N -O {shlex.quote(NODE_FIELDS)}")
    return [n for n in parse_nodes(output, acls=acls, groups=groups) if n.gpu_total]


def load_queue(cfg: Config, partitions: list[str], *, scheduler_tz: tzinfo, user: str | None = None) -> list[QueueRow]:
    user_arg = f"-u {shlex.quote(user)} " if user else ""
    fmt = "%i|%j|%u|%P|%t|%M|%l|%D|%b|%R|%V|%S"
    command = (f"squeue -a {user_arg}-p {shlex.quote(','.join(partitions))} -t RUNNING,PENDING "
               f"-h -o {shlex.quote(fmt)}")
    return parse_queue(query(cfg, command), scheduler_tz=scheduler_tz)


def load_completed_jobs(cfg: Config, partitions: list[str], *, user: str, limit: int = 10) -> list[CompletedJob]:
    fields = "JobIDRaw,JobName,User,State,Elapsed,Timelimit,Start,End,Partition,AllocTRES"
    command = ("start=$(date -d '30 days ago' +%Y-%m-%d); "
               f"sacct -X -n -P -u {shlex.quote(user)} -S \"$start\" --format={shlex.quote(fields)}")
    return parse_completed_jobs(query(cfg, command), partitions=partitions, limit=limit)
