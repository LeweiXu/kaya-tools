"""Summarise the cluster snapshot and print it, as tables or as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from ..config import Config
from ..table import print_table
from .cluster import (
    CompletedJob,
    NodeStatus,
    QueueRow,
    best_access,
    load_completed_jobs,
    load_nodes,
    load_partition_acls,
    load_queue,
    remote_groups,
    remote_timezone,
    remote_user,
    resolve_partitions,
)


def node_is_up(node: NodeStatus) -> bool:
    state = node.state.upper()
    return not any(flag in state for flag in ("DOWN", "DRAIN", "FAIL", "NOT_RESPONDING"))


def summarize_gpu_models(nodes: list[NodeStatus]) -> list[dict[str, Any]]:
    """Per-model totals; only GPUs on nodes that are up count as available."""

    models: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not node.gpu_total:
            continue
        entry = models.setdefault(node.gpu_model, {
            "gpu_model": node.gpu_model, "gpus_total": 0, "gpus_alloc": 0, "gpus_up": 0,
            "nodes": 0, "nodes_up": 0, "access": [],
        })
        entry["gpus_total"] += node.gpu_total
        entry["gpus_alloc"] += node.gpu_alloc
        entry["nodes"] += 1
        if node_is_up(node):
            entry["gpus_up"] += node.gpu_total
            entry["nodes_up"] += 1
        if node.access not in entry["access"]:
            entry["access"].append(node.access)
    for entry in models.values():
        entry["access"] = best_access(entry["access"])
        entry["gpus_free"] = max(0, entry["gpus_up"] - entry["gpus_alloc"])
    return [models[name] for name in sorted(models)]


def summarize(nodes: list[NodeStatus]) -> dict[str, Any]:
    return {
        "nodes": len(nodes),
        "gpus_alloc": sum(n.gpu_alloc for n in nodes),
        "gpus_total": sum(n.gpu_total for n in nodes),
        "cpus_alloc": sum(n.cpu_alloc for n in nodes),
        "cpus_total": sum(n.cpu_total for n in nodes),
        "mem_alloc_gib": round(sum(n.mem_alloc_gib for n in nodes), 1),
        "mem_total_gib": round(sum(n.mem_total_gib for n in nodes), 1),
        "mem_free_gib": round(sum(n.mem_free_gib for n in nodes), 1),
        "by_gpu_model": summarize_gpu_models(nodes),
    }


def others_first_running(queue: list[QueueRow], user: str | None, limit: int) -> tuple[list[QueueRow], int]:
    """Other users' rows, running before pending, cut to `limit`; also how many were cut."""

    others = [row for row in queue if row.user != user]
    others.sort(key=lambda row: row.state != "R")
    return others[:limit], max(0, len(others) - limit)


def print_queue(rows: list[QueueRow], *, show_queue_time: bool = False) -> None:
    headers = ["job_id", "partition", "state", "elapsed", "limit", "nodes", "gres", "name", "user"]
    if show_queue_time:
        headers += ["submitted", "queue_wait"]
    headers.append("reason/nodelist")
    table = []
    for r in rows:
        values: list[object] = [r.job_id, r.partition, r.state, r.elapsed, r.time_limit, r.nodes, r.gres, r.name, r.user]
        if show_queue_time:
            values += [r.submitted, r.queue_wait]
        values.append(r.reason_or_nodelist)
        table.append(values)
    print_table(headers, table)


def print_completed(rows: list[CompletedJob]) -> None:
    print_table(
        ["job_id", "elapsed", "limit", "started", "ended", "name", "partition", "resources"],
        [[r.job_id, r.elapsed, r.time_limit, r.started, r.ended, r.name, r.partition, r.alloc_tres] for r in rows],
    )


def print_text(snapshot: dict[str, Any], *, limit: int) -> None:
    nodes: list[NodeStatus] = snapshot["nodes"]
    summary = summarize(nodes)
    now = datetime.now().astimezone()
    print(f"Kaya status for partitions {', '.join(snapshot['partitions'])} at "
          f"{now:%Y-%m-%d %H:%M:%S} {now.tzname()} (scheduler time zone {snapshot['scheduler_tz']})")
    print(f"Summary: GPUs {summary['gpus_alloc']}/{summary['gpus_total']} allocated, "
          f"CPUs {summary['cpus_alloc']}/{summary['cpus_total']} allocated, "
          f"memory {summary['mem_alloc_gib']}/{summary['mem_total_gib']} GiB allocated "
          f"({summary['mem_free_gib']} GiB reported free)")
    print()
    print("GPUs by model  (free counts only nodes that are up; access compares your Unix groups to the "
          "partition's AllowGroups, '?' when an account list also applies)")
    print_table(
        ["gpu_model", "nodes_up/total", "gpus_up/total", "allocated", "free", "access"],
        [[e["gpu_model"], f"{e['nodes_up']}/{e['nodes']}", f"{e['gpus_up']}/{e['gpus_total']}",
          e["gpus_alloc"], e["gpus_free"], e["access"]] for e in summary["by_gpu_model"]],
    )
    print("\nNodes")
    print_table(
        ["node", "gpu_model", "state", "gpu", "cpu", "mem_alloc/total", "free_mem", "access", "partitions"],
        [[n.name, n.gpu_model, n.state, f"{n.gpu_alloc}/{n.gpu_total}", f"{n.cpu_alloc}/{n.cpu_total}",
          f"{n.mem_alloc_gib}/{n.mem_total_gib}G", f"{n.mem_free_gib}G", n.access, ",".join(n.partitions)]
         for n in nodes],
    )
    others, hidden = others_first_running(snapshot["queue"], snapshot["user"], limit)
    print("\nOther users' running/pending jobs  (running first)")
    print_queue(others)
    if hidden:
        print(f"... {hidden} more not shown (--limit)")
    print("\nYour jobs")
    print_queue(snapshot["mine"], show_queue_time=True)
    print("\nRecently completed jobs")
    print_completed(snapshot["completed"])


def collect(cfg: Config, *, partition: str, user: str | None) -> dict[str, Any]:
    partitions = resolve_partitions(cfg, partition)
    acls = load_partition_acls(cfg)
    groups = remote_groups(cfg)
    scheduler_tz, tz_name = remote_timezone(cfg)
    who = remote_user(cfg, user)
    return {
        "partitions": partitions,
        "user": who,
        "groups": sorted(groups),
        "scheduler_tz": tz_name,
        "nodes": load_nodes(cfg, partitions, acls=acls, groups=groups),
        "queue": load_queue(cfg, partitions, scheduler_tz=scheduler_tz),
        "mine": load_queue(cfg, partitions, scheduler_tz=scheduler_tz, user=who) if who else [],
        "completed": load_completed_jobs(cfg, partitions, user=who) if who else [],
    }


def handle_status(cfg: Config, args: argparse.Namespace) -> int:
    snapshot = collect(cfg, partition=args.partition, user=args.user)
    if args.json:
        out = dict(snapshot)
        out["summary"] = summarize(snapshot["nodes"])
        for key in ("nodes", "queue", "mine", "completed"):
            out[key] = [asdict(row) for row in snapshot[key]]
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print_text(snapshot, limit=max(1, args.limit))
    return 0
