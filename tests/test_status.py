"""The status report: sinfo/squeue/sacct parsing, GPU grouping, access and ordering."""

from datetime import datetime, timezone

from kaya_tools.status.cluster import (
    CompletedJob,
    PartitionAcl,
    QueueRow,
    gpu_partitions,
    parse_completed_jobs,
    parse_gpu_model,
    parse_gres_gpu_count,
    parse_nodes,
    parse_queue,
)
from kaya_tools.status.report import print_text, summarize_gpu_models

# One sinfo -O row per node/partition pair, as `sinfo -h -N -O ...` emits them.
NODE_OUTPUT = "\n".join(
    [
        "k018  ondemand-gpu  mix    gpu:v100-32gb:2  gpu:v100-32gb:1(IDX:1)  32/4/0/36   772000   98304  625222",
        "k018  admin         mix    gpu:v100-32gb:2  gpu:v100-32gb:1(IDX:1)  32/4/0/36   772000   98304  625222",
        "k173  admin         down*  gpu:h100:4       gpu:h100:0(IDX:N/A)     0/0/96/96  1547000       0  N/A",
        "k178  medical       down*  gpu:h100:4       gpu:h100:0(IDX:N/A)     0/0/96/96  1544000       0  N/A",
        "k001  admin         mix    (null)           (null)                  52/44/0/96 1544000  335872  1471626",
    ]
)

ACLS = {
    "ondemand-gpu": PartitionAcl("ondemand-gpu", frozenset({"kaya-users"}), None),
    "admin": PartitionAcl("admin", frozenset({"kaya-admins"}), None),
    "medical": PartitionAcl("medical", frozenset({"medical-users"}), None),
}

GROUPS = frozenset({"lxu", "kaya-users", "ems036"})


def test_queue_wait_uses_start_for_running_and_now_for_pending() -> None:
    """Queue waits stop at start for running jobs and keep growing for pending jobs."""

    output = "\n".join(
        [
            "1|running|me|gpu|R|01:00|02:00|1|gpu:1|node1|2026-07-11T10:00:00|2026-07-11T10:30:00",
            "2|pending|me|gpu|PD|0:00|02:00|1|gpu:1|Priority|2026-07-11T11:00:00|N/A",
        ]
    )

    rows = parse_queue(
        output,
        scheduler_tz=timezone.utc,
        now=datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc),
    )

    assert rows[0].partition == "gpu"
    assert rows[0].submitted == "2026-07-11T10:00:00"
    assert rows[0].queue_wait == "00:30:00"
    assert rows[1].queue_wait == "01:00:00"


def test_completed_jobs_are_newest_first_and_limited() -> None:
    """Completed jobs are ordered by end time and capped at the requested count."""

    output = "\n".join(
        [
            "3|older|me|COMPLETED|01:00:00|02:00:00|2026-07-11T12:00:00|2026-07-11T13:00:00|gpu|cpu=4,gres/gpu=2",
            "4|failed|me|FAILED|00:10:00|02:00:00|2026-07-11T13:00:00|2026-07-11T13:10:00|gpu|cpu=4,gres/gpu=2",
            "2|newer|me|COMPLETED|00:30:00|02:00:00|2026-07-11T14:00:00|2026-07-11T14:30:00|gpu|cpu=4,gres/gpu=2",
        ]
    )

    rows = parse_completed_jobs(output, partitions=["gpu"], limit=1)

    assert [row.job_id for row in rows] == ["2"]


def test_completed_jobs_span_every_selected_partition() -> None:
    """A job counts as long as its partition is one of the selected ones."""

    output = "\n".join(
        [
            "1|a|me|COMPLETED|01:00:00|02:00:00|2026-07-11T12:00:00|2026-07-11T13:00:00|gpu|cpu=4",
            "2|b|me|COMPLETED|01:00:00|02:00:00|2026-07-11T13:00:00|2026-07-11T14:00:00|ondemand-gpu|cpu=4",
            "3|c|me|COMPLETED|01:00:00|02:00:00|2026-07-11T14:00:00|2026-07-11T15:00:00|work|cpu=4",
        ]
    )

    rows = parse_completed_jobs(output, partitions=["gpu", "ondemand-gpu"], limit=10)

    assert [row.job_id for row in rows] == ["2", "1"]


def test_parse_gpu_model_reads_the_model_out_of_gres() -> None:
    """GPU model comes from the middle field of a gres entry."""

    assert parse_gpu_model("gpu:h100:4") == "h100"
    assert parse_gpu_model("gpu:v100-32gb:2") == "v100-32gb"
    assert parse_gpu_model("gpu:v100:2(S:0-1)") == "v100"
    assert parse_gpu_model("gpu:a100:2,gpu:h100:4") == "a100+h100"
    assert parse_gpu_model("gpu:2") == "gpu"
    assert parse_gpu_model("(null)") == ""


def test_parse_gres_gpu_count_handles_used_and_idx_suffixes() -> None:
    """GPU counts parse from both Gres and the GresUsed `(IDX:...)` form."""

    assert parse_gres_gpu_count("gpu:h100:4") == 4
    assert parse_gres_gpu_count("gpu:v100-32gb:1(IDX:1)") == 1
    assert parse_gres_gpu_count("gpu:h100:0(IDX:N/A)") == 0
    assert parse_gres_gpu_count("gpu:a100:2,gpu:h100:4") == 6
    assert parse_gres_gpu_count("(null)") == 0


def test_parse_nodes_dedupes_across_partitions_and_reads_gpu_models() -> None:
    """A node in several partitions appears once, carrying all of them."""

    nodes = {node.name: node for node in parse_nodes(NODE_OUTPUT, acls=ACLS, groups=GROUPS)}

    assert nodes["k018"].partitions == ("admin", "ondemand-gpu")
    assert nodes["k018"].gpu_model == "v100-32gb"
    assert (nodes["k018"].gpu_alloc, nodes["k018"].gpu_total) == (1, 2)
    assert (nodes["k018"].cpu_alloc, nodes["k018"].cpu_total) == (32, 36)
    assert nodes["k173"].gpu_model == "h100"
    assert nodes["k173"].gpu_total == 4
    assert nodes["k001"].gpu_total == 0 and nodes["k001"].gpu_model == "-"


def test_parse_nodes_access_uses_the_most_permissive_partition() -> None:
    """Access is `yes` if any partition holding the node admits your groups."""

    nodes = {node.name: node for node in parse_nodes(NODE_OUTPUT, acls=ACLS, groups=GROUPS)}

    # k018 sits in admin (kaya-admins only) and ondemand-gpu (kaya-users): reachable.
    assert nodes["k018"].access == "yes"
    # The H100s sit only behind groups this user is not in.
    assert nodes["k173"].access == "no"
    assert nodes["k178"].access == "no"


def test_parse_nodes_flags_unknown_access_for_account_restricted_partitions() -> None:
    """An account allowlist cannot be resolved here, so access is reported as unknown."""

    acls = {"x": PartitionAcl("x", None, frozenset({"someacct"}))}
    row = "k900  x  idle  gpu:h100:4  gpu:h100:0(IDX:N/A)  0/0/96/96  1544000  0  N/A"

    assert parse_nodes(row, acls=acls, groups=GROUPS)[0].access == "?"


def test_summarize_gpu_models_counts_only_up_nodes_as_available() -> None:
    """Down nodes contribute to totals but never to the free count."""

    summary = {
        entry["gpu_model"]: entry
        for entry in summarize_gpu_models(parse_nodes(NODE_OUTPUT, acls=ACLS, groups=GROUPS))
    }

    assert summary["h100"]["gpus_total"] == 8
    assert summary["h100"]["gpus_up"] == 0
    assert summary["h100"]["gpus_free"] == 0
    assert summary["h100"]["access"] == "no"
    assert summary["v100-32gb"]["gpus_total"] == 2
    assert summary["v100-32gb"]["gpus_free"] == 1
    assert "-" not in summary  # CPU-only nodes are not a GPU model


def test_gpu_partition_names_includes_restricted_partitions(monkeypatch) -> None:
    """Discovery keeps partitions the caller cannot submit to, so H100s stay visible."""

    output = "\n".join(
        [
            "work|(null)",
            "gpu|gpu:v100:2",
            "medical|gpu:h100:4",
            "ondemand-gpu|gpu:v100-32gb:2",
            "ondemand-gpu|gpu:v100:2",
        ]
    )
    monkeypatch.setattr("kaya_tools.status.cluster.query", lambda cfg, command: output)

    assert gpu_partitions(None) == ["gpu", "medical", "ondemand-gpu"]


def test_your_jobs_are_complete_and_others_are_limited_running_first(capsys) -> None:
    """Your jobs are never cut; the shared table drops your rows, lists running first
    and stops at the limit."""

    def queue_row(job_id: str, user: str, state: str) -> QueueRow:
        return QueueRow(
            job_id,
            "job",
            user,
            "gpu",
            state,
            "0:00",
            "1:00",
            "1",
            "gpu:1",
            "Priority",
            "2026-07-11T12:00:00",
            "N/A",
            "00:10:00",
        )

    def completed_row(job_id: str) -> CompletedJob:
        return CompletedJob(
            job_id,
            "job",
            "me",
            "COMPLETED",
            "01:00:00",
            "02:00:00",
            "2026-07-11T12:00:00",
            "2026-07-11T13:00:00",
            "gpu",
            "cpu=4,gres/gpu=2",
        )

    mine = [queue_row("mine-1", "me", "R"), queue_row("mine-2", "me", "PD")]
    snapshot = {
        "partitions": ["gpu", "ondemand-gpu"],
        "user": "me",
        "groups": [],
        "scheduler_tz": "UTC",
        "nodes": [],
        "queue": [queue_row("other-pd", "you", "PD"), queue_row("other-r", "you", "R"),
                  queue_row("other-pd2", "you", "PD"), *mine],
        "mine": mine,
        "completed": [completed_row("done-1"), completed_row("done-2")],
    }
    print_text(snapshot, limit=2)

    output = capsys.readouterr().out
    shared, _, yours = output.partition("Your jobs")
    assert "mine-" not in shared
    assert shared.index("other-r") < shared.index("other-pd")
    assert "other-pd2" not in shared and "1 more not shown" in shared
    assert "mine-1" in yours and "mine-2" in yours
    assert "done-1" in output and "done-2" in output
