"""What gets sent to Kaya: ssh flags, the shell prelude, sync commands and job scripts."""

from __future__ import annotations

import pytest

from kaya_tools import ssh, sync
from kaya_tools.cli import build_parser, split_forwarded
from kaya_tools.execute import overrides
from kaya_tools.prelude import prelude
from kaya_tools.scripts import RunSettings, local_file, python_command, resolve_settings
from kaya_tools.slurm import explicit_options, sbatch_options, wrapper_script


def test_ssh_uses_the_configured_user_key_and_port(cfg):
    argv = ssh.ssh_argv(cfg)
    assert argv[0] == "ssh" and argv[-1] == "me@kaya.example"
    assert argv[argv.index("-p") + 1] == "22"
    assert argv[argv.index("-i") + 1].endswith(".ssh/id_test")
    assert any(o.startswith("ControlPath=") for o in argv)
    assert ssh.rsync_shell(cfg).startswith("ssh -p 22 -i ")


def test_prelude_cds_loads_modules_exports_and_activates(cfg, tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=secret\nOTHER=x\n")
    cfg.data["secrets"] = {"env_file": str(env), "forward": ["HF_TOKEN"]}

    online = prelude(cfg, activate=True, offline=False, secrets=True)
    assert "cd /group/g1/me/demo" in online
    assert "module load Anaconda3/2024.06\n" in online + "\n"
    assert "module load gcc/9.4.0 2>/dev/null || true" in online
    assert "export PYTHONPATH=/group/g1/me/demo" in online
    assert "export HF_TOKEN=secret" in online and "OTHER" not in online
    assert "conda activate /group/g1/me/demo/envs/core" in online
    assert "HF_HUB_OFFLINE" not in online

    offline = prelude(cfg, activate=False, offline=True, secrets=True)
    assert "export HF_HUB_OFFLINE=1" in offline
    assert "HF_TOKEN" not in offline  # secrets never go offline
    assert "conda activate" not in offline


def test_headers_then_flags_decide_run_settings(cfg, project_dir):
    train = project_dir / "pkg" / "train.py"
    settings = resolve_settings(cfg, train)
    assert settings == RunSettings(target="gpu", activate_env=True, offline=True, job_name="trainer")
    assert resolve_settings(cfg, train, target="login", offline=False).target == "login"
    assert resolve_settings(cfg, project_dir / "loose script.py").target == "login"


def test_python_command_prefers_module_form(cfg, project_dir):
    project = cfg.project
    assert python_command(project, project_dir / "pkg" / "train.py", ["--x"]) == ["python", "-m", "pkg.train", "--x"]
    assert python_command(project, project_dir / "loose script.py", []) == [
        "python", "/group/g1/me/demo/loose script.py"]
    assert local_file(project, "pkg/train.py") == project_dir / "pkg" / "train.py"
    assert local_file(project, "/etc/hostname") is None


def parse(*argv: str):
    own, forwarded = split_forwarded(list(argv))
    args = build_parser().parse_args(own)
    args.forwarded = forwarded
    return args


def test_forwarded_args_split_on_the_first_double_dash():
    args = parse("submit", "pkg/train.py", "--time", "02:00:00", "--", "--time", "x")
    assert args.time == "02:00:00"
    assert args.forwarded == ["--time", "x"]


def test_wrapper_script_has_directives_prelude_exports_and_command(cfg, project_dir):
    args = parse("submit", "pkg/train.py", "--mem", "64G", "--export", "A=b c")
    script = wrapper_script(cfg, ["python", "-m", "pkg.train"], RunSettings("gpu", True, True),
                            overrides(args), job_name="trainer", exports=args.export)
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash --login"
    assert "#SBATCH --mem=64G" in lines and "#SBATCH --partition=gpu" in lines
    assert "#SBATCH --job-name=trainer" in lines
    assert "#SBATCH --output=/group/g1/me/demo/logs/%x_%j.out" in lines
    assert not any(line.startswith("#SBATCH --account") for line in lines)  # empty values are dropped
    at = lines.index("export A='b c'")
    assert any("conda activate" in line for line in lines[:at])
    assert lines[-1] == "python -m pkg.train"


def test_bad_export_is_rejected(cfg):
    with pytest.raises(SystemExit):
        wrapper_script(cfg, ["true"], RunSettings("gpu", True, True), {}, job_name="x", exports=["NOVALUE"])


def test_existing_sbatch_only_gets_flags_the_user_passed(cfg):
    args = parse("submit", "job.sbatch", "--time", "00:10:00")
    assert explicit_options(overrides(args)) == ["--time=00:10:00"]
    assert "--time=01:00:00" in sbatch_options(cfg, {}, job_name="j")


def capture_rsync(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(sync, "run_local", lambda command, **kw: calls.append(list(command)))
    monkeypatch.setattr(sync, "run_script", lambda *a, **kw: None)
    return calls


def test_pull_brings_back_each_pull_folder_unfiltered_by_default(cfg, monkeypatch):
    calls = capture_rsync(monkeypatch)
    sync.pull(cfg)
    assert [c[-2] for c in calls] == ["me@kaya.example:/group/g1/me/demo/results/",
                                      "me@kaya.example:/group/g1/me/demo/logs/"]
    assert all("--exclude=*" not in c for c in calls)


def test_pull_filter_includes_matches_and_excludes_the_rest(cfg, monkeypatch):
    calls = capture_rsync(monkeypatch)
    sync.pull(cfg, ["results"], filter_name="text")
    (command,) = calls
    assert "--include=*.jsonl" in command and "--include=*/" in command
    assert "--prune-empty-dirs" in command
    # rsync reads rules in order: includes must come before the catch-all exclude
    assert command.index("--include=*/") < command.index("--exclude=*")


def test_pull_uses_the_configured_default_filter_unless_told_otherwise(cfg, monkeypatch):
    cfg.data["behaviour"]["pull_filter"] = "text"
    calls = capture_rsync(monkeypatch)
    sync.pull(cfg, ["results"])
    sync.pull(cfg, ["results"], everything=True)
    assert "--exclude=*" in calls[0] and "--exclude=*" not in calls[1]


def test_push_mirrors_with_delete_and_excludes(cfg, monkeypatch):
    calls = capture_rsync(monkeypatch)
    sync.push(cfg)
    (command,) = calls
    assert "--delete" in command and ".git/" in command
    assert command[-1] == "me@kaya.example:/group/g1/me/demo/"


def test_push_refuses_to_mirror_onto_home(cfg, monkeypatch):
    capture_rsync(monkeypatch)
    object.__setattr__(cfg.project, "remote", "/group/g1/me")
    with pytest.raises(Exception, match="refusing"):
        sync.push(cfg)
