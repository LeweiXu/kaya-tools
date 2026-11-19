"""Setup commands: `init` writes config.toml (or adds a project to it), `config` prints the
effective settings, and `doctor` checks each link from this machine to a working env."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import stat
import time
from dataclasses import asdict
from pathlib import Path

from .config import EXAMPLE_CONFIG, Config, default_config_path
from .ssh import RemoteError, query, quote


def toml_str(value: str) -> str:
    """A TOML basic string (JSON string escaping is valid TOML)."""

    return json.dumps(value)


def handle_init(config_path: Path | None, args: argparse.Namespace) -> int:
    path = config_path or default_config_path()
    if not path.exists():
        shutil.copyfile(EXAMPLE_CONFIG, path)
        print(f"Wrote {path} from the example. Fill in [ssh] and [cluster], then run `kaya doctor`.")
    elif not args.project:
        print(f"{path} already exists; leaving it alone.")

    if args.project:
        text = path.read_text()
        header = f"[projects.{args.project}]"
        if header in text:
            raise SystemExit(f"{header} already exists in {path}")
        lines = ["", header, f"local = {toml_str(str(Path.cwd()))}"]
        lines.append(f"remote = {toml_str(args.remote or args.project)}")
        if args.env:
            lines.append(f"env = {toml_str(args.env)}")
        path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n")
        print(f"Added {header} for {Path.cwd()} to {path}. Edit it there to set excludes, pull dirs and so on.")
    return 0


def handle_config(cfg: Config, args: argparse.Namespace) -> int:
    shown = {k: v for k, v in cfg.data.items() if k != "projects"}
    shown["config_file"] = str(cfg.path)
    shown["projects"] = sorted(cfg.data["projects"])
    if cfg.project:
        project = asdict(cfg.project)
        project["local"] = str(project["local"])
        shown["active_project"] = project
    print(json.dumps(shown, indent=2))
    return 0


class Report:
    """Collects check results and prints them as they come in."""

    def __init__(self) -> None:
        self.failed = False

    def ok(self, name: str, detail: str = "") -> None:
        print(f"  ok    {name:<16} {detail}")

    def warn(self, name: str, detail: str) -> None:
        print(f"  warn  {name:<16} {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failed = True
        print(f"  FAIL  {name:<16} {detail}")


def check_local(cfg: Config, report: Report) -> None:
    for tool in ("ssh", "rsync"):
        if shutil.which(tool):
            report.ok(tool, shutil.which(tool) or "")
        else:
            report.fail(tool, "not installed on this machine")
    key = cfg.ssh["key"]
    if not key:
        report.warn("ssh key", "no [ssh] key set; ssh falls back to its own defaults")
        return
    path = Path(key).expanduser()
    if not path.is_file():
        report.fail("ssh key", f"{path} does not exist")
    elif path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        report.fail("ssh key", f"{path} is readable by others; ssh will refuse it (chmod 600)")
    else:
        report.ok("ssh key", str(path))


def check_connection(cfg: Config, report: Report) -> bool:
    host = cfg.ssh["host"]
    try:
        socket.getaddrinfo(host, cfg.ssh["port"])
    except OSError:
        report.fail("dns", f"{host} does not resolve. Off the UWA network? Connect the VPN.")
        return False
    report.ok("dns", host)
    start = time.monotonic()
    try:
        who = query(cfg, "whoami", timeout=float(cfg.ssh["connect_timeout"]) + 20).strip()
    except RemoteError as exc:
        report.fail("ssh", f"{exc}. A timeout usually means the VPN is down; a permission error means the key or user is wrong.")
        return False
    report.ok("ssh", f"logged in as {who} in {time.monotonic() - start:.1f}s")
    return True


def remote_test(cfg: Config, report: Report, name: str, command: str, good: str, bad: str) -> None:
    try:
        query(cfg, command)
        report.ok(name, good)
    except RemoteError:
        report.fail(name, bad)


def check_cluster(cfg: Config, report: Report) -> None:
    if not cfg.home:
        report.fail("home", "[cluster] home is not set")
    else:
        remote_test(cfg, report, "home", f"test -d {quote(cfg.home)} && test -w {quote(cfg.home)}",
                    cfg.home, f"{cfg.home} is missing or not writable")
    remote_test(cfg, report, "slurm", "command -v sbatch && command -v squeue",
                "sbatch and squeue found", "sbatch/squeue not on PATH")
    for module in cfg.modules:
        name = module["name"]
        try:
            query(cfg, f"bash --login -c {quote('module load ' + quote(name))}")
            report.ok("module", name)
        except RemoteError as exc:
            (report.warn if module["optional"] else report.fail)("module", f"{name}: {exc.args[0].splitlines()[-1]}")


def check_project(cfg: Config, report: Report) -> None:
    project = cfg.project
    if project is None:
        report.warn("project", "no project for this folder (fine for status/jobs/shell)")
        return
    report.ok("project", f"{project.name}: {project.local} -> {project.remote}")
    remote_test(cfg, report, "mirror", f"test -d {quote(project.remote)}",
                "remote folder exists", f"{project.remote} not there yet (`kaya push` creates it)")
    if project.env:
        env = project.env if project.env.startswith("/") else project.remote_path(project.env)
        marker = "conda-meta" if project.env_kind == "conda" else "bin/activate"
        remote_test(cfg, report, "env", f"test -e {quote(env + '/' + marker)}",
                    env, f"no {project.env_kind} env at {env}")


def handle_doctor(cfg: Config, args: argparse.Namespace) -> int:
    print(f"kaya doctor  (config: {cfg.path})")
    report = Report()
    check_local(cfg, report)
    if check_connection(cfg, report):
        check_cluster(cfg, report)
        check_project(cfg, report)
    print("\nAll good." if not report.failed else "\nSomething needs fixing (see FAIL above).")
    return 1 if report.failed else 0
