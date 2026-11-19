"""Load `config.toml` from the kaya-tools folder and work out which project the current
directory belongs to. Project sections can override the cluster-wide tables."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The kaya-tools checkout: config, secrets and local state all live under it.
TOOL_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = TOOL_ROOT / ".state"
EXAMPLE_CONFIG = TOOL_ROOT / "config.example.toml"


def default_config_path() -> Path:
    override = os.environ.get("KAYA_CONFIG")
    return Path(override).expanduser() if override else TOOL_ROOT / "config.toml"


DEFAULTS: dict[str, Any] = {
    "ssh": {
        "host": "kaya.hpc.uwa.edu.au",
        "user": "",
        "key": "",
        "port": 22,
        "connect_timeout": 10,
        "keepalive": 15,
        "multiplex": True,
        "persist": "60s",
        "options": [],
    },
    "cluster": {
        "home": "",
        "tmp_dir": ".kaya-tools",
        "modules": [],
    },
    "slurm": {
        "partition": "gpu",
        "gres": "gpu:1",
        "account": "",
        "qos": "",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 4,
        "mem": "16G",
        "time": "01:00:00",
        "poll_seconds": 15,
    },
    "behaviour": {
        "push_before_run": True,
        "push_delete": True,
        "wait": True,
        "pull_after_job": True,
        "pull_filter": "",
        "tail_lines": 120,
        "activate_env": True,
        "gpu_offline": True,
        "forward_secrets": True,
        "confirm_cancel_all": True,
    },
    "pull_filters": {},
    "env_vars": {},
    "offline_env_vars": {},
    "secrets": {"env_file": ".env", "forward": []},
    "projects": {},
}

PROJECT_DEFAULTS: dict[str, Any] = {
    "local": "",
    "remote": "",
    "env": "",
    "env_kind": "conda",
    "python": "python",
    "logs": "logs",
    "pull": ["logs"],
    "excludes": [".git/", ".env", "__pycache__/", "*.pyc", ".venv/", "logs/"],
    "pre_submit": "",
}

#: Tables a project section may override; anything else in a project is a project field.
OVERRIDABLE = ("slurm", "behaviour", "pull_filters", "env_vars", "offline_env_vars", "secrets")


class ConfigError(Exception):
    """A config problem worth showing to the user as-is."""


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass(frozen=True)
class Project:
    """One local folder mirrored to a folder on Kaya."""

    name: str
    local: Path
    remote: str
    env: str
    env_kind: str
    python: str
    logs: str
    pull: list[str]
    excludes: list[str]
    pre_submit: str

    def remote_path(self, rel: str = "") -> str:
        rel = rel.strip("/")
        return f"{self.remote}/{rel}" if rel else self.remote

    @property
    def remote_logs(self) -> str:
        return self.remote_path(self.logs)

    @property
    def local_logs(self) -> Path:
        return self.local / self.logs


@dataclass(frozen=True)
class Config:
    """The effective config: cluster defaults with the active project's overrides applied."""

    data: dict[str, Any]
    path: Path
    project: Project | None

    @property
    def ssh(self) -> dict[str, Any]:
        return self.data["ssh"]

    @property
    def home(self) -> str:
        return str(self.data["cluster"]["home"]).rstrip("/")

    @property
    def tmp_dir(self) -> str:
        return remote_abs(self.home, self.data["cluster"]["tmp_dir"])

    @property
    def modules(self) -> list[dict[str, Any]]:
        return list(self.data["cluster"]["modules"])

    @property
    def slurm(self) -> dict[str, Any]:
        return self.data["slurm"]

    @property
    def behaviour(self) -> dict[str, Any]:
        return self.data["behaviour"]

    @property
    def pull_filters(self) -> dict[str, list[str]]:
        return self.data["pull_filters"]

    @property
    def workdir(self) -> str:
        """Where remote commands start: the project mirror, else the cluster home."""

        if self.project:
            return self.project.remote
        return self.home

    def require_project(self) -> Project:
        if self.project is None:
            names = ", ".join(sorted(self.data["projects"])) or "none configured"
            raise ConfigError(
                f"no project for {Path.cwd()} (projects: {names}). cd into a project folder, "
                "pass --project NAME, or add one with `kaya init --project NAME`."
            )
        return self.project


def remote_abs(home: str, path: str) -> str:
    path = str(path).rstrip("/")
    if path.startswith("/"):
        return path
    if not home:
        raise ConfigError(f"remote path {path!r} is relative but [cluster] home is not set")
    return f"{home}/{path}" if path else home


def normalize_modules(raw: list[Any]) -> list[dict[str, Any]]:
    """Accept plain strings or {name, optional} tables for module entries."""

    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"name": item, "optional": False})
        elif isinstance(item, dict) and "name" in item:
            out.append({"name": str(item["name"]), "optional": bool(item.get("optional", False))})
        else:
            raise ConfigError(f"bad module entry {item!r}; use a name or {{ name = ..., optional = true }}")
    return out


def build_project(name: str, raw: dict[str, Any], home: str) -> Project:
    fields = deep_merge(PROJECT_DEFAULTS, {k: v for k, v in raw.items() if k not in OVERRIDABLE and k != "modules"})
    unknown = set(fields) - set(PROJECT_DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown keys in [projects.{name}]: {', '.join(sorted(unknown))}")
    if not fields["local"]:
        raise ConfigError(f"[projects.{name}] needs `local`")
    remote = fields["remote"] or name
    return Project(
        name=name,
        local=Path(fields["local"]).expanduser().resolve(),
        remote=remote_abs(home, remote),
        env=str(fields["env"]),
        env_kind=str(fields["env_kind"]),
        python=str(fields["python"]),
        logs=str(fields["logs"]).strip("/"),
        pull=[str(p).strip("/") for p in fields["pull"]],
        excludes=list(fields["excludes"]),
        pre_submit=str(fields["pre_submit"]),
    )


def detect_project(projects: dict[str, Any], home: str, cwd: Path) -> str | None:
    """The project whose `local` folder holds `cwd` (the deepest one wins)."""

    best: tuple[int, str] | None = None
    for name, raw in projects.items():
        local = raw.get("local")
        if not local:
            continue
        root = Path(local).expanduser().resolve()
        try:
            cwd.relative_to(root)
        except ValueError:
            continue
        depth = len(root.parts)
        if best is None or depth > best[0]:
            best = (depth, name)
    return best[1] if best else None


def load_config(path: Path | None = None, *, project: str | None = None, cwd: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.is_file():
        raise ConfigError(f"no config at {path}. Run `kaya init` to create one from the example.")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None

    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown top-level tables in {path}: {', '.join(sorted(unknown))}")
    data = deep_merge(DEFAULTS, raw)
    home = str(data["cluster"]["home"]).rstrip("/")
    projects = data["projects"]

    name = project or os.environ.get("KAYA_PROJECT") or detect_project(projects, home, (cwd or Path.cwd()).resolve())
    active: Project | None = None
    if name:
        if name not in projects:
            raise ConfigError(f"no [projects.{name}] in {path}")
        proj_raw = projects[name]
        for table in OVERRIDABLE:
            if table in proj_raw:
                data[table] = deep_merge(data[table], proj_raw[table])
        if "modules" in proj_raw:
            data["cluster"]["modules"] = proj_raw["modules"]
        active = build_project(name, proj_raw, home)

    data["cluster"]["modules"] = normalize_modules(data["cluster"]["modules"])
    filt = data["behaviour"]["pull_filter"]
    if filt and filt not in data["pull_filters"]:
        raise ConfigError(f"behaviour.pull_filter = {filt!r} is not a [pull_filters] entry")
    return Config(data=data, path=path, project=active)
