"""Build the shell prelude every remote run and job starts with: cd into the project,
load modules, export the configured variables and secrets, and activate the env."""

from __future__ import annotations

from pathlib import Path

from .config import TOOL_ROOT, Config
from .ssh import quote


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines (an optional leading `export` is fine)."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        if key.strip():
            values[key.strip()] = value.strip().strip("'\"")
    return values


def expand(value: str, cfg: Config) -> str:
    """Fill `{home}`, `{root}` and `{project}` in a configured value."""

    project = cfg.project
    value = value.replace("{home}", cfg.home)
    value = value.replace("{root}", project.remote if project else cfg.home)
    return value.replace("{project}", project.name if project else "")


def module_lines(cfg: Config) -> list[str]:
    lines = []
    for module in cfg.modules:
        name = quote(module["name"])
        lines.append(f"module load {name} 2>/dev/null || true" if module["optional"] else f"module load {name}")
    if lines:
        lines.append("module list 2>&1 | sed 's/^/[modules] /'")
    return lines


def env_var_lines(cfg: Config, *, offline: bool) -> list[str]:
    values = {k: expand(str(v), cfg) for k, v in cfg.data["env_vars"].items()}
    if offline:
        values.update({k: expand(str(v), cfg) for k, v in cfg.data["offline_env_vars"].items()})
    return [f"export {name}={quote(value)}" for name, value in values.items()]


def secret_lines(cfg: Config) -> list[str]:
    secrets = cfg.data["secrets"]
    env_file = Path(secrets["env_file"]).expanduser()
    if not env_file.is_absolute():
        env_file = TOOL_ROOT / env_file
    dotenv = load_dotenv(env_file)
    return [f"export {name}={quote(dotenv[name])}" for name in secrets["forward"] if dotenv.get(name)]


def activate_lines(cfg: Config) -> list[str]:
    project = cfg.project
    if project is None or not project.env:
        return []
    env = project.env if project.env.startswith("/") else project.remote_path(project.env)
    if project.env_kind == "venv":
        return [f"source {quote(env)}/bin/activate"]
    if project.env_kind == "conda":
        return [
            'source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true',
            "conda deactivate 2>/dev/null || true",
            f"conda activate {quote(env)}",
        ]
    raise SystemExit(f"env_kind must be conda or venv, not {project.env_kind!r}")


def prelude(cfg: Config, *, activate: bool, offline: bool, secrets: bool = False) -> str:
    lines = [
        "set -euo pipefail",
        f"mkdir -p {quote(cfg.workdir)} && cd {quote(cfg.workdir)}" if cfg.workdir else "cd",
        *module_lines(cfg),
        *env_var_lines(cfg, offline=offline),
        *(secret_lines(cfg) if secrets and not offline else []),
    ]
    if activate:
        lines += activate_lines(cfg)
    return "\n".join(lines)
