"""Work out how to run a local script remotely: read its `# kaya:` header hints, map its
path into the project mirror, and build the python command for it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config, Project

HEADER_PREFIX = "# kaya:"
HEADER_LINES = 40
TRUE = {"1", "true", "yes", "y", "on"}
FALSE = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class RunSettings:
    """Where and how one script runs, after headers and CLI flags are applied."""

    target: str  # "login" or "gpu"
    activate_env: bool
    offline: bool
    job_name: str | None = None


def parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE:
        return True
    if normalized in FALSE:
        return False
    raise ValueError(f"invalid boolean for {field}: {value!r}")


def parse_header(path: Path) -> dict[str, str]:
    """Read `# kaya: key=value` lines from the top of a Python file."""

    hints: dict[str, str] = {}
    if path.suffix != ".py" or not path.is_file():
        return hints
    for index, line in enumerate(path.read_text(errors="replace").splitlines()):
        if index >= HEADER_LINES:
            break
        stripped = line.strip()
        if not stripped.startswith(HEADER_PREFIX):
            continue
        key, sep, value = stripped[len(HEADER_PREFIX):].partition("=")
        if sep:
            hints[key.strip().lower().replace("_", "-")] = value.strip()
    return hints


def resolve_settings(
    cfg: Config,
    path: Path | None,
    *,
    target: str = "auto",
    activate: bool | None = None,
    offline: bool | None = None,
    default_target: str = "login",
) -> RunSettings:
    """Combine config defaults, the script header and CLI flags (in that order)."""

    header = parse_header(path) if path else {}
    resolved = header.get("target", default_target)
    if target != "auto":
        resolved = target
    if resolved not in {"login", "gpu"}:
        raise SystemExit(f"invalid target {resolved!r}; expected login or gpu")

    use_env = bool(cfg.behaviour["activate_env"])
    if "env" in header:
        use_env = parse_bool(header["env"], field="env")
    if activate is not None:
        use_env = activate

    is_offline = resolved == "gpu" and bool(cfg.behaviour["gpu_offline"])
    if "offline" in header:
        is_offline = parse_bool(header["offline"], field="offline")
    if offline is not None:
        is_offline = offline

    return RunSettings(target=resolved, activate_env=use_env, offline=is_offline, job_name=header.get("job-name"))


def local_file(project: Project, value: str) -> Path | None:
    """The file `value` names inside the project, or None if it is not one."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve()
        resolved.relative_to(project.local)
    except (FileNotFoundError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def remote_file(project: Project, local: Path) -> str:
    return project.remote_path(local.relative_to(project.local).as_posix())


def module_name(project: Project, local: Path) -> str | None:
    """`pkg.mod` when the file is importable from the project root, else None."""

    if local.suffix != ".py":
        return None
    parts = local.relative_to(project.local).with_suffix("").parts
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def python_command(project: Project, local: Path, args: list[str]) -> list[str]:
    """Run as `python -m` when importable, so package-relative imports work."""

    module = module_name(project, local)
    if module:
        return [project.python, "-m", module, *args]
    return [project.python, remote_file(project, local), *args]
