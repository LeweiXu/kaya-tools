"""SSH transport to Kaya: build the ssh argv from config, run remote commands and
scripts, and share one connection across calls so a slow login node is paid for once."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from .config import STATE_DIR, Config

#: Read-only scheduler queries come back in well under a second on a healthy
#: connection, so this only ever fires on a stalled login node.
QUERY_TIMEOUT = 30


class RemoteError(Exception):
    """A remote command failed; the message is ready to show."""


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def shell_join(command: list[str]) -> str:
    return " ".join(quote(part) for part in command)


def ssh_options(cfg: Config) -> list[str]:
    """Every ssh flag the config implies, without the destination."""

    s = cfg.ssh
    opts = ["-p", str(s["port"])]
    if s["key"]:
        opts += ["-i", str(Path(s["key"]).expanduser()), "-o", "IdentitiesOnly=yes"]
    opts += [
        "-o", f"ConnectTimeout={s['connect_timeout']}",
        "-o", f"ServerAliveInterval={s['keepalive']}",
        "-o", "ServerAliveCountMax=3",
    ]
    if s["multiplex"]:
        sockets = STATE_DIR / "ssh"
        sockets.mkdir(parents=True, exist_ok=True, mode=0o700)
        opts += [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={sockets}/%C",
            "-o", f"ControlPersist={s['persist']}",
        ]
    for option in s["options"]:
        opts += ["-o", str(option)]
    return opts


def destination(cfg: Config) -> str:
    user = cfg.ssh["user"]
    return f"{user}@{cfg.ssh['host']}" if user else str(cfg.ssh["host"])


def ssh_argv(cfg: Config, *, tty: bool = False, batch: bool = False) -> list[str]:
    argv = ["ssh", *ssh_options(cfg)]
    if batch:
        argv += ["-o", "BatchMode=yes"]
    if tty:
        argv.append("-tt")
    return [*argv, destination(cfg)]


def rsync_shell(cfg: Config) -> str:
    """The `rsync -e` value, so rsync reuses the same key, port and connection."""

    return shell_join(["ssh", *ssh_options(cfg)])


def run_local(
    command: list[str],
    *,
    input_text: str | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, input=input_text, text=True, capture_output=capture, check=check, timeout=timeout)


def query(cfg: Config, command: str, *, timeout: float = QUERY_TIMEOUT) -> str:
    """Run a short read-only command and return its stdout, or raise RemoteError."""

    try:
        result = run_local([*ssh_argv(cfg, batch=True), command], capture=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RemoteError(
            f"{cfg.ssh['host']} did not answer `{command}` within {timeout:.0f}s. The login node is slow "
            "or unreachable; try again, or run `kaya doctor`."
        ) from None
    if result.returncode != 0:
        raise RemoteError(result.stderr.strip() or result.stdout.strip() or f"ssh exited {result.returncode}")
    return result.stdout


def run_command(cfg: Config, command: str, *, check: bool = True, tty: bool = False) -> int:
    """Run a remote command with its output going straight to the terminal."""

    return run_local([*ssh_argv(cfg, tty=tty), command], check=check).returncode


def guard_process_group(script: str) -> str:
    """Kill the whole remote process group if the shell is hung up or interrupted.

    A bare `bash -s` child is not a session leader, so a dropped connection would
    otherwise leave its grandchildren (a download, a training loop) running on the
    login node. EXIT is deliberately not trapped: that would SIGTERM the shell on a
    clean exit and turn success into exit 255.
    """

    return "trap 'kill -- -$$ 2>/dev/null || true' HUP TERM INT\n" + script


def write_remote_file(cfg: Config, remote_path: str, content: str) -> None:
    parent = remote_path.rsplit("/", 1)[0] or "/"
    run_local([*ssh_argv(cfg), f"mkdir -p {quote(parent)} && cat > {quote(remote_path)}"], input_text=content)


def remote_tmp_path(cfg: Config, stem: str, suffix: str) -> str:
    return f"{cfg.tmp_dir}/{stem}_{int(time.time())}_{time.monotonic_ns() % 100000}{suffix}"


def run_script(cfg: Config, script: str, *, capture: bool = False, interactive: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a bash login-shell script on Kaya.

    The default pipes the script over stdin to a non-pty shell, which keeps stdout
    clean for parsing (a job id, say). `interactive=True` is for long foreground work:
    the script is staged as a remote file and run under a real pty, because OpenSSH
    only hangs up a pty session's process group, so Ctrl-C or a dead link then tears
    down the remote side too. Running from a file stops the pty echoing the script.
    """

    wrapped = guard_process_group(script)
    if not interactive:
        return run_local([*ssh_argv(cfg), "bash", "--login", "-s"], input_text=wrapped, capture=capture)
    path = remote_tmp_path(cfg, "run", ".sh")
    write_remote_file(cfg, path, wrapped)
    command = f"bash --login {quote(path)}; rc=$?; rm -f {quote(path)}; exit $rc"
    return run_local([*ssh_argv(cfg, tty=True), command], capture=capture)
