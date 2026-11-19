"""Shared fixtures: a config file in a temp folder with one project."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from kaya_tools.config import load_config

BASE = """
[ssh]
host = "kaya.example"
user = "me"
key = "~/.ssh/id_test"

[cluster]
home = "/group/g1/me"
modules = ["Anaconda3/2024.06", {{ name = "gcc/9.4.0", optional = true }}]

[pull_filters]
text = ["*.jsonl", "*.json"]

[env_vars]
PYTHONPATH = "{{root}}"
HF_HOME = "{{root}}/.cache"

[offline_env_vars]
HF_HUB_OFFLINE = "1"

[projects.demo]
local = "{local}"
remote = "demo"
env = "envs/core"
pull = ["results", "logs"]
"""


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "train.py").write_text("# kaya: target=gpu\n# kaya: job-name=trainer\nprint('hi')\n")
    (root / "loose script.py").write_text("print('x')\n")
    (root / "job.sbatch").write_text("#!/bin/bash\n#SBATCH --time=00:05:00\necho hi\n")
    return root


@pytest.fixture
def write_config(tmp_path: Path, project_dir: Path):
    def write(extra: str = "") -> Path:
        path = tmp_path / "config.toml"
        path.write_text(textwrap.dedent(BASE.format(local=project_dir)) + textwrap.dedent(extra))
        return path

    return write


@pytest.fixture
def cfg(write_config, project_dir, monkeypatch):
    monkeypatch.chdir(project_dir)
    return load_config(write_config())
