"""Config loading: project detection, per-project overrides, and the errors users see."""

from __future__ import annotations

import pytest

from kaya_tools.config import ConfigError, load_config


def test_project_is_found_from_a_subfolder(write_config, project_dir):
    cfg = load_config(write_config(), cwd=project_dir / "pkg")
    assert cfg.project is not None and cfg.project.name == "demo"
    assert cfg.project.remote == "/group/g1/me/demo"
    assert cfg.project.remote_logs == "/group/g1/me/demo/logs"
    assert cfg.workdir == "/group/g1/me/demo"


def test_outside_any_project_there_is_none(write_config, tmp_path):
    cfg = load_config(write_config(), cwd=tmp_path)
    assert cfg.project is None
    assert cfg.workdir == "/group/g1/me"
    with pytest.raises(ConfigError, match="no project"):
        cfg.require_project()


def test_deepest_project_wins(write_config, project_dir):
    path = write_config(f"""
        [projects.inner]
        local = "{project_dir / 'pkg'}"
        remote = "/abs/inner"
    """)
    cfg = load_config(path, cwd=project_dir / "pkg")
    assert cfg.project.name == "inner"
    assert cfg.project.remote == "/abs/inner"


def test_project_overrides_merge_into_cluster_tables(write_config, project_dir):
    path = write_config("""
        [projects.demo.slurm]
        mem = "64G"
        [projects.demo.behaviour]
        wait = false
    """)
    cfg = load_config(path, cwd=project_dir)
    assert cfg.slurm["mem"] == "64G"
    assert cfg.slurm["partition"] == "gpu"  # untouched default survives
    assert cfg.behaviour["wait"] is False
    assert cfg.behaviour["push_before_run"] is True


def test_modules_accept_strings_and_tables(cfg):
    assert cfg.modules == [
        {"name": "Anaconda3/2024.06", "optional": False},
        {"name": "gcc/9.4.0", "optional": True},
    ]


def test_explicit_project_flag(write_config, tmp_path):
    cfg = load_config(write_config(), project="demo", cwd=tmp_path)
    assert cfg.project.name == "demo"


@pytest.mark.parametrize("extra, message", [
    ("[typo]\nx = 1\n", "unknown top-level"),
    ("[behaviour]\npull_filter = 'nope'\n", "pull_filter"),
])
def test_bad_config_says_what_is_wrong(write_config, tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write_config(extra), cwd=tmp_path)


def test_unknown_project_key_is_rejected(write_config, project_dir):
    path = write_config("[projects.other]\nlocal = '/x'\nremot = 'y'\n")
    with pytest.raises(ConfigError, match="remot"):
        load_config(path, project="other", cwd=project_dir)


def test_missing_config_points_at_init(tmp_path):
    with pytest.raises(ConfigError, match="kaya init"):
        load_config(tmp_path / "nope.toml")
