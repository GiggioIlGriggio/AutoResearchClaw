import sys

import pytest

from researchclaw.config import DockerSandboxConfig, SandboxConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox
from researchclaw.experiment.sandbox import ExperimentSandbox


def test_docker_cmd_mounts_dataset_and_sets_env(tmp_path):
    sb = DockerSandbox(DockerSandboxConfig(), tmp_path, dataset_dir="/data/abide")
    cmd = sb._build_run_command(
        tmp_path, entry_point="main.py", container_name="t"
    )
    joined = " ".join(cmd)
    assert "/data/abide:/workspace/data:ro" in joined
    assert "RC_DATASET_DIR=/workspace/data" in joined


def test_docker_cmd_no_dataset_env_when_unset(tmp_path):
    sb = DockerSandbox(DockerSandboxConfig(), tmp_path)  # no dataset_dir
    cmd = sb._build_run_command(
        tmp_path, entry_point="main.py", container_name="t"
    )
    assert "RC_DATASET_DIR=/workspace/data" not in " ".join(cmd)


def test_subprocess_run_project_sets_rc_dataset_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "main.py").write_text(
        "import os\nprint('DSDIR=' + os.environ.get('RC_DATASET_DIR', ''))\n",
        encoding="utf-8",
    )
    sb = ExperimentSandbox(
        SandboxConfig(python_path=sys.executable),
        tmp_path / "wd",
        dataset_dir="/data/abide",
    )
    res = sb.run_project(proj, entry_point="main.py", timeout_sec=60)
    assert "DSDIR=/data/abide" in res.stdout
