from researchclaw.config import (
    ScaffoldConfig,
    _parse_scaffold_config,
    _parse_experiment_config,
)


def test_scaffold_config_defaults():
    sc = ScaffoldConfig()
    assert sc.enabled is False
    assert sc.dir == ""
    assert sc.manifest == "SCAFFOLD.md"
    assert sc.dataset_dir == ""
    assert sc.require is True


def test_parse_scaffold_config_from_yaml():
    sc = _parse_scaffold_config(
        {"enabled": True, "dir": "~/gnn_scaffold",
         "dataset_dir": "/data/abide", "require": False}
    )
    assert sc.enabled is True
    assert sc.dir == "~/gnn_scaffold"
    assert sc.dataset_dir == "/data/abide"
    assert sc.require is False
    assert sc.manifest == "SCAFFOLD.md"


def test_experiment_config_includes_scaffold():
    ec = _parse_experiment_config({"scaffold": {"enabled": True, "dir": "/x"}})
    assert ec.scaffold.enabled is True
    assert ec.scaffold.dir == "/x"


def test_experiment_config_scaffold_default_disabled():
    ec = _parse_experiment_config({})
    assert ec.scaffold.enabled is False
