import logging
from pathlib import Path

import pytest

from researchclaw.config import ExperimentConfig, ScaffoldConfig
from researchclaw.experiment import scaffold as sc


def _make_scaffold_dir(root: Path) -> Path:
    d = root / "gnn_scaffold"
    d.mkdir()
    (d / "models.py").write_text("class BaseGNN:\n    pass\n", encoding="utf-8")
    (d / "utils.py").write_text("def set_seed(s):\n    pass\n", encoding="utf-8")
    (d / "SCAFFOLD.md").write_text(
        "# Scaffold\n- `BaseGNN(...)` import from scaffold.models\n", encoding="utf-8"
    )
    return d


def _cfg(scaffold_dir="", dataset_dir="", enabled=True, require=True, mode="sandbox"):
    return ExperimentConfig(
        mode=mode,
        scaffold=ScaffoldConfig(
            enabled=enabled, dir=str(scaffold_dir),
            dataset_dir=str(dataset_dir), require=require,
        ),
    )


def test_resolve_dataset_dir_returns_abspath(tmp_path):
    ds = tmp_path / "abide"
    ds.mkdir()
    cfg = _cfg(dataset_dir=ds)
    assert sc.resolve_dataset_dir(cfg) == str(ds.resolve())


def test_resolve_dataset_dir_empty_when_disabled(tmp_path):
    ds = tmp_path / "abide"
    ds.mkdir()
    assert sc.resolve_dataset_dir(_cfg(dataset_dir=ds, enabled=False)) == ""


def test_resolve_dataset_dir_empty_when_missing(tmp_path):
    assert sc.resolve_dataset_dir(_cfg(dataset_dir=tmp_path / "nope")) == ""


def test_validate_ok(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    ds = tmp_path / "abide"
    ds.mkdir()
    assert sc.validate_scaffold(_cfg(scaffold_dir=d, dataset_dir=ds)) == []


def test_validate_reports_missing_dir(tmp_path):
    errs = sc.validate_scaffold(_cfg(scaffold_dir=tmp_path / "nope"))
    assert any("does not exist" in e for e in errs)


def test_validate_reports_missing_dataset(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    errs = sc.validate_scaffold(_cfg(scaffold_dir=d, dataset_dir=tmp_path / "nope"))
    assert any("dataset_dir does not exist" in e for e in errs)


def test_validate_noop_when_disabled(tmp_path):
    assert sc.validate_scaffold(_cfg(scaffold_dir=tmp_path / "nope", enabled=False)) == []


def test_build_guidance_contains_manifest_and_rules(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    block = sc.build_scaffold_guidance(_cfg(scaffold_dir=d))
    assert "Provided experiment scaffold" in block
    assert "scaffold/models.py" in block
    assert "import from scaffold.models" in block          # from the manifest
    assert "from scaffold.models import BaseGNN" in block   # from the rules
    assert "RC_DATASET_DIR" in block


def test_build_guidance_empty_when_disabled(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    assert sc.build_scaffold_guidance(_cfg(scaffold_dir=d, enabled=False)) == ""


def test_materialize_copies_byte_identical(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    exp = tmp_path / "experiment"
    exp.mkdir()
    assert sc.materialize_scaffold(_cfg(scaffold_dir=d), exp) is True
    assert (exp / "scaffold" / "__init__.py").exists()
    assert (exp / "scaffold" / "models.py").read_bytes() == (d / "models.py").read_bytes()
    assert (exp / "scaffold" / "utils.py").read_bytes() == (d / "utils.py").read_bytes()


def test_materialize_noop_when_disabled(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    exp = tmp_path / "experiment"
    exp.mkdir()
    assert sc.materialize_scaffold(_cfg(scaffold_dir=d, enabled=False), exp) is False
    assert not (exp / "scaffold").exists()


def test_generated_variant_coexists_with_scaffold(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    exp = tmp_path / "experiment"
    exp.mkdir()
    # generated, flat top-level variant of models.py
    (exp / "models.py").write_text("# VARIANT\n", encoding="utf-8")
    sc.materialize_scaffold(_cfg(scaffold_dir=d), exp)
    assert (exp / "models.py").read_text() == "# VARIANT\n"          # untouched
    assert (exp / "scaffold" / "models.py").read_bytes() == (d / "models.py").read_bytes()


def test_materialize_guard_on_scaffold_py_collision(tmp_path):
    d = _make_scaffold_dir(tmp_path)
    exp = tmp_path / "experiment"
    exp.mkdir()
    (exp / "scaffold.py").write_text("# collides\n", encoding="utf-8")
    assert sc.materialize_scaffold(_cfg(scaffold_dir=d), exp) is False
    assert not (exp / "scaffold").is_dir()


def test_enforce_raises_when_require_and_missing(tmp_path):
    with pytest.raises(RuntimeError):
        sc.enforce_scaffold(_cfg(scaffold_dir=tmp_path / "nope", require=True))


def test_enforce_warns_when_not_require(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        sc.enforce_scaffold(_cfg(scaffold_dir=tmp_path / "nope", require=False))
    assert any("Scaffold configuration invalid" in r.message for r in caplog.records)


def test_enforce_noop_when_disabled(tmp_path):
    sc.enforce_scaffold(_cfg(scaffold_dir=tmp_path / "nope", enabled=False))  # no raise


def test_scaffold_import_is_not_an_error():
    # Underpins the design: the repair loop only acts on error-severity issues,
    # so a scaffold import is never stripped or blocked.
    from researchclaw.experiment.validator import validate_code
    v = validate_code("from scaffold.models import BaseGNN\nx = 1\n")
    assert all(i.severity != "error" for i in v.issues)


def _minimal_rc_config(tmp_path, mode="sandbox"):
    # Mirrors tests/test_hep_incremental.py::_make_rc_config
    from researchclaw.config import RCConfig
    data = {
        "project": {"name": "rc-test", "mode": "docs-first", "profile": ""},
        "research": {"topic": "test", "domains": ["ml"]},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "local"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {"use_memory": False, "use_message": False},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "http://localhost:1234/v1",
            "api_key_env": "RC_TEST_KEY", "api_key": "inline-test-key",
            "primary_model": "fake-model", "fallback_models": [],
        },
        "security": {"hitl_required_stages": []},
        "experiment": {"mode": mode},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_runner_calls_enforce_scaffold(monkeypatch, tmp_path):
    # execute_pipeline must fail fast (before any stage) by calling
    # enforce_scaffold. Monkeypatch it to raise and assert it propagates.
    from researchclaw.pipeline import runner

    called = {}

    def fake_enforce(exp_config):
        called["mode"] = exp_config.mode
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "researchclaw.experiment.scaffold.enforce_scaffold", fake_enforce
    )
    cfg = _minimal_rc_config(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        runner.execute_pipeline(
            run_dir=tmp_path, run_id="t", config=cfg, adapters=None,
        )
    assert called.get("mode") == "sandbox"
