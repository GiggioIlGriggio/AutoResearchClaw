"""Experiment scaffold: materialize fixed base code + expose a dataset path.

See docs/specs/2026-06-10-experiment-scaffold-design.md. Every function is a
no-op unless ``config.experiment.scaffold.enabled`` is true.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from researchclaw.config import ExperimentConfig

logger = logging.getLogger(__name__)

SCAFFOLD_PACKAGE = "scaffold"
DATASET_ENV_VAR = "RC_DATASET_DIR"
DATASET_CONTAINER_PATH = "/workspace/data"


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def validate_scaffold(exp_config: ExperimentConfig) -> list[str]:
    """Return error strings; empty list means the scaffold config is usable."""
    s = exp_config.scaffold
    errors: list[str] = []
    if not s.enabled:
        return errors
    if not s.dir:
        errors.append(
            "experiment.scaffold.enabled is true but experiment.scaffold.dir is empty"
        )
    else:
        d = _expand(s.dir)
        if not d.is_dir():
            errors.append(f"experiment.scaffold.dir does not exist: {d}")
        elif not any(d.glob("*.py")):
            errors.append(f"experiment.scaffold.dir has no .py modules: {d}")
    if s.dataset_dir:
        ds = _expand(s.dataset_dir)
        if not ds.is_dir():
            errors.append(f"experiment.scaffold.dataset_dir does not exist: {ds}")
    return errors


def enforce_scaffold(exp_config: ExperimentConfig) -> None:
    """Fail fast (or warn) at pipeline start based on ``scaffold.require``."""
    s = exp_config.scaffold
    if not s.enabled:
        return
    errors = validate_scaffold(exp_config)
    if not errors:
        if exp_config.mode not in ("sandbox", "docker") and s.dataset_dir:
            logger.warning(
                "Scaffold: dataset env var %s is only wired for docker/sandbox "
                "modes; mode=%s will not receive it.",
                DATASET_ENV_VAR, exp_config.mode,
            )
        return
    msg = "Scaffold configuration invalid:\n  - " + "\n  - ".join(errors)
    if s.require:
        raise RuntimeError(msg)
    logger.warning("%s\nContinuing without scaffold (require=false).", msg)


def resolve_dataset_dir(exp_config: ExperimentConfig) -> str:
    """Absolute host path of the scaffold dataset, or '' if unset/missing."""
    s = exp_config.scaffold
    if not s.enabled or not s.dataset_dir:
        return ""
    ds = _expand(s.dataset_dir)
    return str(ds) if ds.is_dir() else ""


def build_scaffold_guidance(exp_config: ExperimentConfig) -> str:
    """Code-gen prompt block: manifest verbatim + file tree + import/dataset rules.

    Returns '' when scaffold is disabled or the dir is unusable.
    """
    s = exp_config.scaffold
    if not s.enabled or not s.dir:
        return ""
    d = _expand(s.dir)
    if not d.is_dir():
        return ""
    modules = sorted(p.name for p in d.glob("*.py") if p.name != "__init__.py")
    if not modules:
        return ""
    file_list = ", ".join(f"{SCAFFOLD_PACKAGE}/{m}" for m in modules)

    manifest_text = ""
    manifest_path = d / s.manifest
    if manifest_path.is_file():
        manifest_text = manifest_path.read_text(
            encoding="utf-8", errors="replace"
        ).strip()
    else:
        logger.warning(
            "Scaffold: manifest %s not found; injecting file tree only.",
            manifest_path,
        )

    block = (
        "\n\n## Provided experiment scaffold (USE THESE)\n"
        f"A ready-made `{SCAFFOLD_PACKAGE}/` package is already in your project. "
        "Prefer importing it over re-implementing. Never modify anything under "
        f"`{SCAFFOLD_PACKAGE}/`.\n\n"
        f"Files available to import: {file_list}\n"
    )
    if manifest_text:
        block += "\n" + manifest_text + "\n"
    block += (
        "\nRules:\n"
        f"- Use a component as-is by importing it, e.g. "
        f"`from {SCAFFOLD_PACKAGE}.models import BaseGNN`.\n"
        "- To vary, write your OWN top-level module (e.g. models.py) with the "
        f"variant and import that instead. Never edit files under "
        f"`{SCAFFOLD_PACKAGE}/`, and never name a top-level module "
        f"`{SCAFFOLD_PACKAGE}`.\n"
        f"- The dataset root is in the environment variable {DATASET_ENV_VAR}. "
        f"Read it with os.environ['{DATASET_ENV_VAR}']; never hardcode dataset paths.\n"
    )
    return block


def materialize_scaffold(exp_config: ExperimentConfig, exp_dir: Path) -> bool:
    """Copy the scaffold dir into ``exp_dir/scaffold/`` verbatim (+ __init__.py).

    Returns True if materialized; no-op (False) when disabled or unusable. The
    package lives in its own subdir, so it never overwrites generated flat files.
    """
    s = exp_config.scaffold
    if not s.enabled or not s.dir:
        return False
    src = _expand(s.dir)
    if not src.is_dir():
        logger.warning("Scaffold: dir not found, skipping: %s", src)
        return False
    # A generated top-level module named scaffold.py would collide with the
    # scaffold/ package on import. Refuse rather than silently shadow it.
    if (exp_dir / f"{SCAFFOLD_PACKAGE}.py").exists():
        logger.error(
            "Scaffold: generated %s.py collides with the scaffold/ package; "
            "skipping materialization.", SCAFFOLD_PACKAGE,
        )
        return False
    dest = exp_dir / SCAFFOLD_PACKAGE
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    init_py = dest / "__init__.py"
    if not init_py.exists():
        init_py.write_text("", encoding="utf-8")
    logger.info(
        "Scaffold: materialized %d module(s) into %s",
        len(list(dest.glob("*.py"))), dest,
    )
    return True
