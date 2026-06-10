# Experiment Scaffold — Design Spec

> Status: approved (design). Date: 2026-06-10. Author: Alessio Comparini + Claude.
> Scope: a single implementation plan. Terminal step: `writing-plans`.

## 1. Problem

Today every experiment the pipeline runs is generated from scratch by the LLM. For
brain-graph GNN work that means model architectures and data-loading logic are
re-invented on each run — not reproducible, and there's no guarantee the LLM uses the
intended baselines or reads the dataset the intended way.

## 2. Goal

Let the user hand the pipeline a fixed, portable **scaffold** — base models, brain-graph
data loaders, shared utilities — plus a pointer to an on-disk dataset. On every run the
pipeline then:

1. Places the scaffold code into the experiment working directory, so generated code can
   `import` it.
2. Exposes the on-disk dataset at a stable location, identically across execution modes.
3. Tells the code-generation LLM, up front, that these exist and must be used rather than
   rebuilt.

The LLM keeps two choices per run: **use-as-is** (import the scaffold directly → that code
is byte-identical across runs, reproducible) or **edit-into-a-variant** (write its own
module that takes precedence, leaving the originals untouched as the reference).

## 3. Locked decisions

| Decision | Choice |
|---|---|
| Execution modes | Mode-agnostic: works for `docker` and local `sandbox` (subprocess) |
| Layout | Dedicated `scaffold/` subpackage; originals never edited; variants are top-level modules |
| LLM awareness | **Manifest-only**: user authors `SCAFFOLD.md`, injected verbatim; code is not parsed |
| Dataset path | Env var `RC_DATASET_DIR` (docker → `/workspace/data`; subprocess → host abspath) |
| Materialization | Pipeline deterministically copies the **whole** `scaffold/`; the LLM decides only what to *import* |
| Approach | **A** — Stage-10 materialization + code-gen prompt injection + sandbox env wiring |

## 4. Configuration surface

New frozen dataclass `ScaffoldConfig` under `ExperimentConfig`, parsed in
`_parse_experiment_config` (`researchclaw/config.py`). Disabled by default, so existing
runs are unaffected.

```yaml
experiment:
  scaffold:
    enabled: true
    dir: "~/gnn_scaffold"        # host dir: your *.py + SCAFFOLD.md → becomes scaffold/
    manifest: "SCAFFOLD.md"      # filename within dir, injected verbatim into the code-gen prompt
    dataset_dir: "/data/abide"   # host dataset dir → exposed as RC_DATASET_DIR
    require: true                # enabled + dir/dataset missing → fail fast (vs warn + disable)
```

```python
@dataclass(frozen=True)
class ScaffoldConfig:
    enabled: bool = False
    dir: str = ""
    manifest: str = "SCAFFOLD.md"
    dataset_dir: str = ""
    require: bool = True
```

The subpackage name is fixed as `scaffold` (not configurable — YAGNI).

## 5. Components & files touched

- **`researchclaw/config.py`** — add `ScaffoldConfig`; add `scaffold:
  ScaffoldConfig = field(default_factory=ScaffoldConfig)` to `ExperimentConfig`; parse the
  `experiment.scaffold` YAML block.
- **`researchclaw/pipeline/stage_impls/_code_generation.py`** — three additions:
  1. **Prompt injection.** A `_build_scaffold_guidance(config)` helper reads the manifest
     verbatim and the scaffold file tree, builds the block in §8, and appends it to the
     `extra_guidance` that feeds `PromptManager.for_stage("code_generation", …)`. Sibling
     of the existing BenchmarkAgent injection block. The same block is also passed into the
     multi-phase code-agent path (architecture planning + per-file generation) when that
     path is active, so behavior is consistent across generation strategies.
  2. **Validation awareness.** Register the top-level module `scaffold` (and its submodules)
     as known-importable so the cross-import validator / repair loop (BUG-184) does **not**
     treat `from scaffold.models import …` as an unresolved import and "fix" it away.
  3. **Materialization.** After the generated files are written to
     `<stage-10>/experiment/`, copy the whole scaffold `dir` → `<stage-10>/experiment/
     scaffold/` verbatim (`shutil.copytree`), and ensure `scaffold/__init__.py` exists.
- **`researchclaw/experiment/docker_sandbox.py`** — in `_build_run_command`, when
  `dataset_dir` is set: mount `-v <dataset_dir>:/workspace/data:ro` (taking priority over
  the existing `/opt/datasets` / `~/.cache/datasets` defaults) and add
  `-e RC_DATASET_DIR=/workspace/data`.
- **`researchclaw/experiment/sandbox.py`** (subprocess `ExperimentSandbox`) — set
  `RC_DATASET_DIR=<abspath(dataset_dir)>` in the subprocess environment.
- **Run-start validation** (runner / config load) — enforce §7 error handling.

## 6. Data flow & ordering

Awareness (prompt) precedes generation; the deterministic copy lands before validation and
execution. The LLM is told about the scaffold *before* it writes a line, so its generated
code already imports the scaffold — the file copy is pure plumbing that only has to be on
disk before validation and execution need it.

```
Stage 10 — code generation
  1. Build prompt  → inject SCAFFOLD.md verbatim + import/dataset rules
  2. LLM generates → writes  from scaffold.models import BaseGNN
                            root = os.environ['RC_DATASET_DIR']
  3. Validate/repair → scaffold.* is known-importable, so imports survive (not "repaired")
  4. Write to disk → generated files  +  copy scaffold/ verbatim  +  scaffold/__init__.py

Stage 12 — execution
  5. run_project carries scaffold/ into the sandbox; RC_DATASET_DIR set; code runs
```

Scaffold **files** ride into *any* sandbox mode for free because they live inside the
experiment dir that `run_project` already copies. The **dataset env var** is wired for
`docker` + `sandbox` in v1; other modes (`ssh_remote`, `colab_drive`, `agentic`, …) emit a
**loud warning** rather than a silent skip — consistent with this repo's anti-silent-fallback
stance.

## 7. Reproducibility guarantee

The multi-file extractor forces generated filenames to be **flat** (it strips subdirs and
path traversal). Therefore the LLM *cannot* emit a file inside `scaffold/` — the originals
are structurally protected and byte-identical on every run. A variant is simply a top-level
`models.py` the LLM writes; `from scaffold.models import …` (original) and `from models
import …` (variant) coexist with no collision. An implementation guard additionally asserts
that no generated file path resolves inside `scaffold/`.

## 8. Injected prompt block (manifest-only)

```
## Provided experiment scaffold (USE THESE)
A ready-made `scaffold/` package is already in your project. Prefer importing it over
re-implementing. Never modify anything under scaffold/.

Files available to import:
  scaffold/models.py, scaffold/data_loader.py, scaffold/utils.py

<<< verbatim contents of SCAFFOLD.md >>>

Rules:
- Use a component as-is by importing it, e.g. `from scaffold.models import BaseGNN`.
- To vary, write your OWN top-level module (e.g. models.py) with the variant and import
  that instead. Never edit files under scaffold/.
- The dataset root is in the environment variable RC_DATASET_DIR. Read it with
  os.environ['RC_DATASET_DIR']; never hardcode dataset paths.
```

If the manifest file is missing, the block is still injected with the file tree only, and a
warning is logged.

## 9. Scaffold authoring contract

This is the contract the user follows when writing a scaffold, and the **single source of
truth** the authoring skill (§11) encodes.

- **Directory.** `dir` is a host directory whose contents become the `scaffold/` package.
  Put your `*.py` modules and `SCAFFOLD.md` here.
- **Package-qualified imports (the one gotcha).** Because the files become a package under
  `scaffold/`, intra-scaffold imports must be package-qualified —
  `from scaffold.utils import x` or relative `from .utils import x` — **not** a bare
  `import utils`. (A bare top-level import would also risk colliding with an LLM-written
  top-level variant of the same name.)
- **Dataset access.** Loaders read the dataset root from `os.environ['RC_DATASET_DIR']`;
  never hardcode a path.
- **Manifest (`SCAFFOLD.md`).** Prose describing each model, each loader (signature +
  returns), the dataset (location relative to `RC_DATASET_DIR`, shape, files), and guidance
  on when to use-as-is vs vary. Injected verbatim, so its quality directly drives how well
  the LLM uses the scaffold.

## 10. Error handling

- `enabled` + missing `dir` or `dataset_dir` → **fail fast** when `require: true` (default);
  otherwise warn and disable the scaffold for the run.
- Missing manifest → warn; inject the file tree only.
- Execution mode without dataset-env support → warn (scaffold files still work; dataset env
  var not set).
- Guard: assert no generated file path resolves inside `scaffold/`.

## 11. Deliverable — scaffold-authoring skill

A Claude Code skill (authored via `/write-a-skill`) that guides a researcher implementing
new scaffold modules to extend the pipeline's capabilities. It encodes the §9 contract:

- How to lay out the scaffold dir and what becomes the `scaffold/` package.
- The package-qualified-import rule (with correct/incorrect examples).
- Reading the dataset via `RC_DATASET_DIR`.
- Writing a high-quality `SCAFFOLD.md` manifest (template + example for a GNN baseline +
  brain-graph loader).
- The use-as-is vs variant model, and the byte-identical reproducibility guarantee.
- A pre-flight checklist (imports package-qualified, manifest entries match public API,
  dataset access via env var, no host-absolute paths).

The skill and the pipeline must stay consistent: §9 is the source of truth for both.

## 12. Testing

- Config parse: `experiment.scaffold` YAML → `ScaffoldConfig`.
- Materialization: scaffold dir → `experiment/scaffold/` is byte-identical (hash compare);
  `__init__.py` present.
- Precedence: a generated top-level `models.py` coexists; `scaffold/models.py` unchanged.
- Prompt injection: `extra_guidance` contains the manifest text, the import/dataset rules,
  and the scaffold file list.
- Validation awareness: a generated file importing `from scaffold.models import …` does not
  trigger a missing-import repair.
- Docker command: includes `-v <dataset_dir>:/workspace/data:ro` and
  `-e RC_DATASET_DIR=/workspace/data`.
- Subprocess env: `RC_DATASET_DIR` set to the dataset abspath.
- Disabled-by-default: with `scaffold.enabled = false`, no behavior change.

## 13. Out of scope (v1)

- Dataset env wiring for `ssh_remote` / `colab_drive` / `agentic` / domain-specific sandboxes
  (warn instead).
- Auto-introspection of scaffold APIs (manifest-only by decision).
- LLM-declared selective copy (whole-`scaffold/` copy by decision).
- Configurable subpackage name.
- Scaffold content hashing / provenance record (possible future nicety).
