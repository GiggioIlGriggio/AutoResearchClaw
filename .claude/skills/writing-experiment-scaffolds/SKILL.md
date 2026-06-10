---
name: writing-experiment-scaffolds
description: Guides writing and extending an AutoResearchClaw experiment scaffold — portable base models, data loaders, and utilities the pipeline materializes into every experiment as an importable `scaffold/` package. Use when adding or editing scaffold modules, authoring a SCAFFOLD.md manifest, wiring dataset access via RC_DATASET_DIR, or when the user mentions experiment scaffolds, scaffold subpackages, reproducible base models/loaders, or extending the pipeline with their own code.
---

# Writing Experiment Scaffolds

An AutoResearchClaw **scaffold** is a fixed, portable set of base models, data loaders, and
utilities the pipeline copies into every experiment as an importable `scaffold/` package. The
code-generation LLM then either imports your code **as-is** (byte-identical across runs,
reproducible) or writes its own top-level variant — but it never edits your originals.

Your job when authoring: keep modules clean and importable, describe them in `SCAFFOLD.md`,
and follow the four rules below. Then run the pre-flight check.

## Layout

Put your modules and manifest in one directory and point `experiment.scaffold.dir` at it:

```
~/gnn_scaffold/
  models.py
  data_loader.py
  utils.py
  SCAFFOLD.md      # describes the API + dataset; injected verbatim into the code-gen prompt
```

The pipeline copies this whole dir into each experiment as `scaffold/` (adding `__init__.py`)
and exposes your dataset at `RC_DATASET_DIR`. Design spec:
`docs/specs/2026-06-10-experiment-scaffold-design.md`.

## The four rules

1. **Package-qualified imports.** Inside scaffold modules, import siblings as
   `from scaffold.utils import normalize_adj` or relative `from .utils import normalize_adj`.
   Never bare `import utils` — it won't resolve under the package and can collide with an
   LLM-written top-level variant of the same name.
2. **Dataset via env var.** Read the dataset root from `os.environ['RC_DATASET_DIR']`; never
   hardcode a host path. Identical in docker (`/workspace/data`) and local-subprocess runs.
3. **Author `SCAFFOLD.md`.** It is injected verbatim into the code-gen prompt — it is how the
   LLM learns what exists. Describe every public class/function (with signatures), the dataset
   (files under `RC_DATASET_DIR`, shapes), and when to use-as-is vs vary. Manifest quality
   directly drives how well the LLM uses your scaffold.
4. **Keep originals reusable, not edited.** The LLM imports or subclasses your code; it never
   modifies `scaffold/`. Parameterize base modules so they're easy to use as-is or extend.

## Workflow: add or edit a scaffold module

1. Write/edit the module in your scaffold dir. Use package-qualified imports (rule 1) and read
   data via `RC_DATASET_DIR` (rule 2).
2. Add/update its entry in `SCAFFOLD.md` (signatures + one-line purpose + any dataset notes).
3. Run the pre-flight check and fix every `FAIL` (review each `WARN`):
   ```
   python .claude/skills/writing-experiment-scaffolds/scripts/preflight.py ~/gnn_scaffold
   ```
4. Optional local smoke test: `RC_DATASET_DIR=/path/to/data python -c "import scaffold.models"`
   (run from a dir where the package is importable).

## Pre-flight checklist

- [ ] Intra-scaffold imports are `from scaffold.x` / `from .x` (no bare sibling imports).
- [ ] No hardcoded dataset paths; loaders read `os.environ['RC_DATASET_DIR']`.
- [ ] Every public class/function you want used appears in `SCAFFOLD.md` with its signature.
- [ ] No host-absolute paths or machine-specific assumptions.
- [ ] Modules import cleanly — no dataset access at import time, only inside functions.

## Examples

Full worked example — a GNN baseline, a brain-graph loader, utils, and a complete
`SCAFFOLD.md` — in [EXAMPLES.md](EXAMPLES.md).
