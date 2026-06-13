# Scaffold: brain functional-connectivity graphs (Schaefer-400)

A ready-made toolkit for **graph-level learning on brain functional-connectivity
(FC) matrices**: build PyG graphs from baked FC bundles, train a depth-safe GCN,
and emit publication plots/tables. Import these instead of re-implementing.
The dataset is exposed at the `RC_DATASET_DIR` environment variable.

Typical experiment: pick a bundle (e.g. classification vs regression), build a
`list[Data]` with `data_loader.load_fc_graphs`, split, batch with a PyG
`DataLoader`, train `models.build_gcn` + a graph-pooling head, and (for the core
claim) also train a parameter-matched MLP on the same data — reporting **test
metric AND parameter count** for both.

## `models` — GCN backbone factory

- `build_gcn(in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.0, act="relu", norm="batch_norm", jk=None, **kwargs)`
  — returns a `torch_geometric.nn.models.GCN`; `forward(x, edge_index, edge_weight=..., batch=...)` gives node embeddings `[num_nodes, out_channels]`. Add your own pooling (e.g. `global_mean_pool`) + linear head for graph-level output.
  - **Default `norm="batch_norm"` is a verified fix** for deep-GCN training collapse; keep it on for `num_layers>=4` unless you are deliberately ablating with `norm=None`.
  - **Signed FC weights + GCN = NaN.** `GCNConv`'s symmetric normalization breaks on negative `edge_weight`; pass `edge_weight_norm="abs"` or `"abs_max"` when building graphs for a GCN, or use unweighted edges.

## `fc_graph` — FC matrix -> PyG graph

- `topk_edge_index(fc, k=10)` — undirected top-k-per-node graph; returns `(edge_index [2,E], edge_weight [E])`, signed FC weights.
- `density_mst_edge_index(fc, density=0.10)` — keep strongest `density` fraction of edges + MST backbone so the graph is connected; returns `(edge_index, edge_weight)`.
- `fc_to_data(fc, y, k=10, node_features="degree", task="classification", edge_construction="topk", density=0.10, edge_weight_norm=None, node_attr=None)` — one FC (or SC) matrix + label -> a PyG `Data`. Works for any node count (`R = fc.shape[0]`), so a 452-node SC matrix flows through unchanged.
  - `node_features`: `"degree"` (weighted-degree z-score, 1 feature/node) or `"identity"` (R one-hot features/node).
  - `node_attr`: optional external per-node features, shape `(R,)` or `(R,F)` (e.g. a parcellated GLM activation vector aligned to the FC nodes). When given it becomes `x` directly, **overriding `node_features`**; its length must equal `R`.
  - `task`: `"classification"` (y as long class index) or `"regression"` (y as float).
  - `edge_weight_norm`: `None` | `"max"` | `"log1p"` | `"abs"` | `"abs_max"`.
- `fc_arrays_to_data_list(fc, y, ..., node_attr=None)` — same options; an `(N,R,R)` stack + `(N,)` labels -> `list[Data]`. `node_attr` is an `(N,R)` / `(N,R,F)` stack sliced per sample.

## `data_loader` — baked FC bundles under `RC_DATASET_DIR`

- `dataset_root()` -> `Path` of `RC_DATASET_DIR` (raises if unset).
- `list_datasets()` -> sorted bundle stems available.
- `load_fc_bundle(name, matrix_key="fc")` -> `(matrix [N,R,R], y [N], meta)`; `meta["task"]` is the baked task, plus every other stored array (`subject_id`, the non-selected modality, `glm_*`, …). `matrix_key="sc"` returns the structural matrix as the graph (its FC stays in `meta`).
- `load_fc_graphs(name, *, k=10, node_features="degree", task=None, edge_construction="topk", density=0.10, edge_weight_norm=None, matrix_key="fc", node_feature_key=None, limit=None, indices=None)` -> `list[Data]`. `task=None` uses the bundle's baked task.
  - `matrix_key="sc"` builds the graph from the SC matrix (452 nodes; use `edge_weight_norm="abs_max"` for GCN — SC weights are large positive streamline counts).
  - `node_feature_key="glm_<contrast>"` attaches a baked per-ROI GLM array `(N,R)` as node features (overrides `node_features`).
  - `limit` (first N) / `indices` (explicit list) subset before building graphs — use for **fast smokes**, since all graphs are built eagerly (PNC's ~970 × 400-node top-k is slow).
- `load_fc_dataset(name, **kwargs)` -> a PyG `InMemoryDataset` (`FCInMemoryDataset`) wrapping the above.

`name` may be a bundle stem (e.g. `"orbit_fc_sex_cls"`), a filename, or an absolute path; the `.npz` suffix is optional.

## `viz` — plots & tables (Agg backend, Okabe-Ito palette)

- `apply_style()` — set the shared matplotlib style in-place. `OKABE_ITO` is the colorblind-safe palette list.
- `curve_with_ci(x, ys, labels, xlabel, ylabel, title, out_path, logx=True)` -> `Path` — mean±std curves; `ys` is a list of `(n_seeds, n_points)` arrays.
- `plot_fc_matrix(fc, out_path, vmax=None)` -> `Path` — 0-centered diverging heatmap of an FC matrix.
- `write_table(df, path_stem)` -> `(md_path, csv_path)` — write a DataFrame as both markdown and CSV.

## `report` — results aggregation

- `read_csv_metrics(metrics_csv)` -> `dict[str,float]` — last non-null value per metric column of a Lightning-style metrics.csv.
- `digest_row(exp_id, sweep, hypothesis, metrics, conclusion)` -> markdown table row.
- `append_digest(digest_path, row, header=None)` -> append a row (writing the header if the file is new).

## Thesis primitives (PNC age→VWM)

These symbols implement the **transfer-vs-concat thesis** (source: age prediction; target: VWM d′). Import from the respective modules — they are thin wrappers over the existing scaffold; no new dependencies.

- **`fc_graph.glm_diagonal_features(glm_vec, normalize=True)`** — builds a `(R,R)` diagonal carrier `diag(z(glm_vec))`: each node's feature is its own GLM z-value, zero elsewhere. This is the thesis `glm_diagonal` carrier. `normalize=True` z-scores `glm_vec` per subject before placing it on the diagonal.
- **loader `glm_diagonal=True` / `glm_normalize=True`** on `data_loader.load_fc_graphs` — when both flags are set, the baked `(N,R)` array named by `node_feature_key` is passed through `glm_diagonal_features` per subject, producing `(R,R)` diagonal node-feature matrices instead of a plain `(R,)` feature vector.
- **loader `graph_feature_key=<str>`** on `data_loader.load_fc_graphs` — attaches the named baked per-graph array as `data.u` (shape `[1,U]`, batched to `[B,U]`). `"age"` gives `U=1`; `"glm_2back_vs_0back"` gives `U=400`. This is the @head side channel.
- **`heads.GraphRegressor(backbone, pooled_dim, global_dim=0, global_hidden=32, head_hidden=64)`** — wraps a GCN backbone: node embeddings → `global_mean_pool` → optional MLP encode of `data.u` (concat) → MLP head → scalar `[B]`. `global_dim=0`: plain pooled-only; `1`: age @head; `400`: GLM-zmap @head. Signature: `forward(x, edge_index, edge_weight, batch, u=None)`.
- **`transfer.save_backbone(model, path)` / `load_backbone(model, path, strict=True)` / `freeze(model)` / `set_trainable(model)`** — save an age-pretrained backbone `state_dict` and reload it into a fresh target model. Input dim is 400 for both `identity` and `glm_diagonal` carriers, so `strict=True` is safe across all experimental cells. `freeze`/`set_trainable` toggle `requires_grad` on backbone parameters for the frozen-feature transfer modes.
- **`baselines.age_vwm_baseline(age, vwm, cv=5)`** → `((slope, intercept), mean_cv_r2)` — the A5 trivial 1-feature age→VWM OLS floor (numpy-only, k-fold CV R²).

> The thesis graph is the **400 cortical Schaefer** SC subset (`sc[:400,:400]` of the 452-node SC). Node-feature carriers: `node_features="identity"` (identity) or `node_feature_key="glm_2back_vs_0back", glm_diagonal=True` (glm_diagonal). The @head side channel comes from `graph_feature_key="age"` (U=1) or `"glm_2back_vs_0back"` (U=400). Load with `matrix_key="sc", edge_weight_norm="abs_max"`.

## Dataset (`RC_DATASET_DIR`)

Baked brain-graph bundles for two cohorts (ORBIT children ~5.5–8.9y; PNC
adolescents 8–21y) × two targets (sex classification, age regression) × **three
modality variants**, so an experiment can pick what it needs at implementation
time. Every bundle stores `y` `(N,)`, `subject_id` `(N,)`, and a scalar `task`.
Call `list_datasets()` for the live list. The three variants per (cohort, target):

- **`<cohort>_fc_<tag>`** — FC graph only: `fc` `(N,400,400)` float32 (Schaefer-400
  functional connectivity = Pearson correlation, symmetric, diag 1.0, **signed**).
- **`<cohort>_sc_<tag>`** — structural connectivity: carries both `fc` `(N,400,400)`
  and `sc` `(N,452,452)` float (streamline-derived, large **positive** weights).
  SC is a **different 452-node** set, not alignable to FC per-node. **Load the SC
  graph with `matrix_key="sc"`** (the default loads the FC view of the same
  subjects, enabling matched FC-vs-SC comparison). N is the FC∩SC subset.
- **`<cohort>_fcglm_<tag>`** — FC graph + a task-activation node feature:
  `fc` `(N,400,400)` plus `glm_2back_vs_0back` `(N,400)` (Schaefer-400 GLM zmap,
  mean agg, 2back-vs-0back contrast). **Attach it with
  `node_feature_key="glm_2back_vs_0back"`** to use per-ROI activation as node
  features. N is the FC∩GLM subset. `sc`/`glm` bundles also carry boolean
  `has_sc`/`has_glm` availability arrays.

| bundle (`name`) | cohort | target | task | N | graph | label distribution |
|---|---|---|---|---|---|---|
| `orbit_fc_sex_cls` | ORBIT | sex | classification | 137 | FC 400 | 0:60 / 1:77 |
| `orbit_sc_sex_cls` | ORBIT | sex | classification | 124 | SC 452 (+FC) | 0:56 / 1:68 |
| `orbit_fcglm_sex_cls` | ORBIT | sex | classification | 100 | FC 400 +GLM | 0:47 / 1:53 |
| `orbit_fc_age_reg` | ORBIT | age (years) | regression | 137 | FC 400 | 5.54 – 8.89 |
| `orbit_sc_age_reg` | ORBIT | age (years) | regression | 125 | SC 452 (+FC) | 5.54 – 8.89 |
| `orbit_fcglm_age_reg` | ORBIT | age (years) | regression | 100 | FC 400 +GLM | 5.54 – 8.89 |
| `pnc_fc_sex_cls` | PNC | sex | classification | 972 | FC 400 | 0:526 / 1:446 |
| `pnc_sc_sex_cls` | PNC | sex | classification | 782 | SC 452 (+FC) | 0:427 / 1:355 |
| `pnc_fcglm_sex_cls` | PNC | sex | classification | 929 | FC 400 +GLM | 0:504 / 1:425 |
| `pnc_fc_age_reg` | PNC | age (years) | regression | 970 | FC 400 | 8.0 – 21.0 |
| `pnc_sc_age_reg` | PNC | age (years) | regression | 781 | SC 452 (+FC) | 8.0 – 21.0 |
| `pnc_fcglm_age_reg` | PNC | age (years) | regression | 927 | FC 400 +GLM | 8.0 – 21.0 |
| `pnc_sc400_age_reg` | PNC | age (years) | regression | 747 | SC 400 cortical (+GLM) | 8.0 – 21.0 |
| `pnc_sc400_vwm_reg` | PNC | VWM d′ | regression | 744 | SC 400 cortical (+GLM, +age) | −0.06 – 4.58 |

Notes:
- FC edges are **signed** (anticorrelations are negative) and SC edges are **large positive** streamline counts — both break `GCNConv` (NaN / instability). Use `edge_weight_norm="abs"`/`"abs_max"` for GCN on either; `"abs_max"` (→[0,1]) is the right default for SC.
- Bundles are class-balanced enough for accuracy/AUC (sex) or MAE/R² (age). Label NaNs were dropped at bake time, so N is the labelled-and-imaged(-and-modality) intersection.
- PNC `age_at_cnb` is in **years** (raw column ranges 8–22; a handful of `0`-coded missings were dropped at bake time — the floor is now a clean `8.0`).
- SC/GLM availability is partial (ORBIT GLM ≈100, SC ≈124; PNC SC ≈781, GLM ≈928) — the `<cohort>_fc_*` bundles keep the full imaged cohort; use the `sc`/`fcglm` variants when you need that modality.
- The two `pnc_sc400_*` bundles are the **PNC age→VWM thesis substrate**. The graph is the **400 cortical Schaefer** SC parcels (`sc[:400,:400]`), distinct from the 452-node `pnc_sc_*` bundles. Both store `sc (N,400,400)`, `glm_2back_vs_0back (N,400)`, `subject_id`, `y`, `task`; `pnc_sc400_vwm_reg` additionally co-stores `age (N,)` (one subject's age is NaN).
- `pnc_sc400_age_reg` (y=age, N=747) provides the **source checkpoints** (A1 identity→age, A4 glm_diagonal→age). `pnc_sc400_vwm_reg` (y=`VWM_overall_dprime`, N=744) is the **target** for the A2/A3/B*/C1/C2 cells and the A5 age floor.
- The 400-d GLM zmap aligns position-by-position to the 400 cortical SC nodes (verified at bake time), so `glm_diagonal` is correctly indexed.
