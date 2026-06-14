"""Single source of truth for the M3 protocol constants + cell map."""
from __future__ import annotations

REPS = 10
OUTER = 5
INNER_TRIALS = 20

AGE_BUNDLE = "pnc_sc400_age_reg"
VWM_BUNDLE = "pnc_sc400_vwm_reg"

# Fixed GCN architecture (Option A). Re-pin from a pre-sweep before the official run.
ARCH = dict(hidden_channels=64, out_channels=32, num_layers=3, norm="batch_norm")

# --- inner Optuna search space (Option A: optimization/reg only; ARCH is fixed) ---
def suggest_hps(trial, *, global_dim: int):
    hps = dict(
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        head_hidden=trial.suggest_categorical("head_hidden", [32, 64, 128]),
    )
    if global_dim > 0:
        hps["global_hidden"] = trial.suggest_categorical("global_hidden", [16, 32, 64])
    return hps


DEFAULT_HPS = dict(lr=1e-3, weight_decay=1e-5, dropout=0.0, head_hidden=64, global_hidden=32)

MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32

# carrier -> load_fc_graphs kwargs (NEVER a `carrier=` kwarg — see SMOKE_RESULTS.md)
def carrier_kwargs(carrier: str) -> dict:
    if carrier == "identity":
        return dict(node_features="identity")
    if carrier == "glm_diagonal":
        return dict(node_feature_key="glm_2back_vs_0back", glm_diagonal=True, glm_normalize=True)
    raise ValueError(f"unknown carrier {carrier!r}")


# cell -> spec. mode: source|scratch|finetune|frozen|trivial. bundle: age|vwm.
CELLS = {
    "A1": dict(carrier="identity",     bundle="age", mode="source",   global_dim=0),
    "A2": dict(carrier="identity",     bundle="vwm", mode="scratch",  global_dim=0),
    "A3": dict(carrier="glm_diagonal", bundle="vwm", mode="scratch",  global_dim=0),
    "A4": dict(carrier="glm_diagonal", bundle="age", mode="source",   global_dim=0),
    "A5": dict(mode="trivial"),
    "B1": dict(carrier="identity",     bundle="vwm", mode="finetune", source="A1", global_dim=0),
    "B2": dict(carrier="identity",     bundle="vwm", mode="frozen",   source="A1", global_dim=0),
    "B3": dict(carrier="glm_diagonal", bundle="vwm", mode="finetune", source="A4", global_dim=0),
    "B4": dict(carrier="glm_diagonal", bundle="vwm", mode="frozen",   source="A4", global_dim=0),
    "C1": dict(carrier="glm_diagonal", bundle="vwm", mode="scratch",  global_dim=1,
               graph_feature_key="age"),
    "C2": dict(carrier="identity",     bundle="vwm", mode="finetune", source="A1", global_dim=400,
               graph_feature_key="glm_2back_vs_0back"),
}
SOURCE_CELLS = [c for c, s in CELLS.items() if s.get("mode") == "source"]          # A1, A4
HPO_CELLS = [c for c, s in CELLS.items() if s.get("mode") in {"scratch", "finetune", "frozen"}]
