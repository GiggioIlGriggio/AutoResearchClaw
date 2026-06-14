"""Single source of truth for the M3 protocol constants + cell map."""
from __future__ import annotations

REPS = 10
OUTER = 5
INNER_TRIALS = 20

AGE_BUNDLE = "pnc_sc400_age_reg"
VWM_BUNDLE = "pnc_sc400_vwm_reg"

# Fixed GCN architecture (Option A). Re-pin from a pre-sweep before the official run.
ARCH = dict(hidden_channels=64, out_channels=32, num_layers=3, norm="batch_norm")
