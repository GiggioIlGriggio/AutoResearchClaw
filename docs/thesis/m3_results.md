# M3 per-cell val_r2

> `task` = the cell's prediction target: **age** (A1/A4 source backbones, held-out age-R²) vs **vwm** (all baseline/transfer cells, the thesis metric). Compare R² only within the same task.

| cell | task | mean R² | std | folds | registered |
|---|---|---|---|---|---|
| A1 | age | 0.5367 | 0.0491 | 50 | — |
| A2 | vwm | 0.0502 | 0.0719 | 50 | ≈ -0.03 (identity floor) |
| A3 | vwm | 0.2143 | 0.0840 | 50 | ≈ 0.18-0.21 (baseline to beat) |
| A4 | age | 0.0434 | 0.1052 | 50 | — |
| A5 | vwm | 0.0556 | 0.0457 | 50 | small positive |
| B1 | vwm | 0.0772 | 0.0473 | 50 | ≈ 0-0.05 (<< A3) |
| B2 | vwm | 0.0313 | 0.0294 | 50 | <= B1 |
| B3 | vwm | 0.1910 | 0.0669 | 50 | ≈ A3 |
| B4 | vwm | 0.1357 | 0.0528 | 50 | <= B3 |
| C1 | vwm | 0.2444 | 0.0652 | 50 | A3 < C1 <= A3⊕A5 |
| C2 | vwm | 0.2228 | 0.0650 | 50 | ≈ A3 |
