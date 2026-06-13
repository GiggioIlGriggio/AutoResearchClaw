"""Bake the PNC age->VWM thesis bundles on a 400-node cortical SC graph.

Two bundles in ~/rc_brain_data, both with the SC-400 graph (sc[:400,:400] of the
452-node schaefer400 SC -- cortical Schaefer-400, verified to align in order with
the 400-d GLM zmap) and the 400-d glm_2back_vs_0back node-feature array:

  pnc_sc400_age_reg : y = age            (A1 identity->age, A4 glm_diagonal->age; source ckpts)
  pnc_sc400_vwm_reg : y = VWM_overall_dprime, + age (N,) co-stored (A5 floor, C1 age@head)

Both filter to has_sc AND has_glm so every cell's carrier is available. Run with a
venv that has torch+torch_geometric+pandas+scipy.
"""
import os, sys, json, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

CORE = "/home/compa/Documents/working_dir/LLM_Playground/core"
sys.path.insert(0, CORE)
from core.data.brain.config import LoadConfig          # noqa: E402
from core.data.brain.records import load_cohort         # noqa: E402

REAL_ROOT = Path("/media/compa/DATA1/Compa/DATA_DERIVATIVES")
OUT = Path(os.path.expanduser("~/rc_brain_data")); OUT.mkdir(parents=True, exist_ok=True)
GLM_CONTRAST, GLM_AGG, GLM_MAPTYPE = "2back_vs_0back", "mean", "zmap"
N_CORTICAL = 400


def stage_pnc():
    """Symlinked PNC root with cleaned labels (Sex->F/M, age coerced + <5y dropped).

    Adapted from scripts/bake_scaffold_dataset.py:stage_pnc (kept local so this
    one-off thesis bake does not import that script's top-level bake).
    """
    stage = Path(tempfile.mkdtemp(prefix="pnc_sc400_"))
    t0 = stage / "PNC" / "T0"; t0.mkdir(parents=True)
    src = REAL_ROOT / "PNC" / "T0"
    (t0 / "Functional_Mats").symlink_to(src / "Functional_Mats")
    (t0 / "Structural_maps").symlink_to(src / "Structural_maps")
    (t0 / "GLM_Maps").symlink_to(src / "GLM_Maps")
    tab = t0 / "Tabular_data"; tab.mkdir()
    df = pd.read_csv(src / "Tabular_data" / "PNC_ALL_SCORES.csv", low_memory=False)
    df["Sex"] = df["Sex"].where(df["Sex"].isin(["F", "M"]))
    age = pd.to_numeric(df["age_at_cnb"], errors="coerce")
    df["age_at_cnb"] = age.where(age >= 5)
    df.to_csv(tab / "PNC_ALL_SCORES.csv", index=False)
    shutil.copy(src / "Tabular_data" / "subject_mapping.tsv", tab / "subject_mapping.tsv")
    return stage


def load(label, root):
    cfg = LoadConfig(cohort="pnc", label_column=label, task="regression", load_sc=True,
                     glm_contrast=GLM_CONTRAST, glm_agg=GLM_AGG, glm_maptype=GLM_MAPTYPE,
                     root=root)
    recs = [r for r in load_cohort(cfg) if r.has_sc and r.has_glm]
    return recs


def arrays(recs):
    sc = np.stack([np.asarray(r.sc[:N_CORTICAL, :N_CORTICAL], np.float32) for r in recs])
    glm = np.stack([np.asarray(r.glm[GLM_CONTRAST], np.float32) for r in recs])
    assert sc.shape[1:] == (N_CORTICAL, N_CORTICAL), sc.shape
    assert glm.shape[1] == N_CORTICAL, glm.shape          # alignment gate
    sid = np.array([r.subject_id for r in recs])
    y = np.array([float(r.label) for r in recs], np.float32)
    return sc, glm, sid, y


def write(name, sc, glm, sid, y, age=None):
    z = dict(sc=sc, glm_2back_vs_0back=glm, subject_id=sid, y=y,
             task=np.array("regression"),
             glm_contrast=np.array(GLM_CONTRAST), glm_agg=np.array(GLM_AGG),
             glm_maptype=np.array(GLM_MAPTYPE))
    if age is not None:
        z["age"] = age
    np.savez(OUT / f"{name}.npz", **z)
    info = {"name": name, "n": int(len(y)), "R_sc": N_CORTICAL,
            "graph_matrix_key": "sc", "node_feature_key": "glm_2back_vs_0back",
            "y_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
            "size_mb": round((OUT / f"{name}.npz").stat().st_size / 1e6, 1)}
    if age is not None:
        info["age_range"] = [round(float(np.nanmin(age)), 2), round(float(np.nanmax(age)), 2)]
        info["n_age_nan"] = int(np.isnan(age).sum())
    print(f"[OK] {name}: N={len(y)} {info['y_range']} ({info['size_mb']} MB)", flush=True)
    return info


def main():
    stage = stage_pnc()
    manifest = []
    try:
        # age->subject_id map (for co-storing age on the VWM bundle)
        age_recs = load("age_at_cnb", stage)
        age_map = {r.subject_id: float(r.label) for r in age_recs}

        # pnc_sc400_age_reg (source checkpoints A1/A4)
        sc, glm, sid, y = arrays(age_recs)
        manifest.append(write("pnc_sc400_age_reg", sc, glm, sid, y))

        # pnc_sc400_vwm_reg (targets; age co-stored)
        vwm_recs = load("VWM_overall_dprime", stage)
        sc, glm, sid, y = arrays(vwm_recs)
        age = np.array([age_map.get(s, np.nan) for s in sid], np.float32)
        manifest.append(write("pnc_sc400_vwm_reg", sc, glm, sid, y, age=age))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    mpath = OUT / "MANIFEST_sc400.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print("\n=== DONE ===\n" + json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
