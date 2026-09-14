# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# 1. 路径设置
# =========================================================
OUTPUT_DIR = Path(
    r"I:\DYF\NPI-4-code\1.NPI\ABIDE2_NPI验证"
)

SUBJECT_LEVEL_CSV = OUTPUT_DIR / "NPI_FC_reproduction_modelFC_empiricalFC.csv"

OUT_GROUP_AVG_CSV = OUTPUT_DIR / "NPI_FC_reproduction_group_average_FC_correlation.csv"
OUT_GROUP_MEAN_EMPIRICAL_FC = OUTPUT_DIR / "group_mean_empirical_FC.npy"
OUT_GROUP_MEAN_MODEL_FC = OUTPUT_DIR / "group_mean_model_FC.npy"


# =========================================================
# 2. FC 上三角相关
# =========================================================
def fc_upper_triangle_corr(fc_empirical, fc_model):
    iu = np.triu_indices_from(fc_empirical, k=1)

    v1 = fc_empirical[iu]
    v2 = fc_model[iu]

    mask = np.isfinite(v1) & np.isfinite(v2)

    if mask.sum() < 10:
        return np.nan

    return float(np.corrcoef(v1[mask], v2[mask])[0, 1])


# =========================================================
# 3. 基于已保存 FC 计算 group-average FC correlation
# =========================================================
def compute_group_average_fc_from_saved_results():
    if not SUBJECT_LEVEL_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find subject-level CSV: {SUBJECT_LEVEL_CSV}"
        )

    df = pd.read_csv(SUBJECT_LEVEL_CSV)

    if "status" in df.columns:
        df = df[df["status"] == "done"].copy()

    if len(df) == 0:
        raise RuntimeError("No completed subjects found in subject-level CSV.")

    empirical_sum = None
    model_sum = None

    n_valid = 0
    missing_subjects = []

    for _, row in df.iterrows():
        subject = str(row["subject"])

        if "output_subject_dir" in df.columns and pd.notna(row["output_subject_dir"]):
            subject_dir = Path(row["output_subject_dir"])
        else:
            subject_dir = OUTPUT_DIR / subject

        empirical_fc_file = subject_dir / "empirical_FC.npy"
        model_fc_file = subject_dir / "model_FC.npy"

        if not empirical_fc_file.exists() or not model_fc_file.exists():
            missing_subjects.append(subject)
            continue

        empirical_fc = np.load(empirical_fc_file).astype(np.float64)
        model_fc = np.load(model_fc_file).astype(np.float64)

        if empirical_sum is None:
            empirical_sum = np.zeros_like(empirical_fc, dtype=np.float64)
            model_sum = np.zeros_like(model_fc, dtype=np.float64)

        empirical_sum += empirical_fc
        model_sum += model_fc
        n_valid += 1

    if n_valid == 0:
        raise RuntimeError("No valid empirical_FC.npy/model_FC.npy files were found.")

    group_mean_empirical_fc = empirical_sum / n_valid
    group_mean_model_fc = model_sum / n_valid

    group_r = fc_upper_triangle_corr(
        group_mean_empirical_fc,
        group_mean_model_fc
    )

    if np.isfinite(group_r):
        group_z = float(np.arctanh(np.clip(group_r, -0.999999, 0.999999)))
    else:
        group_z = np.nan

    np.save(
        OUT_GROUP_MEAN_EMPIRICAL_FC,
        group_mean_empirical_fc.astype(np.float32)
    )

    np.save(
        OUT_GROUP_MEAN_MODEL_FC,
        group_mean_model_fc.astype(np.float32)
    )

    out_df = pd.DataFrame([{
        "N_group_FC": int(n_valid),
        "group_average_model_FC_empirical_FC_r": group_r,
        "group_average_model_FC_empirical_FC_z": group_z,
        "n_missing_FC_files": int(len(missing_subjects)),
        "missing_subjects": ";".join(missing_subjects),
        "group_mean_empirical_FC_file": str(OUT_GROUP_MEAN_EMPIRICAL_FC),
        "group_mean_model_FC_file": str(OUT_GROUP_MEAN_MODEL_FC),
        "note": (
            "This result was computed directly from saved empirical_FC.npy "
            "and model_FC.npy files. No subject-level FC matrices were recomputed."
        )
    }])

    out_df.to_csv(
        OUT_GROUP_AVG_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Finished.")
    print(f"Valid subjects: {n_valid}")
    print(f"Group-average FC r: {group_r:.6f}")
    print(f"Group-average FC Fisher z: {group_z:.6f}")
    print(f"Saved: {OUT_GROUP_AVG_CSV}")
    print(f"Saved: {OUT_GROUP_MEAN_EMPIRICAL_FC}")
    print(f"Saved: {OUT_GROUP_MEAN_MODEL_FC}")

    if len(missing_subjects) > 0:
        print(f"Missing FC files: {len(missing_subjects)}")

    return out_df


# =========================================================
# 4. 主程序
# =========================================================
if __name__ == "__main__":
    compute_group_average_fc_from_saved_results()