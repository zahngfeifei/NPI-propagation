# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ===============================
# 1) PATHS
# ===============================

GRAD_ROOT = r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-独立模板\ABIDE1_结果2_combat"

# 当前 CSV：已经包含 H1-H4 指标 + Group/Age/Sex/Site/FIQ
# 只会使用 Group/Age/Sex/FIQ，不使用 Site，也不使用 MeanFD
SFC_METRICS = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1_combat\EC_SEC_metrics_all_subjects_combat.csv"

H_DIR = r"I:\DYF\NPI-4-code\3.步进分析\种子集合"

OUT_DIR = r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果1"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# ===============================
# 2) SYSTEM DEFINITIONS
# ===============================

SYSTEMS = {
    "H1_sensory": "H1_sensory.csv",
    "H2_attention": "H2_attention.csv",
    "H3_control": "H3_control.csv",
    "H4_DMN": "H4_DMN.csv",
}

EARLY_SUFFIX = "early_slope_1_10"


# ===============================
# 3) SETTINGS
# ===============================

OUTCOME_SCALE = 1000.0


# ===============================
# 4) Utilities
# ===============================

def normalize_sub_id(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    m = re.search(r"Sub0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub-Sub{int(m.group(1)):05d}"

    if re.fullmatch(r"\d+", s):
        return f"sub-Sub{int(s):05d}"

    if re.fullmatch(r"sub-Sub\d{5}", s):
        return s

    return np.nan


def read_roi_list(path):
    df = pd.read_csv(path)

    col = "ROI_index_0based" if "ROI_index_0based" in df.columns else df.columns[0]

    roi = (
        df[col]
        .dropna()
        .astype(int)
        .tolist()
    )

    return sorted(set(roi))


def coerce_sex_to_binary(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()

    if s in ["M", "MALE", "1"]:
        return 1

    if s in ["F", "FEMALE", "0"]:
        return 0

    return np.nan


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def choose_subject_column(df):
    if "sub_id" in df.columns:
        return "sub_id"

    if "SUB_ID_norm" in df.columns:
        return "SUB_ID_norm"

    raise ValueError(
        "当前 CSV 缺少被试 ID 列。需要至少包含 sub_id 或 SUB_ID_norm。"
    )


# ===============================
# 5) Load one gradient file to infer N_ROI
# ===============================

grad_files = sorted(
    glob.glob(
        os.path.join(
            GRAD_ROOT,
            "sub-*",
            "out_G1_procrustes.npy"
        )
    )
)

if len(grad_files) == 0:
    raise FileNotFoundError(
        f"No out_G1_procrustes.npy found under: {GRAD_ROOT}"
    )

sample_grad = np.load(grad_files[0]).squeeze()

if sample_grad.ndim != 1:
    raise ValueError(
        f"out_G1_procrustes.npy must be 1D. Got shape={sample_grad.shape}"
    )

N_ROI = int(sample_grad.shape[0])


# ===============================
# 6) Load system ROIs and bounds check
# ===============================

system_rois = {}
roi_audit = []

for sys_name, fname in SYSTEMS.items():
    roi_path = os.path.join(H_DIR, fname)

    if not os.path.exists(roi_path):
        raise FileNotFoundError(
            f"ROI file not found for {sys_name}: {roi_path}"
        )

    rois = read_roi_list(roi_path)

    if len(rois) > 0 and max(rois) == N_ROI:
        rois = [r - 1 for r in rois]

    bad = [r for r in rois if r < 0 or r >= N_ROI]

    if bad:
        raise ValueError(
            f"{sys_name} ROI index out of bounds 0..{N_ROI - 1}, "
            f"examples: {bad[:10]}"
        )

    system_rois[sys_name] = rois

    roi_audit.append(
        (
            sys_name,
            len(rois),
            min(rois) if rois else None,
            max(rois) if rois else None,
        )
    )


# ===============================
# 7) Build system gradients + within-subject z, G*
# ===============================

rows = []
grad_failed = []

for f in grad_files:
    sub_raw = os.path.basename(os.path.dirname(f))
    sub_norm = normalize_sub_id(sub_raw)

    if pd.isna(sub_norm):
        grad_failed.append((sub_raw, "sub_id_normalize_failed"))
        continue

    g = np.load(f).squeeze()

    if g.ndim != 1 or g.shape[0] != N_ROI:
        grad_failed.append((sub_norm, f"bad_grad_shape:{g.shape}"))
        continue

    if not np.isfinite(g).all():
        grad_failed.append((sub_norm, "grad_has_nan_inf"))
        continue

    g_sys = {
        sys_name: float(np.mean(g[rois])) if len(rois) else np.nan
        for sys_name, rois in system_rois.items()
    }

    vals = np.array(list(g_sys.values()), dtype=float)

    if not np.isfinite(vals).all():
        grad_failed.append((sub_norm, "system_mean_nan"))
        continue

    mean_g = float(np.mean(vals))
    std_g = float(np.std(vals))

    if std_g <= 0:
        grad_failed.append((sub_norm, "system_mean_std_zero"))
        continue

    for sys_name, val in g_sys.items():
        rows.append(
            {
                "sub_id": sub_norm,
                "system": sys_name,
                "G_sys": val,
                "G_star": (val - mean_g) / std_g,
            }
        )

grad_df = pd.DataFrame(rows)

if grad_df.empty:
    raise RuntimeError(
        "grad_df is empty: no valid gradients after QC."
    )

cnt_sys = grad_df.groupby("sub_id")["system"].nunique()
bad_subs = cnt_sys[cnt_sys != len(SYSTEMS)]

if len(bad_subs) > 0:
    grad_df = grad_df.loc[
        ~grad_df["sub_id"].isin(bad_subs.index)
    ].copy()


# ===============================
# 8) Load current CSV
# ===============================

sfc = pd.read_csv(SFC_METRICS)

id_col = choose_subject_column(sfc)

sfc["sub_id"] = sfc[id_col].apply(normalize_sub_id)
sfc = sfc.dropna(subset=["sub_id"]).copy()

if sfc["sub_id"].duplicated().any():
    dups = (
        sfc.loc[sfc["sub_id"].duplicated(), "sub_id"]
        .drop_duplicates()
        .head(10)
        .tolist()
    )
    raise ValueError(
        f"Duplicate sub_id found in current CSV, examples: {dups}"
    )

required_sfc_cols = [
    f"{sys_name}_{EARLY_SUFFIX}"
    for sys_name in SYSTEMS.keys()
]

missing_sfc = [
    c for c in required_sfc_cols
    if c not in sfc.columns
]

if missing_sfc:
    raise ValueError(
        f"SFC_METRICS missing required early_slope columns: {missing_sfc}"
    )


# ===============================
# 9) Wide SEC metrics -> long
# ===============================

long_rows = []

for _, r in sfc.iterrows():
    sub_id = r["sub_id"]

    for sys_name in SYSTEMS.keys():
        long_rows.append(
            {
                "sub_id": sub_id,
                "system": sys_name,
                "early_slope": safe_float(
                    r[f"{sys_name}_{EARLY_SUFFIX}"]
                ),
            }
        )

sfc_long = pd.DataFrame(long_rows)


# ===============================
# 10) Covariates from current CSV
#     不包含 Site 和 MeanFD
# ===============================

need_cov = {
    "sub_id",
    "Group",
    "Age",
    "Sex",
    "FIQ",
}

missing_cov = [
    c for c in need_cov
    if c not in sfc.columns
]

if missing_cov:
    raise ValueError(
        f"当前 CSV 缺少必要协变量列: {missing_cov}"
    )

subinfo = sfc[["sub_id", "Group", "Age", "Sex", "FIQ"]].copy()

subinfo["Sex_bin"] = subinfo["Sex"].apply(coerce_sex_to_binary)

for col in ["Age", "FIQ"]:
    subinfo[col] = pd.to_numeric(subinfo[col], errors="coerce")

subinfo["Group"] = subinfo["Group"].astype(str).str.strip().str.upper()
subinfo = subinfo[subinfo["Group"].isin(["ASD", "HC"])].copy()

subinfo["Group"] = pd.Categorical(
    subinfo["Group"],
    categories=["HC", "ASD"],
    ordered=True,
)

cov_cols = [
    "Group",
    "Age",
    "Sex_bin",
    "FIQ",
]

cov_missing = subinfo[cov_cols].isna().any(axis=1)
n_cov_missing = int(cov_missing.sum())

if n_cov_missing > 0:
    audit_path = os.path.join(
        OUT_DIR,
        "dropped_subjects_missing_covariates.csv"
    )

    subinfo.loc[
        cov_missing,
        ["sub_id"] + cov_cols
    ].to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    subinfo = subinfo.loc[~cov_missing].copy()


# ===============================
# 11) Merge
# ===============================

data = pd.merge(
    grad_df,
    sfc_long,
    on=["sub_id", "system"],
    how="inner",
)

data = pd.merge(
    data,
    subinfo[["sub_id"] + cov_cols],
    on="sub_id",
    how="inner",
)

cnt_final = data.groupby("sub_id")["system"].nunique()
bad_final = cnt_final[cnt_final != len(SYSTEMS)]

if len(bad_final) > 0:
    data = data.loc[
        ~data["sub_id"].isin(bad_final.index)
    ].copy()

if data.empty:
    raise RuntimeError(
        "Final merged table is empty. "
        "Check ID matching between gradients and current CSV."
    )

bad_y = ~np.isfinite(data["early_slope"].to_numpy(dtype=float))
n_bad_y = int(bad_y.sum())

if n_bad_y > 0:
    data = data.loc[~bad_y].copy()

data["system"] = pd.Categorical(
    data["system"],
    categories=list(SYSTEMS.keys()),
    ordered=True,
)

data["early_slope_scaled"] = (
    data["early_slope"] * float(OUTCOME_SCALE)
)

input_path = os.path.join(
    OUT_DIR,
    "result1_input_long_table_strict.csv"
)

data.to_csv(
    input_path,
    index=False,
    encoding="utf-8-sig",
)


# ===============================
# 12) QC report
# ===============================

qc_lines = []

qc_lines.append("Result 1 data-construction-only QC report")
qc_lines.append(f"Generated at: {datetime.now()}")
qc_lines.append("")
qc_lines.append("Purpose:")
qc_lines.append(
    "This script constructs the long-format input table and QC report. "
    "No statistical model, no GEE, and no figure are generated."
)
qc_lines.append("")
qc_lines.append("Current CSV adaptation:")
qc_lines.append(f"  Input CSV: {SFC_METRICS}")
qc_lines.append("  Covariates used: Group, Age, Sex_bin, FIQ")
qc_lines.append("  Covariates not used: Site, MeanFD")
qc_lines.append("")
qc_lines.append(f"N_ROI: {N_ROI}")
qc_lines.append("")
qc_lines.append("ROI audit:")

for sys_name, n, mn, mx in roi_audit:
    qc_lines.append(
        f"  {sys_name}: n={n}, min={mn}, max={mx}"
    )

qc_lines.append("")
qc_lines.append("Sample sizes:")
qc_lines.append(f"  Gradient files found: {len(grad_files)}")
qc_lines.append(f"  Gradient subjects failed QC: {len(grad_failed)}")
qc_lines.append(f"  Subjects dropped due to missing covariates: {n_cov_missing}")
qc_lines.append(f"  Rows dropped due to non-finite early_slope: {n_bad_y}")
qc_lines.append(f"  Final subjects: {data['sub_id'].nunique()}")
qc_lines.append(f"  Final obs: {len(data)}")
qc_lines.append(f"  Expected obs if complete: n_sub × 4 = {data['sub_id'].nunique() * 4}")
qc_lines.append("")
qc_lines.append("Outcome scale:")
qc_lines.append(f"  OUTCOME_SCALE: {OUTCOME_SCALE}")
qc_lines.append("")
qc_lines.append("Columns in output table:")
qc_lines.append(str(list(data.columns)))
qc_lines.append("")
qc_lines.append("early_slope describe:")
qc_lines.append(str(data["early_slope"].describe()))
qc_lines.append("")
qc_lines.append("early_slope_scaled describe:")
qc_lines.append(str(data["early_slope_scaled"].describe()))
qc_lines.append("")

v_within = data.groupby("sub_id")["early_slope"].var()

qc_lines.append("Within-subject variance early_slope:")
qc_lines.append(f"  median var: {v_within.median()}")
qc_lines.append(f"  pct var==0, NaN treated as 0: {(v_within.fillna(0) == 0).mean()}")
qc_lines.append("")
qc_lines.append("Group sample size:")
qc_lines.append(
    str(
        data.drop_duplicates("sub_id")
        .groupby("Group", observed=False)["sub_id"]
        .count()
    )
)
qc_lines.append("")
qc_lines.append("Observation count by Group × system:")
qc_lines.append(str(pd.crosstab(data["Group"], data["system"])))

qc_path = os.path.join(
    OUT_DIR,
    "result1_qc_report.txt"
)

write_text(qc_path, "\n".join(qc_lines))

if grad_failed:
    pd.DataFrame(
        grad_failed,
        columns=["sub_id", "reason"]
    ).to_csv(
        os.path.join(OUT_DIR, "gradient_failed_subjects.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ===============================
# 13) Print
# ===============================

print("Result 1 data-construction-only finished.")
print("Input long table:", input_path)
print("QC report:", qc_path)
print("No statistical model was run.")
print("No GEE was run.")
print("No figure was generated.")
print("Site and MeanFD were not used.")
print("Outputs saved to:", OUT_DIR)
