# -*- coding: utf-8 -*-
"""
Result 4 — behavior-related standalone outputs for selected ADOS analyses

Purpose
-------
This script does NOT modify Result 1/2/3 outputs. It reads the existing Result 1
long table and optional ABIDE I / ABIDE II phenotype CSV files, then creates
standalone selected-ADOS behavior-analysis inputs:

1) result4_brain_subject_metrics_for_behavior.csv
   Subject-level brain metrics:
   - H1/H2/H3/H4 early_slope_scaled
   - H1 vs H3/H4 propagation imbalance
   - individual G_star -> early_slope_scaled coupling index
   - covariates from Result 1 long table

2) result4_behavior_phenotypes_cleaned.csv
   Harmonized selected ADOS phenotype table.

3) result4_brain_behavior_subject_level.csv
   Subject-level brain metrics merged with behavior phenotypes.

4) result4_brain_behavior_long_for_GEE.csv
   Long-format system-level table merged with behavior phenotypes.
   Use this for behavior-modulation GEE models, e.g.:
       early_slope_scaled ~ G_star * ADOS_total_z + C(system) + Age + Sex + FIQ + MeanFD + C(Site)

5) result4_behavior_output_qc_report.txt
   QC summary with sample sizes and missingness counts.

Inputs expected
---------------
- Result 1 long table:
    result1_input_long_table_strict.csv
  Required columns:
    sub_id, system, G_star, early_slope_scaled
  Recommended covariates:
    Group, Age, Sex_bin or Sex, Site, FIQ, MeanFD

- Optional behavior CSVs:
    ABIDEⅠ.csv and ABIDEⅡ.csv
  If unavailable, the script still writes brain-only behavior-ready outputs.

Notes
-----
- Coupling_index_scaled is subject-level slope from:
      early_slope_scaled ~ G_star
- Coupling_index_raw is subject-level slope from:
      early_slope ~ G_star
  if early_slope is available.
- Because each subject has only four system observations, Coupling_index is a compact
  descriptive subject-level index. For primary inference, also consider the long-format
  GEE behavior-modulation model.
"""

import os
import re
import warnings
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ============================================================
# 1) PATH SETTINGS — edit to your local paths
# ============================================================
INPUT_RESULT1_LONG = r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果1\result1_input_long_table_strict.csv"

# Behavior phenotype files are optional. If not found, brain-only outputs are produced.
ABIDE1_PHENO = r"I:\DYF\NPI-4-code\ABIDEⅠ.csv"
ABIDE2_PHENO = r"I:\DYF\NPI-4-code\ABIDEⅡ.csv"

OUT_DIR = r"I:\DYF\NPI-4-code\5.行为分析\ABIDE1_结果1行为相关输出"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ADOS 临床量表默认只保留 ASD 被试。
BEHAVIOR_ASD_ONLY = True

# 是否额外导出指定原始 ADOS 列的 ASD-only 单独 CSV。
EXPORT_SCALE_SPECIFIC_ASD_CSV = True
SCALE_EXPORT_ROOT = r"I:\DYF\NPI-4-code\5.行为分析\ABIDE1_行为量表"

# True：运行前清空 SCALE_EXPORT_ROOT，避免旧版代码输出的 SRS/RRB/Vineland
# 或另一数据集的 ADOS 文件残留。请确认该目录只用于本脚本量表输出。
CLEAR_OLD_SCALE_EXPORTS = True

# 当前分析数据集：
# - "AUTO"：根据 INPUT_RESULT1_LONG 路径中的 ABIDE1 / ABIDE2 自动识别；
# - 也可以手动设置为 "ABIDE1" 或 "ABIDE2"。
ANALYSIS_DATASET = "AUTO"

# 每个数据集仅允许导出的原始 ADOS 列。
SELECTED_ADOS_EXPORT_COLUMNS = {
    "ABIDE1": [
        "ADOS_COMM",
        "ADOS_SOCIAL",
        "ADOS_STEREO_BEHAV",
        "ADOS_TOTAL",
    ],
    "ABIDE2": [
        "ADOS_G_COMM",
        "ADOS_G_SOCIAL",
        "ADOS_G_STEREO_BEHAV",
        "ADOS_G_TOTAL",
    ],
}


# ============================================================
# 2) GLOBAL SETTINGS
# ============================================================
SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]

MISSING_VALUES = [-9999, -999, 9999, 999, "-9999", "-999", "9999", "999"]

# Main variables recommended for behavior association analysis.
PRIMARY_BRAIN_METRICS = [
    "Propagation_imbalance_scaled",
    "H1_minus_H3_scaled",
    "H1_minus_H4_scaled",
    "H3H4_mean_early_slope_scaled",
    "Coupling_index_scaled",
]

PRIMARY_BEHAVIOR_METRICS = [
    "ADOS_communication",
    "ADOS_social",
    "ADOS_stereotyped_behavior",
    "ADOS_total",
]


# ============================================================
# 3) UTILITY FUNCTIONS
# ============================================================
def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_sub_id(x):
    """Normalize IDs to the same format used in the gradient/propagation pipeline: sub-Sub00000."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # Already normalized.
    if re.fullmatch(r"sub-Sub\d{5}", s):
        return s

    # Strings such as Sub0050001 / sub-0050001 / sub_Sub0050001.
    m = re.search(r"Sub0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub-Sub{int(m.group(1)):05d}"

    # Pure numeric IDs.
    if re.fullmatch(r"\d+", s):
        return f"sub-Sub{int(s):05d}"

    return np.nan


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def replace_missing_values(df):
    df = df.copy()
    df = df.replace(MISSING_VALUES, np.nan)
    return df


def to_numeric_if_exists(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def first_available(df, candidates):
    """Return first non-null value across candidate columns row-wise."""
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    out = pd.Series(np.nan, index=df.index)
    for c in existing:
        out = out.fillna(df[c])
    return out


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / sd


def zscore_within(df, value_col, group_col):
    """Dataset-wise z-score. Falls back to all-sample z-score if a group has <3 valid values."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if value_col not in df.columns or group_col not in df.columns:
        return out

    for g, idx in df.groupby(group_col).groups.items():
        s = pd.to_numeric(df.loc[idx, value_col], errors="coerce")
        if s.notna().sum() >= 3 and s.std(skipna=True, ddof=0) > 0:
            out.loc[idx] = (s - s.mean(skipna=True)) / s.std(skipna=True, ddof=0)
        else:
            out.loc[idx] = zscore(s)
    return out


def safe_group_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in ["1", "ASD", "AUTISM", "AUTISM SPECTRUM DISORDER"]:
        return "ASD"
    if s in ["2", "HC", "CONTROL", "TD", "TDC"]:
        return "HC"
    return np.nan


def resolve_analysis_dataset():
    """
    确定本次运行对应的数据集。

    ANALYSIS_DATASET="AUTO" 时，只根据 INPUT_RESULT1_LONG 判断：
    - 路径包含 ABIDE1 -> ABIDE1
    - 路径包含 ABIDE2 -> ABIDE2

    这样 ABIDE1 分析只会读取、整理和导出 ABIDE1 量表，
    ABIDE2 分析只会读取、整理和导出 ABIDE2 量表。
    """
    configured = str(ANALYSIS_DATASET).strip().upper()

    if configured in SELECTED_ADOS_EXPORT_COLUMNS:
        return configured

    if configured != "AUTO":
        raise ValueError(
            'ANALYSIS_DATASET 必须为 "AUTO"、"ABIDE1" 或 "ABIDE2"。'
        )

    path_text = str(INPUT_RESULT1_LONG).upper()
    path_text = path_text.replace("ABIDEⅠ", "ABIDE1").replace("ABIDEⅡ", "ABIDE2")

    has_abide1 = "ABIDE1" in path_text
    has_abide2 = "ABIDE2" in path_text

    if has_abide1 and not has_abide2:
        return "ABIDE1"
    if has_abide2 and not has_abide1:
        return "ABIDE2"

    raise ValueError(
        "无法从 INPUT_RESULT1_LONG 自动识别 ABIDE1/ABIDE2。"
        '请将 ANALYSIS_DATASET 手动设置为 "ABIDE1" 或 "ABIDE2"。'
    )


# ============================================================
# 4) LOAD RESULT 1 LONG TABLE AND CREATE BRAIN OUTPUTS
# ============================================================
def load_result1_long(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result 1 long table not found: {path}")

    df = pd.read_csv(path)
    df = clean_columns(df)

    required = {"sub_id", "system", "G_star", "early_slope_scaled"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Result 1 long table missing required columns: {missing}")

    df["sub_id"] = df["sub_id"].apply(normalize_sub_id)
    df = df.dropna(subset=["sub_id"]).copy()

    numeric_cols = ["G_star", "early_slope_scaled", "early_slope", "Age", "FIQ", "MeanFD", "Sex_bin", "Sex"]
    df = to_numeric_if_exists(df, numeric_cols)

    if "Group" in df.columns:
        df["Group"] = df["Group"].map(safe_group_label)

    df["system"] = df["system"].astype(str).str.strip()

    # Keep only the four expected hierarchy systems when present.
    df = df[df["system"].isin(SYSTEM_ORDER)].copy()

    return df


def audit_system_completeness(df):
    n_sys = df.groupby("sub_id")["system"].nunique()
    valid_subs = n_sys[n_sys == len(SYSTEM_ORDER)].index
    bad = n_sys[n_sys != len(SYSTEM_ORDER)].reset_index().rename(columns={"system": "n_systems"})
    return df[df["sub_id"].isin(valid_subs)].copy(), bad


def estimate_subject_coupling(g, y_col):
    tmp = g[["G_star", y_col]].dropna().copy()
    out = {
        f"Coupling_index_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": np.nan,
        f"Coupling_intercept_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": np.nan,
        f"Coupling_r_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": np.nan,
        f"Coupling_p_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": np.nan,
        f"Coupling_SE_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": np.nan,
        f"n_system_for_coupling_{'scaled' if y_col == 'early_slope_scaled' else 'raw'}": int(tmp.shape[0]),
    }

    if tmp.shape[0] < 3:
        return pd.Series(out)
    if tmp["G_star"].std(ddof=0) == 0 or tmp[y_col].std(ddof=0) == 0:
        return pd.Series(out)

    slope, intercept, r, p, se = linregress(tmp["G_star"], tmp[y_col])
    suffix = "scaled" if y_col == "early_slope_scaled" else "raw"
    out[f"Coupling_index_{suffix}"] = slope
    out[f"Coupling_intercept_{suffix}"] = intercept
    out[f"Coupling_r_{suffix}"] = r
    out[f"Coupling_p_{suffix}"] = p
    out[f"Coupling_SE_{suffix}"] = se

    return pd.Series(out)


def create_system_wide(df, value_col, suffix):
    wide = (
        df.pivot_table(index="sub_id", columns="system", values=value_col, aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    rename = {sys: f"{sys}_{suffix}" for sys in SYSTEM_ORDER if sys in wide.columns}
    wide = wide.rename(columns=rename)
    return wide


def prepare_covariates_from_long(df):
    cov_candidates = [
        "sub_id", "Group", "Age", "Sex_bin", "Sex", "Site", "FIQ", "MeanFD"
    ]
    cols = [c for c in cov_candidates if c in df.columns]
    cov = df[cols].drop_duplicates(subset=["sub_id"]).copy()

    # Unified Sex variable for behavior models.
    if "Sex_bin" in cov.columns:
        cov["Sex_for_model"] = pd.to_numeric(cov["Sex_bin"], errors="coerce")
    elif "Sex" in cov.columns:
        cov["Sex_for_model"] = pd.to_numeric(cov["Sex"], errors="coerce")
    else:
        cov["Sex_for_model"] = np.nan

    return cov


def create_brain_subject_metrics(df):
    df_valid, bad_system = audit_system_completeness(df)

    # Wide early slopes.
    wide_scaled = create_system_wide(df_valid, "early_slope_scaled", "early_slope_scaled")

    if "early_slope" in df_valid.columns:
        wide_raw = create_system_wide(df_valid, "early_slope", "early_slope_raw")
    else:
        wide_raw = pd.DataFrame({"sub_id": wide_scaled["sub_id"]})

    # Wide G_star values for audit and optional use.
    wide_g = create_system_wide(df_valid, "G_star", "G_star")

    brain = wide_scaled.merge(wide_raw, on="sub_id", how="left")
    brain = brain.merge(wide_g, on="sub_id", how="left")

    # Propagation imbalance using scaled outcome.
    h1 = "H1_sensory_early_slope_scaled"
    h2 = "H2_attention_early_slope_scaled"
    h3 = "H3_control_early_slope_scaled"
    h4 = "H4_DMN_early_slope_scaled"

    for c in [h1, h2, h3, h4]:
        if c not in brain.columns:
            brain[c] = np.nan

    brain["H3H4_mean_early_slope_scaled"] = brain[[h3, h4]].mean(axis=1)
    brain["Propagation_imbalance_scaled"] = brain[h1] - brain["H3H4_mean_early_slope_scaled"]
    brain["H1_minus_H3_scaled"] = brain[h1] - brain[h3]
    brain["H1_minus_H4_scaled"] = brain[h1] - brain[h4]
    brain["H1_minus_H2_scaled"] = brain[h1] - brain[h2]

    # Same imbalance using raw early_slope if present.
    raw_cols = [
        "H1_sensory_early_slope_raw",
        "H3_control_early_slope_raw",
        "H4_DMN_early_slope_raw",
    ]
    if all(c in brain.columns for c in raw_cols):
        brain["H3H4_mean_early_slope_raw"] = brain[[raw_cols[1], raw_cols[2]]].mean(axis=1)
        brain["Propagation_imbalance_raw"] = brain[raw_cols[0]] - brain["H3H4_mean_early_slope_raw"]

    # Coupling indices.
    coupling_scaled = (
        df_valid.groupby("sub_id", group_keys=False)
        .apply(lambda g: estimate_subject_coupling(g, "early_slope_scaled"))
        .reset_index()
    )

    if "early_slope" in df_valid.columns:
        coupling_raw = (
            df_valid.groupby("sub_id", group_keys=False)
            .apply(lambda g: estimate_subject_coupling(g, "early_slope"))
            .reset_index()
        )
    else:
        coupling_raw = pd.DataFrame({"sub_id": brain["sub_id"]})

    cov = prepare_covariates_from_long(df_valid)

    brain = cov.merge(brain, on="sub_id", how="inner")
    brain = brain.merge(coupling_scaled, on="sub_id", how="left")
    brain = brain.merge(coupling_raw, on="sub_id", how="left")

    # Z-score primary brain metrics for downstream regression.
    for col in PRIMARY_BRAIN_METRICS:
        if col in brain.columns:
            brain[col + "_z"] = zscore(brain[col])

    return brain, df_valid, bad_system


# ============================================================
# 5) HARMONIZE BEHAVIOR PHENOTYPES
# ============================================================
def read_behavior_file(path, dataset_label):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    df = clean_columns(df)
    df = replace_missing_values(df)
    df["Dataset"] = dataset_label

    # Create normalized sub_id from any available ID column.
    if "sub_id" in df.columns:
        id_raw = df["sub_id"]
    elif "SUB_ID" in df.columns:
        id_raw = df["SUB_ID"]
    elif "FILE_ID" in df.columns:
        id_raw = df["FILE_ID"]
    else:
        raise ValueError(f"No ID column found in behavior file: {path}")

    df["sub_id"] = id_raw.apply(normalize_sub_id)
    df = df.dropna(subset=["sub_id"]).copy()

    return df


def harmonize_behavior(df, analysis_dataset):
    """
    将 ABIDE1 和 ABIDE2 中指定的 ADOS 原始列统一为四个共同指标：

    ABIDE1:
        ADOS_COMM, ADOS_SOCIAL, ADOS_STEREO_BEHAV, ADOS_TOTAL

    ABIDE2:
        ADOS_G_COMM, ADOS_G_SOCIAL, ADOS_G_STEREO_BEHAV, ADOS_G_TOTAL
    """
    df = df.copy()

    numeric_candidates = [
        "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ",
        "ADOS_COMM", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV", "ADOS_TOTAL",
        "ADOS_G_COMM", "ADOS_G_SOCIAL",
        "ADOS_G_STEREO_BEHAV", "ADOS_G_TOTAL",
    ]
    df = to_numeric_if_exists(df, numeric_candidates)

    # 行为表中的组别仅作为审计字段；与脑数据合并后优先使用 Result 1 的 Group。
    if "DX_GROUP" in df.columns:
        df["Group_behavior"] = df["DX_GROUP"].map({1: "ASD", 2: "HC"})
    else:
        df["Group_behavior"] = np.nan

    # 表型文件中的协变量，仅作为审计字段。
    df["Site_behavior"] = first_available(df, ["SITE_ID", "SITE", "site", "Site"])
    df["Age_behavior"] = first_available(df, ["AGE_AT_SCAN", "AGE", "Age"])
    df["Sex_behavior"] = first_available(df, ["SEX", "Sex"])
    df["FIQ_behavior"] = first_available(df, ["FIQ", "FIQ_STANDARD"])

    # 仅使用当前数据集对应的四个 ADOS 原始列，禁止跨数据集回退。
    source_columns = {
        "ABIDE1": {
            "ADOS_communication": "ADOS_COMM",
            "ADOS_social": "ADOS_SOCIAL",
            "ADOS_stereotyped_behavior": "ADOS_STEREO_BEHAV",
            "ADOS_total": "ADOS_TOTAL",
        },
        "ABIDE2": {
            "ADOS_communication": "ADOS_G_COMM",
            "ADOS_social": "ADOS_G_SOCIAL",
            "ADOS_stereotyped_behavior": "ADOS_G_STEREO_BEHAV",
            "ADOS_total": "ADOS_G_TOTAL",
        },
    }

    if analysis_dataset not in source_columns:
        raise ValueError(f"不支持的数据集: {analysis_dataset}")

    for output_col, source_col in source_columns[analysis_dataset].items():
        if source_col in df.columns:
            df[output_col] = pd.to_numeric(df[source_col], errors="coerce")
        else:
            df[output_col] = np.nan
            print(
                f"[WARNING] {analysis_dataset} 表型文件中未找到列 {source_col}。"
            )

    # 同一 sub_id 出现多行时，优先保留四个指定 ADOS 指标非缺失数更多的一行。
    score_cols = [c for c in PRIMARY_BEHAVIOR_METRICS if c in df.columns]
    df["_behavior_nonmissing_count"] = df[score_cols].notna().sum(axis=1)
    df = (
        df.sort_values(
            ["sub_id", "_behavior_nonmissing_count"],
            ascending=[True, False],
        )
        .drop_duplicates(subset=["sub_id"], keep="first")
        .drop(columns=["_behavior_nonmissing_count"])
    )

    keep_cols = [
        "sub_id",
        "Dataset",
        "Group_behavior",
        "Site_behavior",
        "Age_behavior",
        "Sex_behavior",
        "FIQ_behavior",
    ] + PRIMARY_BEHAVIOR_METRICS
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    # 同时提供总体 z 分数和数据集内 z 分数。
    for col in PRIMARY_BEHAVIOR_METRICS:
        if col not in out.columns:
            continue
        out[col + "_z"] = zscore(out[col])
        out[col + "_z_by_dataset"] = zscore_within(
            out,
            col,
            "Dataset",
        )

    return out


def selected_raw_behavior_output(raw_behavior, analysis_dataset):
    """
    仅保留标识信息、协变量和用户指定的 8 个原始 ADOS 列，
    用于 result4_behavior_phenotypes_raw_merged.csv。
    """
    if raw_behavior is None:
        return None

    base_cols = [
        "Dataset",
        "SITE_ID",
        "SITE",
        "SUB_ID",
        "FILE_ID",
        "sub_id",
        "DX_GROUP",
        "AGE_AT_SCAN",
        "SEX",
        "FIQ",
    ]

    if analysis_dataset not in SELECTED_ADOS_EXPORT_COLUMNS:
        raise ValueError(f"不支持的数据集: {analysis_dataset}")

    # 只保留当前数据集对应的四个原始 ADOS 列。
    selected_scale_cols = SELECTED_ADOS_EXPORT_COLUMNS[analysis_dataset]

    keep_cols = [
        c for c in base_cols + selected_scale_cols
        if c in raw_behavior.columns
    ]
    return raw_behavior[keep_cols].copy()


def export_selected_ados_asd_csvs(
    raw_behavior,
    out_root,
    analysis_dataset,
):
    """
    仅导出当前分析数据集对应的四个 ADOS 原始量表，并仅保留 ASD 被试。

    当 analysis_dataset == "ABIDE1" 时，只生成：
        ADOS_ADOS_COMM_ASD_only.csv
        ADOS_ADOS_SOCIAL_ASD_only.csv
        ADOS_ADOS_STEREO_BEHAV_ASD_only.csv
        ADOS_ADOS_TOTAL_ASD_only.csv

    当 analysis_dataset == "ABIDE2" 时，只生成：
        ADOS_ADOS_G_COMM_ASD_only.csv
        ADOS_ADOS_G_SOCIAL_ASD_only.csv
        ADOS_ADOS_G_STEREO_BEHAV_ASD_only.csv
        ADOS_ADOS_G_TOTAL_ASD_only.csv
    """
    summary_columns = [
        "dataset",
        "scale",
        "variable",
        "N",
        "status",
        "output_csv",
    ]

    if raw_behavior is None or raw_behavior.empty:
        return pd.DataFrame(columns=summary_columns)

    if analysis_dataset not in SELECTED_ADOS_EXPORT_COLUMNS:
        raise ValueError(f"不支持的数据集: {analysis_dataset}")

    df = clean_columns(raw_behavior.copy())
    df = replace_missing_values(df)

    required = {"Dataset", "DX_GROUP"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"量表导出缺少必要列: {missing}")

    df["Dataset"] = df["Dataset"].astype(str).str.strip().str.upper()
    df["DX_GROUP"] = pd.to_numeric(df["DX_GROUP"], errors="coerce")

    # 双重限制：必须属于当前数据集，同时必须为 ASD。
    dataset_df = df[
        (df["Dataset"] == analysis_dataset)
        & (df["DX_GROUP"] == 1)
    ].copy()

    root = Path(out_root)

    # 清空旧输出，避免 ABIDE1 分析目录中残留 ABIDE2 文件，反之亦然。
    if CLEAR_OLD_SCALE_EXPORTS and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    ados_dir = root / analysis_dataset / "ADOS"
    ados_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [
        c for c in [
            "Dataset",
            "SITE_ID",
            "SITE",
            "SUB_ID",
            "FILE_ID",
            "sub_id",
            "DX_GROUP",
            "AGE_AT_SCAN",
            "SEX",
            "FIQ",
        ]
        if c in dataset_df.columns
    ]

    export_rows = []

    for col in SELECTED_ADOS_EXPORT_COLUMNS[analysis_dataset]:
        out_csv = ados_dir / f"ADOS_{col}_ASD_only.csv"

        if col not in dataset_df.columns:
            export_rows.append({
                "dataset": analysis_dataset,
                "scale": "ADOS",
                "variable": col,
                "N": 0,
                "status": "column_not_found",
                "output_csv": str(out_csv),
            })
            print(
                f"[WARNING] {analysis_dataset} 中未找到列 {col}，未生成文件。"
            )
            continue

        subset = dataset_df[base_cols + [col]].copy()
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[col])

        if subset.empty:
            export_rows.append({
                "dataset": analysis_dataset,
                "scale": "ADOS",
                "variable": col,
                "N": 0,
                "status": "no_valid_asd_values",
                "output_csv": str(out_csv),
            })
            print(
                f"[WARNING] {analysis_dataset} 的 {col} "
                "没有有效 ASD 数值，未生成文件。"
            )
            continue

        subset.to_csv(
            out_csv,
            index=False,
            encoding="utf-8-sig",
        )

        export_rows.append({
            "dataset": analysis_dataset,
            "scale": "ADOS",
            "variable": col,
            "N": int(subset.shape[0]),
            "status": "exported",
            "output_csv": str(out_csv),
        })

    export_summary = pd.DataFrame(
        export_rows,
        columns=summary_columns,
    )
    export_summary.to_csv(
        root / f"{analysis_dataset}_selected_ADOS_ASD_export_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return export_summary


def prepare_behavior_phenotypes(
    abide1_path,
    abide2_path,
    analysis_dataset,
):
    """
    只读取当前分析数据集的行为表型文件。

    ABIDE1 分析：
        只读取 ABIDE1_PHENO，不读取 ABIDE2_PHENO。

    ABIDE2 分析：
        只读取 ABIDE2_PHENO，不读取 ABIDE1_PHENO。
    """
    phenotype_paths = {
        "ABIDE1": abide1_path,
        "ABIDE2": abide2_path,
    }

    if analysis_dataset not in phenotype_paths:
        raise ValueError(f"不支持的数据集: {analysis_dataset}")

    behavior_raw = read_behavior_file(
        phenotype_paths[analysis_dataset],
        analysis_dataset,
    )

    if behavior_raw is None:
        return None, None, None

    # 仅导出当前数据集白名单中的四个 ADOS 原始列。
    export_summary = None
    if EXPORT_SCALE_SPECIFIC_ASD_CSV:
        export_summary = export_selected_ados_asd_csvs(
            behavior_raw,
            SCALE_EXPORT_ROOT,
            analysis_dataset,
        )

    # 行为相关主输出默认只保留 ASD 组。
    behavior_for_harmonize = behavior_raw.copy()
    if BEHAVIOR_ASD_ONLY and "DX_GROUP" in behavior_for_harmonize.columns:
        behavior_for_harmonize["DX_GROUP"] = pd.to_numeric(
            behavior_for_harmonize["DX_GROUP"],
            errors="coerce",
        )
        behavior_for_harmonize = behavior_for_harmonize[
            behavior_for_harmonize["DX_GROUP"] == 1
        ].copy()

    behavior = harmonize_behavior(
        behavior_for_harmonize,
        analysis_dataset,
    )

    return behavior, behavior_raw, export_summary


# ============================================================
# 6) MERGED OUTPUTS AND QC
# ============================================================
def create_long_behavior_table(df_valid, behavior):
    if behavior is None:
        return df_valid.copy()
    return df_valid.merge(behavior, on="sub_id", how="left")


def add_merged_z_scores(merged):
    merged = merged.copy()

    # 在合并后的被试表中重新计算主要脑指标 z 分数。
    for col in PRIMARY_BRAIN_METRICS:
        if col in merged.columns:
            merged[col + "_z_merged"] = zscore(merged[col])

    return merged


def behavior_counts_table(df, variables):
    rows = []
    if df is None or df.empty:
        return pd.DataFrame(columns=["variable", "N_total", "N_ASD", "N_HC"])

    group_col = "Group" if "Group" in df.columns else "Group_behavior"

    for var in variables:
        if var not in df.columns:
            rows.append({"variable": var, "N_total": 0, "N_ASD": 0, "N_HC": 0})
            continue
        valid = df[df[var].notna()]
        rows.append({
            "variable": var,
            "N_total": int(valid.shape[0]),
            "N_ASD": int((valid[group_col] == "ASD").sum()) if group_col in valid.columns else np.nan,
            "N_HC": int((valid[group_col] == "HC").sum()) if group_col in valid.columns else np.nan,
        })
    return pd.DataFrame(rows)


def write_qc_report(path, df_long, df_valid, bad_system, brain, behavior, merged_subject, merged_long, analysis_dataset):
    lines = []
    lines.append("Result 4 behavior-related output QC report")
    lines.append(f"Generated at: {datetime.now()}")
    lines.append("")

    lines.append("Input:")
    lines.append(f"  Analysis dataset: {analysis_dataset}")
    lines.append(f"  Result 1 long table: {INPUT_RESULT1_LONG}")
    active_pheno = ABIDE1_PHENO if analysis_dataset == "ABIDE1" else ABIDE2_PHENO
    lines.append(
        f"  Active phenotype file: "
        f"{active_pheno if os.path.exists(active_pheno) else 'NOT FOUND'}"
    )
    lines.append(f"  Behavior ASD-only mode: {BEHAVIOR_ASD_ONLY}")
    lines.append(f"  Selected ADOS ASD CSV export: {EXPORT_SCALE_SPECIFIC_ASD_CSV}")
    lines.append(f"  Scale-specific export root: {SCALE_EXPORT_ROOT}")
    lines.append("")

    lines.append("Brain table audit:")
    lines.append(f"  Raw Result 1 rows: {len(df_long)}")
    lines.append(f"  Raw Result 1 subjects: {df_long['sub_id'].nunique()}")
    lines.append(f"  Valid 4-system rows: {len(df_valid)}")
    lines.append(f"  Valid 4-system subjects: {df_valid['sub_id'].nunique()}")
    lines.append(f"  Subjects with system count != 4: {bad_system.shape[0]}")
    if "Group" in df_valid.columns:
        lines.append("  Group counts in valid brain table:")
        for grp, n in df_valid.drop_duplicates("sub_id")["Group"].value_counts(dropna=False).items():
            lines.append(f"    {grp}: {int(n)}")
    lines.append("")

    lines.append("Behavior table audit:")
    if behavior is None:
        lines.append("  No behavior phenotype files found. Behavior outputs were not merged.")
    else:
        lines.append(f"  Behavior subjects: {behavior['sub_id'].nunique()}")
        if "Dataset" in behavior.columns:
            lines.append("  Dataset counts:")
            for ds, n in behavior["Dataset"].value_counts(dropna=False).items():
                lines.append(f"    {ds}: {int(n)}")
        lines.append("  Behavior non-missing counts:")
        counts = behavior_counts_table(behavior, PRIMARY_BEHAVIOR_METRICS)
        for _, r in counts.iterrows():
            lines.append(
                f"    {r['variable']}: N_total={int(r['N_total'])}, "
                f"N_ASD={int(r['N_ASD']) if pd.notna(r['N_ASD']) else 'NA'}, "
                f"N_HC={int(r['N_HC']) if pd.notna(r['N_HC']) else 'NA'}"
            )
    lines.append("")

    lines.append("Merged output audit:")
    lines.append(f"  Subject-level brain rows: {brain.shape[0]}")
    lines.append(f"  Subject-level merged rows: {merged_subject.shape[0]}")
    lines.append(f"  Long-format merged rows: {merged_long.shape[0]}")
    if behavior is not None:
        n_with_any_behavior = int(merged_subject[[c for c in PRIMARY_BEHAVIOR_METRICS if c in merged_subject.columns]].notna().any(axis=1).sum())
        lines.append(f"  Merged subjects with at least one main behavior variable: {n_with_any_behavior}")
    lines.append("")

    lines.append("Key output variables:")
    lines.append("  Brain metrics:")
    for col in PRIMARY_BRAIN_METRICS:
        lines.append(f"    - {col}")
    lines.append("  Behavior metrics:")
    for col in PRIMARY_BEHAVIOR_METRICS:
        lines.append(f"    - {col}")
    lines.append("")

    lines.append("Recommended model examples:")
    lines.append("  ASD subject-level association:")
    lines.append("    ADOS_total_z ~ Coupling_index_scaled_z + Age + Sex_for_model + FIQ + MeanFD + C(Site)")
    lines.append("  ASD long-format behavior modulation GEE:")
    lines.append("    early_slope_scaled ~ G_star * ADOS_total_z + C(system) + Age + Sex_for_model + FIQ + MeanFD + C(Site)")
    lines.append("")

    lines.append("Outputs:")
    lines.append(f"  {os.path.join(OUT_DIR, 'result4_brain_subject_metrics_for_behavior.csv')}")
    lines.append(f"  {os.path.join(OUT_DIR, 'result4_behavior_phenotypes_cleaned.csv')}")
    lines.append(f"  {os.path.join(OUT_DIR, 'result4_brain_behavior_subject_level.csv')}")
    lines.append(f"  {os.path.join(OUT_DIR, 'result4_brain_behavior_long_for_GEE.csv')}")
    lines.append(f"  {os.path.join(OUT_DIR, 'result4_behavior_variable_counts.csv')}")

    write_text(path, "\n".join(lines))


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    analysis_dataset = resolve_analysis_dataset()
    print("Analysis dataset:", analysis_dataset)

    # 1. Brain metrics from Result 1 long table.
    df_long = load_result1_long(INPUT_RESULT1_LONG)
    brain, df_valid, bad_system = create_brain_subject_metrics(df_long)

    brain_out = os.path.join(OUT_DIR, "result4_brain_subject_metrics_for_behavior.csv")
    brain.to_csv(brain_out, index=False, encoding="utf-8-sig")

    long_brain_out = os.path.join(OUT_DIR, "result4_brain_long_base_for_behavior.csv")
    df_valid.to_csv(long_brain_out, index=False, encoding="utf-8-sig")

    if not bad_system.empty:
        bad_system.to_csv(
            os.path.join(OUT_DIR, "audit_result4_bad_system_count.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # 2. Behavior phenotypes.
    behavior, behavior_raw, scale_export_summary = prepare_behavior_phenotypes(
        ABIDE1_PHENO,
        ABIDE2_PHENO,
        analysis_dataset,
    )
    if behavior is not None:
        behavior_out = os.path.join(OUT_DIR, "result4_behavior_phenotypes_cleaned.csv")
        behavior.to_csv(behavior_out, index=False, encoding="utf-8-sig")
        if behavior_raw is not None:
            selected_raw = selected_raw_behavior_output(behavior_raw, analysis_dataset)
            selected_raw.to_csv(
                os.path.join(
                    OUT_DIR,
                    "result4_behavior_phenotypes_raw_merged.csv",
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if scale_export_summary is not None:
            scale_export_summary.to_csv(
                os.path.join(OUT_DIR, "result4_scale_specific_ASD_export_summary.csv"),
                index=False,
                encoding="utf-8-sig"
            )
    else:
        behavior_out = None

    # 3. Subject-level merged table.
    if behavior is not None:
        merged_subject = brain.merge(behavior, on="sub_id", how="left")
    else:
        merged_subject = brain.copy()

    merged_subject = add_merged_z_scores(merged_subject)
    merged_subject_out = os.path.join(OUT_DIR, "result4_brain_behavior_subject_level.csv")
    merged_subject.to_csv(merged_subject_out, index=False, encoding="utf-8-sig")

    # 4. Long-format merged table for GEE behavior modulation.
    merged_long = create_long_behavior_table(df_valid, behavior)
    merged_long = merged_long.merge(
        brain[["sub_id", "Propagation_imbalance_scaled", "Coupling_index_scaled"]],
        on="sub_id",
        how="left"
    )
    merged_long_out = os.path.join(OUT_DIR, "result4_brain_behavior_long_for_GEE.csv")
    merged_long.to_csv(merged_long_out, index=False, encoding="utf-8-sig")

    # 5. Variable counts.
    if behavior is not None:
        counts = behavior_counts_table(merged_subject, PRIMARY_BEHAVIOR_METRICS)
    else:
        counts = pd.DataFrame(columns=["variable", "N_total", "N_ASD", "N_HC"])
    counts_out = os.path.join(OUT_DIR, "result4_behavior_variable_counts.csv")
    counts.to_csv(counts_out, index=False, encoding="utf-8-sig")

    # 6. Dictionary.
    dictionary_rows = [
        {
            "variable": "Coupling_index_scaled",
            "description": (
                "Subject-level slope from early_slope_scaled ~ G_star "
                "across H1-H4."
            ),
        },
        {
            "variable": "Propagation_imbalance_scaled",
            "description": (
                "H1_sensory early_slope_scaled minus mean(H3_control, "
                "H4_DMN) early_slope_scaled."
            ),
        },
        {
            "variable": "ADOS_communication",
            "description": (
                "ABIDE1 ADOS_COMM or ABIDE2 ADOS_G_COMM."
            ),
        },
        {
            "variable": "ADOS_social",
            "description": (
                "ABIDE1 ADOS_SOCIAL or ABIDE2 ADOS_G_SOCIAL."
            ),
        },
        {
            "variable": "ADOS_stereotyped_behavior",
            "description": (
                "ABIDE1 ADOS_STEREO_BEHAV or "
                "ABIDE2 ADOS_G_STEREO_BEHAV."
            ),
        },
        {
            "variable": "ADOS_total",
            "description": (
                "ABIDE1 ADOS_TOTAL or ABIDE2 ADOS_G_TOTAL."
            ),
        },
    ]
    pd.DataFrame(dictionary_rows).to_csv(
        os.path.join(OUT_DIR, "result4_variable_dictionary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 7. QC report.
    write_qc_report(
        os.path.join(OUT_DIR, "result4_behavior_output_qc_report.txt"),
        df_long=df_long,
        df_valid=df_valid,
        bad_system=bad_system,
        brain=brain,
        behavior=behavior,
        merged_subject=merged_subject,
        merged_long=merged_long,
        analysis_dataset=analysis_dataset,
    )

    print("Result 4 selected-ADOS outputs completed.")
    print("Output directory:", OUT_DIR)
    print("Brain subject metrics:", brain_out)
    if behavior_out:
        print("Cleaned behavior phenotypes:", behavior_out)
    print("Merged subject-level table:", merged_subject_out)
    print("Merged long-format GEE table:", merged_long_out)


if __name__ == "__main__":
    main()
