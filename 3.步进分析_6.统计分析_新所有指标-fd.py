# =====================================================
# EC-SEC 组间统计分析（ASD vs HC）
# 敏感性分析版本：H4_minus_H1 contrast + MeanFD
#
# 传播指标 CSV：
#   EC_SEC_metrics_ABIDE*_combat.csv
#
# 协变量 CSV：
#   subject_info_for_stats.csv
#
# 主分析：
#   H4_minus_H1_early_slope_1_10
#     = H4_DMN_early_slope_1_10 - H1_sensory_early_slope_1_10
#
# 主模型：
#   H4_minus_H1 ~ Group + Age + Sex + FIQ + MeanFD
#
# 不控制 Site。
#
# 补充分析：
#   1) H1-H4 单指标 early_slope_1_10 组间 GLM
#   2) temporal_centroid / peak_step / auc 探索性 GLM
#   3) FDR:
#        - supplementary_early_slope_H1H2H3H4_all
#        - exploratory_temporal_centroid_H1H2H3H4
#        - exploratory_peak_step_H1H2H3H4
#        - exploratory_auc_H1H2H3H4
#
# 注意：
#   这是 withFD 敏感性分析代码。
#   主结果是 H4-H1 contrast with FD。
#   单指标结果是 supplementary with FD。
# =====================================================


import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from datetime import datetime


# =====================================================
# 1) 路径设置
# =====================================================
INPUT_CSV = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1_combat\EC_SEC_metrics_all_subjects_combat.csv"

SUBJECT_INFO_CSV = r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"

OUT_DIR = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果4-fd\GroupStats_H4minusH1_contrast_withFD_noSite"
os.makedirs(OUT_DIR, exist_ok=True)


# 主分析：H4-H1 contrast with FD
OUT_CONTRAST_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_withFD.csv"
)

OUT_CONTRAST_DATA_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_withFD_data_used.csv"
)

OUT_CONTRAST_TXT = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_withFD_summary.txt"
)


# 补充分析：单指标 GLM with FD
OUT_SINGLE_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_withFD.csv"
)

OUT_SINGLE_FDR_LONG_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_withFD_FDR_long.csv"
)

OUT_SINGLE_TXT = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_withFD_summary.txt"
)


# =====================================================
# 2) 分析参数
# =====================================================

# True：只做 early_slope_1_10 的补充单指标分析
# False：同时做 temporal_centroid / peak / auc 等探索性指标
ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY = False

# 当 ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY=False 时，
# 是否要求至少存在 temporal_centroid 指标
REQUIRE_TEMPORAL_CENTROID_WHEN_EXPLORATORY = True


# =====================================================
# 3) 工具函数
# =====================================================
def normalize_sub_id(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    m = re.search(r"Sub0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub-Sub{int(m.group(1))}"

    if re.fullmatch(r"\d+", s):
        return f"sub-Sub{int(s)}"

    if re.fullmatch(r"sub-Sub\d+", s):
        return s

    return np.nan


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def drop_constant_and_allzero_cols(X: pd.DataFrame) -> pd.DataFrame:
    keep = []

    for c in X.columns:
        v = X[c].to_numpy(dtype=float)

        if np.all(~np.isfinite(v)):
            continue

        if np.nanstd(v) == 0:
            if c.lower() == "intercept":
                keep.append(c)
            continue

        if np.nanmax(np.abs(v)) == 0:
            continue

        keep.append(c)

    return X[keep]


def resolve_metric_name(df_cols, preferred_name, fallback_names):
    if preferred_name in df_cols:
        return preferred_name

    for fn in fallback_names:
        if fn in df_cols:
            return fn

    return None


def resolve_fd_column(df_cols):
    return resolve_metric_name(
        df_cols=df_cols,
        preferred_name="MeanFD",
        fallback_names=[
            "meanFD",
            "Mean_FD",
            "mean_fd",
            "MEAN_FD",
            "FD",
            "fd",
            "MeanFramewiseDisplacement",
            "mean_framewise_displacement",
            "framewise_displacement",
            "FramewiseDisplacement",
        ],
    )


def normalize_group(series):
    s = series.astype(str).str.strip()
    upper = s.str.upper()

    mapped = pd.Series(np.nan, index=series.index, dtype="float")

    mapped[upper.isin([
        "HC",
        "CONTROL",
        "CONTROLS",
        "TDC",
        "TD",
        "0",
        "2",
    ])] = 0

    mapped[upper.isin([
        "ASD",
        "AUTISM",
        "PATIENT",
        "PATIENTS",
        "1",
    ])] = 1

    return mapped


def normalize_sex(series):
    s = series.astype(str).str.strip()
    upper = s.str.upper()

    mapped = pd.Series(np.nan, index=series.index, dtype="float")

    mapped[upper.isin([
        "M",
        "MALE",
        "1",
    ])] = 1

    mapped[upper.isin([
        "F",
        "FEMALE",
        "2",
        "0",
    ])] = 0

    numeric = pd.to_numeric(series, errors="coerce")
    fill_mask = mapped.isna() & numeric.notna()
    mapped[fill_mask] = numeric[fill_mask]

    return mapped


def get_resolved_actual_col(resolved_configs, metric_name):
    hits = [
        spec for spec in resolved_configs
        if spec["Metric"] == metric_name
    ]

    if len(hits) == 0:
        raise KeyError(
            f"未在 METRIC_CONFIGS 中找到指标: {metric_name}"
        )

    actual_col = hits[0]["Actual_column"]

    if actual_col is None:
        raise KeyError(
            f"无法在输入 CSV 中找到指标列: {metric_name}"
        )

    return actual_col


def fit_group_glm_hc3(
    df_model: pd.DataFrame,
    y_col: str,
    covariate_cols: list,
    min_n: int = 50,
):
    """
    对单个 y 进行：
      y ~ Group + covariates

    其中：
      Group_bin: ASD=1, HC=0

    使用：
      OLS + HC3 robust SE
    """

    needed = [y_col, "Group_bin"] + covariate_cols

    base = df_model[needed].copy()
    base = base.dropna()

    if len(base) < min_n:
        raise RuntimeError(
            f"{y_col} 有效样本过少 N={len(base)}"
        )

    X_dict = {
        "Intercept": 1.0,
        "Group": base["Group_bin"].astype(float),
    }

    for c in covariate_cols:
        X_dict[c] = base[c].astype(float)

    X = pd.DataFrame(X_dict)
    X = drop_constant_and_allzero_cols(X).astype(float)

    if "Group" not in X.columns:
        raise RuntimeError(
            f"{y_col} 模型中 Group 列被删除，无法估计组别效应。"
        )

    for c in covariate_cols:
        if c not in X.columns:
            raise RuntimeError(
                f"{y_col} 模型中协变量 {c} 被删除，可能为常量或全零。"
            )

    rank = np.linalg.matrix_rank(
        X.to_numpy(dtype=float)
    )

    full_rank = rank == X.shape[1]

    fit = sm.OLS(
        base[y_col].astype(float).to_numpy(),
        X.to_numpy(dtype=float),
    ).fit(cov_type="HC3")

    idx_group = list(X.columns).index("Group")

    beta = float(fit.params[idx_group])
    se = float(fit.bse[idx_group])
    tval = float(fit.tvalues[idx_group])
    pval = float(fit.pvalues[idx_group])

    ci_low, ci_high = fit.conf_int()[idx_group]

    ystd = float(base[y_col].std())
    gstd = float(base["Group_bin"].std())

    beta_std = (
        beta * (gstd / ystd)
        if (ystd != 0 and np.isfinite(ystd))
        else np.nan
    )

    return {
        "Beta_ASD_minus_HC": beta,
        "SE_HC3": se,
        "t_value": tval,
        "p_value": pval,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "Beta_std": beta_std,
        "N": int(fit.nobs),
        "R2": float(fit.rsquared),
        "Adj_R2": float(fit.rsquared_adj),
        "Full_rank": bool(full_rank),
        "Rank": int(rank),
        "Num_predictors": int(X.shape[1]),
        "fit": fit,
        "data_used": base,
        "X_columns": list(X.columns),
    }


def add_fdr_scope_columns(res_df, scope_name, mask):
    p_col = f"p_FDR__{scope_name}"
    sig_col = f"sig_FDR_0.05__{scope_name}"
    n_col = f"FDR_n__{scope_name}"

    res_df[p_col] = np.nan
    res_df[sig_col] = False
    res_df[n_col] = np.nan

    n_tests = int(mask.sum())

    if n_tests == 0:
        return res_df

    pvals = res_df.loc[mask, "p_value"].astype(float).to_numpy()
    qvals = multipletests(pvals, method="fdr_bh")[1]

    res_df.loc[mask, p_col] = qvals
    res_df.loc[mask, sig_col] = qvals < 0.05
    res_df.loc[mask, n_col] = n_tests

    return res_df


def make_fdr_long_table(res_df, scope_name, scope_label, mask):
    n_tests = int(mask.sum())

    if n_tests == 0:
        return pd.DataFrame()

    tmp = res_df.loc[mask].copy()
    pvals = tmp["p_value"].astype(float).to_numpy()
    qvals = multipletests(pvals, method="fdr_bh")[1]

    tmp["FDR_scope"] = scope_name
    tmp["FDR_scope_label"] = scope_label
    tmp["FDR_n"] = n_tests
    tmp["p_FDR"] = qvals
    tmp["sig_FDR_0.05"] = qvals < 0.05

    keep_cols = [
        "FDR_scope",
        "FDR_scope_label",
        "FDR_n",
        "Metric",
        "Actual_column",
        "Hierarchy",
        "Family",
        "Analysis_Set",
        "Model",
        "FD_column",
        "Beta_ASD_minus_HC",
        "SE_HC3",
        "t_value",
        "p_value",
        "p_FDR",
        "sig_FDR_0.05",
        "ci_low",
        "ci_high",
        "Beta_std",
        "N",
        "R2",
        "Adj_R2",
        "Full_rank",
        "Rank",
        "Num_predictors",
    ]

    keep_cols = [c for c in keep_cols if c in tmp.columns]

    return tmp[keep_cols]


# =====================================================
# 4) 指标定义
# =====================================================
METRIC_CONFIGS = [
    {
        "Metric": "H1_sensory_early_slope_1_10",
        "Hierarchy": "H1",
        "Family": "early_slope_1_10",
        "Analysis_Set": "Supplementary",
        "Fallbacks": [
            "H1_early_slope_1_10",
            "H1_sensory_early_slope",
            "H1_early_slope",
        ],
    },
    {
        "Metric": "H2_attention_early_slope_1_10",
        "Hierarchy": "H2",
        "Family": "early_slope_1_10",
        "Analysis_Set": "Supplementary",
        "Fallbacks": [
            "H2_early_slope_1_10",
            "H2_attention_early_slope",
            "H2_early_slope",
        ],
    },
    {
        "Metric": "H3_control_early_slope_1_10",
        "Hierarchy": "H3",
        "Family": "early_slope_1_10",
        "Analysis_Set": "Supplementary",
        "Fallbacks": [
            "H3_early_slope_1_10",
            "H3_control_early_slope",
            "H3_early_slope",
        ],
    },
    {
        "Metric": "H4_DMN_early_slope_1_10",
        "Hierarchy": "H4",
        "Family": "early_slope_1_10",
        "Analysis_Set": "Supplementary",
        "Fallbacks": [
            "H4_early_slope_1_10",
            "H4_DMN_early_slope",
            "H4_early_slope",
        ],
    },
    {
        "Metric": "H1_sensory_temporal_centroid",
        "Hierarchy": "H1",
        "Family": "temporal_centroid",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H1_temporal_centroid",
            "H1_sensory_centroid",
            "H1_centroid",
            "H1_sensory_tau",
            "H1_tau",
        ],
    },
    {
        "Metric": "H2_attention_temporal_centroid",
        "Hierarchy": "H2",
        "Family": "temporal_centroid",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H2_temporal_centroid",
            "H2_attention_centroid",
            "H2_centroid",
            "H2_attention_tau",
            "H2_tau",
        ],
    },
    {
        "Metric": "H3_control_temporal_centroid",
        "Hierarchy": "H3",
        "Family": "temporal_centroid",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H3_temporal_centroid",
            "H3_control_centroid",
            "H3_centroid",
            "H3_control_tau",
            "H3_tau",
        ],
    },
    {
        "Metric": "H4_DMN_temporal_centroid",
        "Hierarchy": "H4",
        "Family": "temporal_centroid",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H4_temporal_centroid",
            "H4_DMN_centroid",
            "H4_centroid",
            "H4_DMN_tau",
            "H4_tau",
        ],
    },
    {
        "Metric": "H1_sensory_peak",
        "Hierarchy": "H1",
        "Family": "peak_step",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H1_peak",
            "H1_sensory_peak_step",
            "H1_peak_step",
        ],
    },
    {
        "Metric": "H2_attention_peak",
        "Hierarchy": "H2",
        "Family": "peak_step",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H2_peak",
            "H2_attention_peak_step",
            "H2_peak_step",
        ],
    },
    {
        "Metric": "H3_control_peak",
        "Hierarchy": "H3",
        "Family": "peak_step",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H3_peak",
            "H3_control_peak_step",
            "H3_peak_step",
        ],
    },
    {
        "Metric": "H4_DMN_peak",
        "Hierarchy": "H4",
        "Family": "peak_step",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H4_peak",
            "H4_DMN_peak_step",
            "H4_peak_step",
        ],
    },
    {
        "Metric": "H1_sensory_auc",
        "Hierarchy": "H1",
        "Family": "auc",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H1_auc",
            "H1_sensory_AUC",
            "H1_AUC",
        ],
    },
    {
        "Metric": "H2_attention_auc",
        "Hierarchy": "H2",
        "Family": "auc",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H2_auc",
            "H2_attention_AUC",
            "H2_AUC",
        ],
    },
    {
        "Metric": "H3_control_auc",
        "Hierarchy": "H3",
        "Family": "auc",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H3_auc",
            "H3_control_AUC",
            "H3_AUC",
        ],
    },
    {
        "Metric": "H4_DMN_auc",
        "Hierarchy": "H4",
        "Family": "auc",
        "Analysis_Set": "Exploratory",
        "Fallbacks": [
            "H4_auc",
            "H4_DMN_AUC",
            "H4_AUC",
        ],
    },
]


# =====================================================
# 5) 补充分析 FDR 校正簇
# =====================================================
SUPPLEMENTARY_FDR_SCOPES = [
    {
        "scope_name": "supplementary_early_slope_H1H2H3H4_all",
        "scope_label": "Supplementary full hierarchy: H1-H4 early_slope_1_10",
        "metrics": [
            "H1_sensory_early_slope_1_10",
            "H2_attention_early_slope_1_10",
            "H3_control_early_slope_1_10",
            "H4_DMN_early_slope_1_10",
        ],
    },
]


EXPLORATORY_FDR_FAMILIES = [
    "temporal_centroid",
    "peak_step",
    "auc",
]


# =====================================================
# 6) 读取传播指标 CSV
# =====================================================
df = pd.read_csv(INPUT_CSV)

if "sub_id" not in df.columns:
    if "SUB_ID_norm" in df.columns:
        df["sub_id"] = df["SUB_ID_norm"].apply(normalize_sub_id)
    elif "SUB_ID" in df.columns:
        df["sub_id"] = df["SUB_ID"].apply(normalize_sub_id)
    else:
        raise KeyError(
            "当前传播指标 CSV 缺少 sub_id / SUB_ID_norm / SUB_ID，无法识别被试 ID。"
        )
else:
    df["sub_id"] = df["sub_id"].apply(normalize_sub_id)

n_input = len(df)
n_bad_id = int(df["sub_id"].isna().sum())

df = df.dropna(subset=["sub_id"]).copy()


# =====================================================
# 7) 读取 subject_info_for_stats.csv 并合并 MeanFD
# =====================================================
info = pd.read_csv(SUBJECT_INFO_CSV)

if "sub_id" not in info.columns:
    if "SUB_ID_norm" in info.columns:
        info["sub_id"] = info["SUB_ID_norm"].apply(normalize_sub_id)
    elif "SUB_ID" in info.columns:
        info["sub_id"] = info["SUB_ID"].apply(normalize_sub_id)
    else:
        raise KeyError(
            "subject_info_for_stats.csv 缺少 sub_id / SUB_ID_norm / SUB_ID，无法识别被试 ID。"
        )
else:
    info["sub_id"] = info["sub_id"].apply(normalize_sub_id)

info = info.dropna(subset=["sub_id"]).copy()
info = info.drop_duplicates(subset=["sub_id"], keep="first").copy()

info_required_cols = [
    "sub_id",
    "Group",
    "Age",
    "Sex",
    "Site",
    "FIQ",
    "MeanFD",
]

missing_info_cols = [
    c for c in info_required_cols
    if c not in info.columns
]

if len(missing_info_cols) > 0:
    raise KeyError(
        f"subject_info_for_stats.csv 缺少必要列: {missing_info_cols}"
    )

info_keep = info[info_required_cols].copy()

n_info_subjects = len(info_keep)

df = df.merge(
    info_keep,
    on="sub_id",
    how="left",
    suffixes=("", "_info"),
)

# 如果传播指标 CSV 中已有协变量，则优先使用原列；
# 如果原列缺失，则用 subject_info_for_stats.csv 中的对应列补齐。
for c in [
    "Group",
    "Age",
    "Sex",
    "Site",
    "FIQ",
    "MeanFD",
]:
    info_c = f"{c}_info"

    if c not in df.columns and info_c in df.columns:
        df[c] = df[info_c]

    elif c in df.columns and info_c in df.columns:
        df[c] = df[c].where(df[c].notna(), df[info_c])

drop_info_cols = [
    c for c in df.columns
    if c.endswith("_info")
]

df = df.drop(columns=drop_info_cols)

df_cols = set(df.columns.tolist())

FD_COL = resolve_fd_column(df_cols)

if FD_COL is None:
    raise KeyError(
        "合并 subject_info_for_stats.csv 后仍然缺少 FD / MeanFD 列，"
        "无法执行加入 FD 的敏感性分析。"
    )

required_cols = [
    "Group",
    "Age",
    "Sex",
    "FIQ",
    FD_COL,
]

missing_required = [
    c for c in required_cols
    if c not in df.columns
]

if len(missing_required) > 0:
    raise KeyError(
        f"合并后仍缺少必要协变量列: {missing_required}"
    )

n_missing_fd_after_merge = int(df[FD_COL].isna().sum())

print(f"输入传播指标被试数: {n_input}")
print(f"无法解析 sub_id 并删除: {n_bad_id}")
print(f"subject_info_for_stats.csv 可用被试数: {n_info_subjects}")
print(f"合并后进入分析前被试数: {len(df)}")
print(f"合并的协变量文件: {SUBJECT_INFO_CSV}")
print(f"FD 协变量列: {FD_COL}")
print(f"合并后 FD 缺失数量: {n_missing_fd_after_merge}")
print("Site covariate: not included")


# =====================================================
# 8) 变量编码与 dtype 清理
# =====================================================
df["Group_bin"] = normalize_group(df["Group"])
df["Sex_bin"] = normalize_sex(df["Sex"])
df["Age"] = safe_numeric(df["Age"])
df["FIQ"] = safe_numeric(df["FIQ"])
df["MeanFD_model"] = safe_numeric(df[FD_COL])

n_group_missing = int(df["Group_bin"].isna().sum())
n_sex_missing = int(df["Sex_bin"].isna().sum())
n_fd_missing = int(df["MeanFD_model"].isna().sum())

print(f"Group 无法编码数量: {n_group_missing}")
print(f"Sex 无法编码数量: {n_sex_missing}")
print(f"FD 缺失或无法数值化数量: {n_fd_missing}")


# =====================================================
# 9) 指标列名解析
# =====================================================
df_cols = set(df.columns.tolist())
resolved_metric_configs = []

print("\n[检查] 指标列名解析：")

for spec in METRIC_CONFIGS:
    preferred = spec["Metric"]

    actual = resolve_metric_name(
        df_cols=df_cols,
        preferred_name=preferred,
        fallback_names=spec.get("Fallbacks", []),
    )

    spec = spec.copy()
    spec["Actual_column"] = actual

    if actual is None:
        print(f"  缺失: {preferred}")
    else:
        mark = " fallback" if actual != preferred else ""
        print(f"  {preferred} -> {actual}{mark}")

    resolved_metric_configs.append(spec)


# =====================================================
# 10) 主分析：H4_minus_H1 contrast with FD
# =====================================================
H1_ES_COL = get_resolved_actual_col(
    resolved_metric_configs,
    "H1_sensory_early_slope_1_10",
)

H4_ES_COL = get_resolved_actual_col(
    resolved_metric_configs,
    "H4_DMN_early_slope_1_10",
)

CONTRAST_COL = "H4_minus_H1_early_slope_1_10"

df[CONTRAST_COL] = (
    safe_numeric(df[H4_ES_COL])
    - safe_numeric(df[H1_ES_COL])
)

print("\n[主分析] H4-H1 contrast with FD:")
print(f"  H1 column: {H1_ES_COL}")
print(f"  H4 column: {H4_ES_COL}")
print(f"  Contrast: {CONTRAST_COL} = {H4_ES_COL} - {H1_ES_COL}")


covariate_cols = [
    "Age",
    "Sex_bin",
    "FIQ",
    "MeanFD_model",
]


contrast_fit_result = fit_group_glm_hc3(
    df_model=df,
    y_col=CONTRAST_COL,
    covariate_cols=covariate_cols,
    min_n=50,
)

contrast_data_used = contrast_fit_result["data_used"].copy()
contrast_data_used["sub_id"] = df.loc[contrast_data_used.index, "sub_id"]
contrast_data_used["Group_original"] = df.loc[contrast_data_used.index, "Group"]
contrast_data_used["H1_column_value"] = safe_numeric(
    df.loc[contrast_data_used.index, H1_ES_COL]
)
contrast_data_used["H4_column_value"] = safe_numeric(
    df.loc[contrast_data_used.index, H4_ES_COL]
)
contrast_data_used["FD_column_used"] = FD_COL

contrast_data_used.to_csv(
    OUT_CONTRAST_DATA_CSV,
    index=False,
    encoding="utf-8-sig",
)


contrast_model_label = (
    "H4_minus_H1 ~ Group + Age + Sex_bin + FIQ + MeanFD"
)

contrast_res_df = pd.DataFrame([{
    "Analysis": "Sensitivity_withFD_H4_minus_H1_contrast",
    "Metric": CONTRAST_COL,
    "Definition": f"{H4_ES_COL} - {H1_ES_COL}",
    "H1_column": H1_ES_COL,
    "H4_column": H4_ES_COL,
    "Model": contrast_model_label,
    "FD_column": FD_COL,
    "Beta_ASD_minus_HC": contrast_fit_result["Beta_ASD_minus_HC"],
    "SE_HC3": contrast_fit_result["SE_HC3"],
    "t_value": contrast_fit_result["t_value"],
    "p_value": contrast_fit_result["p_value"],
    "ci_low": contrast_fit_result["ci_low"],
    "ci_high": contrast_fit_result["ci_high"],
    "Beta_std": contrast_fit_result["Beta_std"],
    "N": contrast_fit_result["N"],
    "R2": contrast_fit_result["R2"],
    "Adj_R2": contrast_fit_result["Adj_R2"],
    "Full_rank": contrast_fit_result["Full_rank"],
    "Rank": contrast_fit_result["Rank"],
    "Num_predictors": contrast_fit_result["Num_predictors"],
    "Covariates": "Age, Sex_bin, FIQ, MeanFD",
    "Site": "not included",
}])

contrast_res_df.to_csv(
    OUT_CONTRAST_CSV,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# 11) 主分析 summary
# =====================================================
beta_contrast = contrast_fit_result["Beta_ASD_minus_HC"]
se_contrast = contrast_fit_result["SE_HC3"]
t_contrast = contrast_fit_result["t_value"]
p_contrast = contrast_fit_result["p_value"]
ci_low_contrast = contrast_fit_result["ci_low"]
ci_high_contrast = contrast_fit_result["ci_high"]

with open(OUT_CONTRAST_TXT, "w", encoding="utf-8") as f:
    f.write("EC-SEC Sensitivity Analysis: H4-H1 Contrast with FD\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input files:\n")
    f.write(f"  INPUT_CSV: {INPUT_CSV}\n")
    f.write(f"  SUBJECT_INFO_CSV: {SUBJECT_INFO_CSV}\n\n")

    f.write("Rationale:\n")
    f.write(
        "The initial distance-based finding reflected an altered "
        "Default-VisSomMot relationship, rather than isolated group "
        "differences in single-network gradient values. Therefore, the "
        "primary EC-SEC follow-up tests the relative contrast between "
        "H4 DMN and H1 sensory propagation metrics. This script repeats "
        "that contrast model with MeanFD included as a sensitivity covariate.\n\n"
    )

    f.write("Contrast definition:\n")
    f.write(f"  {CONTRAST_COL} = {H4_ES_COL} - {H1_ES_COL}\n\n")

    f.write("Model specification:\n")
    f.write(f"  Model: {contrast_model_label}\n")
    f.write("  DV: H4_minus_H1 early_slope_1_10 contrast\n")
    f.write("  IV of interest: Group_bin, coded ASD=1 and HC=0\n")
    f.write("  Covariates: Age, Sex_bin, FIQ, MeanFD\n")
    f.write(f"  FD column used: {FD_COL}\n")
    f.write("  Site covariate: not included\n")
    f.write("  Estimator: OLS with HC3 heteroscedasticity-consistent robust standard errors\n\n")

    f.write("ID and coding QC:\n")
    f.write(f"  Input metric rows: {n_input}\n")
    f.write(f"  Unparsable sub_id rows dropped from metric CSV: {n_bad_id}\n")
    f.write(f"  subject_info available subjects: {n_info_subjects}\n")
    f.write(f"  Rows after merge: {len(df)}\n")
    f.write(f"  Group coding missing: {n_group_missing}\n")
    f.write(f"  Sex coding missing: {n_sex_missing}\n")
    f.write(f"  FD missing or non-numeric rows: {n_fd_missing}\n\n")

    f.write("Result:\n")
    f.write(
        f"  beta_ASD_minus_HC = {beta_contrast:.6g}\n"
        f"  SE_HC3 = {se_contrast:.6g}\n"
        f"  t = {t_contrast:.3f}\n"
        f"  p = {p_contrast:.6g}\n"
        f"  95% CI = [{ci_low_contrast:.6g}, {ci_high_contrast:.6g}]\n"
        f"  N = {contrast_fit_result['N']}\n"
        f"  R2 = {contrast_fit_result['R2']:.4f}\n"
        f"  Adj_R2 = {contrast_fit_result['Adj_R2']:.4f}\n"
        f"  Full rank = {contrast_fit_result['Full_rank']}\n"
    )

    f.write("\nInterpretation:\n")
    f.write("  The contrast is defined as H4_DMN minus H1_sensory.\n")
    f.write(
        "  Beta_ASD_minus_HC represents the adjusted group difference "
        "in this H4-H1 contrast after additionally controlling for MeanFD, i.e., "
        "[(H4-H1)_ASD] - [(H4-H1)_HC].\n"
    )

    if beta_contrast > 0:
        f.write(
            "  Beta > 0 means ASD shows a larger H4-H1 contrast than HC "
            "after adjustment for Age, Sex, FIQ, and MeanFD.\n"
        )
    elif beta_contrast < 0:
        f.write(
            "  Beta < 0 means ASD shows a smaller H4-H1 contrast than HC "
            "after adjustment for Age, Sex, FIQ, and MeanFD.\n"
        )
    else:
        f.write(
            "  Beta = 0 means no adjusted ASD-HC difference in the H4-H1 contrast.\n"
        )

    f.write("\nOutput:\n")
    f.write(f"  Contrast CSV: {OUT_CONTRAST_CSV}\n")
    f.write(f"  Contrast data used CSV: {OUT_CONTRAST_DATA_CSV}\n")
    f.write(f"  Contrast summary TXT: {OUT_CONTRAST_TXT}\n")


print("\n=== 敏感性主分析完成：H4-H1 contrast with FD ===")
print(contrast_res_df)
print("H4-H1 contrast with FD 结果已保存到:")
print(OUT_CONTRAST_CSV)
print(OUT_CONTRAST_TXT)


# =====================================================
# 12) 补充分析：单指标 GLM with FD
# =====================================================
if ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY:
    METRICS_TO_RUN = [
        spec for spec in resolved_metric_configs
        if spec["Family"] == "early_slope_1_10"
        and spec["Actual_column"] is not None
    ]
else:
    METRICS_TO_RUN = [
        spec for spec in resolved_metric_configs
        if spec["Actual_column"] is not None
    ]


if (
    not ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY
    and REQUIRE_TEMPORAL_CENTROID_WHEN_EXPLORATORY
):
    n_temporal_centroid = sum(
        (spec["Family"] == "temporal_centroid")
        and (spec["Actual_column"] is not None)
        for spec in resolved_metric_configs
    )

    if n_temporal_centroid == 0:
        raise RuntimeError(
            "未找到任何 temporal_centroid 指标列。当前 CSV 应包含 H1-H4 的 temporal_centroid 列。"
        )


if len(METRICS_TO_RUN) == 0:
    raise RuntimeError(
        "没有找到任何可分析的 EC-SEC 单指标列。"
    )


print("\n[补充分析] 实际纳入单指标 GLM with FD 的指标：")

for spec in METRICS_TO_RUN:
    print(
        f"  {spec['Metric']} | "
        f"actual={spec['Actual_column']} | "
        f"Hierarchy={spec['Hierarchy']} | "
        f"Family={spec['Family']} | "
        f"Analysis_Set={spec['Analysis_Set']}"
    )


single_results = []

for spec in METRICS_TO_RUN:
    metric = spec["Metric"]
    actual_col = spec["Actual_column"]
    hierarchy = spec["Hierarchy"]
    family = spec["Family"]
    analysis_set = spec["Analysis_Set"]

    y_col_tmp = f"__y__{metric}"
    df[y_col_tmp] = safe_numeric(df[actual_col])

    try:
        fit_res = fit_group_glm_hc3(
            df_model=df,
            y_col=y_col_tmp,
            covariate_cols=covariate_cols,
            min_n=50,
        )

    except RuntimeError as e:
        print(f"跳过 {metric}: {e}")
        continue

    single_results.append({
        "Analysis": "Sensitivity_withFD_single_metric_GLM",
        "Metric": metric,
        "Actual_column": actual_col,
        "Hierarchy": hierarchy,
        "Family": family,
        "Analysis_Set": analysis_set,
        "Model": f"{metric} ~ Group + Age + Sex_bin + FIQ + MeanFD",
        "FD_column": FD_COL,
        "Beta_ASD_minus_HC": fit_res["Beta_ASD_minus_HC"],
        "SE_HC3": fit_res["SE_HC3"],
        "t_value": fit_res["t_value"],
        "p_value": fit_res["p_value"],
        "ci_low": fit_res["ci_low"],
        "ci_high": fit_res["ci_high"],
        "Beta_std": fit_res["Beta_std"],
        "N": fit_res["N"],
        "R2": fit_res["R2"],
        "Adj_R2": fit_res["Adj_R2"],
        "Full_rank": fit_res["Full_rank"],
        "Rank": fit_res["Rank"],
        "Num_predictors": fit_res["Num_predictors"],
        "Covariates": "Age, Sex_bin, FIQ, MeanFD",
        "Site": "not included",
    })


res_df = pd.DataFrame(single_results)

if len(res_df) == 0:
    raise RuntimeError(
        "没有任何单指标成功完成回归。请检查指标列名、FD 列与协变量缺失情况。"
    )


# =====================================================
# 13) 补充分析 FDR
# =====================================================
fdr_long_tables = []

for scope in SUPPLEMENTARY_FDR_SCOPES:
    scope_name = scope["scope_name"]
    scope_label = scope["scope_label"]
    metrics = scope["metrics"]

    mask = res_df["Metric"].isin(metrics)

    res_df = add_fdr_scope_columns(
        res_df=res_df,
        scope_name=scope_name,
        mask=mask,
    )

    tmp_long = make_fdr_long_table(
        res_df=res_df,
        scope_name=scope_name,
        scope_label=scope_label,
        mask=mask,
    )

    if len(tmp_long) > 0:
        fdr_long_tables.append(tmp_long)


for fam in EXPLORATORY_FDR_FAMILIES:
    scope_name = f"exploratory_{fam}_H1H2H3H4"
    scope_label = f"Exploratory family: {fam} across H1-H4"

    mask = res_df["Family"].eq(fam)

    res_df = add_fdr_scope_columns(
        res_df=res_df,
        scope_name=scope_name,
        mask=mask,
    )

    tmp_long = make_fdr_long_table(
        res_df=res_df,
        scope_name=scope_name,
        scope_label=scope_label,
        mask=mask,
    )

    if len(tmp_long) > 0:
        fdr_long_tables.append(tmp_long)


if len(fdr_long_tables) > 0:
    fdr_long_df = pd.concat(
        fdr_long_tables,
        axis=0,
        ignore_index=True,
    )
else:
    fdr_long_df = pd.DataFrame()


# =====================================================
# 14) 排序并保存补充分析结果
# =====================================================
analysis_order = {
    "Supplementary": 0,
    "Exploratory": 1,
}

hierarchy_order = {
    "H1": 1,
    "H2": 2,
    "H3": 3,
    "H4": 4,
}

family_order = {
    "early_slope_1_10": 0,
    "temporal_centroid": 1,
    "peak_step": 2,
    "auc": 3,
}

res_df["_Analysis_order"] = (
    res_df["Analysis_Set"]
    .map(analysis_order)
    .fillna(99)
)

res_df["_Family_order"] = (
    res_df["Family"]
    .map(family_order)
    .fillna(99)
)

res_df["_Hierarchy_order"] = (
    res_df["Hierarchy"]
    .map(hierarchy_order)
    .fillna(99)
)

res_df = (
    res_df
    .sort_values(
        [
            "_Analysis_order",
            "_Family_order",
            "_Hierarchy_order",
            "p_value",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )
    .drop(
        columns=[
            "_Analysis_order",
            "_Family_order",
            "_Hierarchy_order",
        ]
    )
    .reset_index(drop=True)
)


if len(fdr_long_df) > 0:
    fdr_long_df["_Scope_order"] = fdr_long_df["FDR_scope"].map({
        "supplementary_early_slope_H1H2H3H4_all": 0,
        "exploratory_temporal_centroid_H1H2H3H4": 1,
        "exploratory_peak_step_H1H2H3H4": 2,
        "exploratory_auc_H1H2H3H4": 3,
    }).fillna(99)

    fdr_long_df["_Hierarchy_order"] = (
        fdr_long_df["Hierarchy"]
        .map(hierarchy_order)
        .fillna(99)
    )

    fdr_long_df = (
        fdr_long_df
        .sort_values(
            [
                "_Scope_order",
                "_Hierarchy_order",
                "p_value",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .drop(
            columns=[
                "_Scope_order",
                "_Hierarchy_order",
            ]
        )
        .reset_index(drop=True)
    )


res_df.to_csv(
    OUT_SINGLE_CSV,
    index=False,
    encoding="utf-8-sig",
)

fdr_long_df.to_csv(
    OUT_SINGLE_FDR_LONG_CSV,
    index=False,
    encoding="utf-8-sig",
)


print("\n=== 补充敏感性分析完成：单指标 GLM with FD + FDR ===")
print(res_df)

print("\n补充单指标 with FD 结果已保存到:")
print(OUT_SINGLE_CSV)

print("\n补充单指标 with FD 长格式 FDR 结果已保存到:")
print(OUT_SINGLE_FDR_LONG_CSV)


# =====================================================
# 15) 补充分析 summary
# =====================================================
with open(OUT_SINGLE_TXT, "w", encoding="utf-8") as f:
    f.write("EC-SEC Supplementary Single-Metric GLM Sensitivity Report with FD\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input files:\n")
    f.write(f"  INPUT_CSV: {INPUT_CSV}\n")
    f.write(f"  SUBJECT_INFO_CSV: {SUBJECT_INFO_CSV}\n\n")

    f.write("Current data format:\n")
    f.write("  EC-SEC metrics were read from INPUT_CSV.\n")
    f.write("  Group, Age, Sex, FIQ, Site, and MeanFD were merged from subject_info_for_stats.csv using sub_id.\n\n")

    f.write("Relationship to primary analysis:\n")
    f.write(
        "  The sensitivity primary EC-SEC analysis is the H4-H1 contrast model with FD. "
        "The analyses in this file are supplementary single-metric GLMs "
        "for H1-H4 and exploratory propagation metrics with FD adjustment.\n\n"
    )

    f.write("ID and coding QC:\n")
    f.write(f"  Input metric rows: {n_input}\n")
    f.write(f"  Unparsable sub_id rows dropped from metric CSV: {n_bad_id}\n")
    f.write(f"  subject_info available subjects: {n_info_subjects}\n")
    f.write(f"  Rows after merge: {len(df)}\n")
    f.write(f"  Group coding missing: {n_group_missing}\n")
    f.write(f"  Sex coding missing: {n_sex_missing}\n")
    f.write(f"  FD column used: {FD_COL}\n")
    f.write(f"  FD missing or non-numeric rows: {n_fd_missing}\n\n")

    f.write("Model specification:\n")
    f.write("  DV: each EC-SEC propagation metric, fitted in a separate model\n")
    f.write("  IV of interest: Group_bin, coded ASD=1 and HC=0\n")
    f.write("  Covariates: Age, Sex_bin, FIQ, MeanFD\n")
    f.write("  Site covariate: not included\n")
    f.write("  Estimator: OLS with HC3 heteroscedasticity-consistent robust standard errors\n")
    f.write("  Inference target: diagnostic-group coefficient for each propagation metric after FD adjustment\n\n")

    f.write("Supplementary FDR strategy:\n")
    f.write(
        "  early_slope_1_10: corrected across H1, H2, H3, and H4 as a full hierarchy family.\n"
    )
    f.write(
        "  temporal_centroid: corrected across available H1-H4 temporal_centroid metrics.\n"
    )
    f.write(
        "  peak_step: corrected across available H1-H4 peak metrics.\n"
    )
    f.write(
        "  auc: corrected across available H1-H4 auc metrics.\n\n"
    )

    f.write("Metric column resolution:\n")
    for spec in resolved_metric_configs:
        f.write(
            f"  - {spec['Metric']} -> {spec['Actual_column']} | "
            f"Hierarchy={spec['Hierarchy']} | "
            f"Family={spec['Family']} | "
            f"Analysis_Set={spec['Analysis_Set']}\n"
        )
    f.write("\n")

    f.write("FDR scopes actually used:\n")
    if len(fdr_long_df) == 0:
        f.write("  None\n")
    else:
        for scope_name in fdr_long_df["FDR_scope"].dropna().unique():
            sub = fdr_long_df[fdr_long_df["FDR_scope"] == scope_name]
            scope_label = sub["FDR_scope_label"].iloc[0]
            fdr_n = int(sub["FDR_n"].iloc[0])
            f.write(f"  - {scope_name}: {scope_label}; n_tests={fdr_n}\n")
    f.write("\n")

    f.write("Single-metric regression results with FD adjustment:\n")
    f.write(
        "  Note: Beta_ASD_minus_HC > 0 means ASD > HC after adjustment for Age, Sex, FIQ, and MeanFD.\n\n"
    )

    for _, row in res_df.iterrows():
        f.write(
            f"[{row['Analysis_Set']} | {row['Family']} | {row['Hierarchy']}] "
            f"{row['Metric']} (actual={row['Actual_column']}): "
            f"beta={row['Beta_ASD_minus_HC']:.6g}, "
            f"SE(HC3)={row['SE_HC3']:.6g}, "
            f"t={row['t_value']:.3f}, "
            f"p={row['p_value']:.3g}, "
            f"95% CI=[{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"N={int(row['N'])}, "
            f"Adj_R2={row['Adj_R2']:.4f}, "
            f"full_rank={bool(row['Full_rank'])}\n"
        )

    f.write("\n")
    f.write("FDR-corrected results by correction scope:\n\n")

    if len(fdr_long_df) == 0:
        f.write("  None\n")
    else:
        for scope_name in fdr_long_df["FDR_scope"].dropna().unique():
            sub = fdr_long_df[fdr_long_df["FDR_scope"] == scope_name].copy()
            scope_label = sub["FDR_scope_label"].iloc[0]
            fdr_n = int(sub["FDR_n"].iloc[0])

            f.write(f"[{scope_name}]\n")
            f.write(f"  {scope_label}\n")
            f.write(f"  FDR_n={fdr_n}\n")

            for _, row in sub.iterrows():
                direction = (
                    "ASD > HC"
                    if row["Beta_ASD_minus_HC"] > 0
                    else "ASD < HC"
                )

                f.write(
                    f"  - {row['Metric']}: "
                    f"{direction}, "
                    f"beta={row['Beta_ASD_minus_HC']:.6g}, "
                    f"p={row['p_value']:.3g}, "
                    f"q_FDR={row['p_FDR']:.3g}, "
                    f"sig_FDR={bool(row['sig_FDR_0.05'])}, "
                    f"N={int(row['N'])}\n"
                )

            f.write("\n")

    f.write("Significant after FDR q<0.05 by correction scope:\n\n")

    if len(fdr_long_df) == 0:
        f.write("  None\n")
    else:
        any_sig = False

        for scope_name in fdr_long_df["FDR_scope"].dropna().unique():
            sub = fdr_long_df[
                (fdr_long_df["FDR_scope"] == scope_name)
                & (fdr_long_df["sig_FDR_0.05"])
            ]

            if len(sub) == 0:
                continue

            any_sig = True
            f.write(f"[{scope_name}]\n")

            for _, row in sub.iterrows():
                direction = (
                    "ASD > HC"
                    if row["Beta_ASD_minus_HC"] > 0
                    else "ASD < HC"
                )

                f.write(
                    f"  - {row['Metric']}: "
                    f"{direction}, "
                    f"beta={row['Beta_ASD_minus_HC']:.6g}, "
                    f"p={row['p_value']:.3g}, "
                    f"q_FDR={row['p_FDR']:.3g}\n"
                )

            f.write("\n")

        if not any_sig:
            f.write("  None\n")

    f.write("\n")
    f.write("Output:\n")
    f.write(f"  Supplementary single-metric CSV: {OUT_SINGLE_CSV}\n")
    f.write(f"  Supplementary long-format FDR CSV: {OUT_SINGLE_FDR_LONG_CSV}\n")
    f.write(f"  Supplementary summary TXT: {OUT_SINGLE_TXT}\n")


print("补充敏感性分析总结报告已保存到:")
print(OUT_SINGLE_TXT)


# =====================================================
# 16) 最终打印
# =====================================================
print("\n全部 withFD 敏感性分析完成。")
print("=" * 70)
print("Sensitivity H4-H1 contrast with FD:")
print(f"  {OUT_CONTRAST_CSV}")
print(f"  {OUT_CONTRAST_TXT}")
print("")
print("Supplementary single-metric results with FD:")
print(f"  {OUT_SINGLE_CSV}")
print(f"  {OUT_SINGLE_FDR_LONG_CSV}")
print(f"  {OUT_SINGLE_TXT}")