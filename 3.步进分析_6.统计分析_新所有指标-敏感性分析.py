# =====================================================
# EC-SEC 组间统计分析（ASD vs HC）
# 参数敏感性分析版本：H4_minus_H1 contrast
#
# 适配对象：
#   1) L_MAX = 30, 40, 50, 60
#   2) early_slope windows:
#        early_slope_1_5
#        early_slope_1_10
#        early_slope_1_15
#
# 输入 CSV：
#   建议使用前一步拆分后的 ABIDE1 或 ABIDE2 总表：
#   EC_SEC_metrics_all_subjects_sensitivity_combat_combined_by_Lmax_ABIDE2.csv
#
# 主分析：
#   对每个 L_MAX 和每个 early window 分别计算：
#
#   H4_minus_H1_early_slope_X
#     = H4_DMN_early_slope_X - H1_sensory_early_slope_X
#
# 主模型：
#   H4_minus_H1 ~ Group + Age + Sex + FIQ
#
# 不控制 MeanFD。
# 不控制 Site。
#
# 补充分析：
#   1) H1-H4 单指标 early_slope_1_5 / 1_10 / 1_15 组间 GLM
#   2) temporal_centroid / peak_step / auc 探索性 GLM
#   3) FDR:
#        - 每个 L_MAX 内，每个 early window 的 H1-H4 一组 FDR
#        - 每个 L_MAX 内，temporal_centroid / peak_step / auc 分别 across H1-H4 FDR
#
# 输出：
#   1) 主 H4-H1 contrast 结果总表
#   2) 主 H4-H1 contrast FDR 长表
#   3) 主 H4-H1 contrast 每个模型使用的数据
#   4) 单指标补充 / 探索性 GLM 总表
#   5) 单指标 FDR 长表
#   6) 总结报告
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

# ABIDE1 或 ABIDE2 参数敏感性分析的 ComBat 拆分总表
INPUT_CSV = (
    r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1-敏感性分析_comnbat\EC_SEC_metrics_all_subjects_sensitivity_combat_combined_by_Lmax.csv"
)

DATASET_LABEL = "ABIDE1"

OUT_DIR = (
    r"I:\DYF\NPI-4-code\3.步进分析"
    rf"\{DATASET_LABEL}_新结果4_参数敏感性分析"
    r"\GroupStats_H4minusH1_contrast_noFD_noSite"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_PRIMARY_DATA_DIR = os.path.join(
    OUT_DIR,
    "primary_H4minusH1_data_used_by_Lmax_and_window"
)

os.makedirs(OUT_PRIMARY_DATA_DIR, exist_ok=True)


# 主分析：H4-H1 contrast
OUT_CONTRAST_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_sensitivity.csv"
)

OUT_CONTRAST_FDR_LONG_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_sensitivity_FDR_long.csv"
)

OUT_CONTRAST_TXT = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_sensitivity_summary.txt"
)


# 补充分析：单指标 GLM
OUT_SINGLE_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_sensitivity.csv"
)

OUT_SINGLE_FDR_LONG_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_sensitivity_FDR_long.csv"
)

OUT_SINGLE_TXT = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary_sensitivity_summary.txt"
)


# 失败模型报告
OUT_FAILED_MODELS_CSV = os.path.join(
    OUT_DIR,
    "EC_SEC_GroupStats_failed_models_sensitivity.csv"
)


# =====================================================
# 2) 敏感性分析参数
# =====================================================

L_MAX_COL = "L_MAX"
L_MAX_LIST = [30, 40, 50, 60]

EARLY_WINDOWS = [
    (1, 5),
    (1, 10),
    (1, 15),
]

EARLY_WINDOW_LABELS = [
    f"{a}_{b}" for a, b in EARLY_WINDOWS
]

# True：只做 early_slope 的补充单指标分析
# False：同时做 temporal_centroid / peak / auc 等探索性指标
ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY = False

# 当 ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY=False 时，
# 是否要求至少存在 temporal_centroid 指标
REQUIRE_TEMPORAL_CENTROID_WHEN_EXPLORATORY = True

# 主模型最小样本量
MIN_N_GLM = 50


# =====================================================
# 3) 工具函数
# =====================================================

def early_label(start_step, end_step):
    return f"{start_step}_{end_step}"


def normalize_sub_id(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    m = re.search(r"Sub0*(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"sub-Sub{int(m.group(1))}"

    if re.fullmatch(r"\d+", s):
        return f"sub-Sub{int(s)}"

    if re.fullmatch(r"\d+\.0", s):
        return f"sub-Sub{int(float(s))}"

    if re.fullmatch(r"sub-Sub\d+", s):
        return s

    m2 = re.search(r"\d+", s)
    if m2:
        return f"sub-Sub{int(m2.group(0))}"

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

    X = pd.DataFrame(X_dict, index=base.index)
    X = drop_constant_and_allzero_cols(X).astype(float)

    if "Group" not in X.columns:
        raise RuntimeError(
            f"{y_col} 模型中 Group 列被删除，无法估计组别效应。"
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


def add_fdr_column_for_mask(
    res_df,
    mask,
    p_col,
    sig_col,
    n_col,
):
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
        L_MAX_COL,
        "Early_Window",
        "Metric",
        "Actual_column",
        "Hierarchy",
        "Family",
        "Analysis_Set",
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

def build_metric_configs():
    configs = []

    hierarchy_specs = [
        ("H1", "H1_sensory", "sensory"),
        ("H2", "H2_attention", "attention"),
        ("H3", "H3_control", "control"),
        ("H4", "H4_DMN", "DMN"),
    ]

    # early_slope 敏感性窗口
    for wlab in EARLY_WINDOW_LABELS:
        for hierarchy, prefix, short_name in hierarchy_specs:
            configs.append({
                "Metric": f"{prefix}_early_slope_{wlab}",
                "Hierarchy": hierarchy,
                "Family": f"early_slope_{wlab}",
                "Early_Window": wlab,
                "Analysis_Set": "Supplementary",
                "Fallbacks": [
                    f"{hierarchy}_early_slope_{wlab}",
                    f"{prefix}_early_slope",
                    f"{hierarchy}_early_slope",
                ],
            })

    # temporal centroid
    for hierarchy, prefix, short_name in hierarchy_specs:
        configs.append({
            "Metric": f"{prefix}_temporal_centroid",
            "Hierarchy": hierarchy,
            "Family": "temporal_centroid",
            "Early_Window": "",
            "Analysis_Set": "Exploratory",
            "Fallbacks": [
                f"{hierarchy}_temporal_centroid",
                f"{prefix}_centroid",
                f"{hierarchy}_centroid",
                f"{prefix}_tau",
                f"{hierarchy}_tau",
            ],
        })

    # peak step
    for hierarchy, prefix, short_name in hierarchy_specs:
        configs.append({
            "Metric": f"{prefix}_peak",
            "Hierarchy": hierarchy,
            "Family": "peak_step",
            "Early_Window": "",
            "Analysis_Set": "Exploratory",
            "Fallbacks": [
                f"{hierarchy}_peak",
                f"{prefix}_peak_step",
                f"{hierarchy}_peak_step",
            ],
        })

    # AUC
    for hierarchy, prefix, short_name in hierarchy_specs:
        configs.append({
            "Metric": f"{prefix}_auc",
            "Hierarchy": hierarchy,
            "Family": "auc",
            "Early_Window": "",
            "Analysis_Set": "Exploratory",
            "Fallbacks": [
                f"{hierarchy}_auc",
                f"{prefix}_AUC",
                f"{hierarchy}_AUC",
            ],
        })

    return configs


METRIC_CONFIGS = build_metric_configs()


# =====================================================
# 5) 读取 CSV
# =====================================================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

if "sub_id" not in df.columns:
    if "SUB_ID_norm" in df.columns:
        df["sub_id"] = df["SUB_ID_norm"].apply(normalize_sub_id)
    elif "SUB_ID" in df.columns:
        df["sub_id"] = df["SUB_ID"].apply(normalize_sub_id)
    else:
        raise KeyError(
            "当前 CSV 缺少 sub_id / SUB_ID_norm / SUB_ID，无法识别被试 ID。"
        )
else:
    df["sub_id"] = df["sub_id"].apply(normalize_sub_id)

n_input = len(df)
n_bad_id = int(df["sub_id"].isna().sum())

df = df.dropna(subset=["sub_id"]).copy()


# =====================================================
# 6) 检查 L_MAX 与协变量
# =====================================================

if L_MAX_COL in df.columns:
    df[L_MAX_COL] = pd.to_numeric(df[L_MAX_COL], errors="coerce")

    n_bad_lmax = int(df[L_MAX_COL].isna().sum())

    if n_bad_lmax > 0:
        print(f"警告：{n_bad_lmax} 行缺少有效 {L_MAX_COL}，这些行会被排除。")

    df = df.dropna(subset=[L_MAX_COL]).copy()
    df[L_MAX_COL] = df[L_MAX_COL].astype(int)

    present_lmax_values = sorted(df[L_MAX_COL].unique().tolist())

    target_lmax_values = [
        int(x) for x in L_MAX_LIST
        if int(x) in present_lmax_values
    ]

    missing_lmax_values = [
        int(x) for x in L_MAX_LIST
        if int(x) not in present_lmax_values
    ]

    if len(missing_lmax_values) > 0:
        print(f"警告：以下 L_MAX 在输入表中未找到：{missing_lmax_values}")

    if len(target_lmax_values) == 0:
        raise RuntimeError(
            f"没有找到任何目标 L_MAX。目标={L_MAX_LIST}，实际={present_lmax_values}"
        )

else:
    print(f"警告：输入 CSV 中没有 {L_MAX_COL} 列，将作为单一分析表处理。")
    target_lmax_values = [None]


required_cols = [
    "Group",
    "Age",
    "Sex",
    "FIQ",
]

missing_required = [
    c for c in required_cols
    if c not in df.columns
]

if len(missing_required) > 0:
    raise KeyError(
        f"当前 CSV 缺少必要协变量列: {missing_required}"
    )


print(f"输入行数: {n_input}")
print(f"无法解析 sub_id 并删除: {n_bad_id}")
print(f"进入分析前行数: {len(df)}")
print(f"目标 L_MAX: {target_lmax_values}")
print("MeanFD covariate: not included")
print("Site covariate: not included")


# =====================================================
# 7) 变量编码与 dtype 清理
# =====================================================

df["Group_bin"] = normalize_group(df["Group"])
df["Sex_bin"] = normalize_sex(df["Sex"])
df["Age"] = safe_numeric(df["Age"])
df["FIQ"] = safe_numeric(df["FIQ"])

n_group_missing = int(df["Group_bin"].isna().sum())
n_sex_missing = int(df["Sex_bin"].isna().sum())

print(f"Group 无法编码数量: {n_group_missing}")
print(f"Sex 无法编码数量: {n_sex_missing}")


# =====================================================
# 8) 指标列名解析
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
# 9) 主分析：H4_minus_H1 contrast 敏感性分析
# =====================================================

covariate_cols = [
    "Age",
    "Sex_bin",
    "FIQ",
]

primary_results = []
failed_rows = []

for lmax_value in target_lmax_values:

    if lmax_value is None:
        df_lmax = df.copy()
        lmax_label = "all"
    else:
        df_lmax = df[df[L_MAX_COL] == lmax_value].copy()
        lmax_label = f"Lmax_{lmax_value}"

    print("\n====================================================")
    print(f"[主分析] {lmax_label}")
    print(f"行数: {len(df_lmax)}")
    print("====================================================")

    for start_step, end_step in EARLY_WINDOWS:
        wlab = early_label(start_step, end_step)

        h1_metric = f"H1_sensory_early_slope_{wlab}"
        h4_metric = f"H4_DMN_early_slope_{wlab}"

        try:
            H1_ES_COL = get_resolved_actual_col(
                resolved_metric_configs,
                h1_metric,
            )

            H4_ES_COL = get_resolved_actual_col(
                resolved_metric_configs,
                h4_metric,
            )

            contrast_col = f"H4_minus_H1_early_slope_{wlab}"

            df_lmax[contrast_col] = (
                safe_numeric(df_lmax[H4_ES_COL])
                - safe_numeric(df_lmax[H1_ES_COL])
            )

            print(f"\n[主分析] {lmax_label}, early_slope_{wlab}")
            print(f"  H1 column: {H1_ES_COL}")
            print(f"  H4 column: {H4_ES_COL}")
            print(f"  Contrast: {contrast_col} = {H4_ES_COL} - {H1_ES_COL}")

            contrast_fit_result = fit_group_glm_hc3(
                df_model=df_lmax,
                y_col=contrast_col,
                covariate_cols=covariate_cols,
                min_n=MIN_N_GLM,
            )

            contrast_data_used = contrast_fit_result["data_used"].copy()
            contrast_data_used["sub_id"] = df_lmax.loc[contrast_data_used.index, "sub_id"]
            contrast_data_used["Group_original"] = df_lmax.loc[contrast_data_used.index, "Group"]

            if L_MAX_COL in df_lmax.columns:
                contrast_data_used[L_MAX_COL] = df_lmax.loc[contrast_data_used.index, L_MAX_COL]

            contrast_data_used["Early_Window"] = wlab
            contrast_data_used["H1_column"] = H1_ES_COL
            contrast_data_used["H4_column"] = H4_ES_COL
            contrast_data_used["H1_column_value"] = safe_numeric(
                df_lmax.loc[contrast_data_used.index, H1_ES_COL]
            )
            contrast_data_used["H4_column_value"] = safe_numeric(
                df_lmax.loc[contrast_data_used.index, H4_ES_COL]
            )

            out_data_csv = os.path.join(
                OUT_PRIMARY_DATA_DIR,
                f"EC_SEC_primary_H4minusH1_data_used_{lmax_label}_early_{wlab}.csv"
            )

            contrast_data_used.to_csv(
                out_data_csv,
                index=False,
                encoding="utf-8-sig",
            )

            contrast_model_label = (
                "H4_minus_H1 ~ Group + "
                + " + ".join(covariate_cols)
            )

            primary_results.append({
                "Analysis": "Primary_H4_minus_H1_contrast",
                L_MAX_COL: lmax_value if lmax_value is not None else np.nan,
                "Early_Window": wlab,
                "Metric": contrast_col,
                "Definition": f"{H4_ES_COL} - {H1_ES_COL}",
                "H1_column": H1_ES_COL,
                "H4_column": H4_ES_COL,
                "Model": contrast_model_label,
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
                "Covariates": ", ".join(covariate_cols),
                "Site": "not included",
                "MeanFD": "not included",
                "Data_used_csv": out_data_csv,
            })

        except Exception as e:
            reason = str(e)
            print(f"跳过主分析 {lmax_label}, early_slope_{wlab}: {reason}")

            failed_rows.append({
                "Analysis": "Primary_H4_minus_H1_contrast",
                L_MAX_COL: lmax_value if lmax_value is not None else np.nan,
                "Early_Window": wlab,
                "Metric": f"H4_minus_H1_early_slope_{wlab}",
                "reason": reason,
            })


contrast_res_df = pd.DataFrame(primary_results)

if len(contrast_res_df) == 0:
    raise RuntimeError(
        "没有任何 H4-H1 contrast 主分析成功完成。请检查指标列名、L_MAX 和协变量。"
    )


# =====================================================
# 10) 主分析 FDR
# =====================================================

contrast_fdr_tables = []

# FDR 1：所有 L_MAX x early window 主 contrast 一起校正
mask_all_primary = contrast_res_df["p_value"].notna()

contrast_res_df = add_fdr_column_for_mask(
    res_df=contrast_res_df,
    mask=mask_all_primary,
    p_col="p_FDR__primary_all_Lmax_x_early_window",
    sig_col="sig_FDR_0.05__primary_all_Lmax_x_early_window",
    n_col="FDR_n__primary_all_Lmax_x_early_window",
)

tmp_primary_all = make_fdr_long_table(
    res_df=contrast_res_df.rename(columns={
        "Definition": "Actual_column",
    }),
    scope_name="primary_all_Lmax_x_early_window",
    scope_label="Primary H4-H1 contrasts across all L_MAX and early-window sensitivity settings",
    mask=mask_all_primary,
)

if len(tmp_primary_all) > 0:
    contrast_fdr_tables.append(tmp_primary_all)


# FDR 2：每个 early window 内 across L_MAX 校正
for wlab in EARLY_WINDOW_LABELS:
    mask = contrast_res_df["Early_Window"].eq(wlab)

    p_col = f"p_FDR__primary_early_{wlab}_across_Lmax"
    sig_col = f"sig_FDR_0.05__primary_early_{wlab}_across_Lmax"
    n_col = f"FDR_n__primary_early_{wlab}_across_Lmax"

    contrast_res_df = add_fdr_column_for_mask(
        res_df=contrast_res_df,
        mask=mask,
        p_col=p_col,
        sig_col=sig_col,
        n_col=n_col,
    )

    tmp = make_fdr_long_table(
        res_df=contrast_res_df.rename(columns={
            "Definition": "Actual_column",
        }),
        scope_name=f"primary_early_{wlab}_across_Lmax",
        scope_label=f"Primary H4-H1 contrast for early_slope_{wlab}, corrected across L_MAX values",
        mask=mask,
    )

    if len(tmp) > 0:
        contrast_fdr_tables.append(tmp)


# FDR 3：每个 L_MAX 内 across early windows 校正
for lmax_value in target_lmax_values:
    if lmax_value is None:
        continue

    mask = contrast_res_df[L_MAX_COL].eq(lmax_value)

    p_col = f"p_FDR__primary_Lmax_{lmax_value}_across_early_windows"
    sig_col = f"sig_FDR_0.05__primary_Lmax_{lmax_value}_across_early_windows"
    n_col = f"FDR_n__primary_Lmax_{lmax_value}_across_early_windows"

    contrast_res_df = add_fdr_column_for_mask(
        res_df=contrast_res_df,
        mask=mask,
        p_col=p_col,
        sig_col=sig_col,
        n_col=n_col,
    )

    tmp = make_fdr_long_table(
        res_df=contrast_res_df.rename(columns={
            "Definition": "Actual_column",
        }),
        scope_name=f"primary_Lmax_{lmax_value}_across_early_windows",
        scope_label=f"Primary H4-H1 contrast for L_MAX={lmax_value}, corrected across early windows",
        mask=mask,
    )

    if len(tmp) > 0:
        contrast_fdr_tables.append(tmp)


if len(contrast_fdr_tables) > 0:
    contrast_fdr_long_df = pd.concat(
        contrast_fdr_tables,
        axis=0,
        ignore_index=True,
    )
else:
    contrast_fdr_long_df = pd.DataFrame()


contrast_res_df["_Early_order"] = (
    contrast_res_df["Early_Window"]
    .map({w: i for i, w in enumerate(EARLY_WINDOW_LABELS)})
    .fillna(99)
)

contrast_res_df = (
    contrast_res_df
    .sort_values(
        [L_MAX_COL, "_Early_order"],
        ascending=[True, True],
    )
    .drop(columns=["_Early_order"])
    .reset_index(drop=True)
)

contrast_res_df.to_csv(
    OUT_CONTRAST_CSV,
    index=False,
    encoding="utf-8-sig",
)

contrast_fdr_long_df.to_csv(
    OUT_CONTRAST_FDR_LONG_CSV,
    index=False,
    encoding="utf-8-sig",
)


print("\n=== 主分析完成：H4-H1 contrast 敏感性分析 ===")
print(contrast_res_df)

print("H4-H1 contrast 结果已保存到:")
print(OUT_CONTRAST_CSV)
print(OUT_CONTRAST_FDR_LONG_CSV)


# =====================================================
# 11) 补充分析：单指标 GLM，按 L_MAX 分层
# =====================================================

if ANALYZE_ONLY_EARLY_SLOPE_SUPPLEMENTARY:
    METRICS_TO_RUN = [
        spec for spec in resolved_metric_configs
        if spec["Family"].startswith("early_slope_")
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


print("\n[补充分析] 实际纳入单指标 GLM 的指标：")

for spec in METRICS_TO_RUN:
    print(
        f"  {spec['Metric']} | "
        f"actual={spec['Actual_column']} | "
        f"Hierarchy={spec['Hierarchy']} | "
        f"Family={spec['Family']} | "
        f"Analysis_Set={spec['Analysis_Set']}"
    )


single_results = []

for lmax_value in target_lmax_values:

    if lmax_value is None:
        df_lmax = df.copy()
        lmax_label = "all"
    else:
        df_lmax = df[df[L_MAX_COL] == lmax_value].copy()
        lmax_label = f"Lmax_{lmax_value}"

    print("\n====================================================")
    print(f"[补充分析] {lmax_label}")
    print(f"行数: {len(df_lmax)}")
    print("====================================================")

    for spec in METRICS_TO_RUN:
        metric = spec["Metric"]
        actual_col = spec["Actual_column"]
        hierarchy = spec["Hierarchy"]
        family = spec["Family"]
        early_window = spec.get("Early_Window", "")
        analysis_set = spec["Analysis_Set"]

        y_col_tmp = f"__y__{metric}"

        df_lmax[y_col_tmp] = safe_numeric(df_lmax[actual_col])

        try:
            fit_res = fit_group_glm_hc3(
                df_model=df_lmax,
                y_col=y_col_tmp,
                covariate_cols=covariate_cols,
                min_n=MIN_N_GLM,
            )

        except RuntimeError as e:
            print(f"跳过 {lmax_label} | {metric}: {e}")

            failed_rows.append({
                "Analysis": "Supplementary_single_metric_GLM",
                L_MAX_COL: lmax_value if lmax_value is not None else np.nan,
                "Early_Window": early_window,
                "Metric": metric,
                "reason": str(e),
            })

            continue

        single_results.append({
            "Analysis": "Supplementary_single_metric_GLM",
            L_MAX_COL: lmax_value if lmax_value is not None else np.nan,
            "Early_Window": early_window,
            "Metric": metric,
            "Actual_column": actual_col,
            "Hierarchy": hierarchy,
            "Family": family,
            "Analysis_Set": analysis_set,
            "Model": f"{metric} ~ Group + " + " + ".join(covariate_cols),
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
            "Covariates": ", ".join(covariate_cols),
            "Site": "not included",
            "MeanFD": "not included",
        })


res_df = pd.DataFrame(single_results)

if len(res_df) == 0:
    raise RuntimeError(
        "没有任何单指标成功完成回归。请检查指标列名与协变量缺失情况。"
    )


# =====================================================
# 12) 补充分析 FDR
# =====================================================

fdr_long_tables = []

for lmax_value in target_lmax_values:
    if lmax_value is None:
        lmax_mask = res_df[L_MAX_COL].isna()
        lmax_label = "all"
    else:
        lmax_mask = res_df[L_MAX_COL].eq(lmax_value)
        lmax_label = f"Lmax_{lmax_value}"

    # early_slope: 每个 early window 在当前 L_MAX 内 across H1-H4
    for wlab in EARLY_WINDOW_LABELS:
        fam = f"early_slope_{wlab}"

        scope_name = f"supplementary_{fam}_H1H2H3H4_{lmax_label}"
        scope_label = (
            f"Supplementary hierarchy: H1-H4 {fam}, "
            f"{lmax_label}"
        )

        mask = (
            lmax_mask
            & res_df["Family"].eq(fam)
            & res_df["Hierarchy"].isin(["H1", "H2", "H3", "H4"])
        )

        p_col = f"p_FDR__{scope_name}"
        sig_col = f"sig_FDR_0.05__{scope_name}"
        n_col = f"FDR_n__{scope_name}"

        res_df = add_fdr_column_for_mask(
            res_df=res_df,
            mask=mask,
            p_col=p_col,
            sig_col=sig_col,
            n_col=n_col,
        )

        tmp_long = make_fdr_long_table(
            res_df=res_df,
            scope_name=scope_name,
            scope_label=scope_label,
            mask=mask,
        )

        if len(tmp_long) > 0:
            fdr_long_tables.append(tmp_long)

    # exploratory families: 每个 family 在当前 L_MAX 内 across H1-H4
    for fam in [
        "temporal_centroid",
        "peak_step",
        "auc",
    ]:
        scope_name = f"exploratory_{fam}_H1H2H3H4_{lmax_label}"
        scope_label = f"Exploratory family: {fam} across H1-H4, {lmax_label}"

        mask = (
            lmax_mask
            & res_df["Family"].eq(fam)
            & res_df["Hierarchy"].isin(["H1", "H2", "H3", "H4"])
        )

        p_col = f"p_FDR__{scope_name}"
        sig_col = f"sig_FDR_0.05__{scope_name}"
        n_col = f"FDR_n__{scope_name}"

        res_df = add_fdr_column_for_mask(
            res_df=res_df,
            mask=mask,
            p_col=p_col,
            sig_col=sig_col,
            n_col=n_col,
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
# 13) 排序并保存补充分析结果
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

family_order = {}

for i, wlab in enumerate(EARLY_WINDOW_LABELS):
    family_order[f"early_slope_{wlab}"] = i

family_order.update({
    "temporal_centroid": 10,
    "peak_step": 11,
    "auc": 12,
})

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
            L_MAX_COL,
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
    fdr_long_df["_Family_order"] = (
        fdr_long_df["Family"]
        .map(family_order)
        .fillna(99)
    )

    fdr_long_df["_Hierarchy_order"] = (
        fdr_long_df["Hierarchy"]
        .map(hierarchy_order)
        .fillna(99)
    )

    fdr_long_df = (
        fdr_long_df
        .sort_values(
            [
                L_MAX_COL,
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
                "_Family_order",
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


print("\n=== 补充分析完成：单指标 GLM + FDR ===")
print(res_df)

print("\n补充单指标结果已保存到:")
print(OUT_SINGLE_CSV)

print("\n补充单指标长格式 FDR 结果已保存到:")
print(OUT_SINGLE_FDR_LONG_CSV)


# =====================================================
# 14) 失败模型输出
# =====================================================

if len(failed_rows) > 0:
    failed_df = pd.DataFrame(failed_rows)
    failed_df.to_csv(
        OUT_FAILED_MODELS_CSV,
        index=False,
        encoding="utf-8-sig",
    )
else:
    failed_df = pd.DataFrame()


# =====================================================
# 15) 主分析 summary
# =====================================================

with open(OUT_CONTRAST_TXT, "w", encoding="utf-8") as f:
    f.write("EC-SEC Primary H4-H1 Contrast Sensitivity Analysis Summary\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input file:\n")
    f.write(f"  INPUT_CSV: {INPUT_CSV}\n")
    f.write(f"  DATASET_LABEL: {DATASET_LABEL}\n\n")

    f.write("Sensitivity settings:\n")
    f.write(f"  L_MAX_LIST: {L_MAX_LIST}\n")
    f.write(f"  EARLY_WINDOWS: {EARLY_WINDOWS}\n")
    f.write(f"  EARLY_WINDOW_LABELS: {EARLY_WINDOW_LABELS}\n\n")

    f.write("Rationale:\n")
    f.write(
        "The primary EC-SEC follow-up tests the relative contrast between "
        "H4 DMN and H1 sensory propagation metrics. This script repeats "
        "the contrast across L_MAX and early-slope window settings as a "
        "parameter sensitivity analysis.\n\n"
    )

    f.write("Contrast definitions:\n")
    for wlab in EARLY_WINDOW_LABELS:
        f.write(
            f"  H4_minus_H1_early_slope_{wlab} = "
            f"H4_DMN_early_slope_{wlab} - H1_sensory_early_slope_{wlab}\n"
        )
    f.write("\n")

    f.write("Model specification:\n")
    f.write("  Model: H4_minus_H1 ~ Group + Age + Sex_bin + FIQ\n")
    f.write("  IV of interest: Group_bin, coded ASD=1 and HC=0\n")
    f.write(f"  Covariates: {', '.join(covariate_cols)}\n")
    f.write("  MeanFD covariate: not included\n")
    f.write("  Site covariate: not included\n")
    f.write("  Estimator: OLS with HC3 heteroscedasticity-consistent robust standard errors\n\n")

    f.write("ID and coding QC:\n")
    f.write(f"  Input rows: {n_input}\n")
    f.write(f"  Unparsable sub_id rows dropped: {n_bad_id}\n")
    f.write(f"  Rows after ID/L_MAX QC: {len(df)}\n")
    f.write(f"  Group coding missing: {n_group_missing}\n")
    f.write(f"  Sex coding missing: {n_sex_missing}\n\n")

    f.write("Primary contrast results:\n")
    f.write(
        "  Note: Beta_ASD_minus_HC represents "
        "[(H4-H1)_ASD] - [(H4-H1)_HC].\n"
    )
    f.write(
        "  Beta > 0 means ASD shows a larger H4-H1 contrast than HC "
        "after covariate adjustment.\n"
    )
    f.write(
        "  Beta < 0 means ASD shows a smaller H4-H1 contrast than HC "
        "after covariate adjustment.\n\n"
    )

    for _, row in contrast_res_df.iterrows():
        f.write(
            f"[L_MAX={row[L_MAX_COL]} | early={row['Early_Window']}] "
            f"{row['Metric']}: "
            f"beta={row['Beta_ASD_minus_HC']:.6g}, "
            f"SE(HC3)={row['SE_HC3']:.6g}, "
            f"t={row['t_value']:.3f}, "
            f"p={row['p_value']:.3g}, "
            f"95% CI=[{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"N={int(row['N'])}, "
            f"Adj_R2={row['Adj_R2']:.4f}, "
            f"full_rank={bool(row['Full_rank'])}\n"
        )

    f.write("\nFDR-corrected primary contrast results:\n\n")

    if len(contrast_fdr_long_df) == 0:
        f.write("  None\n")
    else:
        for scope_name in contrast_fdr_long_df["FDR_scope"].dropna().unique():
            sub = contrast_fdr_long_df[
                contrast_fdr_long_df["FDR_scope"] == scope_name
            ].copy()

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
                    f"  - L_MAX={row[L_MAX_COL]}, early={row['Early_Window']}, "
                    f"{row['Metric']}: "
                    f"{direction}, "
                    f"beta={row['Beta_ASD_minus_HC']:.6g}, "
                    f"p={row['p_value']:.3g}, "
                    f"q_FDR={row['p_FDR']:.3g}, "
                    f"sig_FDR={bool(row['sig_FDR_0.05'])}, "
                    f"N={int(row['N'])}\n"
                )

            f.write("\n")

    f.write("Outputs:\n")
    f.write(f"  Primary contrast CSV: {OUT_CONTRAST_CSV}\n")
    f.write(f"  Primary contrast FDR long CSV: {OUT_CONTRAST_FDR_LONG_CSV}\n")
    f.write(f"  Primary data-used dir: {OUT_PRIMARY_DATA_DIR}\n")
    f.write(f"  Primary summary TXT: {OUT_CONTRAST_TXT}\n")

    if len(failed_df) > 0:
        f.write(f"  Failed model CSV: {OUT_FAILED_MODELS_CSV}\n")


print("H4-H1 contrast 总结报告已保存到:")
print(OUT_CONTRAST_TXT)


# =====================================================
# 16) 补充分析 summary
# =====================================================

with open(OUT_SINGLE_TXT, "w", encoding="utf-8") as f:
    f.write("EC-SEC Supplementary Single-Metric GLM Sensitivity Summary Report\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input file:\n")
    f.write(f"  INPUT_CSV: {INPUT_CSV}\n")
    f.write(f"  DATASET_LABEL: {DATASET_LABEL}\n\n")

    f.write("Current CSV format:\n")
    f.write("  The input CSV already contains EC-SEC metrics and covariates in the same table.\n")
    f.write("  No merge with subject_info_for_stats.csv was performed.\n")
    f.write("  Rows are preserved as subject x L_MAX sensitivity rows.\n\n")

    f.write("Sensitivity settings:\n")
    f.write(f"  L_MAX_LIST: {L_MAX_LIST}\n")
    f.write(f"  EARLY_WINDOWS: {EARLY_WINDOWS}\n")
    f.write(f"  EARLY_WINDOW_LABELS: {EARLY_WINDOW_LABELS}\n\n")

    f.write("Relationship to primary analysis:\n")
    f.write(
        "  The primary EC-SEC analysis is the H4-H1 contrast model. "
        "The analyses in this file are supplementary single-metric GLMs "
        "for H1-H4 and exploratory propagation metrics, repeated across "
        "L_MAX sensitivity settings.\n\n"
    )

    f.write("ID and coding QC:\n")
    f.write(f"  Input rows: {n_input}\n")
    f.write(f"  Unparsable sub_id rows dropped: {n_bad_id}\n")
    f.write(f"  Rows after ID/L_MAX QC: {len(df)}\n")
    f.write(f"  Group coding missing: {n_group_missing}\n")
    f.write(f"  Sex coding missing: {n_sex_missing}\n\n")

    f.write("Model specification:\n")
    f.write("  DV: each EC-SEC propagation metric, fitted in a separate model\n")
    f.write("  IV of interest: Group_bin, coded ASD=1 and HC=0\n")
    f.write(f"  Covariates: {', '.join(covariate_cols)}\n")
    f.write("  MeanFD covariate: not included\n")
    f.write("  Site covariate: not included\n")
    f.write("  Estimator: OLS with HC3 heteroscedasticity-consistent robust standard errors\n")
    f.write("  Inference target: diagnostic-group coefficient for each propagation metric\n\n")

    f.write("Supplementary FDR strategy:\n")
    f.write("  For each L_MAX separately:\n")
    for wlab in EARLY_WINDOW_LABELS:
        f.write(
            f"    - early_slope_{wlab}: corrected across H1, H2, H3, and H4.\n"
        )
    f.write(
        "    - temporal_centroid: corrected across available H1-H4 temporal_centroid metrics.\n"
    )
    f.write(
        "    - peak_step: corrected across available H1-H4 peak metrics.\n"
    )
    f.write(
        "    - auc: corrected across available H1-H4 auc metrics.\n\n"
    )

    f.write("Metric column resolution:\n")
    for spec in resolved_metric_configs:
        f.write(
            f"  - {spec['Metric']} -> {spec['Actual_column']} | "
            f"Hierarchy={spec['Hierarchy']} | "
            f"Family={spec['Family']} | "
            f"Early_Window={spec.get('Early_Window', '')} | "
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

    f.write("Single-metric regression results:\n")
    f.write(
        "  Note: Beta_ASD_minus_HC > 0 means ASD > HC after covariate adjustment.\n\n"
    )

    for _, row in res_df.iterrows():
        f.write(
            f"[L_MAX={row[L_MAX_COL]} | {row['Analysis_Set']} | "
            f"{row['Family']} | {row['Hierarchy']}] "
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
                    f"  - L_MAX={row[L_MAX_COL]}, {row['Metric']}: "
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
                    f"  - L_MAX={row[L_MAX_COL]}, {row['Metric']}: "
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

    if len(failed_df) > 0:
        f.write(f"  Failed model CSV: {OUT_FAILED_MODELS_CSV}\n")


print("补充分析总结报告已保存到:")
print(OUT_SINGLE_TXT)


# =====================================================
# 17) 最终打印
# =====================================================

print("\n全部分析完成。")
print("=" * 70)
print("Primary H4-H1 contrast sensitivity results:")
print(f"  {OUT_CONTRAST_CSV}")
print(f"  {OUT_CONTRAST_FDR_LONG_CSV}")
print(f"  {OUT_CONTRAST_TXT}")
print(f"  Data used dir: {OUT_PRIMARY_DATA_DIR}")
print("")
print("Supplementary single-metric sensitivity results:")
print(f"  {OUT_SINGLE_CSV}")
print(f"  {OUT_SINGLE_FDR_LONG_CSV}")
print(f"  {OUT_SINGLE_TXT}")

if len(failed_df) > 0:
    print("")
    print("Failed model report:")
    print(f"  {OUT_FAILED_MODELS_CSV}")
