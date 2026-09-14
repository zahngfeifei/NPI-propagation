# -*- coding: utf-8 -*-
"""
3A All systems - MeanFD 敏感性分析计算代码
-----------------------------------------
作用：
1. 从基础长表 result1_input_long_table_strict.csv 读取系统层面数据；
2. 从 subject_info_for_stats.csv 读取 Group、Age、Sex、FIQ、MeanFD；
3. 将 MeanFD 合并到长表，并只保留具有 MeanFD 的被试；
4. 保留 H1_sensory、H2_attention、H3_control、H4_DMN 全部系统；
5. 拟合 all-systems G_star × Group 模型；
6. 模型中不控制 Site，但加入 MeanFD 作为敏感性分析协变量；
7. 在同一个模型中计算 HC、ASD、ASD-HC 的 simple slopes；
8. 只输出加入 MeanFD 后的数据和结果；
9. 不绘图。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =========================================================
# 1) 路径设置
# =========================================================
DATA_FILE = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果1"
    r"\result1_input_long_table_strict.csv"
)

SUBJECT_INFO_CSV = Path(
    r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"
)

OUT_DIR = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果2-MeanFD敏感性分析"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_PREFIX = "results2_withFD"

DATA_USED_CSV = OUT_DIR / f"{FILE_PREFIX}_data_used.csv"
MODEL_SUMMARY_PATH = OUT_DIR / f"{FILE_PREFIX}_cluster_OLS_summary.txt"
MODEL_PARAMS_PATH = OUT_DIR / f"{FILE_PREFIX}_cluster_OLS_params.csv"
SIMPLE_SLOPE_PATH = OUT_DIR / f"{FILE_PREFIX}_simple_slopes.csv"
PREDICTION_CSV = OUT_DIR / f"{FILE_PREFIX}_prediction_lines.csv"
SUBJECT_SLOPE_CSV = OUT_DIR / f"{FILE_PREFIX}_subjectSlope_values.csv"
SUBJECT_SLOPE_STATS_TXT = OUT_DIR / f"{FILE_PREFIX}_subjectSlope_stats.txt"
RUN_INFO_TXT = OUT_DIR / f"{FILE_PREFIX}_calculation_info.txt"


# =========================================================
# 2) 基本参数
# =========================================================
OUTCOME_SCALE = 1000.0

# 所有保存到 CSV/TXT/控制台的浮点数统一保留小数点后 7 位
DECIMAL_PLACES = 7
FLOAT_FORMAT = f"%.{DECIMAL_PLACES}f"


def fmt7(value):
    """Format numeric values with exactly 7 decimal places for TXT/print output."""
    if pd.isna(value):
        return "nan"

    try:
        return f"{float(value):.{DECIMAL_PLACES}f}"
    except (TypeError, ValueError):
        return str(value)


def format_df7(table, index=False):
    """Format pandas tables with exactly 7 decimal places for TXT/print output."""
    return table.to_string(
        index=index,
        float_format=lambda x: f"{x:.{DECIMAL_PLACES}f}",
    )


SYSTEM_ORDER = [
    "H1_sensory",
    "H2_attention",
    "H3_control",
    "H4_DMN",
]

GROUP_ORDER = ["HC", "ASD"]


# =========================================================
# 3) 工具函数
# =========================================================
def normalize_sub_id(value):
    return str(value).strip()


def sex_to_binary(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().upper()

    male_values = {"M", "MALE", "男", "1", "1.0"}
    female_values = {"F", "FEMALE", "女", "0", "0.0", "2", "2.0"}

    if s in male_values:
        return 1.0
    if s in female_values:
        return 0.0

    return np.nan


def find_interaction_term(model):
    candidates = [
        "G_star:Group[T.ASD]",
        "Group[T.ASD]:G_star",
    ]

    for term in candidates:
        if term in model.params.index:
            return term

    raise RuntimeError("模型中没有找到 G_star × Group 交互项。")


def get_term_stats(model, term):
    beta = float(model.params.get(term, np.nan))
    se = float(model.bse.get(term, np.nan))
    z_or_t = float(model.tvalues.get(term, np.nan))
    p_value = float(model.pvalues.get(term, np.nan))

    try:
        ci_low, ci_high = model.conf_int().loc[term].tolist()
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    return {
        "beta": beta,
        "SE": se,
        "z_or_t": z_or_t,
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def linear_combo_test(model, term_weights):
    names = list(model.params.index)
    r = np.zeros(len(names), dtype=float)

    for term, weight in term_weights.items():
        if term not in names:
            raise RuntimeError(f"线性组合检验中找不到模型项: {term}")
        r[names.index(term)] = float(weight)

    test = model.t_test(r)
    ci = np.ravel(test.conf_int())

    return {
        "beta": float(np.ravel(test.effect)[0]),
        "SE": float(np.ravel(test.sd)[0]),
        "z_or_t": float(np.ravel(test.tvalue)[0]),
        "p_value": float(np.ravel(test.pvalue)[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
    }


def model_params_to_csv(model, path):
    params = pd.DataFrame({
        "Term": model.params.index,
        "Beta": model.params.values,
        "SE": model.bse.values,
        "z_or_t": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    ci = model.conf_int()
    params["ci_low"] = [ci.loc[t, 0] for t in params["Term"]]
    params["ci_high"] = [ci.loc[t, 1] for t in params["Term"]]

    params.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format=FLOAT_FORMAT,
    )

    return params


def compute_subject_slopes(df):
    rows = []

    for sub_id, d in df.groupby("sub_id"):
        d = d.dropna(
            subset=[
                "G_star",
                "early_slope_scaled",
                "Group",
                "system",
            ]
        ).copy()

        if d["system"].nunique() < 2:
            continue

        x = d["G_star"].to_numpy(dtype=float)
        y = d["early_slope_scaled"].to_numpy(dtype=float)

        if len(np.unique(x)) < 2:
            continue

        slope, intercept = np.polyfit(x, y, 1)

        rows.append({
            "sub_id": sub_id,
            "Group": str(d["Group"].iloc[0]),
            "n_systems": int(d["system"].nunique()),
            "subject_slope": float(slope),
            "subject_intercept": float(intercept),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(
            "无法计算个体斜率，请检查每个被试是否具有至少两个系统的数据。"
        )

    out["Group"] = pd.Categorical(
        out["Group"],
        categories=GROUP_ORDER,
        ordered=True,
    )

    return out


def slope_result_to_sentence(row):
    group = row["Group"]
    beta = row["slope_beta"]
    p = row["p_value"]

    direction = "negative" if beta < 0 else "positive"
    sig = "significant" if p < 0.05 else "not significant"

    return (
        f"{group}: slope = {fmt7(beta)}, p = {fmt7(p)}, "
        f"{direction}, {sig}"
    )


# =========================================================
# 4) 读取基础长表
# =========================================================
df = pd.read_csv(DATA_FILE)

data_required_cols = {
    "sub_id",
    "system",
    "G_star",
    "early_slope",
}

missing = sorted(list(data_required_cols - set(df.columns)))
if missing:
    raise ValueError(f"基础长表缺少必要列: {missing}")

df = df.copy()
df["sub_id"] = df["sub_id"].apply(normalize_sub_id)
df["system"] = df["system"].astype(str).str.strip()

for col in ["G_star", "early_slope"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

covariate_cols_from_long_table = [
    "Group",
    "Age",
    "Sex",
    "Sex_bin",
    "Site",
    "FIQ",
    "MeanFD",
]

df = df.drop(
    columns=[c for c in covariate_cols_from_long_table if c in df.columns],
    errors="ignore",
)

n_long_subjects_before_merge = df["sub_id"].nunique()
n_long_observations_before_merge = len(df)


# =========================================================
# 5) 读取 subject_info_for_stats.csv 并合并 MeanFD
# =========================================================
subject_info = pd.read_csv(SUBJECT_INFO_CSV)

subject_required_cols = {
    "sub_id",
    "Group",
    "Age",
    "Sex",
    "FIQ",
    "MeanFD",
}

missing_subject_cols = sorted(list(subject_required_cols - set(subject_info.columns)))
if missing_subject_cols:
    raise ValueError(f"协变量表缺少必要列: {missing_subject_cols}")

subject_info = subject_info.copy()
subject_info["sub_id"] = subject_info["sub_id"].apply(normalize_sub_id)
subject_info["Group"] = subject_info["Group"].astype(str).str.strip().str.upper()
subject_info["Sex"] = subject_info["Sex"].astype(str).str.strip().str.upper()
subject_info["Sex_bin"] = subject_info["Sex"].apply(sex_to_binary)

for col in ["Age", "Sex_bin", "FIQ", "MeanFD"]:
    subject_info[col] = pd.to_numeric(subject_info[col], errors="coerce")

if "Site" in subject_info.columns:
    subject_info["Site"] = subject_info["Site"].astype(str).str.strip()
    subject_keep_cols = [
        "sub_id",
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "Site",
        "FIQ",
        "MeanFD",
    ]
else:
    subject_keep_cols = [
        "sub_id",
        "Group",
        "Age",
        "Sex",
        "Sex_bin",
        "FIQ",
        "MeanFD",
    ]

n_subject_info_rows_before_dedup = len(subject_info)
n_subject_info_subjects_before_dedup = subject_info["sub_id"].nunique()

subject_info = subject_info[subject_keep_cols].drop_duplicates(
    subset=["sub_id"],
    keep="first",
).copy()

n_subject_info_rows_after_dedup = len(subject_info)

df = df.merge(
    subject_info,
    on="sub_id",
    how="left",
    validate="many_to_one",
    indicator="_subject_info_merge",
)

unmatched_subjects = sorted(
    df.loc[df["_subject_info_merge"] == "left_only", "sub_id"]
    .dropna()
    .unique()
    .tolist()
)

n_unmatched_subjects = len(unmatched_subjects)
df = df.drop(columns=["_subject_info_merge"])


# =========================================================
# 6) 清理数据，只保留带 MeanFD 的完整样本
# =========================================================
required_cols = {
    "sub_id",
    "system",
    "G_star",
    "early_slope",
    "Group",
    "Age",
    "Sex_bin",
    "FIQ",
    "MeanFD",
}

missing = sorted(list(required_cols - set(df.columns)))
if missing:
    raise ValueError(f"合并后数据缺少必要列: {missing}")

df = df[df["system"].isin(SYSTEM_ORDER)].copy()
df = df[df["Group"].isin(GROUP_ORDER)].copy()

n_subjects_after_merge_before_dropna = df["sub_id"].nunique()
n_observations_after_merge_before_dropna = len(df)

for col in ["G_star", "early_slope", "Age", "Sex_bin", "FIQ", "MeanFD"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "sub_id",
        "system",
        "Group",
        "G_star",
        "early_slope",
        "Age",
        "Sex_bin",
        "FIQ",
        "MeanFD",
    ]
).copy()

cnt = df.groupby("sub_id")["system"].nunique()
bad_subs = cnt[cnt != len(SYSTEM_ORDER)]

n_subjects_removed_incomplete_systems = len(bad_subs)

if len(bad_subs) > 0:
    df = df.loc[
        ~df["sub_id"].isin(bad_subs.index)
    ].copy()

df["system"] = pd.Categorical(
    df["system"],
    categories=SYSTEM_ORDER,
    ordered=True,
)

df["Group"] = pd.Categorical(
    df["Group"],
    categories=GROUP_ORDER,
    ordered=True,
)

if "early_slope_scaled" in df.columns:
    df["early_slope_scaled"] = pd.to_numeric(
        df["early_slope_scaled"],
        errors="coerce",
    )
else:
    df["early_slope_scaled"] = df["early_slope"] * OUTCOME_SCALE

df = df.dropna(
    subset=["early_slope_scaled"]
).copy()

if df.empty:
    raise RuntimeError("清理后全系统数据为空，请检查输入文件和 MeanFD 合并结果。")

df.to_csv(
    DATA_USED_CSV,
    index=False,
    encoding="utf-8-sig",
    float_format=FLOAT_FORMAT,
)


# =========================================================
# 7) 拟合 all-systems MeanFD 敏感性分析模型
# =========================================================
formula = (
    "early_slope_scaled ~ G_star * Group "
    "+ Age + Sex_bin + FIQ + MeanFD"
)

model = smf.ols(formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["sub_id"]},
)

interaction_term = find_interaction_term(model)

params = model_params_to_csv(
    model,
    MODEL_PARAMS_PATH,
)


# =========================================================
# 8) HC / ASD simple slopes
# =========================================================
hc_stats = get_term_stats(model, "G_star")

asd_stats = linear_combo_test(
    model,
    {
        "G_star": 1,
        interaction_term: 1,
    }
)

int_stats = get_term_stats(model, interaction_term)

simple_slopes = pd.DataFrame([
    {
        "Group": "HC",
        "slope_beta": hc_stats["beta"],
        "SE": hc_stats["SE"],
        "z_or_t": hc_stats["z_or_t"],
        "p_value": hc_stats["p_value"],
        "ci_low": hc_stats["ci_low"],
        "ci_high": hc_stats["ci_high"],
        "definition": "Beta(G_star)",
        "interpretation": "HC slope of early_slope on G_star across H1-H4, adjusted for MeanFD",
    },
    {
        "Group": "ASD",
        "slope_beta": asd_stats["beta"],
        "SE": asd_stats["SE"],
        "z_or_t": asd_stats["z_or_t"],
        "p_value": asd_stats["p_value"],
        "ci_low": asd_stats["ci_low"],
        "ci_high": asd_stats["ci_high"],
        "definition": f"Beta(G_star) + Beta({interaction_term})",
        "interpretation": "ASD slope of early_slope on G_star across H1-H4, adjusted for MeanFD",
    },
    {
        "Group": "ASD_minus_HC",
        "slope_beta": int_stats["beta"],
        "SE": int_stats["SE"],
        "z_or_t": int_stats["z_or_t"],
        "p_value": int_stats["p_value"],
        "ci_low": int_stats["ci_low"],
        "ci_high": int_stats["ci_high"],
        "definition": f"Beta({interaction_term})",
        "interpretation": "ASD-HC difference in G_star slope across H1-H4, adjusted for MeanFD",
    },
])

simple_slopes.to_csv(
    SIMPLE_SLOPE_PATH,
    index=False,
    encoding="utf-8-sig",
    float_format=FLOAT_FORMAT,
)


# =========================================================
# 9) 保存模型 summary，并追加 simple slope 结果
# =========================================================
with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Result1 all-systems MeanFD sensitivity model\n")
    f.write("Model used: cluster-robust OLS\n")
    f.write("System fixed effects: not included in the sensitivity model\n")
    f.write("Cluster variable: sub_id\n")
    f.write("Outcome: early_slope_scaled = early_slope × OUTCOME_SCALE\n")
    f.write(f"OUTCOME_SCALE: {fmt7(OUTCOME_SCALE)}\n")
    f.write("Site covariate: not included\n")
    f.write("MeanFD covariate: included\n")
    f.write(f"MeanFD source: {SUBJECT_INFO_CSV}\n")
    f.write(f"Formula: {formula}\n")
    f.write("\n")
    f.write("Primary interpretation:\n")
    f.write("  G_star: HC slope of all-systems gradient-propagation association, adjusted for MeanFD\n")
    f.write(f"  {interaction_term}: ASD-HC difference in all-systems G_star slope, adjusted for MeanFD\n")
    f.write(
        f"  G_star + {interaction_term}: ASD slope of all-systems "
        "gradient-propagation association, adjusted for MeanFD\n"
    )
    f.write("\n")
    try:
        f.write(model.summary2(float_format=FLOAT_FORMAT).as_text())
    except Exception:
        # 若当前 statsmodels 版本不支持 summary2(float_format)，保留原 summary。
        f.write(model.summary().as_text())

    f.write("\n\n")
    f.write("Simple slope tests for gradient-propagation coupling\n")
    f.write("=" * 80 + "\n")
    f.write("HC coupling slope:\n")
    f.write("  Beta(G_star)\n")
    f.write("ASD coupling slope:\n")
    f.write(f"  Beta(G_star) + Beta({interaction_term})\n")
    f.write("ASD-HC slope difference:\n")
    f.write(f"  Beta({interaction_term})\n")
    f.write("\n")

    for _, row in simple_slopes.iterrows():
        f.write(f"Group/comparison: {row['Group']}\n")
        f.write(f"  slope_beta:    {fmt7(row['slope_beta'])}\n")
        f.write(f"  SE:            {fmt7(row['SE'])}\n")
        f.write(f"  z_or_t:        {fmt7(row['z_or_t'])}\n")
        f.write(f"  p_value:       {fmt7(row['p_value'])}\n")
        f.write(f"  95% CI:        [{fmt7(row['ci_low'])}, {fmt7(row['ci_high'])}]\n")
        f.write(f"  definition:    {row['definition']}\n")
        f.write(f"  interpretation:{row['interpretation']}\n")
        f.write("\n")

    f.write("\n")
    f.write("Plain-language interpretation\n")
    f.write("=" * 80 + "\n")
    f.write(
        "Use the HC row to determine whether HC shows significant "
        "all-systems gradient-propagation coupling after adjusting for MeanFD.\n"
    )
    f.write(
        "Use the ASD row to determine whether ASD shows significant "
        "all-systems gradient-propagation coupling in the same MeanFD-adjusted model.\n"
    )
    f.write(
        "Use the ASD_minus_HC row to determine whether ASD differs "
        "significantly from HC in coupling slope after adjusting for MeanFD.\n"
    )
    f.write("\n")

    for _, row in simple_slopes.iterrows():
        f.write(slope_result_to_sentence(row) + "\n")


# =========================================================
# 10) 构建 HC / ASD 模型预测线
# =========================================================
x_min = df["G_star"].min()
x_max = df["G_star"].max()
x_grid = np.linspace(x_min, x_max, 250)

age_m = df["Age"].mean()
sex_m = df["Sex_bin"].mean()
fiq_m = df["FIQ"].mean()
meanfd_m = df["MeanFD"].mean()

pred_rows = []

for grp in GROUP_ORDER:
    pred_df = pd.DataFrame({
        "G_star": x_grid,
        "Group": pd.Categorical(
            [grp] * len(x_grid),
            categories=GROUP_ORDER,
            ordered=True,
        ),
        "Age": age_m,
        "Sex_bin": sex_m,
        "FIQ": fiq_m,
        "MeanFD": meanfd_m,
    })

    y_pred = np.asarray(model.predict(pred_df))

    for x, y in zip(x_grid, y_pred):
        pred_rows.append({
            "Group": grp,
            "G_star": float(x),
            "early_slope_pred": float(y),
            "Age_fixed": float(age_m),
            "Sex_bin_fixed": float(sex_m),
            "FIQ_fixed": float(fiq_m),
            "MeanFD_fixed": float(meanfd_m),
        })

pd.DataFrame(pred_rows).to_csv(
    PREDICTION_CSV,
    index=False,
    encoding="utf-8-sig",
    float_format=FLOAT_FORMAT,
)


# =========================================================
# 11) 个体斜率
# =========================================================
subject_slope_df = compute_subject_slopes(df)

subject_slope_df.to_csv(
    SUBJECT_SLOPE_CSV,
    index=False,
    encoding="utf-8-sig",
    float_format=FLOAT_FORMAT,
)

hc = subject_slope_df.loc[
    subject_slope_df["Group"].astype(str) == "HC",
    "subject_slope",
].dropna().to_numpy(dtype=float)

asd = subject_slope_df.loc[
    subject_slope_df["Group"].astype(str) == "ASD",
    "subject_slope",
].dropna().to_numpy(dtype=float)

slope_lines = []
slope_lines.append("Subject-level slope across G_star in all systems")
slope_lines.append("=" * 70)
slope_lines.append(
    "Each subject slope is computed by: early_slope_scaled ~ G_star across H1-H4."
)
slope_lines.append("The main model significance tests are adjusted for MeanFD.")
slope_lines.append("")
slope_lines.append(f"HC n: {len(hc)}")
slope_lines.append(f"ASD n: {len(asd)}")
slope_lines.append("")
slope_lines.append(f"HC mean: {fmt7(np.mean(hc))}")
slope_lines.append(f"HC SEM:  {fmt7(pd.Series(hc).sem())}")
slope_lines.append(f"ASD mean: {fmt7(np.mean(asd))}")
slope_lines.append(f"ASD SEM:  {fmt7(pd.Series(asd).sem())}")
slope_lines.append("")
slope_lines.append("Significance source:")
slope_lines.append("Main MeanFD-adjusted model simple slope tests:")
slope_lines.append(
    f"  HC slope p-value: {fmt7(hc_stats['p_value'])}"
)
slope_lines.append(
    f"  ASD slope p-value: {fmt7(asd_stats['p_value'])}"
)
slope_lines.append(
    f"  ASD-HC interaction term: {interaction_term}"
)
slope_lines.append(
    f"  ASD-HC interaction p-value: {fmt7(model.pvalues[interaction_term])}"
)

with open(SUBJECT_SLOPE_STATS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(slope_lines))


# =========================================================
# 12) 保存运行信息
# =========================================================
lines = []

lines.append("Result1 all-systems MeanFD sensitivity calculation finished")
lines.append("=" * 70)
lines.append(f"Input long data: {DATA_FILE}")
lines.append(f"Subject info / MeanFD data: {SUBJECT_INFO_CSV}")
lines.append(f"Systems: {SYSTEM_ORDER}")
lines.append(f"Formula: {formula}")
lines.append("System fixed effects: not included in the sensitivity model")
lines.append("Site covariate: not included")
lines.append("MeanFD covariate: included")
lines.append(f"OUTCOME_SCALE: {fmt7(OUTCOME_SCALE)}")
lines.append("")
lines.append("Merge and filtering summary:")
lines.append(f"Long-table subjects before merge: {n_long_subjects_before_merge}")
lines.append(f"Long-table observations before merge: {n_long_observations_before_merge}")
lines.append(f"Subject-info rows before deduplication: {n_subject_info_rows_before_dedup}")
lines.append(f"Subject-info unique subjects before deduplication: {n_subject_info_subjects_before_dedup}")
lines.append(f"Subject-info rows after deduplication: {n_subject_info_rows_after_dedup}")
lines.append(f"Unmatched subjects after merge: {n_unmatched_subjects}")
if n_unmatched_subjects > 0:
    lines.append("First unmatched subjects:")
    lines.append(str(unmatched_subjects[:30]))
lines.append(f"Subjects after merge before complete-case filtering: {n_subjects_after_merge_before_dropna}")
lines.append(f"Observations after merge before complete-case filtering: {n_observations_after_merge_before_dropna}")
lines.append(f"Subjects removed for incomplete H1-H4 systems: {n_subjects_removed_incomplete_systems}")
lines.append("")
lines.append(f"N subjects used: {df['sub_id'].nunique()}")
lines.append(f"N observations used: {len(df)}")
lines.append("")
lines.append("Group counts:")
lines.append(
    str(
        df.drop_duplicates("sub_id")
        .groupby("Group", observed=False)["sub_id"]
        .count()
    )
)
lines.append("")
lines.append("MeanFD summary in used data:")
lines.append(
    df.drop_duplicates("sub_id")["MeanFD"].describe().to_string(
        float_format=lambda x: f"{x:.{DECIMAL_PLACES}f}"
    )
)
lines.append("")
lines.append("Primary terms:")
lines.append(f"G_star p-value, HC slope: {fmt7(hc_stats['p_value'])}")
lines.append(
    f"{interaction_term} p-value, ASD-HC slope difference: "
    f"{fmt7(int_stats['p_value'])}"
)
lines.append("")
lines.append("Simple slopes:")
lines.append(format_df7(simple_slopes, index=False))
lines.append("")
lines.append("Interpretation guide:")
lines.append("  HC row: whether HC has significant all-systems coupling after adjusting for MeanFD.")
lines.append("  ASD row: whether ASD has significant all-systems coupling after adjusting for MeanFD.")
lines.append("  ASD_minus_HC row: whether ASD-HC coupling slopes differ after adjusting for MeanFD.")
lines.append("")
lines.append("Saved files:")
lines.append(str(DATA_USED_CSV))
lines.append(str(MODEL_SUMMARY_PATH))
lines.append(str(MODEL_PARAMS_PATH))
lines.append(str(SIMPLE_SLOPE_PATH))
lines.append(str(PREDICTION_CSV))
lines.append(str(SUBJECT_SLOPE_CSV))
lines.append(str(SUBJECT_SLOPE_STATS_TXT))
lines.append(str(RUN_INFO_TXT))

with open(RUN_INFO_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# =========================================================
# 13) Print
# =========================================================
print("All-systems MeanFD sensitivity calculation finished.")
print("Data used:", DATA_USED_CSV)
print("Model summary:", MODEL_SUMMARY_PATH)
print("Model params:", MODEL_PARAMS_PATH)
print("Simple slopes:", SIMPLE_SLOPE_PATH)
print("Prediction lines:", PREDICTION_CSV)
print("Subject slopes:", SUBJECT_SLOPE_CSV)
print("Subject slope stats:", SUBJECT_SLOPE_STATS_TXT)
print("Run info:", RUN_INFO_TXT)

print("")
print("Input files:")
print("Long data:", DATA_FILE)
print("Subject info / MeanFD:", SUBJECT_INFO_CSV)

print("")
print("Model formula:")
print(formula)

print("")
print("Data after MeanFD complete-case filtering:")
print("N subjects:", df["sub_id"].nunique())
print("N observations:", len(df))
print("Group counts:")
print(
    df.drop_duplicates("sub_id")
    .groupby("Group", observed=False)["sub_id"]
    .count()
)

print("")
print("Key model terms:")
for term in ["G_star", interaction_term, "Group[T.ASD]", "MeanFD"]:
    if term in model.params.index:
        print(
            f"{term}: beta={fmt7(model.params[term])}, "
            f"SE={fmt7(model.bse[term])}, "
            f"z/t={fmt7(model.tvalues[term])}, "
            f"p={fmt7(model.pvalues[term])}"
        )
    else:
        print(f"{term}: not found")

print("")
print("Simple slopes:")
print(format_df7(simple_slopes, index=False))

print("")
print("Plain-language simple slope interpretation:")
for _, row in simple_slopes.iterrows():
    print(slope_result_to_sentence(row))
