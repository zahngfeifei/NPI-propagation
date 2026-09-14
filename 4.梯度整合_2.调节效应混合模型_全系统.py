# -*- coding: utf-8 -*-
"""
3A All systems - 计算代码
------------------------
作用：
1. 从基础长表 result1_input_long_table_strict.csv 读取数据；
2. 保留 H1_sensory、H2_attention、H3_control、H4_DMN 全部系统；
3. 拟合 all-systems G_star × Group 模型；
4. 输出模型结果、参数表、HC/ASD simple slopes、预测线数据、个体斜率数据；
5. 不绘图。

正式回答：
1. HC 中是否存在全系统梯度-传播对应关系：看 G_star；
2. ASD 与 HC 是否显著不同：看 G_star:Group[T.ASD]；
3. ASD 中是否仍存在对应关系：看 G_star + G_star:Group[T.ASD] 的线性组合检验。
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

# 计算结果保存地址
OUT_DIR = Path(
    r"I:\DYF\NPI-4-code\4.梯度整合\ABIDE1_结果2-控制系统"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_PREFIX = "results2"

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
def find_interaction_term(model):
    """
    兼容 statsmodels 中交互项名称的两种顺序。
    """
    candidates = [
        "G_star:Group[T.ASD]",
        "Group[T.ASD]:G_star",
    ]

    for term in candidates:
        if term in model.params.index:
            return term

    raise RuntimeError("模型中没有找到 G_star × Group 交互项。")


def get_term_stats(model, term):
    """
    提取单个模型项的 beta、SE、z/t、p 和 CI。
    """
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
    """
    对多个回归系数的线性组合进行检验。

    例如：
        {"G_star": 1, "G_star:Group[T.ASD]": 1}
    """
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
    """
    保存模型所有参数。
    """
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
    )

    return params


def compute_subject_slopes(df):
    """
    对每个被试计算 across-system 的个体斜率：
        early_slope_scaled ~ G_star

    每个被试应包含 H1-H4 四个系统。
    输出 subject_slope，用于柱状散点图。
    """
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


# =========================================================
# 4) 读取并整理数据
# =========================================================
df = pd.read_csv(DATA_FILE)

required_cols = {
    "sub_id",
    "system",
    "G_star",
    "early_slope",
    "Group",
    "Age",
    "Sex_bin",
    "FIQ",
}

missing = sorted(list(required_cols - set(df.columns)))
if missing:
    raise ValueError(f"输入表缺少必要列: {missing}")

df = df.copy()

df["sub_id"] = df["sub_id"].astype(str).str.strip()
df["system"] = df["system"].astype(str).str.strip()
df["Group"] = df["Group"].astype(str).str.strip().str.upper()

for col in ["G_star", "early_slope", "Age", "Sex_bin", "FIQ"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3A：只做全系统 H1-H4
df = df[df["system"].isin(SYSTEM_ORDER)].copy()
df = df[df["Group"].isin(GROUP_ORDER)].copy()

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
    ]
).copy()

# 每个被试必须有 H1-H4 四个系统
cnt = df.groupby("sub_id")["system"].nunique()
bad_subs = cnt[cnt != len(SYSTEM_ORDER)]

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

df["early_slope_scaled"] = df["early_slope"] * OUTCOME_SCALE

if df.empty:
    raise RuntimeError("清理后全系统数据为空，请检查输入文件。")

df.to_csv(
    DATA_USED_CSV,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 5) 拟合 all-systems 主模型
# =========================================================
formula = (
    "early_slope_scaled ~ G_star * Group + C(system) "
    "+ Age + Sex_bin + FIQ"
)

model = smf.ols(formula, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["sub_id"]},
)

interaction_term = find_interaction_term(model)

with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Result1 all-systems model\n")
    f.write("Model used: cluster-robust OLS\n")
    f.write("Cluster variable: sub_id\n")
    f.write("Outcome: early_slope_scaled = early_slope × OUTCOME_SCALE\n")
    f.write(f"OUTCOME_SCALE: {OUTCOME_SCALE}\n")
    f.write("Site covariate: not included\n")
    f.write("MeanFD covariate: not included\n")
    f.write(f"Formula: {formula}\n")
    f.write("\n")
    f.write("Primary interpretation:\n")
    f.write("  G_star: HC slope of all-systems gradient-propagation association\n")
    f.write(f"  {interaction_term}: ASD-HC difference in all-systems G_star slope\n")
    f.write("\n")
    f.write(model.summary().as_text())

params = model_params_to_csv(
    model,
    MODEL_PARAMS_PATH,
)


# =========================================================
# 6) HC / ASD simple slopes
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
        "interpretation": "HC slope of early_slope on G_star across H1-H4",
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
        "interpretation": "ASD slope of early_slope on G_star across H1-H4",
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
        "interpretation": "ASD-HC difference in G_star slope across H1-H4",
    },
])

simple_slopes.to_csv(
    SIMPLE_SLOPE_PATH,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 7) 构建 HC / ASD 模型预测线
# =========================================================
x_min = df["G_star"].min()
x_max = df["G_star"].max()
x_grid = np.linspace(x_min, x_max, 250)

age_m = df["Age"].mean()
sex_m = df["Sex_bin"].mean()
fiq_m = df["FIQ"].mean()
pred_rows = []

for grp in GROUP_ORDER:
    pred_all_systems = []

    for sys_name in SYSTEM_ORDER:
        pred_df = pd.DataFrame({
            "G_star": x_grid,
            "Group": pd.Categorical(
                [grp] * len(x_grid),
                categories=GROUP_ORDER,
                ordered=True,
            ),
            "system": pd.Categorical(
                [sys_name] * len(x_grid),
                categories=SYSTEM_ORDER,
                ordered=True,
            ),
            "Age": age_m,
            "Sex_bin": sex_m,
            "FIQ": fiq_m,
        })

        y_pred = np.asarray(model.predict(pred_df))
        pred_all_systems.append(y_pred)

    y_mean = np.mean(np.vstack(pred_all_systems), axis=0)

    for x, y in zip(x_grid, y_mean):
        pred_rows.append({
            "Group": grp,
            "G_star": float(x),
            "early_slope_pred": float(y),
        })

pd.DataFrame(pred_rows).to_csv(
    PREDICTION_CSV,
    index=False,
    encoding="utf-8-sig",
)


# =========================================================
# 8) 个体斜率
# =========================================================
subject_slope_df = compute_subject_slopes(df)

subject_slope_df.to_csv(
    SUBJECT_SLOPE_CSV,
    index=False,
    encoding="utf-8-sig",
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
slope_lines.append("Subject-level slope across G* in all systems")
slope_lines.append("=" * 70)
slope_lines.append(
    "Each subject slope is computed by: early_slope_scaled ~ G_star across H1-H4."
)
slope_lines.append("")
slope_lines.append(f"HC n: {len(hc)}")
slope_lines.append(f"ASD n: {len(asd)}")
slope_lines.append("")
slope_lines.append(f"HC mean: {np.mean(hc):.10g}")
slope_lines.append(f"HC SEM:  {pd.Series(hc).sem():.10g}")
slope_lines.append(f"ASD mean: {np.mean(asd):.10g}")
slope_lines.append(f"ASD SEM:  {pd.Series(asd).sem():.10g}")
slope_lines.append("")
slope_lines.append("Significance source:")
slope_lines.append(f"Main model interaction term: {interaction_term}")
slope_lines.append(f"Interaction p-value: {model.pvalues[interaction_term]:.12g}")

with open(SUBJECT_SLOPE_STATS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(slope_lines))


# =========================================================
# 9) 保存运行信息
# =========================================================
lines = []

lines.append("Result1 all-systems calculation finished")
lines.append("=" * 70)
lines.append(f"Input data: {DATA_FILE}")
lines.append(f"Systems: {SYSTEM_ORDER}")
lines.append(f"Formula: {formula}")
lines.append(f"OUTCOME_SCALE: {OUTCOME_SCALE}")
lines.append("")
lines.append(f"N subjects: {df['sub_id'].nunique()}")
lines.append(f"N observations: {len(df)}")
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
lines.append("Primary terms:")
lines.append(f"G_star p-value, HC slope: {hc_stats['p_value']}")
lines.append(
    f"{interaction_term} p-value, ASD-HC slope difference: "
    f"{int_stats['p_value']}"
)
lines.append("")
lines.append("Simple slopes:")
lines.append(str(simple_slopes))
lines.append("")
lines.append("Saved files:")
lines.append(str(DATA_USED_CSV))
lines.append(str(MODEL_SUMMARY_PATH))
lines.append(str(MODEL_PARAMS_PATH))
lines.append(str(SIMPLE_SLOPE_PATH))
lines.append(str(PREDICTION_CSV))
lines.append(str(SUBJECT_SLOPE_CSV))
lines.append(str(SUBJECT_SLOPE_STATS_TXT))

with open(RUN_INFO_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# =========================================================
# 10) Print
# =========================================================
print("All-systems calculation finished.")
print("Data used:", DATA_USED_CSV)
print("Model summary:", MODEL_SUMMARY_PATH)
print("Model params:", MODEL_PARAMS_PATH)
print("Simple slopes:", SIMPLE_SLOPE_PATH)
print("Prediction lines:", PREDICTION_CSV)
print("Subject slopes:", SUBJECT_SLOPE_CSV)
print("Run info:", RUN_INFO_TXT)

print("")
print("Key terms:")
for term in ["G_star", interaction_term, "Group[T.ASD]"]:
    if term in model.params.index:
        print(
            f"{term}: beta={model.params[term]:.6g}, "
            f"SE={model.bse[term]:.6g}, "
            f"z/t={model.tvalues[term]:.3f}, "
            f"p={model.pvalues[term]:.6g}"
        )
    else:
        print(f"{term}: not found")

print("")
print("Simple slopes:")
print(simple_slopes)