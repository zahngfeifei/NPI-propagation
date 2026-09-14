# =====================================================
# Step 5 → 统计分析（G1/G2 固定模型版）—— 负向连接适配版
# + 与正向连接 Step 6 脚本保持同构
# + Site 和 MeanFD 已通过 ComBat 控制，不再纳入 GLM
# + 主要结果保存为 CSV
# + 全部模型参数保存为 CSV
# + 单模型 summary 保存为 TXT
# + 群体级总结报告（GLM_summary_report.txt）
# =====================================================

import os
import re
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from datetime import datetime

# -----------------------------------------------------
# 路径设置
# -----------------------------------------------------
PREFIX = "负向连接_"

DATA_PATH = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果3\负向连接_step5_gradient_compression_metrics.csv"
SUB_INFO_PATH = r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"

OUT_DIR = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果4"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------
# 配置
# -----------------------------------------------------
GROUP_TERM = "C(Group, Treatment(reference='HC'))[T.ASD]"
ALPHA_FDR = 0.05

# 与正向连接脚本一致：
# main 指标：G1_P95_P5、G2_P95_P5
# consistency 指标：G1_std、G2_std
DVS = [
    ("G1_P95_P5", "G1", "P95_P5", "main"),
    ("G1_std",    "G1", "std",    "consistency"),
    ("G2_P95_P5", "G2", "P95_P5", "main"),
    ("G2_std",    "G2", "std",    "consistency"),
]

# -----------------------------------------------------
# 工具函数：提取数值型被试 ID
# -----------------------------------------------------
def extract_numeric_id(sub_id):
    """
    sub-Sub0050667 → 50667
    sub-Sub50667   → 50667
    """
    if pd.isna(sub_id):
        return None
    m = re.search(r"Sub0*(\d+)", str(sub_id))
    return int(m.group(1)) if m else None


def fit_glm_for_dv(df, dv):
    """
    对指定 DV 拟合固定 GLM。
    与正向连接脚本一致，但删除 Site 和 MeanFD：
      Y ~ Group + Age + Sex + FIQ
    其中 Group 以 HC 为 reference，Sex 作为分类固定效应。
    """
    formula = (
        f"{dv} ~ "
        "C(Group, Treatment(reference='HC')) "
        "+ Age + C(Sex) + FIQ"
    )
    model = smf.ols(formula, data=df).fit()
    return model, formula


def extract_model_params(model, dv, gradient, metric, metric_role, formula):
    """
    导出模型全部参数。
    """
    ci = model.conf_int()

    rows = []
    for term in model.params.index:
        rows.append({
            "dv": dv,
            "gradient": gradient,
            "metric": metric,
            "metric_role": metric_role,
            "term": term,
            "formula": formula,
            "coef": float(model.params[term]),
            "t": float(model.tvalues[term]),
            "p": float(model.pvalues[term]),
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "adj_r2": float(model.rsquared_adj),
        })

    return rows


def extract_group_effect(model, dv, gradient, metric, metric_role, formula):
    """
    提取 ASD vs HC 组别效应。
    """
    if GROUP_TERM not in model.params.index:
        raise RuntimeError(
            f"模型缺少组别项：{GROUP_TERM}\n"
            f"DV={dv}\nformula={formula}"
        )

    ci = model.conf_int().loc[GROUP_TERM].tolist()

    return {
        "dv": dv,
        "gradient": gradient,
        "metric": metric,
        "metric_role": metric_role,
        "term": GROUP_TERM,
        "formula": formula,
        "beta_ASD_vs_HC": float(model.params[GROUP_TERM]),
        "t": float(model.tvalues[GROUP_TERM]),
        "p": float(model.pvalues[GROUP_TERM]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }


def add_fdr_by_metric_role(df, p_col="p", alpha=0.05):
    """
    与正向连接脚本一致，按 metric_role 分开做 FDR：
      1) main：G1_P95_P5 与 G2_P95_P5 一组
      2) consistency：G1_std 与 G2_std 一组
    """
    if df.empty:
        out = df.copy()
        out["p_FDR"] = pd.NA
        out["sig_FDR"] = pd.NA
        out["sig"] = ""
        return out

    out_list = []

    for _, sub_df in df.groupby("metric_role", sort=False):
        sub_df = sub_df.copy()
        reject, p_fdr = fdrcorrection(sub_df[p_col].values, alpha=alpha)
        sub_df["p_FDR"] = p_fdr
        sub_df["sig_FDR"] = reject.astype(int)
        sub_df["sig"] = ["FDR<0.05" if r else "" for r in reject]
        out_list.append(sub_df)

    return pd.concat(out_list, axis=0, ignore_index=True)


# -----------------------------------------------------
# 读取数据
# -----------------------------------------------------
df_metric = pd.read_csv(DATA_PATH)
df_info = pd.read_csv(SUB_INFO_PATH)

# -----------------------------------------------------
# 统一数值 ID
# -----------------------------------------------------
df_metric["sub_num_id"] = df_metric["sub_id"].apply(extract_numeric_id)
df_info["sub_num_id"] = df_info["sub_id"].apply(extract_numeric_id)

if df_metric["sub_num_id"].isna().any():
    bad = df_metric.loc[df_metric["sub_num_id"].isna(), "sub_id"].head(10).tolist()
    raise ValueError(f"负向梯度文件中存在无法解析的 sub_id，例如: {bad}")

if df_info["sub_num_id"].isna().any():
    bad = df_info.loc[df_info["sub_num_id"].isna(), "sub_id"].head(10).tolist()
    raise ValueError(f"被试信息表中存在无法解析的 sub_id，例如: {bad}")

# -----------------------------------------------------
# 合并
# -----------------------------------------------------
df = pd.merge(
    df_metric,
    df_info,
    on="sub_num_id",
    how="inner",
    suffixes=("_metric", "_info")
)

n_metric = len(df_metric)
n_info = len(df_info)
n_merge = len(df)

# -----------------------------------------------------
# 基础清洗
# -----------------------------------------------------
dv_cols = [x[0] for x in DVS]

required_cols = (
    dv_cols
    + ["Group", "Age", "Sex", "FIQ"]
)

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"缺少列: {col}\n"
            f"如果缺少 G2_P95_P5 或 G2_std，说明负向 Step 5 目前还没有输出 G2 compression 指标；"
            f"请先将负向 Step 5 修改为同时输出 G1 和 G2 指标。"
        )

# 数值列转换
for col in dv_cols + ["Age", "FIQ"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols).copy()
n_final = len(df)

if n_final == 0:
    raise RuntimeError("清洗后没有可用于建模的被试。")

# 分类变量
df["Group"] = df["Group"].astype("category")
df["Sex"] = df["Sex"].astype("category")

if "HC" not in df["Group"].cat.categories:
    raise ValueError("Group 中未找到 HC，无法作为 reference。")

# -----------------------------------------------------
# 逐 DV 拟合模型
# -----------------------------------------------------
group_effect_rows = []
all_param_rows = []

for dv, gradient, metric, metric_role in DVS:
    model, formula = fit_glm_for_dv(df, dv)

    # 保存完整模型 summary txt
    summary_txt = os.path.join(OUT_DIR, f"{PREFIX}GLM_{dv}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # 保存每个模型单独参数 CSV
    param_rows = extract_model_params(
        model=model,
        dv=dv,
        gradient=gradient,
        metric=metric,
        metric_role=metric_role,
        formula=formula,
    )
    pd.DataFrame(param_rows).to_csv(
        os.path.join(OUT_DIR, f"{PREFIX}{dv}_params.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    all_param_rows.extend(param_rows)

    # 提取 ASD vs HC 组别效应
    group_effect_rows.append(
        extract_group_effect(
            model=model,
            dv=dv,
            gradient=gradient,
            metric=metric,
            metric_role=metric_role,
            formula=formula,
        )
    )

# -----------------------------------------------------
# 保存主要结果：组别效应表 + FDR
# -----------------------------------------------------
df_group = pd.DataFrame(group_effect_rows)
df_group = add_fdr_by_metric_role(df_group, p_col="p", alpha=ALPHA_FDR)

main_results_path = os.path.join(
    OUT_DIR,
    f"{PREFIX}GLM_group_effects_G1G2_main_results.csv"
)
df_group.to_csv(main_results_path, index=False, encoding="utf-8-sig")

# -----------------------------------------------------
# 保存所有模型参数
# -----------------------------------------------------
df_all_params = pd.DataFrame(all_param_rows)

all_params_path = os.path.join(
    OUT_DIR,
    f"{PREFIX}GLM_all_model_params_G1G2.csv"
)
df_all_params.to_csv(all_params_path, index=False, encoding="utf-8-sig")

# -----------------------------------------------------
# 群体级总结报告
# -----------------------------------------------------
summary_path = os.path.join(OUT_DIR, f"{PREFIX}GLM_summary_report.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("GLM Summary Report: NEGATIVE G1/G2 Gradient Compression\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Paths:\n")
    f.write(f"  DATA_PATH: {DATA_PATH}\n")
    f.write(f"  SUB_INFO_PATH: {SUB_INFO_PATH}\n")
    f.write(f"  OUT_DIR: {OUT_DIR}\n\n")

    f.write("Sample size:\n")
    f.write(f"  Metric subjects: {n_metric}\n")
    f.write(f"  Info subjects:   {n_info}\n")
    f.write(f"  Merged subjects: {n_merge}\n")
    f.write(f"  Final N after QC: {n_final}\n\n")

    f.write("Model specification:\n")
    f.write("  Model type: fixed-effects GLM (OLS)\n")
    f.write("  Reference group: HC\n")
    f.write("  Group contrast: ASD vs HC\n")
    f.write("  Covariates: Age, Sex, FIQ\n")
    f.write("  Sex modeled as a categorical fixed effect.\n")
    f.write("  Site and MeanFD were not included because they have already been controlled by ComBat.\n\n")

    f.write("Dependent variables:\n")
    for dv, gradient, metric, metric_role in DVS:
        f.write(f"  - {dv}: gradient={gradient}, metric={metric}, role={metric_role}\n")
    f.write("\n")

    f.write("Multiple comparison correction:\n")
    f.write("  Method: Benjamini-Hochberg FDR\n")
    f.write("  Family 1 (main): G1_P95_P5 and G2_P95_P5\n")
    f.write("  Family 2 (consistency): G1_std and G2_std\n\n")

    f.write("Main group effects:\n")
    for _, row in df_group.iterrows():
        f.write(
            f"  {row['dv']}: "
            f"β = {row['beta_ASD_vs_HC']:.4f}, "
            f"t = {row['t']:.3f}, "
            f"p = {row['p']:.4e}, "
            f"p_FDR = {row['p_FDR']:.4e}, "
            f"sig = {row['sig']}\n"
        )
    f.write("\n")

    f.write("Output files:\n")
    f.write(f"  Main group-effect results: {main_results_path}\n")
    f.write(f"  All model parameters: {all_params_path}\n\n")

    f.write("Notes:\n")
    f.write("  - This negative-connectivity Step 6 script was matched to the positive-connectivity G1/G2 script.\n")
    f.write("  - Site and MeanFD were removed from the GLM because they have already been controlled by ComBat.\n")
    f.write("  - G1 and G2 compression metrics were analyzed using the same fixed model.\n")
    f.write("  - Main metrics and consistency metrics were FDR-corrected separately.\n")
    f.write("  - The summary report does not replace the CSV result tables or full model outputs.\n")

print("\n=== Step 6 统计分析完成（负向连接 G1/G2 固定模型版） ===")
print("主要结果 CSV:", main_results_path)
print("全部模型参数 CSV:", all_params_path)
print("Summary report:", summary_path)
