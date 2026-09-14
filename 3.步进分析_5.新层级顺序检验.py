# -*- coding: utf-8 -*-
"""
结果三补充统计（加权传播指标适配版）：
检验基于有效连接的 EC-SEC 传播是否具有显著层级顺序

适配说明：
- 最新加权传播脚本的 EC_SEC_metrics_all_subjects.csv 不再输出 first_arrival_step。
- 当前主检验指标改为 temporal_centroid / propagation center of mass。
- temporal_centroid 数值越小，表示该系统传播质量的时间重心越早。
- Friedman test 仍用于检验 H1/H2/H3/H4 在被试内是否存在整体层级差异。
- post-hoc 仍使用配对 Wilcoxon signed-rank test。

输出说明：
- 为避免破坏下游读取流程，输出目录和文件名保持原脚本不变。
- long-format 文件中保留 first_arrival_step 兼容列，但该列现在存放的是 temporal_centroid 值。
- 所有输出表和 txt 中增加 actual_metric / metric_interpretation，用于明确当前检验指标。

输入：
  I:\\DYF\\NPI-2\\3.步进分析-权重矩阵\\新结果1\\EC_SEC_metrics_all_subjects.csv

输出目录：
  I:\\DYF\\NPI-2\\3.步进分析-权重矩阵\\结果3
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests


# =====================================================
# 路径设置
# =====================================================
# 最新加权传播脚本输出位置：
# OUT_ROOT = I:\DYF\NPI-2\3.步进分析-权重矩阵\新结果1
METRIC_CSV = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1_combat\EC_SEC_metrics_all_subjects_combat.csv"

# 输出目录沿用原脚本设置
OUT_DIR = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果3"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# 文件名沿用既有流程约定
OUT_WIDE_USED = os.path.join(
    OUT_DIR,
    "result3_first_arrival_subject_level_wide_used.csv"
)

OUT_LONG_USED = os.path.join(
    OUT_DIR,
    "result3_first_arrival_subject_level_long_used.csv"
)

OUT_DESC = os.path.join(
    OUT_DIR,
    "result3_first_arrival_descriptive_stats.csv"
)

OUT_FRIEDMAN = os.path.join(
    OUT_DIR,
    "result3_first_arrival_friedman_test.csv"
)

OUT_PAIRWISE = os.path.join(
    OUT_DIR,
    "result3_first_arrival_pairwise_wilcoxon.csv"
)

OUT_TXT = os.path.join(
    OUT_DIR,
    "result3_first_arrival_friedman_summary.txt"
)


# =====================================================
# 参数设置
# =====================================================
ALPHA = 0.05

# post-hoc 多重比较校正方法
# 可选："fdr_bh", "holm", "bonferroni"
PAIRWISE_CORRECTION = "fdr_bh"

# 主检验指标：
# 最新 EC-SEC 加权传播指标表包含：
#   - peak
#   - temporal_centroid
#   - early_slope_1_10
#   - early_auc_1_10
#
# 推荐用于“层级时间顺序”检验的是 temporal_centroid。
# 数值越小，表示传播质量的时间重心越早。
PRIMARY_METRIC = "temporal_centroid"

METRIC_INTERPRETATION = {
    "peak": "smaller value means earlier peak step",
    "temporal_centroid": "smaller value means earlier propagation temporal center of mass",
    "early_slope_1_10": "larger value means steeper early increase; smaller value does not necessarily mean earlier arrival",
    "early_auc_1_10": "larger value means stronger early propagation; smaller value does not necessarily mean earlier arrival",
}

# 对当前主指标，是否按“数值越小 = 越早”解释方向。
# temporal_centroid 和 peak 为 True。
# early_slope / early_auc 若用于 Friedman，方向解释应谨慎。
SMALLER_IS_EARLIER = PRIMARY_METRIC in ["peak", "temporal_centroid"]

# Friedman 检验的四个层级
SYSTEM_CONFIGS = [
    {
        "Level": "H1",
        "System": "H1_sensory",
        "Preferred": f"H1_sensory_{PRIMARY_METRIC}",
        "Fallbacks": [
            f"H1_{PRIMARY_METRIC}",
            f"H1_sensory_{PRIMARY_METRIC.replace('_1_10', '')}",
        ],
    },
    {
        "Level": "H2",
        "System": "H2_attention",
        "Preferred": f"H2_attention_{PRIMARY_METRIC}",
        "Fallbacks": [
            f"H2_{PRIMARY_METRIC}",
            f"H2_attention_{PRIMARY_METRIC.replace('_1_10', '')}",
        ],
    },
    {
        "Level": "H3",
        "System": "H3_control",
        "Preferred": f"H3_control_{PRIMARY_METRIC}",
        "Fallbacks": [
            f"H3_{PRIMARY_METRIC}",
            f"H3_control_{PRIMARY_METRIC.replace('_1_10', '')}",
        ],
    },
    {
        "Level": "H4",
        "System": "H4_DMN",
        "Preferred": f"H4_DMN_{PRIMARY_METRIC}",
        "Fallbacks": [
            f"H4_{PRIMARY_METRIC}",
            f"H4_DMN_{PRIMARY_METRIC.replace('_1_10', '')}",
        ],
    },
]


# =====================================================
# 工具函数
# =====================================================
def resolve_column(df_cols, preferred, fallbacks):
    """
    按 preferred -> fallback 顺序寻找真实存在的列名。
    """
    if preferred in df_cols:
        return preferred

    for c in fallbacks:
        if c in df_cols:
            return c

    return None


def safe_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def iqr(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.quantile(0.75) - x.quantile(0.25))


def rank_biserial_for_paired(a, b):
    """
    配对 Wilcoxon 的 rank-biserial effect size。

    diff = a - b

    当 PRIMARY_METRIC = temporal_centroid 或 peak 时：
    - r_rb < 0：a 通常小于 b，即 a 更早
    - r_rb > 0：a 通常大于 b，即 a 更晚
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    d = a[valid] - b[valid]

    d_nonzero = d[d != 0]

    if len(d_nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(d_nonzero), method="average")

    r_pos = np.sum(ranks[d_nonzero > 0])
    r_neg = np.sum(ranks[d_nonzero < 0])
    denom = np.sum(ranks)

    if denom == 0:
        return np.nan

    return float((r_pos - r_neg) / denom)


def wilcoxon_pairwise(a, b):
    """
    安全执行配对 Wilcoxon signed-rank test。
    如果所有差值均为 0，则返回 p=1。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    n = len(a)

    if n == 0:
        return np.nan, np.nan, 0

    diff = a - b

    if np.all(diff == 0):
        return 0.0, 1.0, n

    try:
        stat, p = stats.wilcoxon(
            a,
            b,
            zero_method="pratt",
            alternative="two-sided"
        )
        return float(stat), float(p), n

    except Exception:
        return np.nan, np.nan, n


def direction_from_median(med_a, med_b, label_a, label_b):
    """
    对方向进行描述。

    对 temporal_centroid / peak：
      数值越小，表示越早。
    对 early_slope / early_auc：
      这里只描述数值大小，不直接解释为早晚。
    """
    if not np.isfinite(med_a) or not np.isfinite(med_b):
        return "NA"

    if SMALLER_IS_EARLIER:
        if med_a < med_b:
            return f"{label_a} earlier than {label_b}"
        elif med_a > med_b:
            return f"{label_a} later than {label_b}"
        else:
            return f"{label_a} = {label_b}"
    else:
        if med_a < med_b:
            return f"{label_a} lower than {label_b}"
        elif med_a > med_b:
            return f"{label_a} higher than {label_b}"
        else:
            return f"{label_a} = {label_b}"


def metric_label_for_text():
    if PRIMARY_METRIC == "temporal_centroid":
        return "temporal_centroid / propagation center of mass"
    return PRIMARY_METRIC


# =====================================================
# 读取数据
# =====================================================
df = pd.read_csv(METRIC_CSV)

if "sub_id" not in df.columns:
    raise ValueError("输入文件缺少 sub_id 列。")

df_cols = set(df.columns.tolist())

resolved = []

print(f"\n[检查] {PRIMARY_METRIC} 指标列名解析：")

for spec in SYSTEM_CONFIGS:
    actual = resolve_column(
        df_cols=df_cols,
        preferred=spec["Preferred"],
        fallbacks=spec["Fallbacks"]
    )

    if actual is None:
        print(f"  ❌ 缺失: {spec['Preferred']} 以及 fallback")
    else:
        mark = "（fallback）" if actual != spec["Preferred"] else ""
        print(f"  ✅ {spec['Preferred']} -> {actual} {mark}")

    spec = spec.copy()
    spec["Actual_column"] = actual
    resolved.append(spec)

missing_specs = [s for s in resolved if s["Actual_column"] is None]

if len(missing_specs) > 0:
    missing_names = [s["Preferred"] for s in missing_specs]
    available_metric_cols = [
        c for c in df.columns
        if any(suffix in c for suffix in ["peak", "temporal_centroid", "early_slope", "early_auc"])
    ]

    raise RuntimeError(
        f"以下 {PRIMARY_METRIC} 指标缺失，无法做 Friedman test："
        + ", ".join(missing_names)
        + "\n当前指标表中可见的相关指标列示例："
        + ", ".join(available_metric_cols[:30])
    )


# =====================================================
# 构建 Friedman 使用的 wide 数据
# =====================================================
wide = pd.DataFrame()
wide["sub_id"] = df["sub_id"]
wide["actual_metric"] = PRIMARY_METRIC
wide["metric_interpretation"] = METRIC_INTERPRETATION.get(PRIMARY_METRIC, "NA")

# 如果有 Group，也保留；但 Friedman 主检验不按组分层
if "Group" in df.columns:
    wide["Group"] = df["Group"]

for spec in resolved:
    wide[spec["System"]] = safe_numeric(df[spec["Actual_column"]])

metric_cols = [spec["System"] for spec in resolved]

n_before = len(wide)

# Friedman 是配对检验，必须四个层级都有值
wide_used = wide.dropna(subset=metric_cols).copy()
n_after = len(wide_used)
n_dropped = n_before - n_after

if n_after < 5:
    raise RuntimeError(
        f"Friedman test 有效样本过少：n={n_after}。请检查 {PRIMARY_METRIC} 列。"
    )

wide_used.to_csv(
    OUT_WIDE_USED,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 转成长格式
# =====================================================
id_cols = ["sub_id", "actual_metric", "metric_interpretation"]
if "Group" in wide_used.columns:
    id_cols.append("Group")

long_used = wide_used.melt(
    id_vars=id_cols,
    value_vars=metric_cols,
    var_name="system",
    value_name="metric_value"
)

# 兼容旧输出字段名：
# 注意：该列现在不是 first_arrival_step，而是 PRIMARY_METRIC 的数值。
# 保留它是为了避免下游脚本因缺列而失败。
long_used["first_arrival_step"] = long_used["metric_value"]

if PRIMARY_METRIC == "temporal_centroid":
    long_used["temporal_centroid"] = long_used["metric_value"]
elif PRIMARY_METRIC == "peak":
    long_used["peak_step"] = long_used["metric_value"]
elif PRIMARY_METRIC == "early_slope_1_10":
    long_used["early_slope_1_10"] = long_used["metric_value"]
elif PRIMARY_METRIC == "early_auc_1_10":
    long_used["early_auc_1_10"] = long_used["metric_value"]

# 加入 hierarchy level 编号
level_map = {
    "H1_sensory": 1,
    "H2_attention": 2,
    "H3_control": 3,
    "H4_DMN": 4,
}

long_used["hierarchy_level"] = long_used["system"].map(level_map)

long_used.to_csv(
    OUT_LONG_USED,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 描述统计
# =====================================================
desc_rows = []

for spec in resolved:
    sys_name = spec["System"]
    x = wide_used[sys_name].dropna().astype(float)

    desc_rows.append({
        "actual_metric": PRIMARY_METRIC,
        "metric_interpretation": METRIC_INTERPRETATION.get(PRIMARY_METRIC, "NA"),
        "system": sys_name,
        "hierarchy_level": level_map[sys_name],
        "actual_column": spec["Actual_column"],
        "n": int(x.shape[0]),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "sem": float(x.std(ddof=1) / np.sqrt(len(x))),
        "median": float(x.median()),
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75)),
        "iqr": iqr(x),
        "min": float(x.min()),
        "max": float(x.max()),
    })

desc_df = pd.DataFrame(desc_rows)

desc_df.to_csv(
    OUT_DESC,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# Friedman test
# =====================================================
arrays = [
    wide_used[sys_name].astype(float).values
    for sys_name in metric_cols
]

friedman_stat, friedman_p = stats.friedmanchisquare(*arrays)

k = len(metric_cols)
n = len(wide_used)

# Kendall's W for Friedman:
# W = chi-square / (n * (k - 1))
kendalls_w = float(friedman_stat / (n * (k - 1)))

# 每个被试内对 H1-H4 排名
# 对 temporal_centroid / peak：数值越小，rank 越小，表示越早
rank_matrix = np.apply_along_axis(
    lambda row: rankdata(row, method="average"),
    axis=1,
    arr=wide_used[metric_cols].astype(float).values
)

mean_ranks = rank_matrix.mean(axis=0)

friedman_rows = []

for idx, sys_name in enumerate(metric_cols):
    friedman_rows.append({
        "test": "Friedman test",
        "actual_metric": PRIMARY_METRIC,
        "metric_interpretation": METRIC_INTERPRETATION.get(PRIMARY_METRIC, "NA"),
        "n_subjects_used": n,
        "n_subjects_dropped_due_to_missing": n_dropped,
        "k_conditions": k,
        "system": sys_name,
        "hierarchy_level": level_map[sys_name],
        "mean_rank": float(mean_ranks[idx]),
        "friedman_chi_square": float(friedman_stat),
        "df": int(k - 1),
        "p_value": float(friedman_p),
        "kendalls_W": kendalls_w,
        "significant_0.05": bool(friedman_p < ALPHA),
        "interpretation": (
            f"{PRIMARY_METRIC} differs across hierarchy levels"
            if friedman_p < ALPHA
            else f"no significant difference in {PRIMARY_METRIC} across hierarchy levels"
        )
    })

friedman_df = pd.DataFrame(friedman_rows)

friedman_df.to_csv(
    OUT_FRIEDMAN,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# Post-hoc pairwise Wilcoxon signed-rank tests
# =====================================================
pairwise_rows = []

for i in range(k):
    for j in range(i + 1, k):
        sys_a = metric_cols[i]
        sys_b = metric_cols[j]

        a = wide_used[sys_a].astype(float).values
        b = wide_used[sys_b].astype(float).values

        stat, p, n_pair = wilcoxon_pairwise(a, b)

        mean_a = float(np.nanmean(a))
        mean_b = float(np.nanmean(b))
        median_a = float(np.nanmedian(a))
        median_b = float(np.nanmedian(b))

        diff = a - b

        pairwise_rows.append({
            "actual_metric": PRIMARY_METRIC,
            "metric_interpretation": METRIC_INTERPRETATION.get(PRIMARY_METRIC, "NA"),
            "comparison": f"{sys_a} vs {sys_b}",
            "system_A": sys_a,
            "system_B": sys_b,
            "n_pairs": int(n_pair),
            "wilcoxon_statistic": stat,
            "p_uncorrected": p,
            "mean_A": mean_a,
            "mean_B": mean_b,
            "mean_diff_A_minus_B": float(np.nanmean(diff)),
            "median_A": median_a,
            "median_B": median_b,
            "median_diff_A_minus_B": float(np.nanmedian(diff)),
            "rank_biserial_effect_size": rank_biserial_for_paired(a, b),
            "direction_by_median": direction_from_median(
                median_a,
                median_b,
                sys_a,
                sys_b
            ),
        })

pairwise_df = pd.DataFrame(pairwise_rows)

if len(pairwise_df) > 0:
    valid_mask = pairwise_df["p_uncorrected"].notna()

    pairwise_df["p_corrected"] = np.nan
    pairwise_df["significant_corrected_0.05"] = False
    pairwise_df["correction_method"] = PAIRWISE_CORRECTION

    if valid_mask.sum() > 0:
        corrected = multipletests(
            pairwise_df.loc[valid_mask, "p_uncorrected"].values,
            alpha=ALPHA,
            method=PAIRWISE_CORRECTION
        )

        pairwise_df.loc[valid_mask, "significant_corrected_0.05"] = corrected[0]
        pairwise_df.loc[valid_mask, "p_corrected"] = corrected[1]

pairwise_df.to_csv(
    OUT_PAIRWISE,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 生成 txt 总结报告
# =====================================================
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("Result 3 Supplementary Statistics: Friedman Test for EC-SEC Propagation Hierarchy\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Input:\n")
    f.write(f"  Metrics CSV: {METRIC_CSV}\n\n")

    f.write("Output directory:\n")
    f.write(f"  {OUT_DIR}\n\n")

    f.write("Adaptation note:\n")
    f.write("  The latest weighted propagation metrics table does not contain first_arrival_step columns.\n")
    f.write(f"  The primary test metric is now: {PRIMARY_METRIC}\n")
    f.write(f"  Metric meaning: {METRIC_INTERPRETATION.get(PRIMARY_METRIC, 'NA')}\n")
    f.write("  Output file names are kept unchanged for compatibility with the previous workflow.\n")
    f.write("  In the long-format output, first_arrival_step is retained as a compatibility alias for metric_value.\n\n")

    f.write("Purpose:\n")
    f.write(
        f"  Test whether individual-level {metric_label_for_text()} values "
        "differ across H1-H4 hierarchy levels.\n"
    )
    f.write(
        "  A significant Friedman test supports non-random hierarchy-level differences "
        "in the temporal propagation metric.\n\n"
    )

    f.write("Column resolution:\n")
    for spec in resolved:
        f.write(
            f"  {spec['System']}: preferred={spec['Preferred']} -> actual={spec['Actual_column']}\n"
        )
    f.write("\n")

    f.write("Sample size:\n")
    f.write(f"  Subjects in input file: {n_before}\n")
    f.write(f"  Subjects used for Friedman test: {n_after}\n")
    f.write(f"  Subjects dropped due to missing H1-H4 metric values: {n_dropped}\n\n")

    f.write("Descriptive statistics:\n")
    for _, row in desc_df.iterrows():
        f.write(
            f"  {row['system']}: "
            f"mean={row['mean']:.4f}, sd={row['sd']:.4f}, "
            f"median={row['median']:.4f}, IQR={row['iqr']:.4f}, "
            f"min={row['min']:.4f}, max={row['max']:.4f}, "
            f"n={int(row['n'])}\n"
        )
    f.write("\n")

    f.write("Friedman test:\n")
    f.write(f"  metric = {PRIMARY_METRIC}\n")
    f.write(f"  chi-square({k - 1}) = {friedman_stat:.6g}\n")
    f.write(f"  p = {friedman_p:.6g}\n")
    f.write(f"  Kendall's W = {kendalls_w:.6g}\n")
    f.write(f"  Significant at alpha={ALPHA}: {bool(friedman_p < ALPHA)}\n\n")

    if SMALLER_IS_EARLIER:
        f.write("Mean ranks, smaller rank means earlier temporal propagation metric:\n")
    else:
        f.write("Mean ranks, smaller rank means lower metric value:\n")

    for sys_name, mr in zip(metric_cols, mean_ranks):
        f.write(f"  {sys_name}: mean_rank={mr:.4f}\n")
    f.write("\n")

    f.write("Post-hoc pairwise Wilcoxon signed-rank tests:\n")
    f.write(f"  Correction method: {PAIRWISE_CORRECTION}\n")
    for _, row in pairwise_df.iterrows():
        f.write(
            f"  {row['comparison']}: "
            f"median_A={row['median_A']:.4f}, "
            f"median_B={row['median_B']:.4f}, "
            f"median_diff_A_minus_B={row['median_diff_A_minus_B']:.4f}, "
            f"W={row['wilcoxon_statistic']:.6g}, "
            f"p_uncorrected={row['p_uncorrected']:.6g}, "
            f"p_corrected={row['p_corrected']:.6g}, "
            f"sig_corrected={bool(row['significant_corrected_0.05'])}, "
            f"rank_biserial={row['rank_biserial_effect_size']:.6g}, "
            f"direction={row['direction_by_median']}\n"
        )
    f.write("\n")

    f.write("Interpretation guide:\n")
    if PRIMARY_METRIC == "temporal_centroid":
        f.write("  - temporal_centroid = sum_t[t * SEC(t)] / sum_t[SEC(t)].\n")
        f.write("  - Smaller temporal_centroid means the system's propagation mass is centered earlier.\n")
        f.write("  - Friedman test evaluates whether H1-H4 temporal centroids differ within subjects.\n")
    elif PRIMARY_METRIC == "peak":
        f.write("  - peak is the step where the system SEC curve reaches its maximum.\n")
        f.write("  - Smaller peak means the system reaches its maximum earlier.\n")
        f.write("  - Friedman test evaluates whether H1-H4 peak steps differ within subjects.\n")
    else:
        f.write(f"  - Current metric is {PRIMARY_METRIC}.\n")
        f.write("  - Direction should be interpreted according to metric_interpretation.\n")
        f.write("  - Friedman test evaluates whether H1-H4 metric values differ within subjects.\n")

    f.write("  - Kendall's W quantifies the strength of hierarchy-level differences.\n")
    f.write("  - Pairwise Wilcoxon tests identify which hierarchy levels differ.\n")
    f.write("  - If H1 < H2 < H3/H4 is supported for temporal_centroid, this indicates a propagation hierarchy from lower-order to higher-order systems.\n\n")

    f.write("Outputs:\n")
    f.write(f"  Wide subject-level data used: {OUT_WIDE_USED}\n")
    f.write(f"  Long subject-level data used: {OUT_LONG_USED}\n")
    f.write(f"  Descriptive statistics: {OUT_DESC}\n")
    f.write(f"  Friedman test result: {OUT_FRIEDMAN}\n")
    f.write(f"  Pairwise Wilcoxon tests: {OUT_PAIRWISE}\n")
    f.write(f"  Summary txt: {OUT_TXT}\n")

print(f"\n=== Friedman test for EC-SEC {PRIMARY_METRIC} hierarchy completed ===")
print("输入文件:", METRIC_CSV)
print("输出目录:", OUT_DIR)
print("当前检验指标:", PRIMARY_METRIC)
print("指标解释:", METRIC_INTERPRETATION.get(PRIMARY_METRIC, "NA"))
print("有效被试数:", n_after)
print("Friedman chi-square:", friedman_stat)
print("Friedman p:", friedman_p)
print("Kendall's W:", kendalls_w)
print("描述统计:", OUT_DESC)
print("Friedman 结果:", OUT_FRIEDMAN)
print("Pairwise Wilcoxon 结果:", OUT_PAIRWISE)
print("总结报告:", OUT_TXT)
