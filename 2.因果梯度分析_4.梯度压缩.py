# -*- coding: utf-8 -*-
# =====================================================
# Step 5: Gradient compression / dedifferentiation
#          NEGATIVE effective-connectivity gradients (G1/G2)
# + 与正向连接 Step5 保持一致
# + 同时计算 Procrustes-aligned G1 和 G2
# + 不做 scaling / normalization
# + 输出文件统一加前缀“负向连接_”
#
# 运行前提：
#   负向连接 Step4 / Procrustes 对齐已经在
#   I:\DYF\NPI-2\2.因果梯度分析\结果2_负向\sub-*
#   下输出：
#       out_G1_procrustes.npy
#       out_G2_procrustes.npy
# =====================================================

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------------------------------
# 路径设置
# -----------------------------------------------------
PREFIX = "负向连接_"

aligned_root = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果2_combat"  # 负向 Step4/Procrustes 输出根目录
out_root     = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果3"  # 负向 Step5 输出目录
os.makedirs(out_root, exist_ok=True)

# -----------------------------------------------------
# 参数（论文需声明）
# -----------------------------------------------------
EPS_STD = 1e-12
MIN_LEN = 10
GRADIENTS = ["G1", "G2"]

# -----------------------------------------------------
# 工具函数
# -----------------------------------------------------
def compute_gradient_metrics(g: np.ndarray, grad_name: str) -> dict:
    """
    计算单个梯度的 compression / dispersion 指标。
    不进行 z-score 或其他归一化，直接使用 Procrustes-aligned 梯度值。
    """
    g = np.asarray(g).reshape(-1)

    if g.ndim != 1:
        raise ValueError(f"{grad_name} not 1D")
    if len(g) < MIN_LEN:
        raise ValueError(f"{grad_name} too short: len={len(g)}")
    if not np.isfinite(g).all():
        raise ValueError(f"{grad_name} contains NaN/Inf")

    g_mean = float(np.mean(g))
    g_std  = float(np.std(g))

    if g_std < EPS_STD:
        raise ValueError(f"{grad_name} std too small: {g_std}")

    g_min = float(np.min(g))
    g_max = float(np.max(g))
    g_range = g_max - g_min

    p5  = float(np.percentile(g, 5))
    p25 = float(np.percentile(g, 25))
    p50 = float(np.percentile(g, 50))
    p75 = float(np.percentile(g, 75))
    p95 = float(np.percentile(g, 95))

    return {
        f"{grad_name}_N": len(g),
        f"{grad_name}_mean": g_mean,
        f"{grad_name}_std": g_std,
        f"{grad_name}_min": g_min,
        f"{grad_name}_max": g_max,
        f"{grad_name}_range": g_range,
        f"{grad_name}_P95": p95,
        f"{grad_name}_P75": p75,
        f"{grad_name}_P50": p50,
        f"{grad_name}_P25": p25,
        f"{grad_name}_P5": p5,
        f"{grad_name}_P95_P5": p95 - p5,
        f"{grad_name}_IQR": p75 - p25,
        f"{grad_name}_MAD": float(np.median(np.abs(g - p50))),
    }


def mean_sd_text(df: pd.DataFrame, col: str) -> str:
    """用于报告中格式化 mean ± sd。"""
    if df.empty or col not in df.columns:
        return "NA"
    return f"{df[col].mean():.4f} ± {df[col].std():.4f}"


# -----------------------------------------------------
# 搜索被试
# -----------------------------------------------------
sub_dirs = sorted([
    d for d in glob.glob(os.path.join(aligned_root, "sub-*"))
    if os.path.isdir(d)
])

print(f"发现 {len(sub_dirs)} 个负向连接被试目录")

# -----------------------------------------------------
# 容器
# -----------------------------------------------------
rows_ok = []
rows_failed = []

# -----------------------------------------------------
# 主循环：同时要求 G1 和 G2 均存在且通过 QC
# -----------------------------------------------------
for sub_dir in sub_dirs:

    sub_id = os.path.basename(sub_dir)
    g1_path = os.path.join(sub_dir, "out_G1_procrustes.npy")
    g2_path = os.path.join(sub_dir, "out_G2_procrustes.npy")

    missing_files = []
    if not os.path.exists(g1_path):
        missing_files.append("out_G1_procrustes.npy")
    if not os.path.exists(g2_path):
        missing_files.append("out_G2_procrustes.npy")

    if missing_files:
        rows_failed.append({
            "sub_id": sub_id,
            "reason": "missing " + ", ".join(missing_files),
        })
        continue

    try:
        g1 = np.load(g1_path).reshape(-1)
        g2 = np.load(g2_path).reshape(-1)

        if len(g1) != len(g2):
            raise ValueError(f"G1/G2 length mismatch: G1={len(g1)} G2={len(g2)}")

        row = {
            "sub_id": sub_id,
            "N": len(g1),
        }
        row.update(compute_gradient_metrics(g1, "G1"))
        row.update(compute_gradient_metrics(g2, "G2"))
        row["qc_pass"] = 1

        rows_ok.append(row)

    except Exception as e:
        rows_failed.append({
            "sub_id": sub_id,
            "reason": str(e),
        })

# -----------------------------------------------------
# 输出主表
# -----------------------------------------------------
df_ok = pd.DataFrame(rows_ok)

csv_ok = os.path.join(out_root, f"{PREFIX}step5_gradient_compression_metrics.csv")
df_ok.to_csv(csv_ok, index=False, encoding="utf-8-sig")

# -----------------------------------------------------
# 输出失败被试
# -----------------------------------------------------
csv_fail = os.path.join(out_root, f"{PREFIX}step5_failed_subjects.csv")

if rows_failed:
    df_fail = pd.DataFrame(rows_failed)
    df_fail.to_csv(csv_fail, index=False, encoding="utf-8-sig")
else:
    df_fail = pd.DataFrame(columns=["sub_id", "reason"])
    df_fail.to_csv(csv_fail, index=False, encoding="utf-8-sig")

# -----------------------------------------------------
# 汇总统计
# -----------------------------------------------------
summary_stats = {
    "n_success": len(df_ok),
    "n_failed": len(rows_failed),
}

for grad in GRADIENTS:
    for metric in ["std", "P95_P5", "IQR", "MAD"]:
        col = f"{grad}_{metric}"
        summary_stats[f"{col}_mean"] = float(df_ok[col].mean()) if (not df_ok.empty and col in df_ok.columns) else np.nan
        summary_stats[f"{col}_sd"]   = float(df_ok[col].std())  if (not df_ok.empty and col in df_ok.columns) else np.nan

summary_stats_path = os.path.join(out_root, f"{PREFIX}step5_summary_stats.txt")
with open(summary_stats_path, "w", encoding="utf-8") as f:
    for k, v in summary_stats.items():
        f.write(f"{k}: {v}\n")

# -----------------------------------------------------
# 群体总结报告（论文 / 审稿用）
# -----------------------------------------------------
summary_report_path = os.path.join(out_root, f"{PREFIX}step5_summary_report.txt")

near_constant_count = 0
if rows_failed:
    near_constant_count = sum("std too small" in str(x["reason"]) for x in rows_failed)

with open(summary_report_path, "w", encoding="utf-8") as f:
    f.write("Step 5 Gradient Compression Summary Report — NEGATIVE G1/G2\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Paths:\n")
    f.write(f"  aligned_root: {aligned_root}\n")
    f.write(f"  out_root: {out_root}\n\n")

    f.write("Sample size:\n")
    f.write(f"  Subject directories scanned: {len(sub_dirs)}\n")
    f.write(f"  Successfully processed: {len(df_ok)}\n")
    f.write(f"  Failed subjects: {len(rows_failed)}\n\n")

    f.write("Compression metrics (group-level):\n")
    if df_ok.empty:
        f.write("  No subjects passed QC.\n\n")
    else:
        for grad in GRADIENTS:
            f.write(f"  {grad}_std        = {mean_sd_text(df_ok, f'{grad}_std')}\n")
            f.write(f"  {grad}_P95-P5     = {mean_sd_text(df_ok, f'{grad}_P95_P5')}\n")
            f.write(f"  {grad}_IQR        = {mean_sd_text(df_ok, f'{grad}_IQR')}\n")
            f.write(f"  {grad}_MAD        = {mean_sd_text(df_ok, f'{grad}_MAD')}\n")
            f.write("\n")

    f.write("Quality control:\n")
    f.write(f"  EPS_STD threshold: {EPS_STD}\n")
    f.write(f"  MIN_LEN threshold: {MIN_LEN}\n")
    f.write(f"  Near-constant gradients detected: {near_constant_count}\n")
    f.write(f"  Failed-subject file: {csv_fail}\n\n")

    f.write("Output files:\n")
    f.write(f"  Subject-level metrics: {csv_ok}\n")
    f.write(f"  Summary stats: {summary_stats_path}\n")
    f.write(f"  This report: {summary_report_path}\n\n")

    f.write("Notes:\n")
    f.write("  - Compression metrics were computed on negative-connection Procrustes-aligned first and second gradients.\n")
    f.write("  - No scaling or normalization was applied prior to compression estimation.\n")
    f.write("  - A subject was retained only if both G1 and G2 files existed and passed QC.\n")
    f.write("  - This script is matched to the positive-connection G1/G2 Step5 pipeline.\n")

print("\n=== Step 5 完成（负向 Procrustes G1/G2 版本） ===")
print("主结果表:", csv_ok)
print("失败被试表:", csv_fail)
print("总结报告:", summary_report_path)
