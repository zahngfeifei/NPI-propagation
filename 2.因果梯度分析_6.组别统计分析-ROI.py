# ============================================================
# Figure 1C
# ROI-wise (Schaefer400) GLM on aligned gradients
# - G1 and G2
# - ASD vs HC
# - Covariates: Age, Sex, FIQ
# - Site and MeanFD removed because they have already been controlled by ComBat
# - FDR correction
# - Plot included
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
from datetime import datetime


# ------------------------------------------------------------
# 路径设置
# ------------------------------------------------------------
G_ROOT = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果2_combat"   # out_G1_procrustes.npy / out_G2_procrustes.npy 所在
SUB_INFO = r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"

OUT_DIR = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果5"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 参数
# ------------------------------------------------------------
ALPHA_FDR = 0.05
GRADIENTS = ["G1", "G2"]


# ------------------------------------------------------------
# Step 1：加载被试信息
# ------------------------------------------------------------
df_info = pd.read_csv(SUB_INFO)

required_cols = ["sub_id", "Group", "Age", "Sex", "FIQ"]
for c in required_cols:
    if c not in df_info.columns:
        raise ValueError(f"缺少列: {c}")


# ------------------------------------------------------------
# Step 2-8：对指定梯度进行 ROI-wise GLM
# ------------------------------------------------------------
def run_gradient_roi_glm(gradient_name):
    """
    对 G1 或 G2 进行 ROI-wise GLM：

        G_roi ~ Group + Age + Sex + FIQ

    其中 Group 以 HC 为 reference，提取 ASD vs HC 的 beta、t 和 p。
    """

    print(f"\n=== 开始分析 {gradient_name} ===")

    # --------------------------------------------------------
    # Step 2：加载所有被试的 out_G*_procrustes.npy
    # --------------------------------------------------------
    subjects, G_list = [], []

    gradient_file_name = f"out_{gradient_name}_procrustes.npy"

    for sub_dir in sorted(glob.glob(os.path.join(G_ROOT, "sub-*"))):
        sub_id = os.path.basename(sub_dir)
        g_path = os.path.join(sub_dir, gradient_file_name)

        if not os.path.exists(g_path):
            continue

        g = np.load(g_path)

        if g.ndim != 1:
            print(f"跳过 {sub_id}: {gradient_file_name} 不是一维数组")
            continue

        subjects.append(sub_id)
        G_list.append(g)

    if len(G_list) == 0:
        raise RuntimeError(f"未找到任何 {gradient_file_name}")

    G_matrix = np.vstack(G_list)  # shape: N × P
    N, P = G_matrix.shape

    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_matrix.npy"),
        G_matrix
    )

    # --------------------------------------------------------
    # Step 3：合并协变量
    # --------------------------------------------------------
    df = df_info[df_info["sub_id"].isin(subjects)].copy()
    df = df.set_index("sub_id").loc[subjects].reset_index()

    if len(df) != N:
        raise RuntimeError(f"{gradient_name}: 协变量与梯度被试数不一致")

    # --------------------------------------------------------
    # Step 4：ROI-wise GLM
    # --------------------------------------------------------
    t_vals = np.zeros(P)
    beta_vals = np.zeros(P)
    p_vals = np.zeros(P)

    term = "C(Group, Treatment(reference='HC'))[T.ASD]"

    for roi in range(P):
        df["G_roi"] = G_matrix[:, roi]

        model = smf.ols(
            "G_roi ~ C(Group, Treatment(reference='HC')) "
            "+ Age + Sex + FIQ",
            data=df
        ).fit()

        if term not in model.params.index:
            raise RuntimeError(
                f"{gradient_name}, ROI {roi}: 模型中没有找到 Group 项：{term}"
            )

        beta_vals[roi] = model.params[term]
        t_vals[roi] = model.tvalues[term]
        p_vals[roi] = model.pvalues[term]

    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_roi_tmap.npy"),
        t_vals
    )
    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_roi_beta.npy"),
        beta_vals
    )
    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_roi_p_uncorrected.npy"),
        p_vals
    )

    # --------------------------------------------------------
    # Step 5：FDR 校正
    # --------------------------------------------------------
    reject_fdr, p_fdr = fdrcorrection(
        p_vals,
        alpha=ALPHA_FDR
    )

    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_roi_p_fdr.npy"),
        p_fdr
    )
    np.save(
        os.path.join(OUT_DIR, f"{gradient_name}_roi_significant_mask.npy"),
        reject_fdr
    )

    # --------------------------------------------------------
    # Step 6：保存 ROI-wise 统计表
    # --------------------------------------------------------
    roi_df = pd.DataFrame({
        "ROI_index": np.arange(P),
        "beta_ASD_vs_HC": beta_vals,
        "t_ASD_vs_HC": t_vals,
        "p_uncorrected": p_vals,
        "p_fdr": p_fdr,
        "significant_fdr": reject_fdr
    })

    roi_csv = os.path.join(
        OUT_DIR,
        f"{gradient_name}_roiwise_GLM_results.csv"
    )

    roi_df.to_csv(
        roi_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Step 7：绘图
    # --------------------------------------------------------
    plt.figure(figsize=(12, 4))

    x = np.arange(P)

    plt.bar(
        x,
        t_vals,
        color="lightgray",
        width=1.0
    )

    plt.scatter(
        x[reject_fdr],
        t_vals[reject_fdr],
        color="red",
        s=10,
        label=f"FDR < {ALPHA_FDR}"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.xlabel("ROI index (Schaefer400)")
    plt.ylabel("t value (ASD vs HC)")
    plt.title(
        f"Figure 1C: ROI-wise group differences in {gradient_name}"
    )
    plt.legend(frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(
        OUT_DIR,
        f"figure_1C_{gradient_name}_group_diff.png"
    )

    plt.savefig(
        fig_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Step 8：统计总结报告 TXT
    # --------------------------------------------------------
    report_path = os.path.join(
        OUT_DIR,
        f"figure1C_{gradient_name}_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Figure 1C ROI-wise GLM report: {gradient_name}\n")
        f.write(f"Generated at: {datetime.now()}\n\n")

        f.write(f"Sample size: N = {N}\n")
        f.write(f"Number of ROIs: {P}\n\n")

        f.write("Model:\n")
        f.write("  G_roi ~ Group + Age + Sex + FIQ\n")
        f.write("  Site and MeanFD were not included because they have already been controlled by ComBat.\n\n")

        f.write("Group contrast:\n")
        f.write("  ASD vs HC\n")
        f.write("  Positive t indicates ASD > HC\n")
        f.write("  Negative t indicates ASD < HC\n\n")

        f.write("Multiple comparison correction:\n")
        f.write("  Method: FDR (Benjamini-Hochberg)\n")
        f.write(f"  Alpha: {ALPHA_FDR}\n\n")

        f.write("Results summary:\n")
        f.write(f"  Significant ROIs (FDR < {ALPHA_FDR}): {int(reject_fdr.sum())}\n")
        f.write(f"  Mean beta (all ROIs): {beta_vals.mean():.4f}\n")
        f.write(f"  Mean |t| (all ROIs): {np.mean(np.abs(t_vals)):.4f}\n")
        f.write(f"  Max |t| (all ROIs): {np.max(np.abs(t_vals)):.4f}\n")
        f.write(f"  Minimum uncorrected p: {np.min(p_vals):.6g}\n")
        f.write(f"  Minimum FDR-corrected p: {np.min(p_fdr):.6g}\n\n")

        f.write("Output files:\n")
        f.write(f"  Gradient matrix: {gradient_name}_matrix.npy\n")
        f.write(f"  t map: {gradient_name}_roi_tmap.npy\n")
        f.write(f"  beta map: {gradient_name}_roi_beta.npy\n")
        f.write(f"  uncorrected p: {gradient_name}_roi_p_uncorrected.npy\n")
        f.write(f"  FDR p: {gradient_name}_roi_p_fdr.npy\n")
        f.write(f"  FDR mask: {gradient_name}_roi_significant_mask.npy\n")
        f.write(f"  ROI-wise CSV: {os.path.basename(roi_csv)}\n")
        f.write(f"  Figure: {os.path.basename(fig_path)}\n\n")

        f.write("Notes:\n")
        f.write("  - Gradients were aligned to a group template using Procrustes.\n")
        f.write("  - FDR correction was performed across Schaefer400 ROIs.\n")

    # --------------------------------------------------------
    # Step 9：返回 summary
    # --------------------------------------------------------
    summary = {
        "gradient": gradient_name,
        "N": int(N),
        "P": int(P),
        "n_significant_fdr": int(reject_fdr.sum()),
        "mean_beta": float(beta_vals.mean()),
        "mean_abs_t": float(np.mean(np.abs(t_vals))),
        "max_abs_t": float(np.max(np.abs(t_vals))),
        "min_p_uncorrected": float(np.min(p_vals)),
        "min_p_fdr": float(np.min(p_fdr)),
        "roi_csv": roi_csv,
        "figure": fig_path,
        "report": report_path
    }

    print(f"=== {gradient_name} 分析完成 ===")
    print(f"Sample size: N = {N}")
    print(f"Number of ROIs: P = {P}")
    print(f"FDR 显著 ROI 数: {int(reject_fdr.sum())}")
    print(f"ROI-wise 结果表: {roi_csv}")
    print(f"统计报告: {report_path}")
    print(f"图像文件: {fig_path}")

    return summary


# ------------------------------------------------------------
# 主程序：分别运行 G1 和 G2
# ------------------------------------------------------------
if __name__ == "__main__":

    all_summaries = []

    for gradient_name in GRADIENTS:
        summary = run_gradient_roi_glm(gradient_name)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    summary_csv = os.path.join(
        OUT_DIR,
        "figure1C_G1_G2_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== Figure 1C G1 + G2 分析全部完成 ===")
    print("输出目录：", OUT_DIR)
    print("总汇总表：", summary_csv)