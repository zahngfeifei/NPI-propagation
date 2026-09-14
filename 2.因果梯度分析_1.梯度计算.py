# ================================
# EC out-gradient（驱动层级）计算 —— 负向连接绝对值版本
# + 群体级 QC 总结报告（summary_report.txt）
#
# 要求实现：
# 1) 仅保留负向连接并取绝对值：M = max(-EC, 0)
# 2) 输出结果与“正向版本”主结果结构一致
# 3) 不再计算、不再保存二值矩阵 A_binary / A_reciprocal
# 4) 文件命名：与正向版本一致，但加前缀“负向连接_”
# ================================

import os
import glob
import numpy as np
from brainspace.gradient import GradientMaps
from datetime import datetime
import shutil

# ------------------------------------------------
# 基础路径
# ------------------------------------------------
data_root = r"I:\DYF\NPI-4-code\1.NPI\ABIDE1_NPI"

# 负向结果目录：仅最后一级不同
out_root = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE1_负向结果1"
failed_root = os.path.join(out_root, "failed_subjects")

os.makedirs(out_root, exist_ok=True)
os.makedirs(failed_root, exist_ok=True)

# 负向结果文件名前缀
PREFIX = "负向连接_"

# ------------------------------------------------
# 全局超参数（论文需声明）
# ------------------------------------------------
p = 0.10
N_COMPONENTS = 10
RANDOM_STATE = 0
EPS_NORM = 1e-12

# ------------------------------------------------
# 搜索被试
# ------------------------------------------------
ec_paths = glob.glob(
    os.path.join(
        data_root,
        "sub-*_task-rest_bold_schaefer400_7yeo_2mm_pipeline_full_36p_gsr_spike_0p01_0p1Hz_timeseries",
        "EC_NPI.npy"
    )
)

print(f"发现 {len(ec_paths)} 个被试")

# ------------------------------------------------
# Summary 容器
# ------------------------------------------------
summary = {
    "n_total": len(ec_paths),
    "n_success": 0,
    "n_failed": 0,
    "failed_subjects": [],
    "min_row_norms": [],
    "median_row_norms": [],
    "outG1_std_list": []
}

# ------------------------------------------------
# 主循环
# ------------------------------------------------
for path in ec_paths:

    sub_dir = os.path.basename(os.path.dirname(path))
    sub_id = sub_dir.split("_")[0]

    print(f"\n========== 处理 {sub_id} ==========")

    try:
        # ============================================================
        # 阶段 1：纯计算 + QC（不保存任何主结果）
        # ============================================================

        # ---------- Step 0：读取 ----------
        EC_raw = np.load(path)

        if EC_raw.ndim != 2 or EC_raw.shape[0] != EC_raw.shape[1]:
            raise ValueError(f"EC 非方阵: shape={EC_raw.shape}")

        P = EC_raw.shape[0]

        # ---------- Step 1：清理 ----------
        EC = EC_raw.astype(np.float64, copy=True)
        np.fill_diagonal(EC, 0.0)
        EC[~np.isfinite(EC)] = 0.0

        # ---------- Step 2：negative-only absolute magnitude ----------
        # 负向连接使用绝对幅值：
        # EC < 0  →  -EC
        # EC >= 0 →  0
        M = np.maximum(-EC, 0.0)

        # ---------- Step 3：row-wise top-p ----------
        K = max(int(np.ceil(p * (P - 1))), 1)
        M_sparse = np.zeros_like(M)

        for i in range(P):
            row = M[i].copy()
            row[i] = 0.0
            pos_idx = np.where(row > 0)[0]

            if pos_idx.size == 0:
                continue

            if pos_idx.size <= K:
                topk = pos_idx
            else:
                part = np.argpartition(row[pos_idx], -K)[-K:]
                topk = pos_idx[part]

            M_sparse[i, topk] = row[topk]

        # ---------- Step 4：row-norm QC ----------
        row_norms = np.linalg.norm(M_sparse, axis=1)
        min_norm = float(np.min(row_norms))
        med_norm = float(np.median(row_norms))
        n_iso = int(np.sum(row_norms < EPS_NORM))

        if n_iso > 0:
            raise ValueError(f"存在孤立/近孤立节点: {n_iso}")

        # ---------- Step 5：梯度 ----------
        gm = GradientMaps(
            n_components=N_COMPONENTS,
            approach="dm",
            kernel="cosine",
            random_state=RANDOM_STATE
        )

        gm.fit(M_sparse, sparsity=None)

        lambdas = np.array(gm.lambdas_, dtype=np.float64)
        if not np.isfinite(lambdas).all() or lambdas.sum() <= 0:
            raise ValueError("无效的 diffusion eigenvalues")

        G = gm.gradients_
        if not np.isfinite(G).all():
            raise ValueError("梯度含 NaN/Inf")

        out_G1 = G[:, 0]
        sd = float(np.std(out_G1))
        if sd < 1e-12:
            raise ValueError("out_G1 近常数")

        out_G1_z = (out_G1 - out_G1.mean()) / sd

        print(gm.lambdas_)

        # ============================================================
        # 阶段 2：所有 QC 通过 → 创建目录 → 保存结果
        # ============================================================

        out_dir = os.path.join(out_root, sub_id)
        os.makedirs(out_dir, exist_ok=True)

        # 保存 EC 梯度主结果
        np.save(os.path.join(out_dir, f"{PREFIX}EC_raw.npy"), EC_raw)
        np.save(os.path.join(out_dir, f"{PREFIX}M_positive.npy"), M)
        np.save(os.path.join(out_dir, f"{PREFIX}M_sparse.npy"), M_sparse)
        np.save(os.path.join(out_dir, f"{PREFIX}row_norms.npy"), row_norms)
        np.save(os.path.join(out_dir, f"{PREFIX}lambdas.npy"), lambdas)
        np.save(os.path.join(out_dir, f"{PREFIX}G_raw.npy"), G)
        np.save(os.path.join(out_dir, f"{PREFIX}out_G1_z.npy"), out_G1_z)

        # 单被试 QC 文本
        with open(os.path.join(out_dir, f"{PREFIX}qc_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Subject: {sub_id}\n")
            f.write(f"ROI: {P}\n")
            f.write(f"p={p}, K={K}, EPS_NORM={EPS_NORM}\n\n")
            f.write(f"row_norm min={min_norm:.3e}, median={med_norm:.3e}\n")
            f.write(f"out_G1 std={sd:.4f}\n")

        # 更新运行摘要
        summary["n_success"] += 1
        summary["min_row_norms"].append(min_norm)
        summary["median_row_norms"].append(med_norm)
        summary["outG1_std_list"].append(sd)

        print(f"✔ {sub_id} 完成")

    except Exception as e:

        print(f"✘ {sub_id} 失败: {e}")

        summary["n_failed"] += 1
        summary["failed_subjects"].append(f"{sub_id}: {e}")

        fail_sub_dir = os.path.join(failed_root, sub_id)
        os.makedirs(fail_sub_dir, exist_ok=True)

        with open(os.path.join(fail_sub_dir, f"{PREFIX}error_log.txt"), "w", encoding="utf-8") as ef:
            ef.write(f"Subject: {sub_id}\n")
            ef.write(f"Error: {str(e)}\n")
            ef.write(f"Original EC path: {path}\n")

        try:
            shutil.copy(path, os.path.join(fail_sub_dir, f"{PREFIX}EC_raw_failed.npy"))
        except Exception:
            pass

# ------------------------------------------------
# 群体 Summary Report
# ------------------------------------------------
report_path = os.path.join(out_root, f"{PREFIX}summary_report.txt")

def stat(x, sci=False):
    if len(x) == 0:
        return "NA"

    if sci:
        return (
            f"mean={np.mean(x):.3e}, sd={np.std(x):.3e}, "
            f"min={np.min(x):.3e}, max={np.max(x):.3e}"
        )
    else:
        return (
            f"mean={np.mean(x):.3f}, sd={np.std(x):.3f}, "
            f"min={np.min(x):.3f}, max={np.max(x):.3f}"
        )

with open(report_path, "w", encoding="utf-8") as f:
    f.write("EC out-gradient pipeline summary report\n")
    f.write("Negative-connection absolute-magnitude version\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write(f"Total subjects: {summary['n_total']}\n")
    f.write(f"Success: {summary['n_success']}\n")
    f.write(f"Failed: {summary['n_failed']}\n\n")

    if summary["n_failed"] > 0:
        f.write("Failed subjects:\n")
        for s in summary["failed_subjects"]:
            f.write(f"  {s}\n")
        f.write("\n")

    f.write("Row-norm minima:\n")
    f.write(f"  {stat(summary['min_row_norms'], sci=True)}\n\n")

    f.write("Row-norm medians:\n")
    f.write(f"  {stat(summary['median_row_norms'], sci=True)}\n\n")

    f.write("out_G1 std:\n")
    f.write(f"  {stat(summary['outG1_std_list'])}\n")

print("\n=== 全部被试处理完成（负向连接绝对值） ===")
print("群体总结报告已生成：")
print(report_path)
print("\n负向主结果根目录：")
print(out_root)
print("\n失败审计目录：")
print(failed_root)
