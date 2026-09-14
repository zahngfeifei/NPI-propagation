# ================================
# EC-SEC 稀疏加权传播矩阵
# ================================

import os
import glob
import numpy as np

data_root = r"I:\DYF\NPI-4-code\1.NPI\ABIDE2_NPI"

weighted_root = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE2_权重矩阵"

os.makedirs(weighted_root, exist_ok=True)

p = 0.10
EPS_NORM = 1e-12

ec_paths = glob.glob(
    os.path.join(
        data_root,
        "sub-*_task-rest_bold_schaefer400_7yeo_2mm_pipeline_full_36p_gsr_spike_0p01_0p1Hz_timeseries",
        "EC_NPI.npy"
    )
)

ec_paths = sorted(ec_paths)

print(f"发现 {len(ec_paths)} 个被试")

n_success = 0
n_failed = 0
failed_subjects = []

for path in ec_paths:

    sub_dir = os.path.basename(os.path.dirname(path))
    sub_id = sub_dir.split("_")[0]

    print(f"\n========== 处理 {sub_id} ==========")

    try:
        EC_raw = np.load(path)

        if EC_raw.ndim != 2 or EC_raw.shape[0] != EC_raw.shape[1]:
            raise ValueError(f"EC 非方阵: shape={EC_raw.shape}")

        P = EC_raw.shape[0]

        EC = EC_raw.astype(np.float64, copy=True)
        np.fill_diagonal(EC, 0.0)
        EC[~np.isfinite(EC)] = 0.0

        M = np.maximum(EC, 0.0)
        np.fill_diagonal(M, 0.0)

        K = max(int(np.ceil(p * (P - 1))), 1)
        M_sparse = np.zeros_like(M, dtype=np.float64)

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

        np.fill_diagonal(M_sparse, 0.0)

        if not np.isfinite(M_sparse).all():
            raise ValueError("M_sparse 含 NaN/Inf")

        if np.min(M_sparse) < 0:
            raise ValueError("M_sparse 存在负权重")

        row_strength = np.sum(M_sparse, axis=1)
        n_zero_strength = int(np.sum(row_strength < EPS_NORM))

        if n_zero_strength > 0:
            raise ValueError(f"存在零/近零出强度节点: {n_zero_strength}")

        weight_dir = os.path.join(weighted_root, sub_id)
        os.makedirs(weight_dir, exist_ok=True)

        W_sparse = M_sparse.astype(np.float32)

        np.save(os.path.join(weight_dir, "W_sparse.npy"), W_sparse)

        n_success += 1

        print(f"完成: {sub_id}")
        print(f"W_sparse saved: {os.path.join(weight_dir, 'W_sparse.npy')}")

    except Exception as e:

        n_failed += 1
        failed_subjects.append(f"{sub_id}: {e}")

        print(f"失败: {sub_id}")
        print(f"原因: {e}")

print("\n=== 全部被试处理完成 ===")
print(f"成功: {n_success}")
print(f"失败: {n_failed}")

if n_failed > 0:
    print("\n失败被试:")
    for item in failed_subjects:
        print(item)