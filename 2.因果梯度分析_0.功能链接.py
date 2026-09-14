import os
import re
import glob
import numpy as np
import pandas as pd

# ============================================================
# 输入：被试 timeseries 所在文件夹
# ============================================================
input_dir = r"I:\DYF\NPI-4-code\1.NPI\ABIDE2_ROI"

# ============================================================
# 输出：FC 矩阵保存路径
# ============================================================
out_dir = r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度\ABIDE2_FC矩阵"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 参数设置
# ============================================================
N_ROI = 400

# ============================================================
# 搜索所有被试 timeseries 文件
# ============================================================
npy_files = sorted(glob.glob(os.path.join(input_dir, "*_timeseries.npy")))

print(f"Found {len(npy_files)} timeseries files in:")
print(input_dir)

if len(npy_files) == 0:
    raise FileNotFoundError(f"No *_timeseries.npy files found in: {input_dir}")

# ============================================================
# 提取被试 ID
# ============================================================
def extract_subject_id(filename):
    base = os.path.basename(filename)
    match = re.search(r"(sub-[^_]+)", base)

    if match:
        return match.group(1)
    else:
        return os.path.splitext(base)[0]

# ============================================================
# 计算完整 Pearson FC
# ============================================================
def compute_full_fc(ts, n_roi=400):
    """
    输入:
        ts: array, shape = time x ROI 或 ROI x time

    输出:
        fc: ROI x ROI 完整 Pearson 功能连接矩阵
            保留正相关和负相关
    """

    ts = np.asarray(ts, dtype=np.float64)

    if ts.ndim != 2:
        raise ValueError(f"Timeseries must be 2D, but got shape {ts.shape}")

    # 自动判断 timeseries 方向
    # 推荐格式通常为 time x ROI，即 T x 400
    if ts.shape[1] == n_roi:
        pass
    elif ts.shape[0] == n_roi:
        ts = ts.T
    else:
        raise ValueError(
            f"Cannot identify ROI dimension. Expected one dimension = {n_roi}, "
            f"but got shape {ts.shape}"
        )

    # 去除每个 ROI 的均值
    ts = ts - np.nanmean(ts, axis=0, keepdims=True)

    # 处理 NaN / Inf
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查零方差 ROI
    std = np.std(ts, axis=0, ddof=1)
    valid = std > 0

    fc = np.zeros((n_roi, n_roi), dtype=np.float64)

    if np.sum(valid) < 2:
        raise ValueError("Less than two valid ROI timeseries after variance check.")

    # 只对有效 ROI 计算相关
    ts_valid = ts[:, valid]
    fc_valid = np.corrcoef(ts_valid, rowvar=False)

    fc[np.ix_(valid, valid)] = fc_valid

    # 数值清理
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

    # 限制相关系数范围
    fc = np.clip(fc, -1.0, 1.0)

    # 对角线设为 1
    np.fill_diagonal(fc, 1.0)

    return fc

# ============================================================
# 批量计算并保存所有被试 FC
# ============================================================
summary = []

for i, f in enumerate(npy_files, 1):
    sub_id = extract_subject_id(f)

    try:
        ts = np.load(f)

        fc = compute_full_fc(
            ts,
            n_roi=N_ROI
        )

        out_name = f"{sub_id}_FC_full.npy"
        out_path = os.path.join(out_dir, out_name)

        np.save(out_path, fc)

        summary.append({
            "subject": sub_id,
            "input_file": f,
            "timeseries_shape": str(ts.shape),
            "fc_shape": str(fc.shape),
            "fc_type": "full Pearson FC",
            "min_fc": np.min(fc),
            "max_fc": np.max(fc),
            "output_file": out_path,
            "status": "success"
        })

        print(f"[{i}/{len(npy_files)}] Saved: {out_name}")

    except Exception as e:
        summary.append({
            "subject": sub_id,
            "input_file": f,
            "timeseries_shape": "NA",
            "fc_shape": "NA",
            "fc_type": "full Pearson FC",
            "min_fc": "NA",
            "max_fc": "NA",
            "output_file": "NA",
            "status": f"failed: {e}"
        })

        print(f"[{i}/{len(npy_files)}] Failed: {sub_id} | {e}")

# ============================================================
# 保存批处理记录
# ============================================================
summary_df = pd.DataFrame(summary)
summary_file = os.path.join(out_dir, "FC_full_batch_summary.csv")
summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

print("\nFinished.")
print(f"Full FC matrices saved to: {out_dir}")
print(f"Summary saved to: {summary_file}")