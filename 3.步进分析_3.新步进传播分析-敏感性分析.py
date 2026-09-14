# =====================================================
# EC-Stepwise Causal Connectivity (EC-SEC) — 加权传播版
# 参数敏感性分析版本
#
# 敏感性分析参数：
#   1) L_MAX: 30, 40, 50, 60
#   2) early slope window:
#        steps 1--5
#        steps 1--10
#        steps 1--15
#
# 当前指标：
#   1) peak step
#   2) temporal centroid / propagation center of mass
#   3) early slope
#   4) full AUC
#
# 输入：
#  1) sub-*/W_sparse.npy
#     W_sparse 为每个节点保留 top 10% positive outgoing NPI weights 后的稀疏加权矩阵
#  2) seeds/Hk CSV
#
# 输出：
#  1) 每被试完整 60 步 SEC 曲线 CSV
#  2) 每个 L_MAX 配置下的每被试截断 SEC 曲线 CSV
#  3) 每被试 QC txt
#  4) 全被试敏感性分析指标汇总 CSV
#  5) 每个 L_MAX 配置单独的指标 CSV
#  6) 全局总结报告 txt
#  7) 关键步数 step-specific 全脑传播向量 v(step)
#
# 核心传播模型：
#  W_sparse -> row-stochastic transition matrix T
#  T_ij = W_ij / sum_j W_ij
#  v_{l+1} = v_l @ T
#
# 方向说明：
#  行表示 source parcel，列表示 target parcel。
#  因此传播使用行向量约定：
#  v_{l+1} = v_l @ T
# =====================================================

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------------------------------------
# 路径设置
# -----------------------------------------------------
W_ROOT = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_权重矩阵"  # sub-*/W_sparse.npy

ROI_DEF_DIR = r"I:\DYF\NPI-4-code\3.步进分析\种子集合"
SEED_CSV = os.path.join(ROI_DEF_DIR, "seeds_S_sensory.csv")
H1_CSV   = os.path.join(ROI_DEF_DIR, "H1_sensory.csv")
H2_CSV   = os.path.join(ROI_DEF_DIR, "H2_attention.csv")
H3_CSV   = os.path.join(ROI_DEF_DIR, "H3_control.csv")
H4_CSV   = os.path.join(ROI_DEF_DIR, "H4_DMN.csv")

OUT_ROOT = r"I:\DYF\NPI-4-code\3.步进分析\ABIDE1_新结果1-参数敏感性分析"
os.makedirs(OUT_ROOT, exist_ok=True)

# 完整 60 步曲线输出
OUT_FULL_CURVE_DIR = os.path.join(OUT_ROOT, "curves_per_subject_full_60")
os.makedirs(OUT_FULL_CURVE_DIR, exist_ok=True)

# 按 L_MAX 配置截断后的曲线输出
OUT_CONFIG_CURVE_ROOT = os.path.join(OUT_ROOT, "curves_per_subject_by_Lmax")
os.makedirs(OUT_CONFIG_CURVE_ROOT, exist_ok=True)

# QC 输出
OUT_QC_DIR = os.path.join(OUT_ROOT, "qc_per_subject")
os.makedirs(OUT_QC_DIR, exist_ok=True)

# 关键步数全脑传播向量输出
OUT_KEYSTEP_DIR = os.path.join(OUT_ROOT, "key_steps")
os.makedirs(OUT_KEYSTEP_DIR, exist_ok=True)

# -----------------------------------------------------
# 参数设置：敏感性分析核心参数
# -----------------------------------------------------
L_MAX_LIST = [30, 40, 50, 60]
L_MAX_PROPAGATION = max(L_MAX_LIST)

EARLY_WINDOWS = [
    (1, 5),
    (1, 10),
    (1, 15)
]

# 推荐主分析使用 markov
# "markov": W -> T(row-stochastic), v_{l+1}=v_l @ T
# "walk"  : v_{l+1}=normalize(v_l @ W)，仅用于兼容旧口径
PROP_MODE = "markov"  # "markov" or "walk"

# 仅在 walk 模式下使用
NORM_MODE = "l1"      # "l1" or "max"

EPS = 1e-8
EPS_OUTSTRENGTH = 1e-12

# markov 模式下是否每步重新归一化
# 理论上 T 行随机且 v0 和为 1 时，不需要每步归一化
MARKOV_RENORM_EACH_STEP = False
MARKOV_RENORM_MODE = "l1"

# visited_count 判断阈值
EPS_VISIT = 1e-5

# 是否保存每一步全脑传播向量
SAVE_VL_FULL = True

# 关键步数，1-based
# 保留原来的早期关键步数，同时加入各 L_MAX 终点步数
KEY_STEPS = sorted(set([1, 3, 5, 7, 9] + L_MAX_LIST))

# 是否保存关键步数的 0/1 mask
# 注意：mask 仅用于可视化，不参与传播计算
SAVE_KEYSTEP_MASK = True
KEYSTEP_MASK_EPS = 1e-12

# -----------------------------------------------------
# 工具函数
# -----------------------------------------------------
def read_roi_index_0based(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if "ROI_index_0based" not in df.columns:
        raise ValueError(f"{csv_path} 缺少列 ROI_index_0based")

    idx = df["ROI_index_0based"].astype(int).values
    idx = np.unique(idx)

    return idx


def normalize_vec(v: np.ndarray, mode: str) -> np.ndarray:
    if mode == "l1":
        denom = np.sum(np.abs(v)) + EPS
        return v / denom

    elif mode == "max":
        denom = np.max(np.abs(v)) + EPS
        return v / denom

    else:
        raise ValueError("normalize mode 只能是 'l1' 或 'max'")


def peak_step(curve: np.ndarray) -> float:
    """
    peak step:
    SEC 曲线达到最大值的步数。
    返回 1-based step。
    """
    curve = np.asarray(curve, dtype=float)

    if np.all(~np.isfinite(curve)):
        return np.nan

    if np.nanmax(curve) <= 0:
        return np.nan

    return float(np.nanargmax(curve) + 1)


def temporal_centroid(curve: np.ndarray) -> float:
    """
    temporal centroid / propagation center of mass:

        tau = sum_t [t * SEC(t)] / sum_t [SEC(t)]

    t 使用 1-based step，即 1..L_MAX。
    该指标表示传播质量在时间维度上的重心。
    数值越大，说明该系统的传播质量整体出现得越晚。
    """
    curve = np.asarray(curve, dtype=float)

    if np.all(~np.isfinite(curve)):
        return np.nan

    valid = np.isfinite(curve)

    if np.sum(valid) == 0:
        return np.nan

    c = curve[valid]
    denom = np.nansum(c)

    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    t_all = np.arange(1, len(curve) + 1, dtype=float)
    t = t_all[valid]

    return float(np.nansum(t * c) / denom)


def auc(curve: np.ndarray) -> float:
    """
    full AUC:
    对完整 SEC 曲线求和。

    在敏感性分析中，curve 已经按指定 L_MAX 截断。
    因此：
      L_MAX=30 时，AUC = sum steps 1--30
      L_MAX=40 时，AUC = sum steps 1--40
      L_MAX=50 时，AUC = sum steps 1--50
      L_MAX=60 时，AUC = sum steps 1--60
    """
    curve = np.asarray(curve, dtype=float)

    if np.all(~np.isfinite(curve)):
        return np.nan

    return float(np.nansum(curve))


def linear_slope(y: np.ndarray) -> float:
    """
    early slope:
    对指定 early window 内 SEC(t) 进行简单线性拟合，返回斜率。
    """
    y = np.asarray(y, dtype=float)

    if len(y) < 2:
        return np.nan

    valid = np.isfinite(y)

    if np.sum(valid) < 2:
        return np.nan

    y = y[valid]

    if np.allclose(y, y[0]):
        return 0.0

    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y2 = y - np.mean(y)

    denom = np.sum(x * x) + EPS

    return float(np.sum(x * y2) / denom)


def early_slope_by_window(curve: np.ndarray, start_step: int, end_step: int) -> float:
    """
    根据 1-based step window 计算 early slope。

    例：
      start_step=1, end_step=5  -> steps 1--5
      start_step=1, end_step=10 -> steps 1--10
      start_step=1, end_step=15 -> steps 1--15
    """
    curve = np.asarray(curve, dtype=float)

    if start_step < 1:
        raise ValueError("start_step 必须 >= 1")

    if end_step < start_step:
        raise ValueError("end_step 必须 >= start_step")

    if end_step > len(curve):
        return np.nan

    s0 = start_step - 1
    s1 = end_step

    return linear_slope(curve[s0:s1])


def window_label(start_step: int, end_step: int) -> str:
    """
    用于列名的 window 标签。
    """
    return f"{start_step}_{end_step}"


def build_transition_matrix_from_weighted(W: np.ndarray):
    """
    W: (P, P) 稀疏加权非负矩阵

    返回:
      T: row-stochastic transition matrix
      outstrength: 每个节点的出强度

    要求:
      1) W 方阵
      2) W 有限
      3) W 非负
      4) 无零出强度行
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W_sparse 不是方阵，shape={W.shape}")

    if not np.isfinite(W).all():
        raise ValueError("W_sparse 含 NaN/Inf")

    if np.min(W) < 0:
        raise ValueError("W_sparse 存在负值；加权 Markov 传播要求非负权重")

    W = W.astype(np.float64, copy=True)
    np.fill_diagonal(W, 0.0)

    outstrength = np.sum(W, axis=1).astype(np.float64)

    if np.any(outstrength <= EPS_OUTSTRENGTH):
        raise ValueError(
            f"存在出强度为0或近0的节点行数={int(np.sum(outstrength <= EPS_OUTSTRENGTH))}"
        )

    T = W / outstrength[:, None]

    if not np.isfinite(T).all():
        raise ValueError("转移矩阵 T 含 NaN/Inf")

    row_sums = np.sum(T, axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"T 不是有效 row-stochastic 矩阵: "
            f"row_sum_min={row_sums.min():.6f}, row_sum_max={row_sums.max():.6f}"
        )

    return T, outstrength


def stat_series(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()

    if len(x) == 0:
        return "NA"

    return (
        f"mean={x.mean():.4f}, sd={x.std():.4f}, "
        f"min={x.min():.4f}, max={x.max():.4f}, n_valid={len(x)}"
    )


# -----------------------------------------------------
# 读取 ROI 定义
# -----------------------------------------------------
seeds = read_roi_index_0based(SEED_CSV)
H1 = read_roi_index_0based(H1_CSV)
H2 = read_roi_index_0based(H2_CSV)
H3 = read_roi_index_0based(H3_CSV)
H4 = read_roi_index_0based(H4_CSV)

H_LIST = [
    ("H1_sensory", H1),
    ("H2_attention", H2),
    ("H3_control", H3),
    ("H4_DMN", H4)
]

print("=== ROI 定义加载完成 ===")
print("Seeds (S) ROI 数量:", len(seeds))

for name, idx in H_LIST:
    print(f"{name} ROI 数量:", len(idx))

# -----------------------------------------------------
# 搜索所有被试 W_sparse.npy
# -----------------------------------------------------
w_paths = sorted(glob.glob(os.path.join(W_ROOT, "sub-*", "W_sparse.npy")))

print(f"\n发现 {len(w_paths)} 个被试的 W_sparse.npy")

if len(w_paths) == 0:
    raise FileNotFoundError("未找到任何 W_sparse.npy，请检查 W_ROOT 路径与文件结构。")

# -----------------------------------------------------
# 主循环：逐被试 EC-SEC 加权传播
# -----------------------------------------------------
summary_rows = []
failed_rows = []

for w_path in w_paths:

    sub_id = os.path.basename(os.path.dirname(w_path))

    print(f"\n========== EC-SEC 加权传播敏感性分析处理 {sub_id} ==========")

    try:
        W = np.load(w_path).astype(np.float64)

        # ---------- QC 1：方阵 ----------
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W_sparse 不是方阵，shape={W.shape}")

        P = W.shape[0]

        # ---------- QC 2：数值合法 ----------
        if not np.isfinite(W).all():
            raise ValueError("W_sparse 含 NaN/Inf")

        # ---------- QC 3：非负 ----------
        if np.min(W) < 0:
            raise ValueError("W_sparse 存在负值；加权传播要求非负矩阵")

        np.fill_diagonal(W, 0.0)

        n_edges = int(np.sum(W > 0))

        if n_edges == 0:
            raise ValueError("W_sparse 没有任何正权重边")

        # ---------- QC 4：ROI 索引越界 ----------
        for name, idx in [
            ("Seeds", seeds),
            ("H1", H1),
            ("H2", H2),
            ("H3", H3),
            ("H4", H4)
        ]:
            if np.any(idx < 0) or np.any(idx >= P):
                raise ValueError(f"{name} ROI 索引超界（0..{P - 1}）")

        # ---------- 构建传播算子 ----------
        if PROP_MODE == "markov":
            T, outstrength = build_transition_matrix_from_weighted(W)
            op = T

        elif PROP_MODE == "walk":
            outstrength = np.sum(W, axis=1).astype(float)

            if np.any(outstrength <= EPS_OUTSTRENGTH):
                raise ValueError(
                    f"存在出强度为0或近0的节点行数="
                    f"{int(np.sum(outstrength <= EPS_OUTSTRENGTH))}"
                )

            op = W.astype(float)

        else:
            raise ValueError("PROP_MODE 只能是 'markov' 或 'walk'")

        # ---------- 初始化 v0 ----------
        v = np.zeros(P, dtype=float)
        v[seeds] = 1.0
        v = normalize_vec(v, "l1")

        if not np.isfinite(v).all():
            raise ValueError("初始化 v0 含 NaN/Inf")

        # ---------- 曲线容器：一次性传播到最大 60 步 ----------
        curves_full = {
            name: np.zeros(L_MAX_PROPAGATION, dtype=float)
            for name, _ in H_LIST
        }

        visited_counts_full = np.zeros(L_MAX_PROPAGATION, dtype=int)

        if SAVE_VL_FULL:
            V_all = np.zeros((L_MAX_PROPAGATION, P), dtype=np.float32)

        # ---------- 递推传播 ----------
        for l in range(L_MAX_PROPAGATION):

            # 行向量传播，rows = source, columns = target
            # v_{l+1} = v_l @ op
            v = v @ op

            if PROP_MODE == "walk":
                v = normalize_vec(v, NORM_MODE)

            if PROP_MODE == "markov" and MARKOV_RENORM_EACH_STEP:
                v = normalize_vec(v, MARKOV_RENORM_MODE)

            if not np.isfinite(v).all():
                raise ValueError(f"传播第 {l + 1} 步出现 NaN/Inf")

            # 记录 SEC 曲线：每个层级系统内平均传播强度
            for name, idx in H_LIST:
                curves_full[name][l] = float(np.mean(v[idx])) if len(idx) > 0 else np.nan

            # visited count 仅作 QC / 可视化辅助
            visited_counts_full[l] = int(np.sum(v > EPS_VISIT))

            if SAVE_VL_FULL:
                V_all[l, :] = v.astype(np.float32)

        # ---------- 保存完整 60 步 SEC 曲线 ----------
        full_curve_df = pd.DataFrame({
            "step": np.arange(1, L_MAX_PROPAGATION + 1),
            "SEC_H1_sensory": curves_full["H1_sensory"],
            "SEC_H2_attention": curves_full["H2_attention"],
            "SEC_H3_control": curves_full["H3_control"],
            "SEC_H4_DMN": curves_full["H4_DMN"],
            "visited_count": visited_counts_full
        })

        full_curve_out = os.path.join(
            OUT_FULL_CURVE_DIR,
            f"{sub_id}_EC_SEC_curves_full_{L_MAX_PROPAGATION}.csv"
        )

        full_curve_df.to_csv(full_curve_out, index=False, encoding="utf-8-sig")

        # ---------- 保存全脑 v_l 与关键步数 ----------
        v_all_out = None

        if SAVE_VL_FULL:
            v_all_out = os.path.join(
                OUT_FULL_CURVE_DIR,
                f"{sub_id}_V_all_full_{L_MAX_PROPAGATION}.npy"
            )
            np.save(v_all_out, V_all)

            for step in KEY_STEPS:
                if step < 1 or step > L_MAX_PROPAGATION:
                    continue

                v_k = V_all[step - 1, :].astype(np.float32)

                out_v = os.path.join(OUT_KEYSTEP_DIR, f"{sub_id}_step{step}.npy")
                np.save(out_v, v_k)

                if SAVE_KEYSTEP_MASK:
                    mask = (v_k > KEYSTEP_MASK_EPS).astype(np.int8)
                    out_m = os.path.join(OUT_KEYSTEP_DIR, f"{sub_id}_step{step}_mask.npy")
                    np.save(out_m, mask)

        # ---------- 针对每个 L_MAX 配置保存曲线并提取指标 ----------
        for L_MAX_CURRENT in L_MAX_LIST:

            config_label = f"Lmax_{L_MAX_CURRENT}"

            config_curve_dir = os.path.join(OUT_CONFIG_CURVE_ROOT, config_label)
            os.makedirs(config_curve_dir, exist_ok=True)

            # 截断到当前 L_MAX
            curves_current = {
                name: curves_full[name][:L_MAX_CURRENT].copy()
                for name, _ in H_LIST
            }

            visited_counts_current = visited_counts_full[:L_MAX_CURRENT].copy()

            curve_df = pd.DataFrame({
                "step": np.arange(1, L_MAX_CURRENT + 1),
                "SEC_H1_sensory": curves_current["H1_sensory"],
                "SEC_H2_attention": curves_current["H2_attention"],
                "SEC_H3_control": curves_current["H3_control"],
                "SEC_H4_DMN": curves_current["H4_DMN"],
                "visited_count": visited_counts_current
            })

            curve_out = os.path.join(
                config_curve_dir,
                f"{sub_id}_EC_SEC_curves_{config_label}.csv"
            )

            curve_df.to_csv(curve_out, index=False, encoding="utf-8-sig")

            # ---------- 提取当前 L_MAX 下的 SEC 指标 ----------
            metrics = {
                "sub_id": sub_id,
                "L_MAX": int(L_MAX_CURRENT),
                "L_MAX_PROPAGATION": int(L_MAX_PROPAGATION),
                "full_auc_window": f"1_{L_MAX_CURRENT}",
                "early_windows": ";".join([f"{a}_{b}" for a, b in EARLY_WINDOWS])
            }

            for name, _ in H_LIST:
                c = curves_current[name]

                # 1) peak step, 当前 L_MAX 窗口内
                metrics[f"{name}_peak"] = peak_step(c)

                # 2) temporal centroid, 当前 L_MAX 窗口内
                metrics[f"{name}_temporal_centroid"] = temporal_centroid(c)

                # 3) early slope, 多个窗口
                for early_start, early_end in EARLY_WINDOWS:
                    wlab = window_label(early_start, early_end)
                    metrics[f"{name}_early_slope_{wlab}"] = early_slope_by_window(
                        c,
                        early_start,
                        early_end
                    )

                # 4) full AUC, 当前 L_MAX 窗口内
                metrics[f"{name}_auc"] = auc(c)

            # ---------- QC 指标 ----------
            metrics["P"] = int(P)
            metrics["W_edges"] = int(np.sum(W > 0))

            metrics["W_outstrength_min"] = float(np.min(outstrength))
            metrics["W_outstrength_median"] = float(np.median(outstrength))
            metrics["W_outstrength_max"] = float(np.max(outstrength))
            metrics["W_outstrength_mean"] = float(np.mean(outstrength))

            metrics["visited_count_last"] = int(visited_counts_current[-1])
            metrics["visited_count_max"] = int(np.max(visited_counts_current))

            metrics["PROP_MODE"] = PROP_MODE
            metrics["curve_out_path"] = curve_out
            metrics["full_curve_out_path"] = full_curve_out
            metrics["w_path"] = w_path

            if SAVE_VL_FULL:
                metrics["V_all_path"] = v_all_out
                metrics["key_steps_dir"] = OUT_KEYSTEP_DIR

            summary_rows.append(metrics)

        # ---------- 保存每被试 QC 文本 ----------
        qc_txt = os.path.join(OUT_QC_DIR, f"{sub_id}_qc.txt")

        with open(qc_txt, "w", encoding="utf-8") as f:
            f.write(f"EC-SEC weighted propagation sensitivity QC: {sub_id}\n")
            f.write(f"P (ROI): {P}\n")
            f.write(f"L_MAX_LIST: {L_MAX_LIST}\n")
            f.write(f"L_MAX_PROPAGATION: {L_MAX_PROPAGATION}\n")
            f.write(f"EARLY_WINDOWS: {EARLY_WINDOWS}\n")
            f.write(f"PROP_MODE: {PROP_MODE}\n\n")

            f.write("Input:\n")
            f.write(f"  W_sparse: {w_path}\n\n")

            f.write("Weighted graph QC:\n")
            f.write(f"  W_edges(>0): {int(np.sum(W > 0))}\n")
            f.write(f"  W_outstrength_min: {float(np.min(outstrength)):.6e}\n")
            f.write(f"  W_outstrength_median: {float(np.median(outstrength)):.6e}\n")
            f.write(f"  W_outstrength_max: {float(np.max(outstrength)):.6e}\n")
            f.write(f"  W_outstrength_mean: {float(np.mean(outstrength)):.6e}\n\n")

            if PROP_MODE == "walk":
                f.write(f"NORM_MODE: {NORM_MODE}\n")

            if PROP_MODE == "markov":
                f.write(f"MARKOV_RENORM_EACH_STEP: {MARKOV_RENORM_EACH_STEP}\n")
                f.write(f"MARKOV_RENORM_MODE: {MARKOV_RENORM_MODE}\n\n")

            f.write("SEC metric parameters:\n")
            f.write("  Metrics:\n")
            f.write("    - peak step\n")
            f.write("    - temporal centroid / propagation center of mass\n")
            f.write("    - early slope\n")
            f.write("    - full AUC\n")
            f.write(f"  L_MAX_LIST: {L_MAX_LIST}\n")
            f.write(f"  FULL_AUC_WINDOWS: {[f'1..{x}' for x in L_MAX_LIST]}\n")
            f.write(f"  EARLY_SLOPE_WINDOWS: {[f'{a}..{b}' for a, b in EARLY_WINDOWS]}\n\n")

            f.write("Propagation QC:\n")
            f.write(f"  EPS_VISIT: {EPS_VISIT}\n")
            f.write(f"  Visited count max over full {L_MAX_PROPAGATION} steps: {int(np.max(visited_counts_full))}\n")
            f.write(f"  Visited count last at step {L_MAX_PROPAGATION}: {int(visited_counts_full[-1])}\n")
            f.write(f"  Full curves saved: {full_curve_out}\n\n")

            f.write("Per-L_MAX curve outputs:\n")
            for L_MAX_CURRENT in L_MAX_LIST:
                config_label = f"Lmax_{L_MAX_CURRENT}"
                config_curve_dir = os.path.join(OUT_CONFIG_CURVE_ROOT, config_label)
                curve_out = os.path.join(
                    config_curve_dir,
                    f"{sub_id}_EC_SEC_curves_{config_label}.csv"
                )
                f.write(f"  {config_label}: {curve_out}\n")

            f.write("\n")

            if SAVE_VL_FULL:
                f.write(f"V_all saved: {v_all_out}\n")
                f.write(f"Key steps saved in: {OUT_KEYSTEP_DIR}\n")
                f.write(f"KEY_STEPS: {KEY_STEPS}\n")
                f.write(f"SAVE_KEYSTEP_MASK: {SAVE_KEYSTEP_MASK}\n")
                f.write(f"KEYSTEP_MASK_EPS: {KEYSTEP_MASK_EPS}\n")

        print(f"已保存完整 {L_MAX_PROPAGATION} 步曲线: {full_curve_out}")
        print(
            f"QC: weighted_edges={int(np.sum(W > 0))}, "
            f"outstrength_min={float(np.min(outstrength)):.3e}, "
            f"visited_max_full={int(np.max(visited_counts_full))}"
        )

        if SAVE_VL_FULL:
            print(f"已保存 V_all: {v_all_out}")
            print(f"已保存关键步数文件到: {OUT_KEYSTEP_DIR}")

    except Exception as e:
        print(f"✘ {sub_id} 失败：{str(e)}")

        failed_rows.append({
            "sub_id": sub_id,
            "reason": str(e),
            "w_path": w_path
        })

        continue

# -----------------------------------------------------
# 汇总输出：所有被试 x 所有 L_MAX 配置指标表
# -----------------------------------------------------
summary_df = pd.DataFrame(summary_rows)

summary_csv = os.path.join(OUT_ROOT, "EC_SEC_metrics_all_subjects_sensitivity.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

# 每个 L_MAX 单独保存一份指标表，便于后续统计分析
for L_MAX_CURRENT in L_MAX_LIST:
    if len(summary_df) > 0 and "L_MAX" in summary_df.columns:
        sub_df = summary_df[summary_df["L_MAX"] == L_MAX_CURRENT].copy()
    else:
        sub_df = pd.DataFrame()

    out_csv = os.path.join(
        OUT_ROOT,
        f"EC_SEC_metrics_all_subjects_Lmax_{L_MAX_CURRENT}.csv"
    )

    sub_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# 失败被试
failed_csv = None

if failed_rows:
    failed_df = pd.DataFrame(failed_rows)
    failed_csv = os.path.join(OUT_ROOT, "EC_SEC_failed_subjects.csv")
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")

# -----------------------------------------------------
# 全局总结报告
# -----------------------------------------------------
report_path = os.path.join(OUT_ROOT, "EC_SEC_sensitivity_summary_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("EC-SEC Weighted Stepwise Propagation Sensitivity Analysis Summary Report\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Paths:\n")
    f.write(f"  W_ROOT: {W_ROOT}\n")
    f.write(f"  ROI_DEF_DIR: {ROI_DEF_DIR}\n")
    f.write(f"  OUT_ROOT: {OUT_ROOT}\n")
    f.write(f"  OUT_FULL_CURVE_DIR: {OUT_FULL_CURVE_DIR}\n")
    f.write(f"  OUT_CONFIG_CURVE_ROOT: {OUT_CONFIG_CURVE_ROOT}\n")
    f.write(f"  OUT_QC_DIR: {OUT_QC_DIR}\n")
    f.write(f"  OUT_KEYSTEP_DIR: {OUT_KEYSTEP_DIR}\n\n")

    f.write("Propagation model:\n")
    f.write("  Matrix orientation: rows = source parcels, columns = target parcels\n")
    f.write("  Transition matrix: row-stochastic T\n")
    f.write("  Propagation convention: row vector, v_{l+1} = v_l @ T\n\n")

    f.write("Sensitivity parameters:\n")
    f.write(f"  L_MAX_LIST: {L_MAX_LIST}\n")
    f.write(f"  L_MAX_PROPAGATION: {L_MAX_PROPAGATION}\n")
    f.write(f"  EARLY_WINDOWS: {EARLY_WINDOWS}\n")
    f.write(f"  FULL_AUC_WINDOWS: {[f'1..{x}' for x in L_MAX_LIST]}\n\n")

    f.write("Other parameters:\n")
    f.write(f"  PROP_MODE: {PROP_MODE}\n")

    if PROP_MODE == "walk":
        f.write(f"  NORM_MODE: {NORM_MODE}\n")

    if PROP_MODE == "markov":
        f.write(f"  MARKOV_RENORM_EACH_STEP: {MARKOV_RENORM_EACH_STEP}\n")
        f.write(f"  MARKOV_RENORM_MODE: {MARKOV_RENORM_MODE}\n")

    f.write(f"  EPS: {EPS}\n")
    f.write(f"  EPS_OUTSTRENGTH: {EPS_OUTSTRENGTH}\n")
    f.write(f"  EPS_VISIT: {EPS_VISIT}\n")
    f.write(f"  SAVE_VL_FULL: {SAVE_VL_FULL}\n")
    f.write(f"  KEY_STEPS: {KEY_STEPS}\n")
    f.write(f"  SAVE_KEYSTEP_MASK: {SAVE_KEYSTEP_MASK}\n")
    f.write(f"  KEYSTEP_MASK_EPS: {KEYSTEP_MASK_EPS}\n\n")

    f.write("SEC metrics:\n")
    f.write("  peak step: step at which SEC reaches its maximum value within current L_MAX window\n")
    f.write("  temporal centroid: sum_t[t * SEC(t)] / sum_t[SEC(t)] within current L_MAX window\n")
    f.write("  early slope: linear slope of SEC across selected early window\n")
    f.write("  full AUC: summed SEC across current full propagation window, steps 1..L_MAX\n\n")

    f.write("ROI sets:\n")
    f.write(f"  Seeds n={len(seeds)}\n")
    f.write(f"  H1 n={len(H1)}, H2 n={len(H2)}, H3 n={len(H3)}, H4 n={len(H4)}\n\n")

    total = len(w_paths)
    success_subjects = summary_df["sub_id"].nunique() if len(summary_df) > 0 else 0
    success_rows = len(summary_df)
    failed = len(failed_rows)

    f.write("Sample size:\n")
    f.write(f"  Total subjects found: {total}\n")
    f.write(f"  Success subjects: {success_subjects}\n")
    f.write(f"  Success rows subject x L_MAX: {success_rows}\n")
    f.write(f"  Failed subjects: {failed}\n")

    if failed_csv:
        f.write(f"  Failed list: {failed_csv}\n")

    f.write("\n")

    if success_rows > 0:
        f.write("Weighted graph / propagation QC, all sensitivity rows:\n")

        qc_cols = [
            "W_edges",
            "W_outstrength_min",
            "W_outstrength_median",
            "W_outstrength_max",
            "W_outstrength_mean",
            "visited_count_max",
            "visited_count_last"
        ]

        for col in qc_cols:
            if col in summary_df.columns:
                f.write(f"  {col}: {stat_series(summary_df[col])}\n")

        f.write("\n")

        f.write("Stepwise metrics, grouped by L_MAX:\n")

        for L_MAX_CURRENT in L_MAX_LIST:
            f.write(f"\n  ---- L_MAX = {L_MAX_CURRENT} ----\n")

            tmp = summary_df[summary_df["L_MAX"] == L_MAX_CURRENT].copy()

            if len(tmp) == 0:
                f.write("  No valid rows.\n")
                continue

            metric_cols = [
                c for c in tmp.columns
                if (
                    c.endswith("_peak")
                    or c.endswith("_temporal_centroid")
                    or c.endswith("_auc")
                    or "_early_slope_" in c
                )
            ]

            for col in metric_cols:
                x = pd.to_numeric(tmp[col], errors="coerce").dropna()

                if len(x) == 0:
                    continue

                f.write(f"  {col}: {stat_series(tmp[col])}\n")

        f.write("\n")

    f.write("Outputs:\n")
    f.write(f"  Sensitivity metrics CSV: {summary_csv}\n")

    for L_MAX_CURRENT in L_MAX_LIST:
        f.write(
            f"  Metrics CSV L_MAX={L_MAX_CURRENT}: "
            f"{os.path.join(OUT_ROOT, f'EC_SEC_metrics_all_subjects_Lmax_{L_MAX_CURRENT}.csv')}\n"
        )

    f.write(f"  Full curves dir: {OUT_FULL_CURVE_DIR}\n")
    f.write(f"  Per-L_MAX curves root dir: {OUT_CONFIG_CURVE_ROOT}\n")
    f.write(f"  QC dir: {OUT_QC_DIR}\n")
    f.write(f"  Key-step v(step) dir: {OUT_KEYSTEP_DIR}\n")

print("\n=== EC-SEC 加权传播参数敏感性分析全部完成 ===")
print("敏感性分析汇总指标表已保存：", summary_csv)

for L_MAX_CURRENT in L_MAX_LIST:
    print(
        f"L_MAX={L_MAX_CURRENT} 指标表已保存：",
        os.path.join(OUT_ROOT, f"EC_SEC_metrics_all_subjects_Lmax_{L_MAX_CURRENT}.csv")
    )

if failed_csv:
    print("失败被试表已保存：", failed_csv)

print("总结报告已保存：", report_path)
print("完整 60 步曲线目录：", OUT_FULL_CURVE_DIR)
print("每个 L_MAX 截断曲线目录：", OUT_CONFIG_CURVE_ROOT)
print("每被试 QC 目录：", OUT_QC_DIR)
print("关键步数输出目录：", OUT_KEYSTEP_DIR)