import os
import glob
import numpy as np
from brainspace.gradient import GradientMaps
from scipy.linalg import orthogonal_procrustes
from datetime import datetime
import pandas as pd

# ------------------------------------------------
# 路径设置（负向版：仅最后一级不同，其余与正向一致）
# ------------------------------------------------
sub_root   = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果1"
out_root   = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果2"
os.makedirs(out_root, exist_ok=True)

group_dir = os.path.join(out_root, "group")
os.makedirs(group_dir, exist_ok=True)

# 补充审计输出目录（使用独立末级目录）
supp2_root = r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果2-补充"
os.makedirs(supp2_root, exist_ok=True)

# 负向输入文件名前缀（仅用于读取输入，输出命名不变）
PREFIX = "负向连接_"

# ------------------------------------------------
# 参数（论文需声明）
# ------------------------------------------------
N_COMPONENTS = 10
K_ALIGN      = 5
EPS_NORM     = 1e-12

# QC阈值（审稿人常问：对齐质量阈值）
R_LOW_THRESH = 0.20

# 列标准化参数
EPS_Z = 1e-12  # 防止 std=0

# ------------------------------------------------
# 工具函数
# ------------------------------------------------
def is_square_matrix(M: np.ndarray) -> bool:
    return (isinstance(M, np.ndarray) and M.ndim == 2 and M.shape[0] == M.shape[1])

def zscore_cols(X: np.ndarray, eps: float = EPS_Z) -> np.ndarray:
    """
    对每一列做 z-score： (x - mean)/std
    若 std 过小，则用 eps 替代，避免数值爆炸。
    """
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    return (X - mu) / sd

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        return np.nan
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    sa = np.std(a)
    sb = np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

# ------------------------------------------------
# Step 1：加载 M_sparse（逐被试 QC + 失败审计）
# ------------------------------------------------
sub_ids, sub_dirs, M_list = [], [], []
m_failed = []  # 审计：M_sparse失败

all_sub_dirs = sorted(glob.glob(os.path.join(sub_root, "sub-*")))
n_total_dirs = len(all_sub_dirs)

P_ref = None

for d in all_sub_dirs:
    sub_id = os.path.basename(d)
    # 负向输入文件：负向连接_M_sparse.npy
    m_path = os.path.join(d, f"{PREFIX}M_sparse.npy")

    if not os.path.exists(m_path):
        m_failed.append({"sub_id": sub_id, "reason": "M_sparse_missing", "path": m_path})
        continue

    try:
        M = np.load(m_path)

        # QC: shape
        if not is_square_matrix(M):
            m_failed.append({"sub_id": sub_id, "reason": f"M_not_square_shape={getattr(M,'shape',None)}", "path": m_path})
            continue

        # QC: consistent P
        P = M.shape[0]
        if P_ref is None:
            P_ref = P
        elif P != P_ref:
            m_failed.append({"sub_id": sub_id, "reason": f"M_inconsistent_P={P}_expected={P_ref}", "path": m_path})
            continue

        # QC: finite
        if not np.isfinite(M).all():
            m_failed.append({"sub_id": sub_id, "reason": "M_has_nan_inf", "path": m_path})
            continue

        # QC: non-negative（负向绝对值矩阵同样应为非负）
        if np.min(M) < -1e-12:
            m_failed.append({"sub_id": sub_id, "reason": "M_has_negative_values", "path": m_path})
            continue

        # QC: row norms (near-isolated rows)
        rn = np.linalg.norm(M, axis=1)
        if np.sum(rn < EPS_NORM) > 0:
            m_failed.append({"sub_id": sub_id, "reason": f"M_has_near_zero_rows_n={int(np.sum(rn<EPS_NORM))}", "path": m_path})
            continue

        # pass
        sub_dirs.append(d)
        sub_ids.append(sub_id)
        M_list.append(M)

    except Exception as e:
        m_failed.append({"sub_id": sub_id, "reason": f"M_load_failed:{type(e).__name__}:{e}", "path": m_path})
        continue

# 输出 Step1 失败审计（补充目录；文件名与正向一致）
pd.DataFrame(m_failed).to_csv(
    os.path.join(supp2_root, "audit_m_sparse_failed_subjects.csv"),
    index=False, encoding="utf-8-sig"
)

if len(M_list) == 0:
    raise RuntimeError("没有任何被试通过 M_sparse QC（请查看 audit_m_sparse_failed_subjects.csv）")

M_array = np.stack(M_list, axis=0)
N, P, _ = M_array.shape

# ------------------------------------------------
# Step 2：群体模板 + 模板 QC（输出命名与正向一致）
# ------------------------------------------------
M_bar = M_array.mean(axis=0)
np.save(os.path.join(group_dir, "M_bar.npy"), M_bar)

row_norms = np.linalg.norm(M_bar, axis=1)
template_qc = {
    "row_norm_min": float(row_norms.min()),
    "row_norm_median": float(np.median(row_norms)),
    "n_near_zero_rows": int((row_norms < EPS_NORM).sum())
}

gm_template = GradientMaps(
    n_components=N_COMPONENTS,
    approach="dm",
    kernel="cosine",
    random_state=0
)
gm_template.fit(M_bar, sparsity=None)

G_ref = gm_template.gradients_
np.save(os.path.join(group_dir, "G_template.npy"), G_ref)

g1_ref = G_ref[:, 0]
template_qc["G1_std"] = float(np.std(g1_ref))

# ======================================================
# 梯度“解释百分比”输出（与正向一致：spectrum share）
# NOTE: 这是 diffusion eigen-spectrum share（归一化扩散特征值占比），不是 PCA 方差解释率
# ======================================================
lambdas = np.array(gm_template.lambdas_)[:N_COMPONENTS]
if np.any(~np.isfinite(lambdas)) or np.sum(lambdas) <= 0:
    lambdas = np.where(np.isfinite(lambdas), lambdas, 0.0)
    if np.sum(lambdas) <= 0:
        lambdas = np.ones(N_COMPONENTS, dtype=float)

explained_pct = lambdas / lambdas.sum() * 100.0

explained_csv = os.path.join(group_dir, "gradient_explained_percentage.csv")
with open(explained_csv, "w") as f:
    f.write("gradient,lambda,explained_percentage\n")
    for i, (lam, pct) in enumerate(zip(lambdas, explained_pct), start=1):
        f.write(f"G{i},{lam:.6e},{pct:.4f}\n")

# ------------------------------------------------
# Step 3：逐被试对齐（仅 Procrustes：K dims + 列zscore）
# ------------------------------------------------
qc_rows = []          # (sub_id, r_procrustes)
g_failed = []         # G_raw.npy失败审计
norm_audit_rows = []  # 记录 X/Y 列范数（对齐前审计）

for sub_id, sub_dir in zip(sub_ids, sub_dirs):
    # 负向输入文件：负向连接_G_raw.npy
    g_path = os.path.join(sub_dir, f"{PREFIX}G_raw.npy")

    try:
        if not os.path.exists(g_path):
            g_failed.append({"sub_id": sub_id, "reason": "G_raw_missing", "path": g_path})
            continue

        G = np.load(g_path)

        # QC: G 应该是 (P, n_components>=K_ALIGN)
        if not (isinstance(G, np.ndarray) and G.ndim == 2 and G.shape[0] == P and G.shape[1] >= K_ALIGN):
            g_failed.append({
                "sub_id": sub_id,
                "reason": f"G_dim_mismatch_shape={getattr(G,'shape',None)}_expected_P={P}_K={K_ALIGN}",
                "path": g_path
            })
            continue

        if not np.isfinite(G).all():
            g_failed.append({"sub_id": sub_id, "reason": "G_has_nan_inf", "path": g_path})
            continue

        # ---------- Procrustes (K dims, with column z-score) ----------
        X = G[:, :K_ALIGN].copy()
        Y = G_ref[:, :K_ALIGN].copy()

        # 记录列范数（对齐前）
        x_col_norm = np.linalg.norm(X, axis=0)
        y_col_norm = np.linalg.norm(Y, axis=0)

        # 列标准化（防止某维尺度支配 Procrustes）
        Xz = zscore_cols(X)
        Yz = zscore_cols(Y)

        R, _ = orthogonal_procrustes(Xz, Yz)
        X_aligned = Xz @ R

        g1_proc = X_aligned[:, 0]
        g2_proc = X_aligned[:, 1] if K_ALIGN >= 2 else None

        r_proc = safe_corr(g1_ref, g1_proc)

        # 输出（命名与正向一致；只输出 procrustes）
        out_sub = os.path.join(out_root, sub_id)
        os.makedirs(out_sub, exist_ok=True)

        np.save(os.path.join(out_sub, "out_G1_procrustes.npy"), g1_proc)
        if g2_proc is not None:
            np.save(os.path.join(out_sub, "out_G2_procrustes.npy"), g2_proc)

        qc_rows.append((sub_id, r_proc))

        norm_audit_rows.append({
            "sub_id": sub_id,
            "K_ALIGN": int(K_ALIGN),
            "X_col_norms": "|".join([f"{v:.6g}" for v in x_col_norm]),
            "Y_col_norms": "|".join([f"{v:.6g}" for v in y_col_norm]),
        })

    except Exception as e:
        g_failed.append({"sub_id": sub_id, "reason": f"G_process_failed:{type(e).__name__}:{e}", "path": g_path})
        continue

# 输出 G_raw 失败审计（补充目录；命名与正向一致）
pd.DataFrame(g_failed).to_csv(
    os.path.join(supp2_root, "audit_graw_failed_subjects.csv"),
    index=False, encoding="utf-8-sig"
)

# 输出 Procrustes 列范数审计（补充目录；命名与正向一致）
pd.DataFrame(norm_audit_rows).to_csv(
    os.path.join(supp2_root, "audit_procrustes_column_norms.csv"),
    index=False, encoding="utf-8-sig"
)

if len(qc_rows) == 0:
    raise RuntimeError("没有任何被试成功完成对齐（请查看补充目录中的 audit_graw_failed_subjects.csv）")

# ------------------------------------------------
# Step 4：保存 QC CSV（仅 Procrustes）
# ------------------------------------------------
qc_csv = os.path.join(group_dir, "alignment_qc_procrustes.csv")
with open(qc_csv, "w") as f:
    f.write("sub_id,r_procrustes\n")
    for sid, rp in qc_rows:
        rp_out = rp if np.isfinite(rp) else np.nan
        f.write(f"{sid},{rp_out:.4f}\n")

# ------------------------------------------------
# Step 5：summary_report.txt（仅 Procrustes）
# ------------------------------------------------
summary_path = os.path.join(group_dir, "summary_report.txt")

r_proc = np.array([r[1] for r in qc_rows], dtype=float)

def stat(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "NA"
    return f"mean={x.mean():.4f}, sd={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}"

n_success = len(qc_rows)

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Supplementary: Group template & alignment robustness report (Procrustes-only)\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Paths:\n")
    f.write(f"  sub_root: {sub_root}\n")
    f.write(f"  out_root: {out_root}\n")
    f.write(f"  supplementary_audit_dir: {supp2_root}\n\n")

    f.write("Sample size audit:\n")
    f.write(f"  Total sub-* dirs scanned: {n_total_dirs}\n")
    f.write(f"  M_sparse QC passed (template inputs): {len(M_list)}\n")
    f.write(f"  M_sparse QC failed: {len(m_failed)}  (see audit_m_sparse_failed_subjects.csv)\n")
    f.write(f"  Alignment success (G_raw read + alignment done): {n_success}\n")
    f.write(f"  Alignment failed after M_sparse pass: {len(g_failed)} (see audit_graw_failed_subjects.csv)\n\n")

    f.write("Template QC:\n")
    f.write(f"  ROI (P): {P}\n")
    for k, v in template_qc.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n")

    f.write("Diffusion eigen-spectrum share (Top components):\n")
    f.write("  NOTE: The percentages below are normalized diffusion eigenvalues (spectrum share),\n")
    f.write("        NOT 'variance explained' in the PCA sense.\n")
    for i, pct in enumerate(explained_pct, start=1):
        f.write(f"  G{i}: {pct:.2f}%\n")
    f.write("\n")

    f.write("Alignment quality (correlation with template G1):\n")
    f.write(f"  Method: Procrustes (multi-gradient, K_ALIGN={K_ALIGN}, column-zscore before fit)\n")
    f.write(f"  Low quality threshold: r < {R_LOW_THRESH}\n")
    f.write(f"  {stat(r_proc)}\n")
    f.write(f"  r<{R_LOW_THRESH} count = {int(np.sum(np.isfinite(r_proc) & (r_proc < R_LOW_THRESH)))}\n")
    f.write(f"  r is NaN count = {int(np.sum(~np.isfinite(r_proc)))}\n\n")

    f.write("K_ALIGN rationale (report-ready text):\n")
    f.write(f"  We aligned subject gradients to the group template using the first K_ALIGN={K_ALIGN} components.\n")
    f.write("  This choice balances capturing shared low-dimensional structure while avoiding noise and\n")
    f.write("  potential overfitting from higher-order components.\n\n")

    f.write("Failed list (examples, up to 30 each):\n")
    if len(m_failed) > 0:
        f.write("  M_sparse failures:\n")
        for row in m_failed[:30]:
            f.write(f"    - {row['sub_id']}: {row['reason']}\n")
    else:
        f.write("  M_sparse failures: None\n")

    if len(g_failed) > 0:
        f.write("  G_raw/alignment failures:\n")
        for row in g_failed[:30]:
            f.write(f"    - {row['sub_id']}: {row['reason']}\n")
    else:
        f.write("  G_raw/alignment failures: None\n")
    f.write("\n")

    f.write("Outputs:\n")
    f.write(f"  Template: {os.path.join(group_dir, 'M_bar.npy')}\n")
    f.write(f"  Template gradients: {os.path.join(group_dir, 'G_template.npy')}\n")
    f.write(f"  Spectrum share CSV: {explained_csv}\n")
    f.write(f"  Alignment QC CSV: {qc_csv}\n")
    f.write(f"  This summary: {summary_path}\n\n")

# ------------------------------------------------
# 补充结果报告文件（独立目录）
# ------------------------------------------------
supp_report = os.path.join(supp2_root, "supplementary_report.txt")
with open(supp_report, "w", encoding="utf-8") as f:
    f.write("Supplementary audit report (supp2, Procrustes-only)\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("High-level:\n")
    f.write(f"  Total sub-* dirs scanned: {n_total_dirs}\n")
    f.write(f"  M_sparse QC passed: {len(M_list)}\n")
    f.write(f"  Alignment success: {n_success}\n")
    f.write(f"  M_sparse failures: {len(m_failed)}\n")
    f.write(f"  G_raw/alignment failures: {len(g_failed)}\n\n")

    f.write("Key notes:\n")
    f.write("  - Procrustes alignment uses column-wise z-scoring of X and Y prior to orthogonal_procrustes.\n")
    f.write("  - Column norms before standardization are saved to audit_procrustes_column_norms.csv.\n")
    f.write("  - 'explained_percentage' is reported as diffusion eigen-spectrum share (normalized eigenvalues).\n\n")

    f.write("Audit files:\n")
    f.write("  - audit_m_sparse_failed_subjects.csv\n")
    f.write("  - audit_graw_failed_subjects.csv\n")
    f.write("  - audit_procrustes_column_norms.csv\n")
    f.write("  - (Original) group\\summary_report.txt\n\n")

print("分析完成，结果保存至：")
print("负向输出目录（结构与正向一致，仅最后一级不同）：", out_root)
print("summary_report.txt：", summary_path)
print("补充审计输出目录：", supp2_root)
print("补充报告：", supp_report)
print("梯度谱份额 CSV：", explained_csv)
print("对齐 QC CSV：", qc_csv)
