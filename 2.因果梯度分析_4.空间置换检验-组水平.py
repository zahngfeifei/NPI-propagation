import os
import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from brainspace.datasets import load_conte69, load_parcellation
from brainspace.null_models import SpinPermutations


# ============================================================
# 1. 路径设置
# ============================================================

DATASET = "ABIDE1"

# EC 数据目录：ComBat 后的正向连接 / 因果梯度结果
ec_root = r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-独立模板\ABIDE1_结果2_combat"

# FC 数据目录：ComBat 后的功能梯度结果
# FC 数据目录沿用现有文件夹名 comabt
fc_root = r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度\ABIDE1_功能梯度_群体模板_comabt"

# ------------------------------------------------------------
# 群体模板目录（与 ComBat 后的单被试梯度目录分开配置）
# ------------------------------------------------------------
# EC 群体模板位于未进行 ComBat 的独立模板结果目录中。
# 这里直接填写 group 文件夹本身，而不是其上一级目录。
ec_template_dir = (
    r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-独立模板"
    r"\ABIDE1_结果2\group"
)

# FC 群体模板位于未进行 ComBat 的功能梯度群体模板目录。
# 这里直接填写 group 文件夹本身，而不是其上一级目录。
fc_template_dir = (
    r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度"
    r"\ABIDE1_功能梯度_群体模板\group"
)

# 输出目录
out_root = r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度\ABIDE1_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平"

os.makedirs(out_root, exist_ok=True)

all_dir = os.path.join(out_root, "all_results")
metric_dir = os.path.join(out_root, "metric_separate_results")
comparison_dir = os.path.join(out_root, "comparison_separate_results")
template_dir = os.path.join(out_root, "template_results")
failed_dir = os.path.join(out_root, "failed")
spin_dir = os.path.join(out_root, "spin_model")
group_level_dir = os.path.join(out_root, "group_level_results")

for d in [
    all_dir,
    metric_dir,
    comparison_dir,
    template_dir,
    failed_dir,
    spin_dir,
    group_level_dir
]:
    os.makedirs(d, exist_ok=True)

if not os.path.isdir(ec_root):
    raise FileNotFoundError(f"EC root does not exist: {ec_root}")

if not os.path.isdir(fc_root):
    raise FileNotFoundError(f"FC root does not exist: {fc_root}")


# ============================================================
# 2. 参数
# ============================================================

EPS = 1e-12

K_EC = 2
K_FC = 2

DO_TEMPLATE = True

# 模板检验开启时，提前检查两个独立群体模板目录。
if DO_TEMPLATE:
    if not os.path.isdir(ec_template_dir):
        raise FileNotFoundError(
            f"EC template directory does not exist: {ec_template_dir}"
        )
    if not os.path.isdir(fc_template_dir):
        raise FileNotFoundError(
            f"FC template directory does not exist: {fc_template_dir}"
        )

EXPECTED_N_ROI = 400

N_PERM = 5000
SEED = 42

SPIN_UNIQUE = None

# "A": 旋转 EC 梯度，固定 FC 梯度
# "B": 固定 EC 梯度，旋转 FC 梯度
SPIN_TARGET = "A"

TAIL = "two-sided"

SAVE_SUBJECT_NULL_DISTRIBUTIONS = False
SAVE_TEMPLATE_NULL_DISTRIBUTIONS = True
SAVE_SPIN_INDICES = True

# 是否运行组水平空间置换检验
# 组水平统计量定义为：mean_subject_rho = mean(Spearman_i), 即先对每个被试计算 EC-FC rho，
# 再在被试维度取平均。组水平 p_spin 由 mean_subject_rho 与其 spin-null 分布比较得到。
DO_GROUP_LEVEL_SUBJECT_MEAN_SPIN = True
SAVE_GROUP_LEVEL_NULL_DISTRIBUTIONS = True

# 可选：是否额外对“组平均梯度图”做一次空间置换检验。
# 该检验统计量是 Spearman(mean_EC_map, mean_FC_map)，与 mean(Spearman_i) 不是同一个量。
DO_GROUP_MEAN_MAP_SPIN = True
SAVE_GROUP_MEAN_MAP_NULL_DISTRIBUTIONS = True


# ============================================================
# 3. 工具函数
# ============================================================

def normalize_sub_id(name):
    """
    将不同被试文件夹命名统一成可匹配的数字 ID。

    兼容示例：
        sub-Sub50007 -> 50007
        sub-50007    -> 50007
        Sub50007     -> 50007
        50007        -> 50007
        sub-0050007  -> 50007
    """

    name = os.path.basename(str(name)).strip()
    digits = re.findall(r"\d+", name)

    if len(digits) == 0:
        return name.lower()

    sid = digits[-1].lstrip("0")

    if sid == "":
        sid = "0"

    return sid


def safe_name(x):
    x = str(x)

    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        x = x.replace(ch, "_")

    return x


def project_to_unit_sphere(points, eps=1e-12):
    pts = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)

    if np.any(norms < eps):
        bad = np.where(norms.ravel() < eps)[0].tolist()
        raise RuntimeError(f"Near-zero centroid norm for indices: {bad[:10]}")

    return pts / norms


def spearman_correlation(a, b, eps=EPS):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")

    if len(a) < 3:
        raise ValueError(f"Spearman 相关至少需要 3 个 ROI，当前 n={len(a)}")

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("向量包含 NaN 或 Inf")

    ar = rankdata(a, method="average").astype(np.float64)
    br = rankdata(b, method="average").astype(np.float64)

    ar -= np.mean(ar)
    br -= np.mean(br)

    sa = np.sqrt(np.sum(ar ** 2))
    sb = np.sqrt(np.sum(br ** 2))

    if sa < eps or sb < eps:
        raise ValueError(
            f"秩向量近零，可能是常数向量: rank_norm_a={sa:.3e}, rank_norm_b={sb:.3e}"
        )

    rho = float(np.sum(ar * br) / (sa * sb))
    rho = max(min(rho, 1.0), -1.0)

    return rho


def spearman_matrix_vs_vector(mat, vec, eps=EPS):
    mat = np.asarray(mat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)

    if mat.ndim != 2:
        raise ValueError(f"mat 必须为二维矩阵，当前 shape={mat.shape}")

    if mat.shape[1] != len(vec):
        raise ValueError(f"mat 和 vec ROI 数不一致: {mat.shape[1]} vs {len(vec)}")

    if not np.isfinite(mat).all() or not np.isfinite(vec).all():
        raise ValueError("mat 或 vec 包含 NaN/Inf")

    mat_r = rankdata(mat, axis=1, method="average").astype(np.float64)
    vec_r = rankdata(vec, method="average").astype(np.float64)

    mat_r -= np.mean(mat_r, axis=1, keepdims=True)
    vec_r -= np.mean(vec_r)

    mat_norm = np.sqrt(np.sum(mat_r ** 2, axis=1))
    vec_norm = np.sqrt(np.sum(vec_r ** 2))

    if vec_norm < eps:
        raise ValueError("固定向量秩近零，可能是常数向量")

    if np.any(mat_norm < eps):
        bad_n = int(np.sum(mat_norm < eps))
        raise ValueError(f"{bad_n} 个置换向量秩近零，可能是常数向量")

    rho = (mat_r @ vec_r) / (mat_norm * vec_norm)
    rho = np.clip(rho, -1.0, 1.0)

    return rho.astype(np.float64)


def compute_p_spin(rho_obs, null_rhos, tail="two-sided"):
    null_rhos = np.asarray(null_rhos, dtype=np.float64).reshape(-1)
    n_perm = len(null_rhos)

    if tail == "two-sided":
        p_spin = (1.0 + np.sum(np.abs(null_rhos) >= abs(rho_obs))) / (n_perm + 1.0)

    elif tail == "greater":
        p_spin = (1.0 + np.sum(null_rhos >= rho_obs)) / (n_perm + 1.0)

    elif tail == "less":
        p_spin = (1.0 + np.sum(null_rhos <= rho_obs)) / (n_perm + 1.0)

    else:
        raise ValueError("TAIL 必须为 'two-sided', 'greater', 或 'less'")

    return float(p_spin)


def summarize_null_distribution(null_rhos):
    null_rhos = np.asarray(null_rhos, dtype=np.float64).reshape(-1)

    return {
        "null_mean": float(np.nanmean(null_rhos)),
        "null_sd": float(np.nanstd(null_rhos, ddof=1)),
        "null_min": float(np.nanmin(null_rhos)),
        "null_max": float(np.nanmax(null_rhos))
    }


def check_gradient_matrix(G, name, k_required):
    if G.ndim != 2:
        raise ValueError(f"{name} 不是二维矩阵: shape={G.shape}")

    if G.shape[1] < k_required:
        raise ValueError(
            f"{name} 梯度轴数量不足: shape={G.shape}, k_required={k_required}"
        )

    if EXPECTED_N_ROI is not None and G.shape[0] != EXPECTED_N_ROI:
        raise ValueError(
            f"{name} ROI 数不是 {EXPECTED_N_ROI}: gradient_ROI={G.shape[0]}"
        )

    if not np.isfinite(G).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")


def load_brainspace_schaefer400_spin_centroids():
    print("\n[INFO] Building BrainSpace Schaefer400 sphere centroids...")

    sphere_lh, sphere_rh = load_conte69(as_sphere=True, join=False)
    parc_lh, parc_rh = load_parcellation(name="schaefer", scale=400, join=False)

    lh_ids = np.unique(parc_lh)
    lh_ids = lh_ids[lh_ids != 0]

    rh_ids = np.unique(parc_rh)
    rh_ids = rh_ids[rh_ids != 0]

    if not (len(lh_ids) == 200 and lh_ids.min() == 1 and lh_ids.max() == 200):
        raise RuntimeError(
            f"Unexpected LH parcel IDs: min={lh_ids.min()}, max={lh_ids.max()}, n={len(lh_ids)}"
        )

    if not (len(rh_ids) == 200 and rh_ids.min() == 201 and rh_ids.max() == 400):
        raise RuntimeError(
            f"Unexpected RH parcel IDs: min={rh_ids.min()}, max={rh_ids.max()}, n={len(rh_ids)}"
        )

    print("[QC] BrainSpace parcellation IDs OK: LH=1..200, RH=201..400")

    centroids_lh = np.zeros((200, 3), dtype=np.float64)

    for pid in range(1, 201):
        verts = np.where(parc_lh == pid)[0]

        if len(verts) == 0:
            raise RuntimeError(f"LH parcel {pid} has no vertices.")

        centroids_lh[pid - 1] = sphere_lh.Points[verts].mean(axis=0)

    centroids_rh = np.zeros((200, 3), dtype=np.float64)

    for pid in range(201, 401):
        verts = np.where(parc_rh == pid)[0]

        if len(verts) == 0:
            raise RuntimeError(f"RH parcel {pid} has no vertices.")

        centroids_rh[pid - 201] = sphere_rh.Points[verts].mean(axis=0)

    centroids_lh = project_to_unit_sphere(centroids_lh)
    centroids_rh = project_to_unit_sphere(centroids_rh)

    return centroids_lh, centroids_rh


def fit_brainspace_spin_model(centroids_lh, centroids_rh):
    print("\n[INFO] Fitting BrainSpace SpinPermutations...")

    if SPIN_UNIQUE is None:
        sp = SpinPermutations(
            n_rep=N_PERM,
            random_state=SEED
        )
    else:
        sp = SpinPermutations(
            n_rep=N_PERM,
            random_state=SEED,
            unique=SPIN_UNIQUE
        )

    sp.fit(
        centroids_lh,
        points_rh=centroids_rh
    )

    spin_lh = np.asarray(sp.spin_lh_, dtype=np.int64)
    spin_rh = np.asarray(sp.spin_rh_, dtype=np.int64)

    if spin_lh.shape != (N_PERM, 200):
        raise RuntimeError(f"spin_lh_ shape 异常: {spin_lh.shape}")

    if spin_rh.shape != (N_PERM, 200):
        raise RuntimeError(f"spin_rh_ shape 异常: {spin_rh.shape}")

    full_spin_indices = np.empty((N_PERM, 400), dtype=np.int32)
    full_spin_indices[:, 0:200] = spin_lh
    full_spin_indices[:, 200:400] = spin_rh + 200

    print("[INFO] BrainSpace spin model fitted.")
    print("  spin_lh_:", spin_lh.shape)
    print("  spin_rh_:", spin_rh.shape)
    print("  full_spin_indices:", full_spin_indices.shape)

    if SAVE_SPIN_INDICES:
        tag = f"n400_perm{N_PERM}_seed{SEED}_unique{SPIN_UNIQUE}"

        np.save(
            os.path.join(spin_dir, f"brainspace_spin_lh_{tag}.npy"),
            spin_lh
        )

        np.save(
            os.path.join(spin_dir, f"brainspace_spin_rh_{tag}.npy"),
            spin_rh
        )

        np.save(
            os.path.join(spin_dir, f"brainspace_spin_indices_full_{tag}.npy"),
            full_spin_indices
        )

        centroid_df = pd.DataFrame({
            "ROI_ID": np.arange(1, 401),
            "hemi": ["L"] * 200 + ["R"] * 200,
            "x": np.r_[centroids_lh[:, 0], centroids_rh[:, 0]],
            "y": np.r_[centroids_lh[:, 1], centroids_rh[:, 1]],
            "z": np.r_[centroids_lh[:, 2], centroids_rh[:, 2]],
        })

        centroid_df.to_csv(
            os.path.join(spin_dir, "brainspace_schaefer400_sphere_centroids.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    return sp, full_spin_indices


def spin_spearman_test(map_a, map_b, full_spin_indices, spin_target="A", tail="two-sided"):
    """
    map_a: EC gradient
    map_b: FC gradient

    spin_target:
        "A": 旋转 EC，固定 FC
        "B": 固定 EC，旋转 FC
    """

    map_a = np.asarray(map_a, dtype=np.float64).reshape(-1)
    map_b = np.asarray(map_b, dtype=np.float64).reshape(-1)

    if map_a.shape != map_b.shape:
        raise ValueError(f"Map shape 不一致: {map_a.shape} vs {map_b.shape}")

    if len(map_a) != 400:
        raise ValueError(f"当前脚本要求 400 ROI，实际 len={len(map_a)}")

    if not np.isfinite(map_a).all() or not np.isfinite(map_b).all():
        raise ValueError("map_a 或 map_b 包含 NaN/Inf")

    rho_obs = spearman_correlation(map_a, map_b)

    spin_target = str(spin_target).upper()

    if spin_target == "A":
        spun_maps = map_a[full_spin_indices]
        null_rhos = spearman_matrix_vs_vector(spun_maps, map_b)

    elif spin_target == "B":
        spun_maps = map_b[full_spin_indices]
        null_rhos = spearman_matrix_vs_vector(spun_maps, map_a)

    else:
        raise ValueError("SPIN_TARGET 必须为 'A' 或 'B'")

    p_spin = compute_p_spin(rho_obs, null_rhos, tail=tail)

    return {
        "rho_obs": float(rho_obs),
        "p_spin": float(p_spin),
        "null_mean": float(np.nanmean(null_rhos)),
        "null_sd": float(np.nanstd(null_rhos, ddof=1)),
        "null_min": float(np.nanmin(null_rhos)),
        "null_max": float(np.nanmax(null_rhos)),
        "null_rhos": null_rhos
    }


def load_gradient_from_subject_dir(sub_dir, modality, k_use):
    """
    读取单被试梯度。

    EC 默认读取：
        out_G1_procrustes.npy
        out_G2_procrustes.npy

    FC 默认读取：
        G_procrustes_z.npy

    回退读取：
        FC_G1_procrustes_z.npy
        FC_G2_procrustes_z.npy
    """

    modality = modality.upper()

    matrix_candidates = []

    if modality == "FC":
        matrix_candidates = [
            "G_procrustes_z.npy",
            "G_procrustes.npy",
            "FC_G_aligned_K5.npy",
            "G_aligned_K5.npy",
            "G_raw.npy"
        ]

    elif modality == "EC":
        matrix_candidates = [
            "G_procrustes_z.npy",
            "G_procrustes.npy",
            "EC_G_aligned_K5.npy",
            "NPI_G_aligned_K5.npy",
            "G_aligned_K5.npy",
            "G_raw.npy"
        ]

    else:
        raise ValueError("modality 必须为 'EC' 或 'FC'")

    for fname in matrix_candidates:
        fpath = os.path.join(sub_dir, fname)

        if os.path.exists(fpath):
            G = np.load(fpath)

            if G.ndim == 2 and G.shape[1] >= k_use:
                return G[:, :k_use], fpath

    cols = []
    used_files = []

    for k in range(1, k_use + 1):

        if modality == "EC":
            axis_candidates = [
                f"out_G{k}_procrustes_z.npy",
                f"out_G{k}_procrustes_combat.npy",
                f"out_G{k}_procrustes.npy",
                f"out_G{k}_z.npy",
                f"EC_G{k}_procrustes_z.npy",
                f"EC_G{k}_procrustes.npy",
                f"EC_G{k}_z.npy"
            ]

        else:
            axis_candidates = [
                f"FC_G{k}_procrustes_z.npy",
                f"FC_G{k}_procrustes_combat.npy",
                f"FC_G{k}_procrustes.npy",
                f"FC_G{k}_z.npy"
            ]

        found = None

        for fname in axis_candidates:
            fpath = os.path.join(sub_dir, fname)

            if os.path.exists(fpath):
                found = fpath
                break

        if found is None:
            raise FileNotFoundError(
                f"{modality} 缺少 G{k} 文件: {sub_dir}"
            )

        vec = np.load(found)

        if vec.ndim != 1:
            vec = np.asarray(vec).reshape(-1)

        cols.append(vec)
        used_files.append(found)

    G = np.column_stack(cols)

    return G, "|".join(used_files)


def load_template_gradient(group_dir_local, modality, k_use):
    """
    从指定的群体模板 group 文件夹读取 G1/G2。

    注意：
    group_dir_local 必须是 group 文件夹本身，例如：
        I:\\...\\ABIDE1_结果2\\group

    该路径与 ComBat 后的单被试梯度目录相互独立。
    """
    group_dir_local = os.path.abspath(
        os.path.normpath(group_dir_local)
    )

    if not os.path.isdir(group_dir_local):
        raise FileNotFoundError(
            f"{modality} 群体模板目录不存在: {group_dir_local}"
        )

    matrix_candidates = [
        "G_template.npy",
        f"{modality}_G_template.npy",
        "FC_G_template.npy",
        "EC_G_template.npy",
        "NPI_G_template.npy"
    ]

    for fname in matrix_candidates:
        fpath = os.path.join(group_dir_local, fname)

        if os.path.exists(fpath):
            G = np.load(fpath)

            if G.ndim == 2 and G.shape[1] >= k_use:
                return G[:, :k_use], fpath

    cols = []
    used_files = []

    for k in range(1, k_use + 1):

        axis_candidates = [
            f"{modality}_G{k}_template.npy",
            f"{modality}_G{k}_template_combat.npy",
            f"FC_G{k}_template.npy",
            f"EC_G{k}_template.npy",
            f"out_G{k}_template.npy"
        ]

        found = None

        for fname in axis_candidates:
            fpath = os.path.join(group_dir_local, fname)

            if os.path.exists(fpath):
                found = fpath
                break

        if found is None:
            raise FileNotFoundError(
                f"{modality} group 模板缺少 G{k} 文件: {group_dir_local}"
            )

        vec = np.load(found)

        if vec.ndim != 1:
            vec = np.asarray(vec).reshape(-1)

        cols.append(vec)
        used_files.append(found)

    G = np.column_stack(cols)

    return G, "|".join(used_files)


def build_subject_map(root, modality, k_use):
    exclude_dirs = {
        "group",
        "audit",
        "audit_group_template",
        "failed_subjects",
        "failed",
        "spin_model",
        "all_results",
        "metric_separate_results",
        "gradient_separate_results",
        "comparison_separate_results",
        "template_results",
        "__pycache__"
    }

    subject_map = {}
    failed = []

    for d in sorted(glob.glob(os.path.join(root, "*"))):

        if not os.path.isdir(d):
            continue

        name = os.path.basename(d)

        if name in exclude_dirs:
            continue

        try:
            G, src = load_gradient_from_subject_dir(
                d,
                modality=modality,
                k_use=k_use
            )

            check_gradient_matrix(
                G,
                f"{modality}_G",
                k_required=k_use
            )

            norm_id = normalize_sub_id(name)

            if norm_id in subject_map:
                failed.append({
                    "modality": modality,
                    "raw_id": name,
                    "dir": d,
                    "error": (
                        f"被试 ID 归一化后重复: {norm_id}; "
                        f"已存在 {subject_map[norm_id]['raw_id']}"
                    )
                })
                continue

            subject_map[norm_id] = {
                "raw_id": name,
                "dir": d,
                "source": src
            }

        except Exception as e:
            failed.append({
                "modality": modality,
                "raw_id": name,
                "dir": d,
                "error": str(e)
            })

    return subject_map, failed


# ============================================================
# 4. 建立 EC / FC 被试索引
# ============================================================

ec_map, ec_index_failed = build_subject_map(
    ec_root,
    modality="EC",
    k_use=K_EC
)

fc_map, fc_index_failed = build_subject_map(
    fc_root,
    modality="FC",
    k_use=K_FC
)

print(f"EC 可用被试数: {len(ec_map)}")
print(f"FC 可用被试数: {len(fc_map)}")

common_ids = sorted(set(ec_map.keys()) & set(fc_map.keys()))

print(f"EC 与 FC 共同被试数: {len(common_ids)}")

index_failed_csv = os.path.join(failed_dir, "index_failed_subjects.csv")

pd.DataFrame(ec_index_failed + fc_index_failed).to_csv(
    index_failed_csv,
    index=False,
    encoding="utf-8-sig"
)

if len(common_ids) == 0:
    raise RuntimeError(
        "没有匹配到 EC 与 FC 共同被试。请检查两个目录中的 sub-* 文件夹命名是否一致。"
    )


# ============================================================
# 5. 构建 BrainSpace spin model
# ============================================================

centroids_lh, centroids_rh = load_brainspace_schaefer400_spin_centroids()
sp, full_spin_indices = fit_brainspace_spin_model(centroids_lh, centroids_rh)


# ============================================================
# 6. 逐被试 2 × 2 cross-axis spin test
# ============================================================

rows = []
failed_rows = []

# 组水平 subject-mean spin test 的累加器。
# 对每个 comparison，保存：
#   1) 所有被试的 rho_obs，用于计算 observed mean_subject_rho；
#   2) 每个 permutation 下所有被试 null rho 的累加和，用于得到 group null distribution。
group_obs_rhos = {}
group_null_sum = {}
group_null_n = {}

# 可选：保存成功进入分析的 EC/FC 梯度矩阵，用于组平均梯度图 spin test。
group_ec_mats = []
group_fc_mats = []
group_match_ids = []

for sid in common_ids:

    ec_info = ec_map[sid]
    fc_info = fc_map[sid]

    try:
        ec_G, ec_source = load_gradient_from_subject_dir(
            ec_info["dir"],
            modality="EC",
            k_use=K_EC
        )

        fc_G, fc_source = load_gradient_from_subject_dir(
            fc_info["dir"],
            modality="FC",
            k_use=K_FC
        )

        check_gradient_matrix(ec_G, "EC_G", k_required=K_EC)
        check_gradient_matrix(fc_G, "FC_G", k_required=K_FC)

        if ec_G.shape[0] != fc_G.shape[0]:
            raise ValueError(
                f"EC 与 FC ROI 数不一致: EC={ec_G.shape}, FC={fc_G.shape}"
            )

        n_roi = ec_G.shape[0]

        if DO_GROUP_MEAN_MAP_SPIN:
            group_ec_mats.append(ec_G.copy())
            group_fc_mats.append(fc_G.copy())
            group_match_ids.append(sid)

        for i_ec in range(K_EC):
            for j_fc in range(K_FC):

                ec_gname = f"EC_G{i_ec + 1}"
                fc_gname = f"FC_G{j_fc + 1}"
                comparison = f"{ec_gname}_vs_{fc_gname}"

                ec_vec = ec_G[:, i_ec]
                fc_vec = fc_G[:, j_fc]

                spin_res = spin_spearman_test(
                    map_a=ec_vec,
                    map_b=fc_vec,
                    full_spin_indices=full_spin_indices,
                    spin_target=SPIN_TARGET,
                    tail=TAIL
                )

                if DO_GROUP_LEVEL_SUBJECT_MEAN_SPIN:
                    if comparison not in group_obs_rhos:
                        group_obs_rhos[comparison] = []
                        group_null_sum[comparison] = np.zeros(N_PERM, dtype=np.float64)
                        group_null_n[comparison] = 0

                    group_obs_rhos[comparison].append(spin_res["rho_obs"])
                    group_null_sum[comparison] += spin_res["null_rhos"]
                    group_null_n[comparison] += 1

                row = {
                    "match_id": sid,
                    "ec_id": ec_info["raw_id"],
                    "fc_id": fc_info["raw_id"],
                    "ec_source": ec_source,
                    "fc_source": fc_source,
                    "roi_scope": "whole_brain",
                    "n_roi": n_roi,

                    "ec_gradient": ec_gname,
                    "fc_gradient": fc_gname,
                    "comparison": comparison,

                    "spearman_r": spin_res["rho_obs"],
                    "p_spin": spin_res["p_spin"],
                    "n_perm": N_PERM,
                    "tail": TAIL,
                    "spin_target": SPIN_TARGET,
                    "spin_null_mean": spin_res["null_mean"],
                    "spin_null_sd": spin_res["null_sd"],
                    "spin_null_min": spin_res["null_min"],
                    "spin_null_max": spin_res["null_max"],

                    "status": "success",
                    "error": ""
                }

                rows.append(row)

                if SAVE_SUBJECT_NULL_DISTRIBUTIONS:
                    null_dir = os.path.join(
                        out_root,
                        "subject_null_distributions",
                        safe_name(comparison)
                    )

                    os.makedirs(null_dir, exist_ok=True)

                    null_csv = os.path.join(
                        null_dir,
                        f"{safe_name(ec_info['raw_id'])}_vs_{safe_name(fc_info['raw_id'])}_{safe_name(comparison)}_null.csv"
                    )

                    pd.DataFrame({
                        "perm_id": np.arange(1, N_PERM + 1),
                        "rho_null": spin_res["null_rhos"]
                    }).to_csv(
                        null_csv,
                        index=False,
                        encoding="utf-8-sig"
                    )

    except Exception as e:
        failed_rows.append({
            "match_id": sid,
            "ec_id": ec_info["raw_id"],
            "fc_id": fc_info["raw_id"],
            "roi_scope": "whole_brain",
            "comparison": "ALL",
            "error": str(e)
        })


df_long = pd.DataFrame(rows)
df_failed = pd.DataFrame(failed_rows)


# ============================================================
# 7. 保存逐被试 2 × 2 cross-axis 结果
# ============================================================

all_long_csv = os.path.join(
    all_dir,
    "ALL_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test_long.csv"
)

df_long.to_csv(
    all_long_csv,
    index=False,
    encoding="utf-8-sig"
)

if len(df_long) > 0:

    base_cols = [
        "match_id",
        "ec_id",
        "fc_id",
        "roi_scope",
        "n_roi",
        "ec_gradient",
        "fc_gradient",
        "comparison"
    ]

    spin_test_csv = os.path.join(
        metric_dir,
        "spin_test_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis.csv"
    )

    df_long[
        base_cols + [
            "spearman_r",
            "p_spin",
            "n_perm",
            "tail",
            "spin_target",
            "spin_null_mean",
            "spin_null_sd",
            "spin_null_min",
            "spin_null_max"
        ]
    ].to_csv(
        spin_test_csv,
        index=False,
        encoding="utf-8-sig"
    )

    for comparison in sorted(df_long["comparison"].unique()):
        sub_df = df_long[df_long["comparison"] == comparison].copy()

        file_name = f"wholebrain_{safe_name(comparison)}_spin_test.csv"

        sub_df.to_csv(
            os.path.join(comparison_dir, file_name),
            index=False,
            encoding="utf-8-sig"
        )

    wide_parts = []

    metrics = [
        "spearman_r",
        "p_spin",
        "spin_null_mean",
        "spin_null_sd",
        "spin_null_min",
        "spin_null_max"
    ]

    for metric in metrics:
        tmp = df_long.pivot_table(
            index=["match_id", "ec_id", "fc_id"],
            columns="comparison",
            values=metric,
            aggfunc="first"
        )

        tmp.columns = [f"{metric}_{comp}" for comp in tmp.columns]
        wide_parts.append(tmp)

    df_wide = pd.concat(wide_parts, axis=1).reset_index()

else:
    spin_test_csv = ""
    df_wide = pd.DataFrame()


wide_csv = os.path.join(
    all_dir,
    "ALL_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test_wide.csv"
)

df_wide.to_csv(
    wide_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 保存 comparison 层面的统计摘要
# ============================================================

summary_rows = []

if len(df_long) > 0:

    for comparison in sorted(df_long["comparison"].unique()):

        sub_df = df_long[df_long["comparison"] == comparison].copy()

        if len(sub_df) == 0:
            continue

        pvals = sub_df["p_spin"].to_numpy(dtype=float)

        summary_rows.append({
            "roi_scope": "whole_brain",
            "comparison": comparison,
            "ec_gradient": sub_df["ec_gradient"].iloc[0],
            "fc_gradient": sub_df["fc_gradient"].iloc[0],
            "n_subjects": len(sub_df),
            "n_roi": int(sub_df["n_roi"].iloc[0]),

            "spearman_mean": float(np.nanmean(sub_df["spearman_r"])),
            "spearman_sd": float(np.nanstd(sub_df["spearman_r"])),
            "spearman_min": float(np.nanmin(sub_df["spearman_r"])),
            "spearman_max": float(np.nanmax(sub_df["spearman_r"])),

            "p_spin_mean": float(np.nanmean(pvals)),
            "p_spin_median": float(np.nanmedian(pvals)),
            "p_spin_min": float(np.nanmin(pvals)),
            "p_spin_max": float(np.nanmax(pvals)),
            "n_p_spin_lt_0p05": int(np.sum(pvals < 0.05)),
            "n_p_spin_lt_0p01": int(np.sum(pvals < 0.01)),
            "n_p_spin_lt_0p001": int(np.sum(pvals < 0.001)),

            "spin_null_mean_mean": float(np.nanmean(sub_df["spin_null_mean"])),
            "spin_null_sd_mean": float(np.nanmean(sub_df["spin_null_sd"]))
        })


df_summary = pd.DataFrame(summary_rows)

summary_csv = os.path.join(
    all_dir,
    "ALL_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_summary_statistics.csv"
)

df_summary.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. 组水平 2 × 2 cross-axis spin test
# ============================================================
# 组水平检验包括以下两种：
#   A) group-level subject-mean spin test：
#      统计量 = 所有被试 subject-level Spearman rho 的平均值。
#      这是对前面 spearman_mean 的正式组水平空间置换检验。
#
#   B) group-mean map spin test：
#      统计量 = Spearman(组平均 EC 梯度图, 组平均 FC 梯度图)。
#      这是对组平均空间图本身的检验，和 A 的统计量不同。
# ============================================================

group_subject_mean_rows = []

if DO_GROUP_LEVEL_SUBJECT_MEAN_SPIN and len(group_obs_rhos) > 0:

    for comparison in sorted(group_obs_rhos.keys()):

        obs_rhos = np.asarray(group_obs_rhos[comparison], dtype=np.float64)
        n_sub = int(group_null_n[comparison])

        if n_sub == 0:
            continue

        group_null_rhos = group_null_sum[comparison] / float(n_sub)
        group_rho_obs = float(np.nanmean(obs_rhos))
        group_p_spin = compute_p_spin(group_rho_obs, group_null_rhos, tail=TAIL)
        null_summary = summarize_null_distribution(group_null_rhos)

        ec_gradient, fc_gradient = comparison.split("_vs_")

        group_subject_mean_rows.append({
            "roi_scope": "whole_brain",
            "comparison": comparison,
            "ec_gradient": ec_gradient,
            "fc_gradient": fc_gradient,
            "n_subjects": n_sub,
            "n_roi": EXPECTED_N_ROI,

            # 关键结果：这是组水平统计量及其组水平空间置换 p 值
            "group_statistic": "mean_subject_spearman_r",
            "group_spearman_mean": group_rho_obs,
            "group_p_spin": group_p_spin,

            # 描述 subject-level rho 的分布
            "subject_spearman_sd": float(np.nanstd(obs_rhos, ddof=1)),
            "subject_spearman_min": float(np.nanmin(obs_rhos)),
            "subject_spearman_max": float(np.nanmax(obs_rhos)),

            # 描述 group-level null distribution
            "n_perm": N_PERM,
            "tail": TAIL,
            "spin_target": SPIN_TARGET,
            "group_spin_null_mean": null_summary["null_mean"],
            "group_spin_null_sd": null_summary["null_sd"],
            "group_spin_null_min": null_summary["null_min"],
            "group_spin_null_max": null_summary["null_max"]
        })

        if SAVE_GROUP_LEVEL_NULL_DISTRIBUTIONS:
            null_csv = os.path.join(
                group_level_dir,
                f"group_level_subject_mean_spin_null_distribution_wholebrain_{safe_name(comparison)}.csv"
            )

            pd.DataFrame({
                "perm_id": np.arange(1, N_PERM + 1),
                "mean_subject_rho_null": group_null_rhos
            }).to_csv(
                null_csv,
                index=False,
                encoding="utf-8-sig"
            )


df_group_subject_mean = pd.DataFrame(group_subject_mean_rows)

group_subject_mean_csv = os.path.join(
    group_level_dir,
    "group_level_subject_mean_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
)

df_group_subject_mean.to_csv(
    group_subject_mean_csv,
    index=False,
    encoding="utf-8-sig"
)


group_mean_map_rows = []

if DO_GROUP_MEAN_MAP_SPIN and len(group_ec_mats) > 0:

    ec_stack = np.stack(group_ec_mats, axis=0)  # shape = n_subjects x n_roi x K_EC
    fc_stack = np.stack(group_fc_mats, axis=0)  # shape = n_subjects x n_roi x K_FC

    ec_group_mean = np.nanmean(ec_stack, axis=0)  # shape = n_roi x K_EC
    fc_group_mean = np.nanmean(fc_stack, axis=0)  # shape = n_roi x K_FC

    check_gradient_matrix(ec_group_mean, "EC_group_mean", k_required=K_EC)
    check_gradient_matrix(fc_group_mean, "FC_group_mean", k_required=K_FC)

    np.save(
        os.path.join(group_level_dir, "EC_group_mean_gradient_map.npy"),
        ec_group_mean
    )

    np.save(
        os.path.join(group_level_dir, "FC_group_mean_gradient_map.npy"),
        fc_group_mean
    )

    pd.DataFrame(
        ec_group_mean,
        columns=[f"EC_G{k + 1}" for k in range(K_EC)]
    ).assign(ROI_ID=np.arange(1, ec_group_mean.shape[0] + 1)).to_csv(
        os.path.join(group_level_dir, "EC_group_mean_gradient_map.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        fc_group_mean,
        columns=[f"FC_G{k + 1}" for k in range(K_FC)]
    ).assign(ROI_ID=np.arange(1, fc_group_mean.shape[0] + 1)).to_csv(
        os.path.join(group_level_dir, "FC_group_mean_gradient_map.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    for i_ec in range(K_EC):
        for j_fc in range(K_FC):

            ec_gname = f"EC_G{i_ec + 1}"
            fc_gname = f"FC_G{j_fc + 1}"
            comparison = f"{ec_gname}_vs_{fc_gname}"

            spin_res = spin_spearman_test(
                map_a=ec_group_mean[:, i_ec],
                map_b=fc_group_mean[:, j_fc],
                full_spin_indices=full_spin_indices,
                spin_target=SPIN_TARGET,
                tail=TAIL
            )

            group_mean_map_rows.append({
                "roi_scope": "whole_brain",
                "comparison": comparison,
                "ec_gradient": ec_gname,
                "fc_gradient": fc_gname,
                "n_subjects": int(ec_stack.shape[0]),
                "n_roi": int(ec_group_mean.shape[0]),

                # 注意：这是组平均图之间的相关，不等于 mean(subject-level rho)
                "group_statistic": "spearman_of_group_mean_maps",
                "group_mean_map_spearman_r": spin_res["rho_obs"],
                "group_mean_map_p_spin": spin_res["p_spin"],
                "n_perm": N_PERM,
                "tail": TAIL,
                "spin_target": SPIN_TARGET,
                "group_mean_map_spin_null_mean": spin_res["null_mean"],
                "group_mean_map_spin_null_sd": spin_res["null_sd"],
                "group_mean_map_spin_null_min": spin_res["null_min"],
                "group_mean_map_spin_null_max": spin_res["null_max"]
            })

            if SAVE_GROUP_MEAN_MAP_NULL_DISTRIBUTIONS:
                null_csv = os.path.join(
                    group_level_dir,
                    f"group_mean_map_spin_null_distribution_wholebrain_{safe_name(comparison)}.csv"
                )

                pd.DataFrame({
                    "perm_id": np.arange(1, N_PERM + 1),
                    "rho_null": spin_res["null_rhos"]
                }).to_csv(
                    null_csv,
                    index=False,
                    encoding="utf-8-sig"
                )


df_group_mean_map = pd.DataFrame(group_mean_map_rows)

group_mean_map_csv = os.path.join(
    group_level_dir,
    "group_mean_map_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
)

df_group_mean_map.to_csv(
    group_mean_map_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 模板层面 2 × 2 cross-axis spin test
# ============================================================

template_rows = []

if DO_TEMPLATE:

    try:
        # 模板路径与 ComBat 后的单被试梯度路径分开使用。
        ec_T, ec_template_source = load_template_gradient(
            ec_template_dir,
            modality="EC",
            k_use=K_EC
        )

        fc_T, fc_template_source = load_template_gradient(
            fc_template_dir,
            modality="FC",
            k_use=K_FC
        )

        check_gradient_matrix(ec_T, "EC_G_template", k_required=K_EC)
        check_gradient_matrix(fc_T, "FC_G_template", k_required=K_FC)

        if ec_T.shape[0] != fc_T.shape[0]:
            raise ValueError(
                f"模板 ROI 数不一致: EC={ec_T.shape}, FC={fc_T.shape}"
            )

        n_roi = ec_T.shape[0]

        for i_ec in range(K_EC):
            for j_fc in range(K_FC):

                ec_gname = f"EC_G{i_ec + 1}"
                fc_gname = f"FC_G{j_fc + 1}"
                comparison = f"{ec_gname}_vs_{fc_gname}"

                ec_vec = ec_T[:, i_ec]
                fc_vec = fc_T[:, j_fc]

                spin_res = spin_spearman_test(
                    map_a=ec_vec,
                    map_b=fc_vec,
                    full_spin_indices=full_spin_indices,
                    spin_target=SPIN_TARGET,
                    tail=TAIL
                )

                template_rows.append({
                    "roi_scope": "whole_brain",
                    "n_roi": n_roi,

                    "ec_gradient": ec_gname,
                    "fc_gradient": fc_gname,
                    "comparison": comparison,

                    "ec_template_source": ec_template_source,
                    "fc_template_source": fc_template_source,

                    "template_spearman_r": spin_res["rho_obs"],
                    "template_p_spin": spin_res["p_spin"],
                    "n_perm": N_PERM,
                    "tail": TAIL,
                    "spin_target": SPIN_TARGET,
                    "template_spin_null_mean": spin_res["null_mean"],
                    "template_spin_null_sd": spin_res["null_sd"],
                    "template_spin_null_min": spin_res["null_min"],
                    "template_spin_null_max": spin_res["null_max"]
                })

                if SAVE_TEMPLATE_NULL_DISTRIBUTIONS:
                    null_csv = os.path.join(
                        template_dir,
                        f"template_spin_null_distribution_wholebrain_{safe_name(comparison)}.csv"
                    )

                    pd.DataFrame({
                        "perm_id": np.arange(1, N_PERM + 1),
                        "rho_null": spin_res["null_rhos"]
                    }).to_csv(
                        null_csv,
                        index=False,
                        encoding="utf-8-sig"
                    )

    except Exception as e:
        print(f"模板空间置换检验失败，跳过模板结果: {e}")


df_template = pd.DataFrame(template_rows)

template_all_csv = os.path.join(
    template_dir,
    "template_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
)

df_template.to_csv(
    template_all_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 保存失败记录和报告
# ============================================================

failed_csv = os.path.join(
    failed_dir,
    "failed_subjects.csv"
)

df_failed.to_csv(
    failed_csv,
    index=False,
    encoding="utf-8-sig"
)

summary_report = os.path.join(
    out_root,
    "EC_FC_positive_sparse_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test_summary_report.txt"
)

with open(summary_report, "w", encoding="utf-8") as f:

    f.write("EC vs positive-sparse FC whole-brain 2 x 2 cross-axis BrainSpace spin permutation report\n")
    f.write(f"Generated at: {datetime.now()}\n\n")

    f.write("Paths:\n")
    f.write(f"  DATASET: {DATASET}\n")
    f.write(f"  ec_root (ComBat subjects): {ec_root}\n")
    f.write(f"  fc_root (ComBat subjects): {fc_root}\n")
    f.write(f"  ec_template_dir: {ec_template_dir}\n")
    f.write(f"  fc_template_dir: {fc_template_dir}\n")
    f.write(f"  out_root: {out_root}\n\n")

    f.write("Cross-axis comparisons:\n")
    f.write("  EC_G1_vs_FC_G1\n")
    f.write("  EC_G1_vs_FC_G2\n")
    f.write("  EC_G2_vs_FC_G1\n")
    f.write("  EC_G2_vs_FC_G2\n\n")

    f.write("Input gradient files:\n")
    f.write("  EC subject gradients:\n")
    f.write("    Preferred: out_G1_procrustes_combat.npy, out_G2_procrustes_combat.npy\n")
    f.write("    Also supported: out_G1_procrustes.npy, out_G2_procrustes.npy, G_procrustes_z.npy, G_procrustes.npy, G_raw.npy\n")
    f.write("  FC subject gradients:\n")
    f.write("    Preferred: G_procrustes_z.npy\n")
    f.write("    Also supported: FC_G1_procrustes_z.npy, FC_G2_procrustes_z.npy\n")
    f.write("  Template gradients:\n")
    f.write(f"    EC template group directory: {ec_template_dir}\n")
    f.write(f"    FC template group directory: {fc_template_dir}\n")
    f.write("    Supported: G_template.npy or separate *_Gx_template.npy files\n\n")

    f.write("ROI scope:\n")
    f.write("  Whole-brain ROI. No Yeo7 network split is used.\n")
    f.write(f"  Expected ROI number: {EXPECTED_N_ROI}\n\n")

    f.write("BrainSpace spin model:\n")
    f.write("  Surface: Conte69 sphere from brainspace.datasets.load_conte69(as_sphere=True, join=False)\n")
    f.write("  Parcellation: brainspace.datasets.load_parcellation(name='schaefer', scale=400, join=False)\n")
    f.write("  Parcel centroid: average sphere coordinates within each Schaefer400 parcel, projected to unit sphere\n")
    f.write(f"  N_PERM: {N_PERM}\n")
    f.write(f"  SEED: {SEED}\n")
    f.write(f"  SPIN_UNIQUE: {SPIN_UNIQUE}\n")
    f.write(f"  SPIN_TARGET: {SPIN_TARGET}\n")
    f.write(f"  TAIL: {TAIL}\n\n")

    f.write("Definition:\n")
    f.write("  Subject-level test:\n")
    f.write("    For each subject and each EC-FC gradient pair, rho_obs = Spearman(EC gradient, FC gradient) across 400 ROIs.\n")
    f.write("    Subject-level p_spin is computed by rotating one gradient map with BrainSpace spin permutation and comparing rho_obs with the subject-specific spin-null distribution.\n")
    f.write("  Group-level subject-mean test:\n")
    f.write("    group_spearman_mean = mean(subject-level rho_obs) across matched subjects.\n")
    f.write("    group_p_spin is computed by comparing group_spearman_mean with the spin-null distribution of mean(subject-level null rho).\n")
    f.write("    This is the recommended group-level p value for the reported spearman_mean.\n")
    f.write("  Group-mean map test:\n")
    f.write("    group_mean_map_spearman_r = Spearman(mean EC gradient map, mean FC gradient map) across 400 ROIs.\n")
    f.write("    This is a group-map spatial test and is not identical to mean(subject-level rho_obs).\n\n")

    f.write("Covariate note:\n")
    f.write("  Spin permutation controls cortical spatial autocorrelation only.\n")
    f.write("  It does not control age, sex, FD, site, or other subject-level covariates.\n")
    f.write("  Covariates should be controlled before this step if needed.\n\n")

    f.write("Sample size:\n")
    f.write(f"  EC available subjects: {len(ec_map)}\n")
    f.write(f"  FC available subjects: {len(fc_map)}\n")
    f.write(f"  matched subjects: {len(common_ids)}\n")
    f.write(f"  successful subject-comparison rows: {len(df_long)}\n")
    f.write(f"  failed rows: {len(df_failed)}\n")
    f.write(f"  EC/FC index failed csv: {index_failed_csv}\n\n")

    f.write("Main outputs:\n")
    f.write(f"  long csv: {all_long_csv}\n")
    f.write(f"  wide csv: {wide_csv}\n")
    f.write(f"  summary statistics csv: {summary_csv}\n")
    f.write(f"  spin test csv: {spin_test_csv}\n")
    f.write(f"  group-level subject-mean spin csv: {group_subject_mean_csv}\n")
    f.write(f"  group-mean map spin csv: {group_mean_map_csv}\n")
    f.write(f"  group-level result dir: {group_level_dir}\n")
    f.write(f"  comparison separate dir: {comparison_dir}\n")
    f.write(f"  template csv: {template_all_csv}\n")
    f.write(f"  failed csv: {failed_csv}\n")
    f.write(f"  spin model dir: {spin_dir}\n")

print("\n=== EC 与 FC 2 × 2 cross-axis BrainSpace 空间置换检验完成 ===")
print("输出目录：", out_root)
print("共同被试数：", len(common_ids))
print("长表行数：", len(df_long))
print("失败行数：", len(df_failed))
print("空间置换长表：", all_long_csv)
print("空间置换宽表：", wide_csv)
print("空间置换统计汇总：", summary_csv)
print("空间置换单独结果：", spin_test_csv)
print("组水平 subject-mean 空间置换结果：", group_subject_mean_csv)
print("组平均图空间置换结果：", group_mean_map_csv)
print("组水平结果目录：", group_level_dir)
print("comparison 单独结果目录：", comparison_dir)
print("EC 群体模板目录：", ec_template_dir)
print("FC 群体模板目录：", fc_template_dir)
print("模板结果：", template_all_csv)
print("失败记录：", failed_csv)
print("索引失败记录：", index_failed_csv)
print("spin model 目录：", spin_dir)
print("总结报告：", summary_report)
