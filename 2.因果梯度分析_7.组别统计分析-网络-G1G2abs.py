import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

# =========================================================
# 路径设置
# =========================================================
BASE_DIR = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\补充敏感性分析\ABIDE1_结果2_combat"
)
LABEL_FILE = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\Schaefer400_7Yeo_network_labels_with_ID.csv"
)
INFO_FILE = Path(
    r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"
)

# 使用新目录，避免覆盖 signed difference 版本结果
OUT_DIR = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\补充敏感性分析"
    r"\ABIDE1_结果6_integrated_G1G2_absDistance_DMN_VisSMN_noDistanceFDR"
)
OUT_DIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# 配置
# =========================================================
NETWORKS = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]

# 网络间层级距离定义：
# 分别在 G1 和 G2 上计算 Default 与 Vis+SomMot 的 unsigned distance。
#
# Default 对应 DMN。
# ("Vis", "SomMot") 表示 VIS 与 SomMot 联合后的全部 ROI。
#
# 距离定义：
# |mean(Default) - mean(Vis + SomMot)|
#
# 该指标不保留方向信息，只表示两个网络集合在梯度轴上的分离幅度。
DIST_DEFS = [
    (
        "Default_dist_VisSomMot",
        "Default",
        ("Vis", "SomMot"),
    ),
]

TERM_GROUP = "C(Group, Treatment(reference='HC'))[T.ASD]"
TERM_INT = "C(Group, Treatment(reference='HC'))[T.ASD]:Age"

ALPHA_FDR = 0.05

# 是否执行 ROI-wise Group × Age 交互分析
RUN_ROIWISE_INTERACTION = True

# 可设置为 ["G1"]、["G2"] 或 ["G1", "G2"]
ROIWISE_GRADIENTS = ["G1", "G2"]


# =========================================================
# 工具函数
# =========================================================
def find_g_file(sub_id: str, g: int) -> Path | None:
    """
    根据被试编号查找 G1 或 G2 的 Procrustes 对齐结果。
    """
    fname = f"out_G{g}_procrustes.npy"

    candidates = [
        BASE_DIR / sub_id / fname,
    ]

    if not sub_id.startswith("sub-"):
        candidates.append(
            BASE_DIR / f"sub-{sub_id}" / fname
        )

    if sub_id.startswith("sub-"):
        stripped = sub_id.replace("sub-", "", 1)
        candidates.append(
            BASE_DIR / f"sub-{stripped}" / fname
        )

    for path in candidates:
        if path.exists():
            return path

    return None


def safe_zscore(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    对单个被试的全脑梯度进行 z-score。

    若标准差异常或接近 0，则返回全 0 数组。
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    mu = np.mean(x)
    sd = np.std(x)

    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)

    return (x - mu) / sd


def as_network_tuple(
    network_group,
) -> tuple[str, ...]:
    """
    将单个网络名称或多个网络名称统一转换为 tuple。
    """
    if isinstance(network_group, str):
        return (network_group,)

    return tuple(network_group)


def get_group_indices(
    net2idx: dict,
    network_group,
) -> np.ndarray:
    """
    返回单个网络或多个网络联合后的 ROI 索引。

    例如：
    ("Vis", "SomMot")
    返回 Vis 和 SomMot 网络中全部 ROI 的索引。
    """
    networks = as_network_tuple(network_group)

    idx_list = []

    for network in networks:
        if network not in net2idx:
            raise KeyError(
                f"未知网络名称：{network}"
            )

        idx_list.append(
            net2idx[network]
        )

    if len(idx_list) == 0:
        return np.array([], dtype=int)

    return np.concatenate(idx_list)


def network_group_mean(
    grad_z: np.ndarray,
    network_group,
    net2idx: dict,
) -> float:
    """
    计算单个网络或联合网络 ROI 集合的平均梯度值。
    """
    idx = get_group_indices(
        net2idx=net2idx,
        network_group=network_group,
    )

    if len(idx) == 0:
        return np.nan

    return float(
        np.mean(grad_z[idx])
    )


def unsigned_network_distance(
    grad_z: np.ndarray,
    group_a,
    group_b,
    net2idx: dict,
) -> float:
    """
    计算两个网络或联合网络在单一梯度轴上的 unsigned distance。

    定义：
    D(A, B) = |mean_zG(A) - mean_zG(B)|

    该指标不保留 A 相对于 B 位于梯度正端还是负端的信息，
    只表示两个网络集合在该梯度轴上的分离幅度。
    """
    a_mean = network_group_mean(
        grad_z=grad_z,
        network_group=group_a,
        net2idx=net2idx,
    )

    b_mean = network_group_mean(
        grad_z=grad_z,
        network_group=group_b,
        net2idx=net2idx,
    )

    if (
        not np.isfinite(a_mean)
        or not np.isfinite(b_mean)
    ):
        return np.nan

    return float(
        np.abs(a_mean - b_mean)
    )


def fdr_table(
    df: pd.DataFrame,
    p_col: str = "p",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    对一个完整的检验家族执行 Benjamini-Hochberg FDR 校正。
    """
    if df.empty:
        out = df.copy()
        out["p_FDR"] = np.nan
        out["sig"] = ""
        return out

    p_values = pd.to_numeric(
        df[p_col],
        errors="coerce",
    ).to_numpy()

    if np.any(~np.isfinite(p_values)):
        raise ValueError(
            f"用于 FDR 校正的列 {p_col} 中存在无效 P 值。"
        )

    reject, p_fdr = fdrcorrection(
        p_values,
        alpha=alpha,
    )

    out = df.copy()
    out["p_FDR"] = p_fdr
    out["sig"] = [
        "FDR<0.05" if value else ""
        for value in reject
    ]

    return out


def fdr_by_dv(
    df: pd.DataFrame,
    dv_col: str = "dv",
    p_col: str = "p",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    按 dv 分组分别执行 FDR 校正。

    本函数用于网络层分析，使：
    G1_network 的 7 个网络在内部校正；
    G2_network 的 7 个网络在内部校正。
    """
    if df.empty:
        out = df.copy()
        out["p_FDR"] = np.nan
        out["sig"] = ""
        return out

    out_list = []

    for _, sub_df in df.groupby(
        dv_col,
        sort=False,
    ):
        corrected = fdr_table(
            df=sub_df.copy(),
            p_col=p_col,
            alpha=alpha,
        )

        out_list.append(corrected)

    return pd.concat(
        out_list,
        axis=0,
        ignore_index=True,
    )


def fit_and_extract_term(
    formula: str,
    data: pd.DataFrame,
    dv_name: str,
    item_name: str,
    term: str,
) -> dict:
    """
    拟合 OLS 模型并提取指定模型项的：
    beta、t、P、95% CI 和样本量。
    """
    model = smf.ols(
        formula=formula,
        data=data,
    ).fit()

    if term not in model.params.index:
        raise RuntimeError(
            f"模型缺少指定项：{term}\n"
            f"formula={formula}"
        )

    ci = model.conf_int().loc[term].tolist()

    return {
        "dv": dv_name,
        "item": item_name,
        "term": term,
        "formula": formula,
        "beta": float(model.params[term]),
        "t": float(model.tvalues[term]),
        "p": float(model.pvalues[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n": int(model.nobs),
    }


def save_split_tables(
    df: pd.DataFrame,
    prefix: str,
    out_dir: Path,
) -> None:
    """
    保存完整结果表，并按 dv 分别保存单独结果表。
    """
    combined_file = out_dir / f"{prefix}.csv"

    df.to_csv(
        combined_file,
        index=False,
    )

    if "dv" not in df.columns or df.empty:
        return

    for dv_value, sub_df in df.groupby(
        "dv",
        sort=False,
    ):
        safe_dv = (
            str(dv_value)
            .replace("/", "_")
            .replace(" ", "_")
        )

        split_file = (
            out_dir
            / f"{prefix}_{safe_dv}.csv"
        )

        sub_df.to_csv(
            split_file,
            index=False,
        )


def run_roiwise_groupxage(
    grad_matrix: np.ndarray,
    grad_name: str,
    df_roi_base: pd.DataFrame,
    label_df: pd.DataFrame,
    atlas_label_col: str | None,
    out_dir: Path,
    alpha: float = 0.05,
) -> None:
    """
    执行 ROI-wise Group × Age 交互分析。

    参数
    ----------
    grad_matrix
        N × P 梯度矩阵。
        N 为被试数，P 为 ROI 数。

    grad_name
        "G1" 或 "G2"。

    df_roi_base
        按 grad_matrix 被试顺序排列的协变量表。
    """
    n_subjects, n_rois = grad_matrix.shape

    np.save(
        out_dir / f"{grad_name}_matrix.npy",
        grad_matrix,
    )

    df_roi = df_roi_base.copy()

    df_roi["Group"] = (
        df_roi["Group"].astype("category")
    )
    df_roi["Sex"] = (
        df_roi["Sex"].astype("category")
    )
    df_roi["Site"] = (
        df_roi["Site"].astype("category")
    )

    beta_vals = np.zeros(
        n_rois,
        dtype=float,
    )
    t_vals = np.zeros(
        n_rois,
        dtype=float,
    )
    p_vals = np.zeros(
        n_rois,
        dtype=float,
    )
    ci_low = np.zeros(
        n_rois,
        dtype=float,
    )
    ci_high = np.zeros(
        n_rois,
        dtype=float,
    )

    roi_var = f"{grad_name}_roi"

    formula_roi_int = (
        f"{roi_var} ~ "
        "C(Group, Treatment(reference='HC')) * Age "
        "+ FIQ + C(Sex)"
    )

    for roi_idx in range(n_rois):
        dfr = df_roi.copy()

        dfr[roi_var] = (
            grad_matrix[:, roi_idx]
        )

        model = smf.ols(
            formula=formula_roi_int,
            data=dfr,
        ).fit()

        if TERM_INT not in model.params.index:
            raise RuntimeError(
                f"{grad_name} ROI {roi_idx + 1} "
                f"缺少交互项：{TERM_INT}"
            )

        beta_vals[roi_idx] = float(
            model.params[TERM_INT]
        )
        t_vals[roi_idx] = float(
            model.tvalues[TERM_INT]
        )
        p_vals[roi_idx] = float(
            model.pvalues[TERM_INT]
        )

        roi_ci = (
            model.conf_int()
            .loc[TERM_INT]
            .tolist()
        )

        ci_low[roi_idx] = float(
            roi_ci[0]
        )
        ci_high[roi_idx] = float(
            roi_ci[1]
        )

    reject_fdr, p_fdr = fdrcorrection(
        p_vals,
        alpha=alpha,
    )

    np.save(
        out_dir
        / f"{grad_name}_roi_GroupxAge_beta.npy",
        beta_vals,
    )
    np.save(
        out_dir
        / f"{grad_name}_roi_GroupxAge_t.npy",
        t_vals,
    )
    np.save(
        out_dir
        / f"{grad_name}_roi_GroupxAge_p.npy",
        p_vals,
    )
    np.save(
        out_dir
        / f"{grad_name}_roi_GroupxAge_p_fdr.npy",
        p_fdr,
    )
    np.save(
        out_dir
        / f"{grad_name}_roi_GroupxAge_sig_mask.npy",
        reject_fdr.astype(np.int8),
    )

    roi_table = pd.DataFrame({
        "ROI_ID": np.arange(
            1,
            n_rois + 1,
            dtype=int,
        ),
        "ROI_idx": np.arange(
            n_rois,
            dtype=int,
        ),
        "gradient": grad_name,
        "term": TERM_INT,
        "beta": beta_vals,
        "t": t_vals,
        "p": p_vals,
        "p_FDR": p_fdr,
        "sig_FDR": reject_fdr.astype(int),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "Yeo7_network": (
            label_df["Yeo7_network"].values
        ),
    })

    if atlas_label_col is not None:
        roi_table["Atlas_Label"] = (
            label_df[atlas_label_col]
            .astype(str)
            .values
        )

    front_cols = ["ROI_ID"]

    if atlas_label_col is not None:
        front_cols.append(
            "Atlas_Label"
        )

    front_cols += [
        "Yeo7_network",
        "gradient",
        "term",
        "beta",
        "t",
        "p",
        "p_FDR",
        "sig_FDR",
        "ci_low",
        "ci_high",
        "ROI_idx",
    ]

    roi_table = roi_table[
        front_cols
    ]

    roi_table.to_csv(
        out_dir
        / (
            f"{grad_name}_GroupxAge_"
            "interaction_roi_table.csv"
        ),
        index=False,
    )

    roi_summary = pd.DataFrame([
        {
            "gradient": grad_name,
            "N_subjects": int(n_subjects),
            "P_rois": int(n_rois),
            "n_sig_fdr": int(
                reject_fdr.sum()
            ),
            "mean_beta": float(
                np.mean(beta_vals)
            ),
            "mean_abs_t": float(
                np.mean(np.abs(t_vals))
            ),
            "n_beta_positive": int(
                (beta_vals > 0).sum()
            ),
            "n_beta_negative": int(
                (beta_vals < 0).sum()
            ),
        }
    ])

    roi_summary.to_csv(
        out_dir
        / (
            f"{grad_name}_GroupxAge_"
            "interaction_roi_summary.csv"
        ),
        index=False,
    )


# =========================================================
# 1) 读取 Yeo7 标签
# =========================================================
label_df = pd.read_csv(
    LABEL_FILE
)

required_cols = {
    "ROI_ID",
    "Yeo7_network",
}

missing_cols = (
    required_cols
    - set(label_df.columns)
)

if missing_cols:
    raise ValueError(
        f"Label file 缺少列：{missing_cols}"
    )

label_df["ROI_ID"] = pd.to_numeric(
    label_df["ROI_ID"],
    errors="raise",
).astype(int)

label_df = (
    label_df
    .sort_values("ROI_ID")
    .reset_index(drop=True)
)

expected_roi_ids = np.arange(
    1,
    len(label_df) + 1,
)

if not np.array_equal(
    label_df["ROI_ID"].to_numpy(),
    expected_roi_ids,
):
    raise ValueError(
        "ROI_ID 必须从 1 开始连续排列，"
        "以保证梯度数组与标签顺序一致。"
    )

label_df["ROI_idx"] = (
    label_df["ROI_ID"] - 1
)

atlas_label_col = None

for candidate in [
    "Atlas_Label",
    "Label",
    "ROI_Label",
    "Parcel",
    "parcel_name",
]:
    if candidate in label_df.columns:
        atlas_label_col = candidate
        break

net2idx = {
    network: label_df.loc[
        label_df["Yeo7_network"] == network,
        "ROI_idx",
    ].to_numpy(dtype=int)
    for network in NETWORKS
}

for network, idx in net2idx.items():
    if len(idx) == 0:
        raise ValueError(
            f"网络 {network} 没有对应的 ROI，"
            "请检查标签文件。"
        )


# =========================================================
# 2) 读取被试信息
# =========================================================
info = pd.read_csv(
    INFO_FILE
)

required_info_cols = {
    "sub_id",
    "Group",
    "Age",
    "Sex",
    "FIQ",
}

missing_info_cols = (
    required_info_cols
    - set(info.columns)
)

if missing_info_cols:
    raise ValueError(
        f"Subject info 缺少列："
        f"{missing_info_cols}"
    )

if "Site" not in info.columns:
    info["Site"] = np.nan

if "MeanFD" not in info.columns:
    info["MeanFD"] = np.nan

info["FIQ"] = pd.to_numeric(
    info["FIQ"],
    errors="coerce",
)
info["Age"] = pd.to_numeric(
    info["Age"],
    errors="coerce",
)
info["MeanFD"] = pd.to_numeric(
    info["MeanFD"],
    errors="coerce",
)

# 只根据当前模型实际使用的变量剔除缺失值。
# Site 和 MeanFD 当前不进入模型。
info = info.dropna(
    subset=[
        "sub_id",
        "Group",
        "Age",
        "Sex",
        "FIQ",
    ]
).copy()

info["sub_id"] = (
    info["sub_id"]
    .astype(str)
    .str.strip()
)

if info["sub_id"].duplicated().any():
    duplicated_ids = (
        info.loc[
            info["sub_id"].duplicated(
                keep=False
            ),
            "sub_id",
        ]
        .tolist()
    )

    raise ValueError(
        "Subject info 中存在重复 sub_id："
        f"{duplicated_ids[:20]}"
    )

valid_groups = set(
    info["Group"]
    .astype(str)
    .unique()
)

if "HC" not in valid_groups:
    raise ValueError(
        "Group 中未找到 HC，"
        "无法将 HC 设置为参考组。"
    )

if "ASD" not in valid_groups:
    raise ValueError(
        "Group 中未找到 ASD。"
    )


# =========================================================
# 3) 逐被试计算网络均值和网络间距离
#    同时缓存 ROI-wise G1/G2 原始梯度矩阵
# =========================================================
rows_net_long = []
rows_dist_wide = []
missing_subs = []

roi_subjects = []
g1_matrix_list = []
g2_matrix_list = []

for _, sub in info.iterrows():
    sub_id = str(
        sub["sub_id"]
    )

    g1_file = find_g_file(
        sub_id=sub_id,
        g=1,
    )
    g2_file = find_g_file(
        sub_id=sub_id,
        g=2,
    )

    if (
        g1_file is None
        or g2_file is None
    ):
        missing_subs.append(
            sub_id
        )
        continue

    g1 = np.load(
        g1_file
    ).reshape(-1)

    g2 = np.load(
        g2_file
    ).reshape(-1)

    if (
        len(g1) != len(label_df)
        or len(g2) != len(label_df)
    ):
        raise ValueError(
            f"{sub_id} 梯度长度与标签数量不匹配："
            f"G1={len(g1)}, "
            f"G2={len(g2)}, "
            f"label={len(label_df)}"
        )

    if (
        np.any(~np.isfinite(g1))
        or np.any(~np.isfinite(g2))
    ):
        raise ValueError(
            f"{sub_id} 的 G1 或 G2 中存在 NaN/Inf。"
        )

    # ROI-wise 分析保留 Procrustes 对齐后的原始梯度值。
    roi_subjects.append(
        sub_id
    )
    g1_matrix_list.append(
        g1.copy()
    )
    g2_matrix_list.append(
        g2.copy()
    )

    # 网络层均值和网络距离使用被试内全脑 z-score。
    g1z = safe_zscore(
        g1
    )
    g2z = safe_zscore(
        g2
    )

    g1_net = {
        network: network_group_mean(
            grad_z=g1z,
            network_group=network,
            net2idx=net2idx,
        )
        for network in NETWORKS
    }

    g2_net = {
        network: network_group_mean(
            grad_z=g2z,
            network_group=network,
            net2idx=net2idx,
        )
        for network in NETWORKS
    }

    for network in NETWORKS:
        rows_net_long.append({
            "sub_id": sub_id,
            "network": network,
            "g1_net": g1_net[network],
            "g2_net": g2_net[network],
            "Group": sub["Group"],
            "Age": float(sub["Age"]),
            "FIQ": float(sub["FIQ"]),
            "Sex": sub["Sex"],
            "Site": sub.get(
                "Site",
                np.nan,
            ),
            "MeanFD": (
                float(sub["MeanFD"])
                if pd.notna(
                    sub.get(
                        "MeanFD",
                        np.nan,
                    )
                )
                else np.nan
            ),
        })

    dist_row = {
        "sub_id": sub_id,
        "Group": sub["Group"],
        "Age": float(sub["Age"]),
        "FIQ": float(sub["FIQ"]),
        "Sex": sub["Sex"],
        "Site": sub.get(
            "Site",
            np.nan,
        ),
        "MeanFD": (
            float(sub["MeanFD"])
            if pd.notna(
                sub.get(
                    "MeanFD",
                    np.nan,
                )
            )
            else np.nan
        ),
    }

    for name, group_a, group_b in DIST_DEFS:
        # G1 距离
        dist_row[f"G1_{name}"] = (
            unsigned_network_distance(
                grad_z=g1z,
                group_a=group_a,
                group_b=group_b,
                net2idx=net2idx,
            )
        )

        # G2 距离
        dist_row[f"G2_{name}"] = (
            unsigned_network_distance(
                grad_z=g2z,
                group_a=group_a,
                group_b=group_b,
                net2idx=net2idx,
            )
        )

    rows_dist_wide.append(
        dist_row
    )


if missing_subs:
    pd.Series(
        missing_subs,
        name="sub_id",
    ).to_csv(
        OUT_DIR
        / "missing_subjects_no_G1_or_G2.csv",
        index=False,
    )

    print(
        "找不到 G1 或 G2 的被试数：",
        len(missing_subs),
    )

if (
    len(g1_matrix_list) == 0
    or len(g2_matrix_list) == 0
):
    raise RuntimeError(
        "没有可用于分析的被试，"
        "请检查梯度文件路径和 sub_id 匹配。"
    )

df_net = pd.DataFrame(
    rows_net_long
)
df_dist = pd.DataFrame(
    rows_dist_wide
)

if df_net.empty or df_dist.empty:
    raise RuntimeError(
        "生成的数据表为空，"
        "请检查输入文件和被试匹配情况。"
    )

df_net.to_csv(
    OUT_DIR
    / "Yeo7_network_level_G1G2_long.csv",
    index=False,
)

df_dist.to_csv(
    OUT_DIR
    / "Yeo7_hierarchy_distances_G1G2.csv",
    index=False,
)

for dataframe in [
    df_net,
    df_dist,
]:
    dataframe["Group"] = (
        dataframe["Group"]
        .astype("category")
    )
    dataframe["Sex"] = (
        dataframe["Sex"]
        .astype("category")
    )
    dataframe["Site"] = (
        dataframe["Site"]
        .astype("category")
    )

if (
    "HC"
    not in df_net["Group"].cat.categories
):
    raise ValueError(
        "Group 中未找到 HC，"
        "无法作为参考组。"
    )


# =========================================================
# 4) 网络层 GLM：G1/G2 组间主效应
#
# G1 的 7 个网络内部执行 FDR。
# G2 的 7 个网络内部执行 FDR。
# G1 和 G2 不联合校正。
# =========================================================
net_results = []

for dv, dv_name in [
    ("g1_net", "G1_network"),
    ("g2_net", "G2_network"),
]:
    for network in NETWORKS:
        dfn = df_net.loc[
            df_net["network"] == network
        ].copy()

        formula = (
            f"{dv} ~ "
            "C(Group, Treatment(reference='HC')) "
            "+ Age + FIQ + C(Sex)"
        )

        result = fit_and_extract_term(
            formula=formula,
            data=dfn,
            dv_name=dv_name,
            item_name=network,
            term=TERM_GROUP,
        )

        net_results.append(
            result
        )

glm_net = fdr_by_dv(
    df=pd.DataFrame(net_results),
    dv_col="dv",
    p_col="p",
    alpha=ALPHA_FDR,
)

save_split_tables(
    df=glm_net,
    prefix=(
        "GLM_Yeo7_network_G1G2_"
        "main_effect_splitFDR"
    ),
    out_dir=OUT_DIR,
)


# =========================================================
# 5) 网络层 GLM：G1/G2 Group × Age 交互
#
# G1 的 7 个网络内部执行 FDR。
# G2 的 7 个网络内部执行 FDR。
# G1 和 G2 不联合校正。
# =========================================================
net_int_results = []

for dv, dv_name in [
    ("g1_net", "G1_network"),
    ("g2_net", "G2_network"),
]:
    for network in NETWORKS:
        dfn = df_net.loc[
            df_net["network"] == network
        ].copy()

        formula = (
            f"{dv} ~ "
            "C(Group, Treatment(reference='HC')) "
            "* Age "
            "+ FIQ + C(Sex)"
        )

        result = fit_and_extract_term(
            formula=formula,
            data=dfn,
            dv_name=dv_name,
            item_name=network,
            term=TERM_INT,
        )

        net_int_results.append(
            result
        )

glm_net_int = fdr_by_dv(
    df=pd.DataFrame(
        net_int_results
    ),
    dv_col="dv",
    p_col="p",
    alpha=ALPHA_FDR,
)

save_split_tables(
    df=glm_net_int,
    prefix=(
        "GLM_Yeo7_network_G1G2_"
        "GroupxAge_splitFDR"
    ),
    out_dir=OUT_DIR,
)


# =========================================================
# 6) 网络间层级距离 GLM：组间主效应
#
# 分析指标：
# 1. G1_Default_dist_VisSomMot
# 2. G2_Default_dist_VisSomMot
#
# G1 和 G2 作为两个独立的梯度结果分别建模、分别保存。
# 距离分析不执行 FDR 校正，只报告原始 P 值。
# =========================================================
dist_main_tables = []

for gradient_name in ["G1", "G2"]:
    gradient_results = []

    for distance_name, _, _ in DIST_DEFS:
        col = f"{gradient_name}_{distance_name}"

        formula = (
            f"{col} ~ "
            "C(Group, Treatment(reference='HC')) "
            "+ Age + FIQ + C(Sex)"
        )

        result = fit_and_extract_term(
            formula=formula,
            data=df_dist,
            dv_name=f"{gradient_name}_distance",
            item_name=col,
            term=TERM_GROUP,
        )

        gradient_results.append(
            result
        )

    # 每个梯度单独保存，不进行任何多重比较校正。
    gradient_table = pd.DataFrame(
        gradient_results
    )

    gradient_table.to_csv(
        OUT_DIR
        / (
            f"GLM_hierarchy_distance_{gradient_name}_"
            "Default_dist_VisSomMot_main_effect_rawP.csv"
        ),
        index=False,
    )

    dist_main_tables.append(
        gradient_table
    )

# 额外保存一个仅用于汇总查看的合并表。
# 该合并表同样只包含原始 P 值，不执行 FDR。
glm_dist = pd.concat(
    dist_main_tables,
    axis=0,
    ignore_index=True,
)

glm_dist.to_csv(
    OUT_DIR
    / (
        "GLM_hierarchy_distances_G1G2_"
        "Default_dist_VisSomMot_main_effect_rawP.csv"
    ),
    index=False,
)


# =========================================================
# 7) 网络间层级距离 GLM：Group × Age 交互
#
# 分析指标：
# 1. G1_Default_dist_VisSomMot
# 2. G2_Default_dist_VisSomMot
#
# G1 和 G2 作为两个独立的梯度结果分别建模、分别保存。
# 距离交互效应不执行 FDR 校正，只报告原始 P 值。
# =========================================================
dist_interaction_tables = []

for gradient_name in ["G1", "G2"]:
    gradient_results = []

    for distance_name, _, _ in DIST_DEFS:
        col = f"{gradient_name}_{distance_name}"

        formula = (
            f"{col} ~ "
            "C(Group, Treatment(reference='HC')) "
            "* Age "
            "+ FIQ + C(Sex)"
        )

        result = fit_and_extract_term(
            formula=formula,
            data=df_dist,
            dv_name=f"{gradient_name}_distance",
            item_name=col,
            term=TERM_INT,
        )

        gradient_results.append(
            result
        )

    # 每个梯度单独保存，不进行任何多重比较校正。
    gradient_table = pd.DataFrame(
        gradient_results
    )

    gradient_table.to_csv(
        OUT_DIR
        / (
            f"GLM_hierarchy_distance_{gradient_name}_"
            "Default_dist_VisSomMot_GroupxAge_rawP.csv"
        ),
        index=False,
    )

    dist_interaction_tables.append(
        gradient_table
    )

# 额外保存一个仅用于汇总查看的合并表。
# 该合并表同样只包含原始 P 值，不执行 FDR。
glm_dist_int = pd.concat(
    dist_interaction_tables,
    axis=0,
    ignore_index=True,
)

glm_dist_int.to_csv(
    OUT_DIR
    / (
        "GLM_hierarchy_distances_G1G2_"
        "Default_dist_VisSomMot_GroupxAge_rawP.csv"
    ),
    index=False,
)


# =========================================================
# 8) ROI-wise G1/G2 Group × Age 交互分析
#
# 每个梯度内部在 400 个 ROI 之间执行 FDR。
# G1 和 G2 不联合校正。
# =========================================================
if RUN_ROIWISE_INTERACTION:
    g1_matrix = np.vstack(
        g1_matrix_list
    )
    g2_matrix = np.vstack(
        g2_matrix_list
    )

    n_g1, p_g1 = g1_matrix.shape
    n_g2, p_g2 = g2_matrix.shape

    if (
        n_g1 != n_g2
        or p_g1 != p_g2
    ):
        raise RuntimeError(
            "G1_matrix 与 G2_matrix "
            "维度不一致。"
        )

    df_roi_base = (
        info.loc[
            info["sub_id"].isin(
                roi_subjects
            )
        ]
        .copy()
        .set_index("sub_id")
        .loc[roi_subjects]
        .reset_index()
    )

    if len(df_roi_base) != n_g1:
        raise RuntimeError(
            "ROI-wise 协变量表与梯度矩阵"
            "被试数不一致。"
        )

    if "G1" in ROIWISE_GRADIENTS:
        run_roiwise_groupxage(
            grad_matrix=g1_matrix,
            grad_name="G1",
            df_roi_base=df_roi_base,
            label_df=label_df,
            atlas_label_col=atlas_label_col,
            out_dir=OUT_DIR,
            alpha=ALPHA_FDR,
        )

    if "G2" in ROIWISE_GRADIENTS:
        run_roiwise_groupxage(
            grad_matrix=g2_matrix,
            grad_name="G2",
            df_roi_base=df_roi_base,
            label_df=label_df,
            atlas_label_col=atlas_label_col,
            out_dir=OUT_DIR,
            alpha=ALPHA_FDR,
        )


# =========================================================
# 完成提示
# =========================================================
print(
    "完成：G1/G2 网络层分析。"
)

print(
    "网络层模型协变量："
    "Group + Age + FIQ + Sex。"
)

print(
    "网络层 FDR："
    "G1 的 7 个网络内部校正；"
    "G2 的 7 个网络内部校正；"
    "主效应和 Group × Age 交互效应分别校正。"
)

print(
    "完成：G1/G2 网络间层级距离分析。"
)

print(
    "距离指标："
    "G1_Default_dist_VisSomMot 和 "
    "G2_Default_dist_VisSomMot。"
)

print(
    "距离定义："
    "|mean(Default) - mean(Vis + SomMot)|。"
)

print(
    "距离分析多重比较："
    "不执行 FDR 校正；"
    "G1 和 G2 分别建模、分别保存；"
    "主效应和 Group × Age 交互效应均报告原始 P 值。"
)

if RUN_ROIWISE_INTERACTION:
    print(
        "完成：ROI-wise Group × Age 交互分析。"
    )

    print(
        "ROI-wise FDR："
        "G1 和 G2 各自在全部 ROI 内独立校正。"
    )

print(
    "输出目录：",
    OUT_DIR,
)
