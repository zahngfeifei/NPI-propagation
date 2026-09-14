import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

# =========================================================
# 路径设置
# =========================================================
BASE_DIR   = Path(r"I:\DYF\NPI-4-code\2.梯度分析\补充敏感性分析\ABIDE1_结果2_combat")
LABEL_FILE = Path(r"I:\DYF\NPI-4-code\2.梯度分析\Schaefer400_7Yeo_network_labels_with_ID.csv")
INFO_FILE  = Path(r"I:\DYF\NPI-4-code\subject_info_for_stats.csv")

# 使用新目录，避免覆盖 signed difference 版本结果
OUT_DIR    = Path(r"I:\DYF\NPI-4-code\2.梯度分析\补充敏感性分析\ABIDE1_结果6_integrated_G1G2_absDistance_DMN_VisSMN-fd")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# =========================================================
# 配置
# =========================================================
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# 网络间层级距离定义：只保留 G1_Default_dist_VisSomMot
# 说明：Default 对应 DMN；("Vis", "SomMot") 表示 VIS 与 SMN/SomMot 联合后的 ROI 集合。
# 注意：该版本为 unsigned distance，不保留方向信息。
DIST_DEFS = [
    ("Default_dist_VisSomMot", "Default", ("Vis", "SomMot")),
]

TERM_GROUP = "C(Group, Treatment(reference='HC'))[T.ASD]"
TERM_INT   = "C(Group, Treatment(reference='HC'))[T.ASD]:Age"
ALPHA_FDR  = 0.05

# 是否做 ROI-wise Group × Age 交互分析
# 仅运行原第 5 个脚本的 G2 ROI-wise 分析时，设为 ["G2"]
RUN_ROIWISE_INTERACTION = True
ROIWISE_GRADIENTS = ["G1", "G2"]

# =========================================================
# 工具函数
# =========================================================
def find_g_file(sub_id: str, g: int) -> Path | None:
    fname = f"out_G{g}_procrustes.npy"
    candidates = [BASE_DIR / sub_id / fname]

    if not sub_id.startswith("sub-"):
        candidates.append(BASE_DIR / f"sub-{sub_id}" / fname)

    if sub_id.startswith("sub-"):
        stripped = sub_id.replace("sub-", "", 1)
        candidates.append(BASE_DIR / f"sub-{stripped}" / fname)

    for p in candidates:
        if p.exists():
            return p
    return None


def safe_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    mu = np.mean(x)
    sd = np.std(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def as_network_tuple(network_group) -> tuple[str, ...]:
    """把单个网络名或多个网络名统一转为 tuple。"""
    if isinstance(network_group, str):
        return (network_group,)
    return tuple(network_group)


def get_group_indices(net2idx: dict, network_group) -> np.ndarray:
    """
    返回一个网络或多个网络联合后的 ROI 索引。
    例如 ("Vis", "SomMot") 会返回 VIS 与 SMN/SomMot 的所有 ROI。
    """
    nets = as_network_tuple(network_group)
    idx_list = []
    for net in nets:
        if net not in net2idx:
            raise KeyError(f"未知网络名称: {net}")
        idx_list.append(net2idx[net])
    return np.concatenate(idx_list)


def network_group_mean(grad_z: np.ndarray, network_group, net2idx: dict) -> float:
    """计算单个网络或联合网络 ROI 集合的梯度均值。"""
    idx = get_group_indices(net2idx, network_group)
    if len(idx) == 0:
        return np.nan
    return float(grad_z[idx].mean())


def unsigned_network_distance(grad_z: np.ndarray, group_a, group_b, net2idx: dict) -> float:
    """
    计算两个网络/联合网络在单一梯度轴上的 unsigned distance。

    D(A, B) = |mean_zG(A) - mean_zG(B)|

    该指标不保留 A 相对于 B 位于梯度正端还是负端的信息，只表示一维梯度轴上的分离幅度。
    """
    a_mean = network_group_mean(grad_z, group_a, net2idx)
    b_mean = network_group_mean(grad_z, group_b, net2idx)
    if not np.isfinite(a_mean) or not np.isfinite(b_mean):
        return np.nan
    return float(np.abs(a_mean - b_mean))


def fdr_table(df: pd.DataFrame, p_col: str = "p", alpha: float = 0.05) -> pd.DataFrame:
    """对一个检验家族做 FDR。"""
    if df.empty:
        out = df.copy()
        out["p_FDR"] = np.nan
        out["sig"] = ""
        return out

    rej, p_fdr = fdrcorrection(df[p_col].values, alpha=alpha)
    out = df.copy()
    out["p_FDR"] = p_fdr
    out["sig"] = ["FDR<0.05" if r else "" for r in rej]
    return out


def fdr_by_dv(df: pd.DataFrame, dv_col: str = "dv", p_col: str = "p", alpha: float = 0.05) -> pd.DataFrame:
    """
    按 dv 分组分别做 FDR。
    本脚本中用于保证 G1 和 G2 不联合校正。
    """
    if df.empty:
        out = df.copy()
        out["p_FDR"] = np.nan
        out["sig"] = ""
        return out

    out_list = []
    for _, sub_df in df.groupby(dv_col, sort=False):
        out_list.append(fdr_table(sub_df.copy(), p_col=p_col, alpha=alpha))
    return pd.concat(out_list, axis=0, ignore_index=True)


def fit_and_extract_term(
    formula: str,
    data: pd.DataFrame,
    dv_name: str,
    item_name: str,
    term: str,
) -> dict:
    """拟合 OLS 并提取指定 term 的 beta/t/p/CI。"""
    m = smf.ols(formula, data=data).fit()
    if term not in m.params.index:
        raise RuntimeError(f"模型缺少指定项：{term}\nformula={formula}")

    ci = m.conf_int().loc[term].tolist()
    return {
        "dv": dv_name,
        "item": item_name,
        "term": term,
        "formula": formula,
        "beta": float(m.params[term]),
        "t": float(m.tvalues[term]),
        "p": float(m.pvalues[term]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n": int(m.nobs),
    }


def save_split_tables(df: pd.DataFrame, prefix: str, out_dir: Path) -> None:
    """保存合并表，并按 dv 额外拆成单独 CSV。"""
    df.to_csv(out_dir / f"{prefix}.csv", index=False)
    if "dv" in df.columns and not df.empty:
        for dv_value, sub_df in df.groupby("dv", sort=False):
            safe_dv = str(dv_value).replace("/", "_").replace(" ", "_")
            sub_df.to_csv(out_dir / f"{prefix}_{safe_dv}.csv", index=False)


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
    ROI-wise Group × Age 交互分析。
    grad_matrix: N × P，按 df_roi_base 的被试顺序排列。
    grad_name: "G1" 或 "G2"。
    """
    N, P = grad_matrix.shape
    np.save(out_dir / f"{grad_name}_matrix.npy", grad_matrix)

    df_roi = df_roi_base.copy()
    df_roi["Group"] = df_roi["Group"].astype("category")
    df_roi["Sex"]   = df_roi["Sex"].astype("category")
    df_roi["Site"]  = df_roi["Site"].astype("category")

    beta_vals = np.zeros(P, dtype=float)
    t_vals    = np.zeros(P, dtype=float)
    p_vals    = np.zeros(P, dtype=float)
    ci_low    = np.zeros(P, dtype=float)
    ci_high   = np.zeros(P, dtype=float)

    roi_var = f"{grad_name}_roi"
    formula_roi_int = (
        f"{roi_var} ~ C(Group, Treatment(reference='HC')) * Age "
        "+ FIQ + C(Sex) + MeanFD"
    )

    for roi in range(P):
        dfr = df_roi.copy()
        dfr[roi_var] = grad_matrix[:, roi]

        model = smf.ols(formula_roi_int, data=dfr).fit()
        if TERM_INT not in model.params.index:
            raise RuntimeError(f"{grad_name} ROI {roi + 1} 缺少交互项：{TERM_INT}")

        beta_vals[roi] = float(model.params[TERM_INT])
        t_vals[roi]    = float(model.tvalues[TERM_INT])
        p_vals[roi]    = float(model.pvalues[TERM_INT])

        roi_ci = model.conf_int().loc[TERM_INT].tolist()
        ci_low[roi]  = float(roi_ci[0])
        ci_high[roi] = float(roi_ci[1])

    reject_fdr, p_fdr = fdrcorrection(p_vals, alpha=alpha)

    np.save(out_dir / f"{grad_name}_roi_GroupxAge_beta.npy", beta_vals)
    np.save(out_dir / f"{grad_name}_roi_GroupxAge_t.npy", t_vals)
    np.save(out_dir / f"{grad_name}_roi_GroupxAge_p.npy", p_vals)
    np.save(out_dir / f"{grad_name}_roi_GroupxAge_p_fdr.npy", p_fdr)
    np.save(out_dir / f"{grad_name}_roi_GroupxAge_sig_mask.npy", reject_fdr.astype(np.int8))

    roi_table = pd.DataFrame({
        "ROI_ID": np.arange(1, P + 1, dtype=int),
        "ROI_idx": np.arange(P, dtype=int),
        "gradient": grad_name,
        "term": TERM_INT,
        "beta": beta_vals,
        "t": t_vals,
        "p": p_vals,
        "p_FDR": p_fdr,
        "sig_FDR": reject_fdr.astype(int),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "Yeo7_network": label_df["Yeo7_network"].values,
    })

    if atlas_label_col is not None:
        roi_table["Atlas_Label"] = label_df[atlas_label_col].astype(str).values

    front_cols = ["ROI_ID"]
    if atlas_label_col is not None:
        front_cols.append("Atlas_Label")
    front_cols += [
        "Yeo7_network", "gradient", "term", "beta", "t", "p", "p_FDR",
        "sig_FDR", "ci_low", "ci_high", "ROI_idx"
    ]
    roi_table = roi_table[front_cols]
    roi_table.to_csv(out_dir / f"{grad_name}_GroupxAge_interaction_roi_table.csv", index=False)

    roi_summary = pd.DataFrame([{
        "gradient": grad_name,
        "N_subjects": int(N),
        "P_rois": int(P),
        "n_sig_fdr": int(reject_fdr.sum()),
        "mean_beta": float(np.mean(beta_vals)),
        "mean_abs_t": float(np.mean(np.abs(t_vals))),
        "n_beta_positive": int((beta_vals > 0).sum()),
        "n_beta_negative": int((beta_vals < 0).sum()),
    }])
    roi_summary.to_csv(out_dir / f"{grad_name}_GroupxAge_interaction_roi_summary.csv", index=False)


# =========================================================
# 1) 读取 Yeo7 标签
# =========================================================
label_df = pd.read_csv(LABEL_FILE)

required_cols = {"ROI_ID", "Yeo7_network"}
miss = required_cols - set(label_df.columns)
if miss:
    raise ValueError(f"Label file 缺少列: {miss}")

label_df["ROI_idx"] = label_df["ROI_ID"].astype(int) - 1

atlas_label_col = None
for candidate in ["Atlas_Label", "Label", "ROI_Label", "Parcel", "parcel_name"]:
    if candidate in label_df.columns:
        atlas_label_col = candidate
        break

net2idx = {
    net: label_df.loc[label_df["Yeo7_network"] == net, "ROI_idx"].values
    for net in NETWORKS
}

for net, idx in net2idx.items():
    if len(idx) == 0:
        raise ValueError(f"网络 {net} 没有对应的 ROI，请检查标签文件")

# =========================================================
# 2) 读取被试信息
# =========================================================
info = pd.read_csv(INFO_FILE)

need = {"sub_id", "Group", "Age", "Sex", "FIQ"}
miss = need - set(info.columns)
if miss:
    raise ValueError(f"Subject info 缺少列: {miss}")

if "Site" not in info.columns:
    info["Site"] = np.nan

if "MeanFD" not in info.columns:
    fd_candidates = [
        c for c in info.columns
        if str(c).strip().lower().replace("_", "") in ["fd", "meanfd", "fdmean"]
    ]
    if fd_candidates:
        info["MeanFD"] = info[fd_candidates[0]]
    else:
        raise ValueError("Subject info 缺少 FD 协变量列：需要 MeanFD、FD、fd、mean_fd 或 FD_mean。")

info["FIQ"]    = pd.to_numeric(info["FIQ"], errors="coerce")
info["Age"]    = pd.to_numeric(info["Age"], errors="coerce")
info["MeanFD"] = pd.to_numeric(info["MeanFD"], errors="coerce")

# FD 作为正式协变量，按实际建模变量剔除缺失
info = info.dropna(subset=["sub_id", "Group", "Age", "Sex", "FIQ", "MeanFD"]).copy()
info["sub_id"] = info["sub_id"].astype(str)

# =========================================================
# 3) 逐被试计算：网络均值、网络间 unsigned distance
#    同时缓存 ROI-wise G1/G2 原始梯度矩阵
# =========================================================
rows_net_long = []
rows_dist_wide = []
missing_subs = []

roi_subjects = []
g1_matrix_list = []
g2_matrix_list = []

for _, sub in info.iterrows():
    sub_id = str(sub["sub_id"])

    p1 = find_g_file(sub_id, 1)
    p2 = find_g_file(sub_id, 2)
    if p1 is None or p2 is None:
        missing_subs.append(sub_id)
        continue

    g1 = np.load(p1).reshape(-1)
    g2 = np.load(p2).reshape(-1)

    if len(g1) != len(label_df) or len(g2) != len(label_df):
        raise ValueError(
            f"{sub_id} 梯度长度与 label 不匹配：G1={len(g1)} G2={len(g2)} label={len(label_df)}"
        )

    # ROI-wise 分析沿用 procrustes 原始梯度值；网络层指标使用全脑 z-score
    roi_subjects.append(sub_id)
    g1_matrix_list.append(g1.copy())
    g2_matrix_list.append(g2.copy())

    g1z = safe_zscore(g1)
    g2z = safe_zscore(g2)

    g1_net = {net: network_group_mean(g1z, net, net2idx) for net in NETWORKS}
    g2_net = {net: network_group_mean(g2z, net, net2idx) for net in NETWORKS}

    for net in NETWORKS:
        rows_net_long.append({
            "sub_id": sub_id,
            "network": net,
            "g1_net": g1_net[net],
            "g2_net": g2_net[net],
            "Group": sub["Group"],
            "Age": float(sub["Age"]),
            "FIQ": float(sub["FIQ"]),
            "Sex": sub["Sex"],
            "Site": sub.get("Site", np.nan),
            "MeanFD": float(sub["MeanFD"]) if pd.notna(sub.get("MeanFD", np.nan)) else np.nan,
        })

    dist_row = {
        "sub_id": sub_id,
        "Group": sub["Group"],
        "Age": float(sub["Age"]),
        "FIQ": float(sub["FIQ"]),
        "Sex": sub["Sex"],
        "Site": sub.get("Site", np.nan),
        "MeanFD": float(sub["MeanFD"]) if pd.notna(sub.get("MeanFD", np.nan)) else np.nan,
    }
    for name, A, B in DIST_DEFS:
        # 只保留 G1_Default_dist_VisSomMot；不再生成 G1_Default_dist_Vis、
        # G1_Default_dist_SomMot 以及所有 G2 distance 指标。
        dist_row[f"G1_{name}"] = unsigned_network_distance(g1z, A, B, net2idx)
    rows_dist_wide.append(dist_row)


if missing_subs:
    pd.Series(missing_subs).to_csv(
        OUT_DIR / "missing_subjects_no_G1_or_G2.csv",
        index=False,
        header=False,
    )
    print(f"⚠️ 找不到 G1 或 G2 的被试数: {len(missing_subs)}")

if len(g1_matrix_list) == 0 or len(g2_matrix_list) == 0:
    raise RuntimeError("没有可用于分析的被试，请检查梯度文件路径和 sub_id 匹配。")

df_net   = pd.DataFrame(rows_net_long)
df_dist  = pd.DataFrame(rows_dist_wide)

if df_net.empty or df_dist.empty:
    raise RuntimeError("生成的数据表为空，请检查输入文件和被试匹配情况。")

df_net.to_csv(OUT_DIR / "Yeo7_network_level_G1G2_long.csv", index=False)
df_dist.to_csv(OUT_DIR / "Yeo7_hierarchy_distances_G1G2.csv", index=False)

for d in [df_net, df_dist]:
    d["Group"] = d["Group"].astype("category")
    d["Sex"]   = d["Sex"].astype("category")
    d["Site"]  = d["Site"].astype("category")

if "HC" not in df_net["Group"].cat.categories:
    raise ValueError("Group 中未找到 HC，无法作为 baseline。")

# =========================================================
# 4) 网络层 GLM：G1 / G2 主效应
#    FDR：G1_network 和 G2_network 分开校正
# =========================================================
net_results = []
for dv, dv_name in [("g1_net", "G1_network"), ("g2_net", "G2_network")]:
    for net in NETWORKS:
        dfn = df_net[df_net["network"] == net].copy()
        formula = (
            f"{dv} ~ C(Group, Treatment(reference='HC')) "
            "+ Age + FIQ + C(Sex) + MeanFD"
        )
        net_results.append(
            fit_and_extract_term(formula, dfn, dv_name=dv_name, item_name=net, term=TERM_GROUP)
        )

glm_net = fdr_by_dv(pd.DataFrame(net_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_net, "GLM_Yeo7_network_G1G2_main_effect_splitFDR", OUT_DIR)

# =========================================================
# 5) 网络层 GLM：G1 / G2 Group × Age 交互
#    FDR：G1_network 和 G2_network 分开校正
# =========================================================
net_int_results = []
for dv, dv_name in [("g1_net", "G1_network"), ("g2_net", "G2_network")]:
    for net in NETWORKS:
        dfn = df_net[df_net["network"] == net].copy()
        formula = (
            f"{dv} ~ C(Group, Treatment(reference='HC')) * Age "
            "+ FIQ + C(Sex) + MeanFD"
        )
        net_int_results.append(
            fit_and_extract_term(formula, dfn, dv_name=dv_name, item_name=net, term=TERM_INT)
        )

glm_net_int = fdr_by_dv(pd.DataFrame(net_int_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_net_int, "GLM_Yeo7_network_G1G2_GroupxAge_splitFDR", OUT_DIR)

# =========================================================
# 6) 网络间层级距离 GLM：只保留 G1_Default_dist_VisSomMot 主效应
# =========================================================
dist_results = []
dist_cols = ["G1_Default_dist_VisSomMot"]

for col in dist_cols:
    formula = (
        f"{col} ~ C(Group, Treatment(reference='HC')) "
        "+ Age + FIQ + C(Sex) + MeanFD"
    )
    dv_name = "G1_distance"
    dist_results.append(
        fit_and_extract_term(formula, df_dist, dv_name=dv_name, item_name=col, term=TERM_GROUP)
    )

glm_dist = fdr_by_dv(pd.DataFrame(dist_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_dist, "GLM_hierarchy_distances_G1G2_main_effect_splitFDR", OUT_DIR)

# =========================================================
# 7) 网络间层级距离 GLM：只保留 G1_Default_dist_VisSomMot Group × Age 交互
# =========================================================
dist_int_results = []
for col in dist_cols:
    formula = (
        f"{col} ~ C(Group, Treatment(reference='HC')) * Age "
        "+ FIQ + C(Sex) + MeanFD"
    )
    dv_name = "G1_distance"
    dist_int_results.append(
        fit_and_extract_term(formula, df_dist, dv_name=dv_name, item_name=col, term=TERM_INT)
    )

glm_dist_int = fdr_by_dv(pd.DataFrame(dist_int_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_dist_int, "GLM_hierarchy_distances_G1G2_GroupxAge_splitFDR", OUT_DIR)

# =========================================================
# 8) ROI-wise G1/G2 Group × Age 交互分析，可用于 ctab / 全皮层 β 图
#    每个梯度内部按 ROI 做一次 FDR；G1 与 G2 不联合校正
# =========================================================
if RUN_ROIWISE_INTERACTION:
    G1_matrix = np.vstack(g1_matrix_list)   # N × P
    G2_matrix = np.vstack(g2_matrix_list)   # N × P
    N1, P1 = G1_matrix.shape
    N2, P2 = G2_matrix.shape

    if N1 != N2 or P1 != P2:
        raise RuntimeError("G1_matrix 与 G2_matrix 维度不一致。")

    df_roi_base = info[info["sub_id"].isin(roi_subjects)].copy()
    df_roi_base = df_roi_base.set_index("sub_id").loc[roi_subjects].reset_index()

    if len(df_roi_base) != N1:
        raise RuntimeError("ROI-wise 协变量表与梯度矩阵被试数不一致。")

    if "G1" in ROIWISE_GRADIENTS:
        run_roiwise_groupxage(
            grad_matrix=G1_matrix,
            grad_name="G1",
            df_roi_base=df_roi_base,
            label_df=label_df,
            atlas_label_col=atlas_label_col,
            out_dir=OUT_DIR,
            alpha=ALPHA_FDR,
        )

    if "G2" in ROIWISE_GRADIENTS:
        run_roiwise_groupxage(
            grad_matrix=G2_matrix,
            grad_name="G2",
            df_roi_base=df_roi_base,
            label_df=label_df,
            atlas_label_col=atlas_label_col,
            out_dir=OUT_DIR,
            alpha=ALPHA_FDR,
        )

print("✅ 完成：G1/G2 网络层分析，模型协变量为 Group + Age + FIQ + Sex + MeanFD；G1 与 G2 分开 FDR 校正")
print("✅ 完成：网络间层级距离分析；仅保留 G1_Default_dist_VisSomMot，层级指标为 |A - B| unsigned distance")
if RUN_ROIWISE_INTERACTION:
    print("✅ 完成：ROI-wise Group × Age 交互分析；G1 与 G2 各自独立 FDR")
print("输出目录：", OUT_DIR)
