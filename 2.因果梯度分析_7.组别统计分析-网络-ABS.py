import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

# =========================================================
# 路径设置 —— 负向连接适配版
# =========================================================
BASE_DIR   = Path(r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_负向结果2_combat")
LABEL_FILE = Path(r"I:\DYF\NPI-4-code\2.梯度分析\Schaefer400_7Yeo_network_labels_with_ID.csv")
INFO_FILE  = Path(r"I:\DYF\NPI-3\subject_info_for_stats.csv")

OUT_DIR    = Path(r"I:\DYF\NPI-4-code\2.梯度分析\负向链接-独立模板\ABIDE2_结果6_integrated_G1G2_absDistance_DMN_VisSMN_负向")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 负向连接输出前缀
PREFIX = "负向连接_"

# =========================================================
# 配置
# Site 和 MeanFD 已通过 ComBat 控制，不再纳入 GLM 模型。
# =========================================================
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# 网络间层级距离定义：仅保留 G1_Default_dist_VisSomMot
# Default 对应 DMN；("Vis", "SomMot") 表示 VIS 与 SMN/SomMot 联合后的 ROI 集合。
# 该版本使用绝对值，因此为 unsigned distance，不保留方向信息。
DIST_DEFS = [
    ("Default_dist_VisSomMot", "Default", ("Vis", "SomMot")),
]

TERM_GROUP = "C(Group, Treatment(reference='HC'))[T.ASD]"
TERM_INT   = "C(Group, Treatment(reference='HC'))[T.ASD]:Age"
ALPHA_FDR  = 0.05

# 是否做 ROI-wise Group × Age 交互分析
RUN_ROIWISE_INTERACTION = True
ROIWISE_GRADIENTS = ["G1", "G2"]

# =========================================================
# 工具函数
# =========================================================
def out_file(name: str) -> Path:
    """统一给负向输出文件加前缀。"""
    return OUT_DIR / f"{PREFIX}{name}"

def find_g_file(sub_id: str, g: int) -> Path | None:
    """
    查找 Procrustes 对齐后的 G1/G2 文件。

    兼容两种命名：
      1) out_G1_procrustes.npy
      2) 负向连接_out_G1_procrustes.npy

    也兼容 sub_id 是否带 sub- 前缀。
    """
    fnames = [
        f"{PREFIX}out_G{g}_procrustes.npy",
        f"out_G{g}_procrustes.npy",
    ]

    sub_candidates = [sub_id]

    if not sub_id.startswith("sub-"):
        sub_candidates.append(f"sub-{sub_id}")

    if sub_id.startswith("sub-"):
        stripped = sub_id.replace("sub-", "", 1)
        sub_candidates.append(f"sub-{stripped}")

    # 去重但保持顺序
    sub_candidates = list(dict.fromkeys(sub_candidates))

    for sid in sub_candidates:
        for fname in fnames:
            p = BASE_DIR / sid / fname
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

def network_axis_distance(grad_z: np.ndarray, group_a, group_b, net2idx: dict) -> float:
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
    用于保证 G1 和 G2 不联合校正。
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

def save_split_tables(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    """保存合并表，并按 dv 额外拆成单独 CSV；所有输出加负向前缀。"""
    df.to_csv(out_dir / f"{PREFIX}{name}.csv", index=False)

    if "dv" in df.columns and not df.empty:
        for dv_value, sub_df in df.groupby("dv", sort=False):
            safe_dv = str(dv_value).replace("/", "_").replace(" ", "_")
            sub_df.to_csv(out_dir / f"{PREFIX}{name}_{safe_dv}.csv", index=False)

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
    np.save(out_dir / f"{PREFIX}{grad_name}_matrix.npy", grad_matrix)

    df_roi = df_roi_base.copy()
    df_roi["Group"] = df_roi["Group"].astype("category")
    df_roi["Sex"]   = df_roi["Sex"].astype("category")

    beta_vals = np.zeros(P, dtype=float)
    t_vals    = np.zeros(P, dtype=float)
    p_vals    = np.zeros(P, dtype=float)
    ci_low    = np.zeros(P, dtype=float)
    ci_high   = np.zeros(P, dtype=float)

    roi_var = f"{grad_name}_roi"
    formula_roi_int = (
        f"{roi_var} ~ C(Group, Treatment(reference='HC')) * Age "
        "+ FIQ + C(Sex)"
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

    np.save(out_dir / f"{PREFIX}{grad_name}_roi_GroupxAge_beta.npy", beta_vals)
    np.save(out_dir / f"{PREFIX}{grad_name}_roi_GroupxAge_t.npy", t_vals)
    np.save(out_dir / f"{PREFIX}{grad_name}_roi_GroupxAge_p.npy", p_vals)
    np.save(out_dir / f"{PREFIX}{grad_name}_roi_GroupxAge_p_fdr.npy", p_fdr)
    np.save(out_dir / f"{PREFIX}{grad_name}_roi_GroupxAge_sig_mask.npy", reject_fdr.astype(np.int8))

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
    roi_table.to_csv(out_dir / f"{PREFIX}{grad_name}_GroupxAge_interaction_roi_table.csv", index=False)

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
    roi_summary.to_csv(out_dir / f"{PREFIX}{grad_name}_GroupxAge_interaction_roi_summary.csv", index=False)

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

info["FIQ"]    = pd.to_numeric(info["FIQ"], errors="coerce")
info["Age"]    = pd.to_numeric(info["Age"], errors="coerce")

# 严格剔除缺失，保证不同模型样本一致
info = info.dropna(subset=["sub_id", "Group", "Age", "Sex", "FIQ"]).copy()
info["sub_id"] = info["sub_id"].astype(str)

# =========================================================
# 3) 逐被试计算：网络均值、网络间绝对距离
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
        missing_subs.append({
            "sub_id": sub_id,
            "missing_G1": int(p1 is None),
            "missing_G2": int(p2 is None),
        })
        continue

    g1 = np.load(p1).reshape(-1)
    g2 = np.load(p2).reshape(-1)

    if len(g1) != len(label_df) or len(g2) != len(label_df):
        raise ValueError(
            f"{sub_id} 梯度长度与 label 不匹配："
            f"G1={len(g1)} G2={len(g2)} label={len(label_df)}\n"
            f"G1 file: {p1}\n"
            f"G2 file: {p2}"
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
            "G1_file": str(p1),
            "G2_file": str(p2),
        })

    dist_row = {
        "sub_id": sub_id,
        "Group": sub["Group"],
        "Age": float(sub["Age"]),
        "FIQ": float(sub["FIQ"]),
        "Sex": sub["Sex"],
        "G1_file": str(p1),
        "G2_file": str(p2),
    }

    for name, A, B in DIST_DEFS:
        # 仅保留 G1_Default_dist_VisSomMot；不再生成 G1_Vis、G1_SomMot 或任何 G2 distance 指标。
        dist_row[f"G1_{name}"] = network_axis_distance(g1z, A, B, net2idx)

    rows_dist_wide.append(dist_row)

if missing_subs:
    pd.DataFrame(missing_subs).to_csv(
        out_file("missing_subjects_no_G1_or_G2.csv"),
        index=False,
    )
    print(f"⚠️ 找不到 G1 或 G2 的被试数: {len(missing_subs)}")
    print(f"缺失名单已保存: {out_file('missing_subjects_no_G1_or_G2.csv')}")

if len(g1_matrix_list) == 0 or len(g2_matrix_list) == 0:
    raise RuntimeError(
        "没有可用于分析的被试。请检查：\n"
        f"1) BASE_DIR 是否正确: {BASE_DIR}\n"
        "2) 每个被试目录下是否存在 out_G1_procrustes.npy / out_G2_procrustes.npy\n"
        "3) 或是否存在 负向连接_out_G1_procrustes.npy / 负向连接_out_G2_procrustes.npy\n"
        "4) subject_info_for_stats.csv 中的 sub_id 是否能匹配结果2_负向中的目录名。"
    )

df_net   = pd.DataFrame(rows_net_long)
df_dist  = pd.DataFrame(rows_dist_wide)

if df_net.empty or df_dist.empty:
    raise RuntimeError("生成的数据表为空，请检查输入文件和被试匹配情况。")

df_net.to_csv(out_file("Yeo7_network_level_G1G2_long.csv"), index=False)
df_dist.to_csv(out_file("Yeo7_hierarchy_distances_G1G2.csv"), index=False)

for d in [df_net, df_dist]:
    d["Group"] = d["Group"].astype("category")
    d["Sex"]   = d["Sex"].astype("category")

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
            "+ Age + FIQ + C(Sex)"
        )
        net_results.append(
            fit_and_extract_term(
                formula=formula,
                data=dfn,
                dv_name=dv_name,
                item_name=net,
                term=TERM_GROUP,
            )
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
            "+ FIQ + C(Sex)"
        )
        net_int_results.append(
            fit_and_extract_term(
                formula=formula,
                data=dfn,
                dv_name=dv_name,
                item_name=net,
                term=TERM_INT,
            )
        )

glm_net_int = fdr_by_dv(pd.DataFrame(net_int_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_net_int, "GLM_Yeo7_network_G1G2_GroupxAge_splitFDR", OUT_DIR)

# =========================================================
# 6) 网络间层级距离 GLM：仅 G1_Default_dist_VisSomMot 主效应
#    只保留一个 distance 指标，因此 FDR 仅作用于该单项结果
# =========================================================
dist_results = []
dist_cols = [c for c in df_dist.columns if c.startswith("G1_") or c.startswith("G2_")]

# 排除文件路径列，避免被误认为 G1/G2 指标
dist_cols = [c for c in dist_cols if c not in ["G1_file", "G2_file"]]

for col in dist_cols:
    formula = (
        f"{col} ~ C(Group, Treatment(reference='HC')) "
        "+ Age + FIQ + C(Sex)"
    )
    dv_name = "G1_distance" if col.startswith("G1_") else "G2_distance"
    dist_results.append(
        fit_and_extract_term(
            formula=formula,
            data=df_dist,
            dv_name=dv_name,
            item_name=col,
            term=TERM_GROUP,
        )
    )

glm_dist = fdr_by_dv(pd.DataFrame(dist_results), dv_col="dv", p_col="p", alpha=ALPHA_FDR)
save_split_tables(glm_dist, "GLM_hierarchy_distances_G1G2_main_effect_splitFDR", OUT_DIR)

# =========================================================
# 7) 网络间层级距离 GLM：仅 G1_Default_dist_VisSomMot Group × Age 交互
#    只保留一个 distance 指标，因此 FDR 仅作用于该单项结果
# =========================================================
dist_int_results = []

for col in dist_cols:
    formula = (
        f"{col} ~ C(Group, Treatment(reference='HC')) * Age "
        "+ FIQ + C(Sex)"
    )
    dv_name = "G1_distance" if col.startswith("G1_") else "G2_distance"
    dist_int_results.append(
        fit_and_extract_term(
            formula=formula,
            data=df_dist,
            dv_name=dv_name,
            item_name=col,
            term=TERM_INT,
        )
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

# =========================================================
# 9) 简要运行摘要
# =========================================================
run_summary = pd.DataFrame([{
    "BASE_DIR": str(BASE_DIR),
    "OUT_DIR": str(OUT_DIR),
    "N_subjects_used": int(len(roi_subjects)),
    "N_missing_G1_or_G2": int(len(missing_subs)),
    "P_rois": int(len(label_df)),
    "n_networks": int(len(NETWORKS)),
    "n_distance_metrics": int(len(dist_cols)),
    "distance_definition": "|mean_zG(A) - mean_zG(B)|",
    "run_roiwise_interaction": bool(RUN_ROIWISE_INTERACTION),
    "roiwise_gradients": ",".join(ROIWISE_GRADIENTS),
}])
run_summary.to_csv(out_file("run_summary.csv"), index=False)

print("✅ 完成：负向连接 G1/G2 网络层分析，G1 与 G2 分开 FDR 校正")
print("✅ 完成：负向连接网络间层级距离分析；仅保留 G1_Default_dist_VisSomMot，层级指标为 |A - B| unsigned distance")
if RUN_ROIWISE_INTERACTION:
    print("✅ 完成：负向连接 ROI-wise Group × Age 交互分析；G1 与 G2 各自独立 FDR")
print("使用被试数：", len(roi_subjects))
print("缺失 G1/G2 被试数：", len(missing_subs))
print("输出目录：", OUT_DIR)