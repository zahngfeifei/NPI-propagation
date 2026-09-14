import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
import re

# =========================================================
# 路径设置
# =========================================================
BASE_ROOT = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-参数敏感性分析\ABIDE1_结果2_sensitivity_combat"
)

LABEL_FILE = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\Schaefer400_7Yeo_network_labels_with_ID.csv"
)

INFO_FILE = Path(
    r"I:\DYF\NPI-4-code\subject_info_for_stats.csv"
)

OUT_ROOT = Path(
    r"I:\DYF\NPI-4-code\2.梯度分析\正向连接-参数敏感性分析\ABIDE1_结果6_sensitivity_G1_Default_dist_VisSomMot_only_noFD"
)
OUT_ROOT.mkdir(exist_ok=True, parents=True)

# =========================================================
# 敏感性分析参数
# =========================================================
P_THRESHOLDS = [0.10, 0.15, 0.20, 0.25]
K_ALIGN_LIST = [3, 5, 7, 9]

EXPECTED_FEATURES = 400

# =========================================================
# 只分析这一个指标
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

DIST_NAME = "G1_Default_dist_VisSomMot"
DIST_A = "Default"
DIST_B = ("Vis", "SomMot")

DV_NAME = "G1_distance"
ITEM_NAME = "G1_Default_dist_VisSomMot"

TERM_GROUP = "C(Group, Treatment(reference='HC'))[T.ASD]"

FORMULA = (
    "G1_Default_dist_VisSomMot ~ "
    "C(Group, Treatment(reference='HC')) + Age + FIQ + C(Sex)"
)

# =========================================================
# 工具函数
# =========================================================
def p_label(p):
    return f"p{int(round(p * 100)):02d}"


def k_label(k):
    return f"k{k:02d}"


def normalize_sub_id(raw_sub_id):
    """
    统一不同 sub_id 格式。

    Examples:
        sub-Sub28741 -> 28741
        Sub28741     -> 28741
        sub-28741    -> 28741
        28741        -> 28741
        28741.0      -> 28741
    """
    if pd.isna(raw_sub_id):
        return ""

    sid = str(raw_sub_id).strip()

    if re.fullmatch(r"\d+\.0", sid):
        sid = sid[:-2]

    sid = re.sub(r"^sub-Sub", "", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^sub-", "", sid, flags=re.IGNORECASE)
    sid = re.sub(r"^Sub", "", sid, flags=re.IGNORECASE)

    digit_match = re.findall(r"\d+", sid)
    if digit_match:
        sid = digit_match[-1].lstrip("0")

    return sid


def get_sub_id_from_folder(folder_name: str):
    first_entity = str(folder_name).split("_", 1)[0]
    return normalize_sub_id(first_entity)


def build_subject_folder_map(combo_dir: Path) -> dict[str, Path]:
    """
    为某个 p/k 组合一次性建立：
        normalized_sub_id -> subject_folder

    性能优化：
        每个 p/k 组合只扫描一次 sub-* 文件夹，
        不再为每个被试重复扫描整个目录。
    """
    folder_map = {}

    for folder in sorted(combo_dir.glob("sub-*")):
        if not folder.is_dir():
            continue

        sid_norm = get_sub_id_from_folder(folder.name)

        if sid_norm == "":
            continue

        if sid_norm not in folder_map:
            folder_map[sid_norm] = folder

    return folder_map


def find_g1_file(
    combo_dir: Path,
    sub_id: str,
    folder_map: dict[str, Path],
) -> Path | None:
    """
    在某个 p/k 组合目录下查找 G1 文件。

    优先读取：
        out_G1_procrustes.npy

    若不存在，再读取：
        out_G1_procrustes_combat.npy

    注意：
        folder_map 已在当前 p/k 组合开始时建立，
        此函数不会再次扫描 combo_dir。
    """
    sid_norm = normalize_sub_id(sub_id)

    candidate_folders = []

    if sid_norm in folder_map:
        candidate_folders.append(folder_map[sid_norm])

    # 保留原代码中的候选路径规则，避免改变文件匹配逻辑
    candidate_folders.extend([
        combo_dir / str(sub_id),
        combo_dir / f"sub-{sub_id}",
        combo_dir / f"sub-Sub{sub_id}",
        combo_dir / f"sub-{sid_norm}",
        combo_dir / f"sub-Sub{sid_norm}",
    ])

    file_names = [
        "out_G1_procrustes.npy",
        "out_G1_procrustes_combat.npy",
    ]

    seen = set()

    for folder in candidate_folders:
        folder_key = str(folder)

        if folder_key in seen:
            continue

        seen.add(folder_key)

        for fname in file_names:
            path = folder / fname

            if path.exists():
                return path

    return None


def safe_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    mu = np.mean(x)
    sd = np.std(x)

    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)

    return (x - mu) / sd


def as_network_tuple(network_group) -> tuple[str, ...]:
    if isinstance(network_group, str):
        return (network_group,)
    return tuple(network_group)


def get_group_indices(net2idx: dict, network_group) -> np.ndarray:
    nets = as_network_tuple(network_group)
    idx_list = []

    for net in nets:
        if net not in net2idx:
            raise KeyError(f"未知网络名称: {net}")

        idx = net2idx[net]

        if len(idx) == 0:
            raise ValueError(f"网络 {net} 没有 ROI")

        idx_list.append(idx)

    return np.concatenate(idx_list)


def unsigned_network_distance_from_indices(
    grad_z: np.ndarray,
    group_a_indices: np.ndarray,
    group_b_indices: np.ndarray,
) -> float:
    """
    D(A, B) = |mean_zG(A) - mean_zG(B)|

    本脚本中：
        A = Default
        B = Vis + SomMot

    计算定义与原代码完全一致，只是网络索引已预先计算，
    避免每个被试重复调用 np.concatenate。
    """
    a_mean = float(grad_z[group_a_indices].mean())
    b_mean = float(grad_z[group_b_indices].mean())

    if not np.isfinite(a_mean) or not np.isfinite(b_mean):
        return np.nan

    return float(np.abs(a_mean - b_mean))


def load_gradient_vector(path: Path, expected_features: int = EXPECTED_FEATURES):
    arr = np.load(path).reshape(-1)

    if expected_features is not None and len(arr) != expected_features:
        raise ValueError(
            f"梯度长度错误: {path}, length={len(arr)}, expected={expected_features}"
        )

    if not np.isfinite(arr).all():
        raise ValueError(f"梯度包含 NaN/Inf: {path}")

    return arr


def fit_single_model(df: pd.DataFrame) -> dict:
    """
    只拟合这一条模型：

    G1_Default_dist_VisSomMot ~
        C(Group, Treatment(reference='HC')) + Age + FIQ + C(Sex)
    """
    model = smf.ols(FORMULA, data=df).fit()

    if TERM_GROUP not in model.params.index:
        raise RuntimeError(f"模型缺少指定项：{TERM_GROUP}")

    ci = model.conf_int().loc[TERM_GROUP].tolist()

    return {
        "dv": DV_NAME,
        "item": ITEM_NAME,
        "term": TERM_GROUP,
        "formula": FORMULA,
        "beta": float(model.params[TERM_GROUP]),
        "t": float(model.tvalues[TERM_GROUP]),
        "p": float(model.pvalues[TERM_GROUP]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


# =========================================================
# 读取 Yeo7 标签
# =========================================================
label_df = pd.read_csv(LABEL_FILE)

required_cols = {"ROI_ID", "Yeo7_network"}
miss = required_cols - set(label_df.columns)

if miss:
    raise ValueError(f"Label file 缺少列: {miss}")

label_df["ROI_idx"] = label_df["ROI_ID"].astype(int) - 1

net2idx = {
    net: label_df.loc[label_df["Yeo7_network"] == net, "ROI_idx"].values
    for net in NETWORKS
}

for net, idx in net2idx.items():
    if len(idx) == 0:
        raise ValueError(f"网络 {net} 没有对应 ROI，请检查标签文件。")

# 性能优化：网络索引只计算一次
DIST_A_INDICES = get_group_indices(net2idx, DIST_A)
DIST_B_INDICES = get_group_indices(net2idx, DIST_B)

# =========================================================
# 读取被试信息
# =========================================================
info = pd.read_csv(INFO_FILE)

need = {"sub_id", "Group", "Age", "Sex", "FIQ"}
miss = need - set(info.columns)

if miss:
    raise ValueError(f"Subject info 缺少列: {miss}")

info["FIQ"] = pd.to_numeric(info["FIQ"], errors="coerce")
info["Age"] = pd.to_numeric(info["Age"], errors="coerce")

# 这个模型实际使用变量：
# Group + Age + FIQ + Sex
info = info.dropna(
    subset=["sub_id", "Group", "Age", "Sex", "FIQ"]
).copy()

info["sub_id"] = info["sub_id"].astype(str)
info["SUB_ID_norm"] = info["sub_id"].apply(normalize_sub_id)

info = info[info["SUB_ID_norm"] != ""].copy()
info = info.drop_duplicates(subset="SUB_ID_norm", keep="first").copy()

print(f"Subject info after covariate QC: {len(info)}")


# =========================================================
# 单个 p × K_ALIGN 组合分析
# =========================================================
def run_one_combination(p: float, k_align: int) -> dict:
    ptag = p_label(p)
    ktag = k_label(k_align)

    combo_dir = BASE_ROOT / ptag / ktag
    out_dir = OUT_ROOT / ptag / ktag
    out_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 100)
    print(f"开始分析: {ptag}, {ktag}")
    print(f"输入目录: {combo_dir}")
    print(f"输出目录: {out_dir}")
    print("=" * 100)

    if not combo_dir.exists():
        raise RuntimeError(f"组合输入目录不存在: {combo_dir}")

    # 性能优化：
    # 每个 p/k 组合只扫描一次被试文件夹
    folder_map = build_subject_folder_map(combo_dir)

    print(f"当前组合扫描到的被试文件夹数: {len(folder_map)}")

    rows = []
    missing_subs = []
    failed_subs = []

    # 性能优化：itertuples 比 iterrows 更快
    for sub in info.itertuples(index=False):
        sub_id_raw = str(sub.sub_id)
        sub_id_norm = str(sub.SUB_ID_norm)

        try:
            g1_path = find_g1_file(
                combo_dir=combo_dir,
                sub_id=sub_id_raw,
                folder_map=folder_map,
            )

            if g1_path is None:
                missing_subs.append({
                    "sub_id": sub_id_raw,
                    "SUB_ID_norm": sub_id_norm,
                    "missing_gradient": "G1",
                })
                raise FileNotFoundError("G1 文件不存在")

            g1 = load_gradient_vector(g1_path)

            if len(g1) != len(label_df):
                raise ValueError(
                    f"{sub_id_raw} G1 长度与 label 不匹配: "
                    f"{len(g1)} vs {len(label_df)}"
                )

            g1z = safe_zscore(g1)

            dist_value = unsigned_network_distance_from_indices(
                grad_z=g1z,
                group_a_indices=DIST_A_INDICES,
                group_b_indices=DIST_B_INDICES,
            )

            row = {
                "sub_id": sub_id_raw,
                "SUB_ID_norm": sub_id_norm,
                "Group": sub.Group,
                "Age": float(sub.Age),
                "FIQ": float(sub.FIQ),
                "Sex": sub.Sex,
                "G1_file": str(g1_path),
                DIST_NAME: dist_value,
            }

            rows.append(row)

        except Exception as e:
            failed_subs.append({
                "sub_id": sub_id_raw,
                "SUB_ID_norm": sub_id_norm,
                "error": f"{type(e).__name__}: {e}",
            })
            continue

    if missing_subs:
        pd.DataFrame(missing_subs).to_csv(
            out_dir / "missing_subjects_no_G1.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if failed_subs:
        pd.DataFrame(failed_subs).to_csv(
            out_dir / "failed_subjects_G1_distance.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"{ptag}/{ktag}: 没有可用于分析的被试。")

    df = df.dropna(
        subset=[
            "Group",
            "Age",
            "FIQ",
            "Sex",
            DIST_NAME,
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(f"{ptag}/{ktag}: 剔除缺失值后数据为空。")

    df["Group"] = df["Group"].astype("category")
    df["Sex"] = df["Sex"].astype("category")

    if "HC" not in df["Group"].cat.categories:
        raise ValueError(f"{ptag}/{ktag}: Group 中未找到 HC，无法作为 baseline。")

    # 保存该组合下的距离指标
    metric_csv = out_dir / "G1_Default_dist_VisSomMot_subject_values.csv"
    df.to_csv(
        metric_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # 拟合指定单一模型
    result = fit_single_model(df)

    result.update({
        "p_threshold": p,
        "p_label": ptag,
        "K_ALIGN": k_align,
        "k_label": ktag,
        "n_info_after_covariate_qc": len(info),
        "n_metric_subjects_before_model_na_drop": len(rows),
        "n_model_subjects": len(df),
        "n_failed_or_missing_subjects": len(failed_subs),
        "metric_csv": str(metric_csv),
        "out_dir": str(out_dir),
        "status": "success",
        "error": "",
    })

    result_csv = out_dir / "GLM_G1_Default_dist_VisSomMot_main_effect.csv"

    pd.DataFrame([result]).to_csv(
        result_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # 简要报告
    summary_path = out_dir / "summary_report.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Sensitivity analysis: G1_Default_dist_VisSomMot only, no FD\n")
        f.write("===========================================================\n\n")

        f.write("Combination:\n")
        f.write(f"  p_threshold: {p}\n")
        f.write(f"  p_label: {ptag}\n")
        f.write(f"  K_ALIGN: {k_align}\n")
        f.write(f"  k_label: {ktag}\n\n")

        f.write("Metric:\n")
        f.write("  G1_Default_dist_VisSomMot = |mean_zG1(Default) - mean_zG1(Vis + SomMot)|\n")
        f.write("  This is an unsigned distance metric.\n\n")

        f.write("Model:\n")
        f.write(f"  {FORMULA}\n")
        f.write(f"  Extracted term: {TERM_GROUP}\n\n")

        f.write("Sample size:\n")
        f.write(f"  info subjects after covariate QC: {len(info)}\n")
        f.write(f"  metric subjects before model NA drop: {len(rows)}\n")
        f.write(f"  model subjects: {len(df)}\n")
        f.write(f"  failed or missing subjects: {len(failed_subs)}\n\n")

        f.write("Model result:\n")
        f.write(f"  beta: {result['beta']}\n")
        f.write(f"  t: {result['t']}\n")
        f.write(f"  p: {result['p']}\n")
        f.write(f"  ci_low: {result['ci_low']}\n")
        f.write(f"  ci_high: {result['ci_high']}\n")
        f.write(f"  n: {result['n']}\n")
        f.write(f"  r2: {result['r2']}\n")
        f.write(f"  adj_r2: {result['adj_r2']}\n")
        f.write(f"  aic: {result['aic']}\n")
        f.write(f"  bic: {result['bic']}\n\n")

        f.write("Outputs:\n")
        f.write(f"  subject metric CSV: {metric_csv}\n")
        f.write(f"  GLM result CSV: {result_csv}\n")
        f.write(f"  summary: {summary_path}\n")

    result["result_csv"] = str(result_csv)
    result["summary_path"] = str(summary_path)

    return result


# =========================================================
# 主程序：循环 p × K_ALIGN
# =========================================================
overall_rows = []

for p in P_THRESHOLDS:
    for k_align in K_ALIGN_LIST:
        try:
            row = run_one_combination(p, k_align)
            overall_rows.append(row)

            print(
                f"✅ 完成: {row['p_label']} / {row['k_label']} / "
                f"N={row['n_model_subjects']} / beta={row['beta']:.6f} / p={row['p']:.6g}"
            )

        except Exception as e:
            print(f"❌ 失败: {p_label(p)} / {k_label(k_align)}: {e}")

            overall_rows.append({
                "dv": DV_NAME,
                "item": ITEM_NAME,
                "term": TERM_GROUP,
                "formula": FORMULA,
                "p_threshold": p,
                "p_label": p_label(p),
                "K_ALIGN": k_align,
                "k_label": k_label(k_align),
                "status": "failed",
                "beta": np.nan,
                "t": np.nan,
                "p": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": 0,
                "r2": np.nan,
                "adj_r2": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "n_info_after_covariate_qc": len(info),
                "n_metric_subjects_before_model_na_drop": 0,
                "n_model_subjects": 0,
                "n_failed_or_missing_subjects": np.nan,
                "metric_csv": "",
                "out_dir": str(OUT_ROOT / p_label(p) / k_label(k_align)),
                "result_csv": "",
                "summary_path": "",
                "error": f"{type(e).__name__}: {e}",
            })

# =========================================================
# 总体汇总
# =========================================================
overall_df = pd.DataFrame(overall_rows)

overall_csv = OUT_ROOT / "overall_G1_Default_dist_VisSomMot_sensitivity_summary.csv"

overall_df.to_csv(
    overall_csv,
    index=False,
    encoding="utf-8-sig",
)

overall_txt = OUT_ROOT / "overall_G1_Default_dist_VisSomMot_sensitivity_report.txt"

with open(overall_txt, "w", encoding="utf-8") as f:
    f.write("Overall sensitivity report: G1_Default_dist_VisSomMot only, no FD\n")
    f.write("=================================================================\n\n")

    f.write("Design:\n")
    f.write(f"  Thresholds: {P_THRESHOLDS}\n")
    f.write(f"  K_ALIGN_LIST: {K_ALIGN_LIST}\n")
    f.write("  Only one planned metric is tested in each sensitivity condition.\n\n")

    f.write("Metric:\n")
    f.write("  G1_Default_dist_VisSomMot = |mean_zG1(Default) - mean_zG1(Vis + SomMot)|\n\n")

    f.write("Model:\n")
    f.write(f"  {FORMULA}\n")
    f.write(f"  Extracted term: {TERM_GROUP}\n\n")

    f.write("Paths:\n")
    f.write(f"  BASE_ROOT: {BASE_ROOT}\n")
    f.write(f"  OUT_ROOT: {OUT_ROOT}\n")
    f.write(f"  LABEL_FILE: {LABEL_FILE}\n")
    f.write(f"  INFO_FILE: {INFO_FILE}\n\n")

    f.write("Combination results:\n")
    f.write("=" * 100 + "\n\n")

    for row in overall_df.itertuples(index=False):
        f.write(
            f"{row.p_label} | {row.k_label} | "
            f"status={row.status} | "
            f"N={row.n_model_subjects} | "
            f"beta={row.beta} | "
            f"t={row.t} | "
            f"p={row.p}\n"
        )

        if row.status != "success":
            f.write(f"  error: {row.error}\n")

        f.write("\n")

    f.write("Outputs:\n")
    f.write(f"  overall CSV: {overall_csv}\n")
    f.write(f"  overall report: {overall_txt}\n")

print("\n✅ 单一指标敏感性分析完成")
print("总体汇总 CSV：", overall_csv)
print("总体报告：", overall_txt)
print("输出目录：", OUT_ROOT)