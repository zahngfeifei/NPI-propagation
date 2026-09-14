import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import rankdata


# ============================================================
# 1. 路径设置
# ============================================================

ABIDE2_ROOT = (
    r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度"
    r"\ABIDE2_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平"
)

# 直接使用指定的 group-level 结果文件
ABIDE2_GROUP_CSV = (
    r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度"
    r"\ABIDE2_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平"
    r"\group_level_results"
    r"\group_level_subject_mean_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
)

ABIDE2_GROUP_MEAN_MAP_CSV = (
    r"I:\DYF\NPI-4-code\2.梯度分析\功能梯度"
    r"\ABIDE2_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平"
    r"\group_level_results"
    r"\group_mean_map_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
)

SAVE_DIR = r"I:\DYF\NPI-4-code\补充内容新\S2"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. 输入文件名
# ============================================================

LONG_FILE = (
    "ALL_wholebrain_EC-G1G2_vs_FC-G1G2_"
    "cross_axis_spin_test_long.csv"
)

SUMMARY_FILE = (
    "ALL_wholebrain_EC-G1G2_vs_FC-G1G2_"
    "cross_axis_summary_statistics.csv"
)

EC_GROUP_MEAN_MAP_FILE = "EC_group_mean_gradient_map.csv"
FC_GROUP_MEAN_MAP_FILE = "FC_group_mean_gradient_map.csv"


# ============================================================
# 3. 输出文件
# ============================================================

OUT_PNG = os.path.join(
    SAVE_DIR,
    "S2_ABIDE2_EC_FC_2x2_subject_group_spin_and_group_mean_map.png"
)

OUT_SVG = os.path.join(
    SAVE_DIR,
    "S2_ABIDE2_EC_FC_2x2_subject_group_spin_and_group_mean_map.svg"
)

OUT_NUMERIC = os.path.join(
    SAVE_DIR,
    "S2_ABIDE2_EC_FC_2x2_subject_group_spin_and_group_mean_map_values.csv"
)


# ============================================================
# 4. 字体设置
# ============================================================

FONT_FAMILY = "Arial"
FONT_SIZE = 21

rcParams["font.family"] = FONT_FAMILY
rcParams["font.sans-serif"] = [FONT_FAMILY]
rcParams["font.size"] = FONT_SIZE

rcParams["axes.titlesize"] = FONT_SIZE
rcParams["axes.labelsize"] = FONT_SIZE

rcParams["xtick.labelsize"] = FONT_SIZE
rcParams["ytick.labelsize"] = FONT_SIZE

rcParams["legend.fontsize"] = FONT_SIZE
rcParams["figure.titlesize"] = FONT_SIZE

rcParams["svg.fonttype"] = "none"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

rcParams["axes.unicode_minus"] = False


# ============================================================
# 5. 绘图参数
# ============================================================

COMPARISON_ORDER = [
    "EC_G1_vs_FC_G1",
    "EC_G1_vs_FC_G2",
    "EC_G2_vs_FC_G1",
    "EC_G2_vs_FC_G2"
]

COMPARISON_LABELS = [
    "EC-G1 vs FC-G1",
    "EC-G1 vs FC-G2",
    "EC-G2 vs FC-G1",
    "EC-G2 vs FC-G2"
]

HEATMAP_ROWS = [
    "EC-G1",
    "EC-G2"
]

HEATMAP_COLS = [
    "FC-G1",
    "FC-G2"
]

POINT_COLOR = "#1f5fbf"
BOX_COLOR = "#d8e6ff"

MEAN_COLOR = "#b30000"
MEDIAN_COLOR = "#083d9c"

SCATTER_COLOR = "#2F74D0"
SCATTER_LINE_COLOR = "#1E5BB8"

HEATMAP_CMAP = "RdBu_r"

TITLE_Y = 1.13
TITLE_PAD = 2

PANEL_LABEL_X = -0.14
PANEL_LABEL_Y = 1.22


# ============================================================
# 6. 工具函数
# ============================================================

def check_file(path, label):
    """检查文件是否存在。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label} 不存在：\n{path}"
        )


def find_group_mean_gradient_map_file(
    result_root,
    file_name,
    dataset_label
):
    """
    搜索 EC 或 FC 的 ROI 级组平均梯度图文件。
    """

    candidates = [
        os.path.join(
            result_root,
            "group_level_results",
            file_name
        ),
        os.path.join(
            result_root,
            "all_results",
            file_name
        ),
        os.path.join(
            result_root,
            file_name
        ),
        os.path.join(
            SAVE_DIR,
            file_name
        )
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"{dataset_label} 的 {file_name} 不存在。\n"
        f"组平均图散点图需要该 ROI 级组平均梯度图文件。\n"
        f"尝试过以下路径：\n"
        + "\n".join(candidates)
    )


def spearman_correlation(a, b):
    """
    手动计算 Spearman 相关系数，
    避免不同 scipy 版本造成的接口差异。
    """

    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {a.shape} vs {b.shape}"
        )

    finite_mask = np.isfinite(a) & np.isfinite(b)

    a = a[finite_mask]
    b = b[finite_mask]

    if len(a) < 3:
        raise ValueError(
            "有效 ROI 数量少于 3，无法计算 Spearman 相关。"
        )

    ar = rankdata(
        a,
        method="average"
    ).astype(float)

    br = rankdata(
        b,
        method="average"
    ).astype(float)

    ar -= np.mean(ar)
    br -= np.mean(br)

    denominator = (
        np.sqrt(np.sum(ar ** 2))
        * np.sqrt(np.sum(br ** 2))
    )

    if denominator <= 1e-12:
        raise ValueError(
            "排序向量接近常量，无法计算 Spearman 相关。"
        )

    rho = np.sum(ar * br) / denominator

    return float(
        np.clip(
            rho,
            -1.0,
            1.0
        )
    )


def get_group_row(group, comparison):
    """获取指定 comparison 的 group-level 行。"""

    sub = group[
        group["comparison"].astype(str) == comparison
    ]

    if len(sub) == 0:
        return None

    return sub.iloc[0]


def get_group_mean_map_row(group_mean_map, comparison):
    """获取指定 comparison 的 group-mean map 行。"""

    sub = group_mean_map[
        group_mean_map["comparison"].astype(str) == comparison
    ]

    if len(sub) == 0:
        return None

    return sub.iloc[0]


def format_p_group(p, n_perm=None):
    """格式化图中显示的 group-level spin-test p 值。"""

    if p is None or not np.isfinite(p):
        return r"$p_{\mathrm{spin}} = \mathrm{NA}$"

    if n_perm is not None and np.isfinite(n_perm):
        min_p = 1.0 / (float(n_perm) + 1.0)

        if p <= min_p + 1e-12:
            return r"$p_{\mathrm{spin}} < 0.001$"

    if p < 0.001:
        return r"$p_{\mathrm{spin}} < 0.001$"

    return rf"$p_{{\mathrm{{spin}}}} = {p:.3f}$"


def set_uniform_title(ax, title):
    """设置统一的子图标题位置。"""

    ax.set_title(
        title,
        y=TITLE_Y,
        pad=TITLE_PAD,
        fontweight="bold"
    )


def add_panel_label(
    ax,
    label,
    x=PANEL_LABEL_X,
    y=PANEL_LABEL_Y
):
    """添加小写面板标签。"""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE + 1,
        fontweight="bold"
    )


def clean_axis(ax):
    """统一普通坐标轴格式。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        length=5,
        width=1.2
    )


# ============================================================
# 7. 数据读取
# ============================================================

def load_abide2_dataset():
    """读取 ABIDE II 数据。"""

    dataset_label = "ABIDE II"

    long_csv = os.path.join(
        ABIDE2_ROOT,
        "all_results",
        LONG_FILE
    )

    summary_csv = os.path.join(
        ABIDE2_ROOT,
        "all_results",
        SUMMARY_FILE
    )

    group_csv = ABIDE2_GROUP_CSV
    group_mean_map_csv = ABIDE2_GROUP_MEAN_MAP_CSV

    ec_group_map_csv = find_group_mean_gradient_map_file(
        ABIDE2_ROOT,
        EC_GROUP_MEAN_MAP_FILE,
        dataset_label
    )

    fc_group_map_csv = find_group_mean_gradient_map_file(
        ABIDE2_ROOT,
        FC_GROUP_MEAN_MAP_FILE,
        dataset_label
    )

    check_file(
        long_csv,
        f"{dataset_label} long CSV"
    )

    check_file(
        summary_csv,
        f"{dataset_label} summary CSV"
    )

    check_file(
        group_csv,
        f"{dataset_label} group-level subject-mean CSV"
    )

    check_file(
        group_mean_map_csv,
        f"{dataset_label} group-mean map spin CSV"
    )

    check_file(
        ec_group_map_csv,
        f"{dataset_label} EC group-mean map CSV"
    )

    check_file(
        fc_group_map_csv,
        f"{dataset_label} FC group-mean map CSV"
    )

    df_long = pd.read_csv(
        long_csv,
        encoding="utf-8-sig"
    )

    df_summary = pd.read_csv(
        summary_csv,
        encoding="utf-8-sig"
    )

    df_group = pd.read_csv(
        group_csv,
        encoding="utf-8-sig"
    )

    df_group_mean_map = pd.read_csv(
        group_mean_map_csv,
        encoding="utf-8-sig"
    )

    df_ec_map = pd.read_csv(
        ec_group_map_csv,
        encoding="utf-8-sig"
    )

    df_fc_map = pd.read_csv(
        fc_group_map_csv,
        encoding="utf-8-sig"
    )

    required_long = [
        "comparison",
        "spearman_r",
        "p_spin"
    ]

    required_summary = [
        "comparison",
        "spearman_mean",
        "spearman_sd",
        "p_spin_mean",
        "p_spin_median",
        "n_subjects"
    ]

    required_group = [
        "comparison",
        "group_spearman_mean",
        "group_p_spin",
        "subject_spearman_sd",
        "subject_spearman_min",
        "subject_spearman_max",
        "n_subjects",
        "n_perm",
        "tail",
        "spin_target"
    ]

    required_group_mean_map = [
        "comparison",
        "group_mean_map_spearman_r",
        "group_mean_map_p_spin",
        "n_subjects",
        "n_perm"
    ]

    required_ec_map = [
        "EC_G1",
        "EC_G2"
    ]

    required_fc_map = [
        "FC_G1",
        "FC_G2"
    ]

    dataframe_requirements = [
        (
            df_long,
            required_long,
            f"{dataset_label} long CSV"
        ),
        (
            df_summary,
            required_summary,
            f"{dataset_label} summary CSV"
        ),
        (
            df_group,
            required_group,
            f"{dataset_label} group-level CSV"
        ),
        (
            df_group_mean_map,
            required_group_mean_map,
            f"{dataset_label} group-mean map spin CSV"
        ),
        (
            df_ec_map,
            required_ec_map,
            f"{dataset_label} EC group-mean map CSV"
        ),
        (
            df_fc_map,
            required_fc_map,
            f"{dataset_label} FC group-mean map CSV"
        )
    ]

    for dataframe, required_columns, file_label in dataframe_requirements:
        missing_columns = [
            col
            for col in required_columns
            if col not in dataframe.columns
        ]

        if missing_columns:
            raise KeyError(
                f"{file_label} 缺少字段："
                + ", ".join(missing_columns)
            )

    # 只保留四种 cross-axis comparison
    df_long = df_long[
        df_long["comparison"].isin(COMPARISON_ORDER)
    ].copy()

    df_summary = df_summary[
        df_summary["comparison"].isin(COMPARISON_ORDER)
    ].copy()

    df_group = df_group[
        df_group["comparison"].isin(COMPARISON_ORDER)
    ].copy()

    df_group_mean_map = df_group_mean_map[
        df_group_mean_map["comparison"].isin(COMPARISON_ORDER)
    ].copy()

    if len(df_long) == 0:
        raise RuntimeError(
            f"{dataset_label} long CSV 中没有目标 comparison。"
        )

    if len(df_summary) == 0:
        raise RuntimeError(
            f"{dataset_label} summary CSV 中没有目标 comparison。"
        )

    if len(df_group) == 0:
        raise RuntimeError(
            f"{dataset_label} group-level CSV 中没有目标 comparison。"
        )

    if len(df_group_mean_map) == 0:
        raise RuntimeError(
            f"{dataset_label} group-mean map CSV 中没有目标 comparison。"
        )

    # 设置 comparison 顺序
    for dataframe in [
        df_long,
        df_summary,
        df_group,
        df_group_mean_map
    ]:
        dataframe["comparison"] = pd.Categorical(
            dataframe["comparison"],
            categories=COMPARISON_ORDER,
            ordered=True
        )

    df_long = (
        df_long
        .sort_values("comparison")
        .reset_index(drop=True)
    )

    df_summary = (
        df_summary
        .sort_values("comparison")
        .reset_index(drop=True)
    )

    df_group = (
        df_group
        .sort_values("comparison")
        .reset_index(drop=True)
    )

    df_group_mean_map = (
        df_group_mean_map
        .sort_values("comparison")
        .reset_index(drop=True)
    )

    # mean subject-level rho 矩阵
    rho_mat = np.full(
        (2, 2),
        np.nan,
        dtype=float
    )

    # group-level p_spin 矩阵
    p_mat = np.full(
        (2, 2),
        np.nan,
        dtype=float
    )

    n_mat = np.full(
        (2, 2),
        np.nan,
        dtype=float
    )

    matrix_mapping = {
        "EC_G1_vs_FC_G1": (0, 0),
        "EC_G1_vs_FC_G2": (0, 1),
        "EC_G2_vs_FC_G1": (1, 0),
        "EC_G2_vs_FC_G2": (1, 1)
    }

    for _, row in df_group.iterrows():
        comparison = str(row["comparison"])

        if comparison not in matrix_mapping:
            continue

        row_index, column_index = matrix_mapping[comparison]

        rho_mat[row_index, column_index] = float(
            row["group_spearman_mean"]
        )

        p_mat[row_index, column_index] = float(
            row["group_p_spin"]
        )

        n_mat[row_index, column_index] = float(
            row["n_subjects"]
        )

    return {
        "label": dataset_label,
        "long": df_long,
        "summary": df_summary,
        "group": df_group,
        "group_mean_map": df_group_mean_map,
        "ec_group_map": df_ec_map,
        "fc_group_map": df_fc_map,
        "rho_mat": rho_mat,
        "p_mat": p_mat,
        "n_mat": n_mat,
        "long_csv": long_csv,
        "summary_csv": summary_csv,
        "group_csv": group_csv,
        "group_mean_map_csv": group_mean_map_csv,
        "ec_group_map_csv": ec_group_map_csv,
        "fc_group_map_csv": fc_group_map_csv
    }


# ============================================================
# 8. 绘图函数
# ============================================================

def draw_boxplot(
    ax,
    data,
    group,
    title,
    ylabel
):
    """
    绘制四种 comparison 的逐被试 Spearman rho 分布。
    红色菱形表示 mean subject-level rho。
    """

    values = []
    means = []

    positions = np.arange(
        1,
        len(COMPARISON_ORDER) + 1
    )

    for comparison in COMPARISON_ORDER:
        vals = data.loc[
            data["comparison"].astype(str) == comparison,
            "spearman_r"
        ].dropna().to_numpy(dtype=float)

        values.append(vals)

        row = get_group_row(
            group,
            comparison
        )

        if row is None:
            means.append(np.nan)
        else:
            means.append(
                float(row["group_spearman_mean"])
            )

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": MEDIAN_COLOR,
            "linewidth": 2.4
        },
        boxprops={
            "facecolor": BOX_COLOR,
            "color": "black",
            "linewidth": 1.2
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.2
        },
        capprops={
            "color": "black",
            "linewidth": 1.2
        }
    )

    # 固定随机种子，使 jitter 每次运行一致
    rng = np.random.default_rng(42)

    for x_position, vals in zip(
        positions,
        values
    ):
        if len(vals) == 0:
            continue

        jitter = rng.normal(
            loc=0.0,
            scale=0.055,
            size=len(vals)
        )

        ax.scatter(
            np.full(len(vals), x_position) + jitter,
            vals,
            s=18,
            alpha=0.38,
            color=POINT_COLOR,
            edgecolors="none"
        )

    ax.scatter(
        positions,
        means,
        marker="D",
        s=90,
        color=MEAN_COLOR,
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
        label="Mean"
    )

    ax.axhline(
        0,
        color="gray",
        linewidth=1.1,
        linestyle="--"
    )

    ax.set_xticks(positions)

    ax.set_xticklabels(
        COMPARISON_LABELS,
        rotation=35,
        ha="right"
    )

    set_uniform_title(
        ax,
        title
    )

    ax.set_ylabel(ylabel)

    clean_axis(ax)

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=FONT_SIZE - 4
    )


def draw_heatmap(
    ax,
    group,
    rho_mat,
    p_mat,
    title
):
    """
    热图显示：
    mean subject-level rho = group_spearman_mean
    p_spin = group_p_spin
    """

    norm = TwoSlopeNorm(
        vmin=-1.0,
        vcenter=0.0,
        vmax=1.0
    )

    image = ax.imshow(
        rho_mat,
        cmap=HEATMAP_CMAP,
        norm=norm,
        aspect="equal"
    )

    set_uniform_title(
        ax,
        title
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels(
        HEATMAP_COLS,
        fontweight="bold"
    )

    ax.set_yticks([0, 1])

    ax.set_yticklabels(
        HEATMAP_ROWS,
        fontweight="bold"
    )

    ax.tick_params(
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0
    )

    reverse_mapping = {
        (0, 0): "EC_G1_vs_FC_G1",
        (0, 1): "EC_G1_vs_FC_G2",
        (1, 0): "EC_G2_vs_FC_G1",
        (1, 1): "EC_G2_vs_FC_G2"
    }

    for row_index in range(2):
        for column_index in range(2):
            rho = rho_mat[
                row_index,
                column_index
            ]

            p_value = p_mat[
                row_index,
                column_index
            ]

            comparison = reverse_mapping[
                (row_index, column_index)
            ]

            group_row = get_group_row(
                group,
                comparison
            )

            if group_row is None:
                n_perm = None
            else:
                n_perm = float(
                    group_row["n_perm"]
                )

            if np.isfinite(rho):
                rho_text = (
                    rf"$\rho_{{mean}} = {rho:.4f}$"
                )
            else:
                rho_text = (
                    r"$\rho_{mean} = \mathrm{NA}$"
                )

            if np.isfinite(rho) and abs(rho) >= 0.45:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                column_index,
                row_index,
                rho_text
                + "\n"
                + format_p_group(
                    p_value,
                    n_perm=n_perm
                ),
                ha="center",
                va="center",
                fontsize=FONT_SIZE - 5,
                fontweight="bold",
                color=text_color
            )

    ax.set_xticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 2, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="black",
        linestyle="-",
        linewidth=1.0
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def draw_group_map_scatter(
    ax,
    dataset,
    ec_axis,
    fc_axis,
    comparison,
    title,
    xlabel,
    ylabel
):
    """
    展示 group-average EC map 和 group-average FC map
    在 ROI 层面的对应关系。

    统计值优先读取 group_mean_map spin-test CSV。
    """

    ec_map = dataset["ec_group_map"]
    fc_map = dataset["fc_group_map"]
    group_mean_map = dataset["group_mean_map"]

    x_all = ec_map[
        ec_axis
    ].to_numpy(dtype=float)

    y_all = fc_map[
        fc_axis
    ].to_numpy(dtype=float)

    if x_all.shape != y_all.shape:
        raise ValueError(
            f"{ec_axis} 与 {fc_axis} ROI 数量不一致："
            f"{x_all.shape} vs {y_all.shape}"
        )

    finite_mask = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
    )

    x = x_all[finite_mask]
    y = y_all[finite_mask]

    if len(x) < 3:
        raise ValueError(
            f"{comparison} 有效 ROI 数量少于 3。"
        )

    group_mean_row = get_group_mean_map_row(
        group_mean_map,
        comparison
    )

    if group_mean_row is not None:
        rho = float(
            group_mean_row[
                "group_mean_map_spearman_r"
            ]
        )

        p_value = float(
            group_mean_row[
                "group_mean_map_p_spin"
            ]
        )

        n_perm = float(
            group_mean_row["n_perm"]
        )
    else:
        rho = spearman_correlation(
            x,
            y
        )

        p_value = np.nan
        n_perm = None

    ax.scatter(
        x,
        y,
        s=28,
        color=SCATTER_COLOR,
        alpha=0.85,
        edgecolors="none"
    )

    # 线性拟合线仅作为视觉辅助
    if np.nanstd(x) > 1e-12:
        coefficients = np.polyfit(
            x,
            y,
            1
        )

        xx = np.linspace(
            np.nanmin(x),
            np.nanmax(x),
            200
        )

        yy = (
            coefficients[0] * xx
            + coefficients[1]
        )

        ax.plot(
            xx,
            yy,
            color=SCATTER_LINE_COLOR,
            linewidth=2.0
        )

    set_uniform_title(
        ax,
        title
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    annotation = (
        "Group-average map"
        + "\n"
        + rf"$\rho_{{map}} = {rho:.4f}$"
        + "\n"
        + format_p_group(
            p_value,
            n_perm=n_perm
        )
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_SIZE - 5
    )

    clean_axis(ax)


# ============================================================
# 9. 保存数值表
# ============================================================

def save_numeric_table(dataset):
    """保存 ABIDE II 图中使用的数值。"""

    rows = []

    group = dataset["group"].copy()
    summary = dataset["summary"].copy()
    group_mean_map = dataset[
        "group_mean_map"
    ].copy()

    for comparison in COMPARISON_ORDER:
        group_sub = group[
            group["comparison"].astype(str)
            == comparison
        ]

        summary_sub = summary[
            summary["comparison"].astype(str)
            == comparison
        ]

        map_sub = group_mean_map[
            group_mean_map["comparison"].astype(str)
            == comparison
        ]

        if len(group_sub) == 0:
            continue

        group_row = group_sub.iloc[0]

        output_row = {
            "dataset": dataset["label"],
            "comparison": comparison,
            "main_text_group_statistic": str(
                group_row.get(
                    "group_statistic",
                    "mean_subject_spearman_r"
                )
            ),
            "main_text_mean_subject_level_rho": float(
                group_row[
                    "group_spearman_mean"
                ]
            ),
            "main_text_group_p_spin": float(
                group_row[
                    "group_p_spin"
                ]
            ),
            "subject_spearman_sd": float(
                group_row[
                    "subject_spearman_sd"
                ]
            ),
            "subject_spearman_min": float(
                group_row[
                    "subject_spearman_min"
                ]
            ),
            "subject_spearman_max": float(
                group_row[
                    "subject_spearman_max"
                ]
            ),
            "n_subjects": int(
                group_row["n_subjects"]
            ),
            "n_perm": int(
                group_row["n_perm"]
            ),
            "tail": str(
                group_row["tail"]
            ),
            "spin_target": str(
                group_row["spin_target"]
            )
        }

        if len(summary_sub) > 0:
            summary_row = summary_sub.iloc[0]

            output_row.update({
                "descriptive_subject_p_spin_mean_not_group_p": float(
                    summary_row[
                        "p_spin_mean"
                    ]
                ),
                "descriptive_subject_p_spin_median_not_group_p": float(
                    summary_row[
                        "p_spin_median"
                    ]
                )
            })

        if len(map_sub) > 0:
            map_row = map_sub.iloc[0]

            output_row.update({
                "illustrative_group_average_map_statistic_not_main_text": str(
                    map_row.get(
                        "group_statistic",
                        "spearman_of_group_mean_maps"
                    )
                ),
                "illustrative_group_average_map_rho_not_main_text": float(
                    map_row[
                        "group_mean_map_spearman_r"
                    ]
                ),
                "illustrative_group_average_map_p_spin_not_main_text": float(
                    map_row[
                        "group_mean_map_p_spin"
                    ]
                )
            })

        rows.append(output_row)

    output_dataframe = pd.DataFrame(rows)

    output_dataframe.to_csv(
        OUT_NUMERIC,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():
    # 仅加载 ABIDE II
    abide2 = load_abide2_dataset()

    # ABIDE II 被试数量
    n_subjects = int(
        abide2["group"]["n_subjects"].iloc[0]
    )

    # 2 行 × 2 列布局
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(19, 16),
        constrained_layout=False
    )

    # --------------------------------------------------------
    # a. 逐被试 Spearman rho 分布
    # --------------------------------------------------------

    draw_boxplot(
        ax=axes[0, 0],
        data=abide2["long"],
        group=abide2["group"],
        title=(
            f"ABIDE II | Subject ρ "
            f"(n = {n_subjects})"
        ),
        ylabel=r"Subject-level Spearman $\rho$"
    )

    add_panel_label(
        axes[0, 0],
        "a"
    )

    # --------------------------------------------------------
    # b. 2×2 correlation matrix
    # --------------------------------------------------------

    heatmap_image = draw_heatmap(
        ax=axes[0, 1],
        group=abide2["group"],
        rho_mat=abide2["rho_mat"],
        p_mat=abide2["p_mat"],
        title="ABIDE II | Matrix"
    )

    add_panel_label(
        axes[0, 1],
        "b"
    )

    # --------------------------------------------------------
    # c. EC-G1 vs FC-G1 group-average map
    # --------------------------------------------------------

    draw_group_map_scatter(
        ax=axes[1, 0],
        dataset=abide2,
        ec_axis="EC_G1",
        fc_axis="FC_G1",
        comparison="EC_G1_vs_FC_G1",
        title="ABIDE II | EC-G1 vs FC-G1",
        xlabel="Group-average EC-G1",
        ylabel="Group-average FC-G1"
    )

    add_panel_label(
        axes[1, 0],
        "c"
    )

    # --------------------------------------------------------
    # d. EC-G2 vs FC-G2 group-average map
    # --------------------------------------------------------

    draw_group_map_scatter(
        ax=axes[1, 1],
        dataset=abide2,
        ec_axis="EC_G2",
        fc_axis="FC_G2",
        comparison="EC_G2_vs_FC_G2",
        title="ABIDE II | EC-G2 vs FC-G2",
        xlabel="Group-average EC-G2",
        ylabel="Group-average FC-G2"
    )

    add_panel_label(
        axes[1, 1],
        "d"
    )

    # --------------------------------------------------------
    # 子图间距
    # --------------------------------------------------------

    plt.subplots_adjust(
        left=0.09,
        right=0.91,
        top=0.91,
        bottom=0.09,
        wspace=0.36,
        hspace=0.46
    )

    # --------------------------------------------------------
    # 热图色条
    # --------------------------------------------------------

    heatmap_position = axes[
        0,
        1
    ].get_position()

    colorbar_axis = fig.add_axes([
        heatmap_position.x1 + 0.015,
        heatmap_position.y0,
        0.015,
        heatmap_position.height
    ])

    colorbar = fig.colorbar(
        heatmap_image,
        cax=colorbar_axis
    )

    colorbar.set_label(
        r"Mean subject-level $\rho$",
        rotation=90,
        labelpad=15,
        fontsize=FONT_SIZE
    )

    colorbar.ax.tick_params(
        labelsize=FONT_SIZE
    )

    # --------------------------------------------------------
    # 保存图片
    # --------------------------------------------------------

    fig.savefig(
        OUT_PNG,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        OUT_SVG,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 保存数值表
    # --------------------------------------------------------

    save_numeric_table(
        abide2
    )

    # --------------------------------------------------------
    # 输出信息
    # --------------------------------------------------------

    print("\nABIDE II 2×2 绘图完成。")

    print("\n输出 PNG：")
    print(OUT_PNG)

    print("\n输出 SVG：")
    print(OUT_SVG)

    print("\n输出数值表：")
    print(OUT_NUMERIC)

    print("\nABIDE II long CSV：")
    print(abide2["long_csv"])

    print("\nABIDE II summary CSV：")
    print(abide2["summary_csv"])

    print("\nABIDE II group-level subject-mean CSV：")
    print(abide2["group_csv"])

    print("\nABIDE II group-mean map spin CSV：")
    print(abide2["group_mean_map_csv"])

    print("\nABIDE II EC group-mean gradient map：")
    print(abide2["ec_group_map_csv"])

    print("\nABIDE II FC group-mean gradient map：")
    print(abide2["fc_group_map_csv"])

    print(
        "\nABIDE II mean subject-level rho matrix："
    )

    print(
        pd.DataFrame(
            abide2["rho_mat"],
            index=HEATMAP_ROWS,
            columns=HEATMAP_COLS
        )
    )

    print(
        "\nABIDE II group-level p_spin matrix："
    )

    print(
        pd.DataFrame(
            abide2["p_mat"],
            index=HEATMAP_ROWS,
            columns=HEATMAP_COLS
        )
    )


if __name__ == "__main__":
    main()