from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FONT_FAMILY = "Arial"

# 固定导出画布尺寸：宽度严格为 1800 px。
# 高度设为 1350 px，以保证 2×2 布局下内容清晰可读。
FIGURE_WIDTH_PX = 1800
FIGURE_HEIGHT_PX = 1350
PNG_DPI = 100
FIGURE_WIDTH_INCHES = FIGURE_WIDTH_PX / PNG_DPI
FIGURE_HEIGHT_INCHES = FIGURE_HEIGHT_PX / PNG_DPI

# Matplotlib 字号单位为 point；在 100 dpi 下，25 px 对应 18 pt。
FONT_SIZE_PX = 25
FONT_SIZE = FONT_SIZE_PX * 72.0 / PNG_DPI

# 面板字母、标题和 y 轴标题的强制对齐参数
# 通过固定 y 轴标题坐标和标题起始坐标，实现精确左对齐。
PANEL_LABEL_X = -0.29
TITLE_AND_YLABEL_X = -0.14
TITLE_X = -0.17
PANEL_HEADER_Y = 1.18
TITLE_LINE_SPACING = 1.05

# 仅按最左/最右非白色像素校正横向布局，不裁剪顶部或底部。
EXPORT_PADDING_INCHES = 0.06
HORIZONTAL_FIT_MAX_ITERATIONS = 8
HORIZONTAL_FIT_TOLERANCE_PX = 0

SCATTER_YLABEL_X = TITLE_AND_YLABEL_X
SLOPE_YLABEL_X = TITLE_AND_YLABEL_X

GROUP_COLORS = {"HC": "#0072B2", "ASD": "#D55E00"}

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORIES = {
    "unadjusted": SCRIPT_DIRECTORY / "ABIDE2_结果2-不控制系统差异",
    "adjusted": SCRIPT_DIRECTORY / "ABIDE2_结果2-控制系统",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY],
            "font.size": FONT_SIZE,
            "font.weight": "normal",
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_SIZE,
            "axes.labelweight": "normal",
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def load_model_results(model_directory: Path) -> dict[str, pd.DataFrame]:
    required_files = {
        "observations": "results2_data_used.csv",
        "predictions": "results2_prediction_lines.csv",
        "slopes": "results2_simple_slopes.csv",
    }
    missing_files = [
        filename
        for filename in required_files.values()
        if not (model_directory / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required result files in {model_directory}: {missing_files}"
        )
    return {
        result_name: pd.read_csv(model_directory / filename)
        for result_name, filename in required_files.items()
    }


def style_scatter_axis(axis: plt.Axes) -> None:
    axis.set_xlim(-1.85, 1.75)
    axis.set_ylim(-0.235, 0.215)
    axis.set_xticks(np.arange(-1.5, 1.6, 0.5))
    axis.set_yticks(np.arange(-0.2, 0.21, 0.1))
    axis.set_xlabel("Within-subject EC G1 (G*)", labelpad=8)
    axis.set_ylabel("Early SEC slope\n(scaled ×1,000)", labelpad=10)
    axis.yaxis.set_label_coords(SCATTER_YLABEL_X, 0.5)
    axis.tick_params(direction="out", length=7, width=1.4, pad=6)


def add_prediction_line(
    axis: plt.Axes,
    predictions: pd.DataFrame,
    group: str,
    linestyle: str,
    color: str = "#222222",
) -> None:
    group_predictions = predictions.loc[predictions["Group"] == group].sort_values(
        "G_star"
    )
    axis.plot(
        group_predictions["G_star"],
        group_predictions["early_slope_pred"],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        zorder=4,
    )


def add_axis_title(
    axis: plt.Axes,
    title: str,
    title_x: float = TITLE_X,
    title_y: float = PANEL_HEADER_Y,
) -> None:
    title_artist = axis.text(
        title_x,
        title_y,
        title,
        transform=axis.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
        linespacing=TITLE_LINE_SPACING,
        clip_on=False,
    )
    title_artist.set_gid("panel-title")


def align_titles_with_y_axis_labels(
    figure: plt.Figure,
    axes: np.ndarray,
) -> None:
    """将标题与 y 轴标题的实际渲染左边缘精确对齐。"""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for axis in np.asarray(axes).flat:
        title_artist = next(
            artist
            for artist in axis.texts
            if artist.get_gid() == "panel-title"
        )
        y_label_left_px = axis.yaxis.label.get_window_extent(renderer).x0
        aligned_title_x = axis.transAxes.inverted().transform(
            (y_label_left_px, 0)
        )[0]
        title_artist.set_x(aligned_title_x)

    figure.canvas.draw()


def plot_groups_combined(
    axis: plt.Axes,
    model_results: dict[str, pd.DataFrame],
    model_title: str,
    title_x: float = TITLE_X,
    title_y: float = PANEL_HEADER_Y,
) -> None:
    observations = model_results["observations"]
    for group in ["HC", "ASD"]:
        group_observations = observations.loc[observations["Group"] == group]
        axis.scatter(
            group_observations["G_star"],
            group_observations["early_slope_scaled"],
            s=28,
            marker="s",
            color=GROUP_COLORS[group],
            alpha=0.28,
            linewidths=0,
            rasterized=False,
        )

    add_prediction_line(axis, model_results["predictions"], "HC", "-")
    add_prediction_line(
        axis,
        model_results["predictions"],
        "ASD",
        (0, (3, 2)),
    )

    style_scatter_axis(axis)
    add_axis_title(axis, model_title, title_x, title_y)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=GROUP_COLORS["HC"],
                markeredgewidth=0,
                markersize=10,
                label="HC",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=GROUP_COLORS["ASD"],
                markeredgewidth=0,
                markersize=10,
                label="ASD",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=2.0,
                label="HC fit",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=2.0,
                linestyle=(0, (3, 2)),
                label="ASD fit",
            ),
        ],
        loc="lower right",
        borderaxespad=0.5,
        handlelength=2.0,
        labelspacing=0.4,
    )


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "P < 0.001"
    if p_value < 0.01:
        return f"P = {p_value:.5f}"
    return f"P = {p_value:.3f}"


def plot_slope_summary(
    axis: plt.Axes,
    slopes: pd.DataFrame,
    model_title: str,
    title_x: float = TITLE_X,
    title_y: float = PANEL_HEADER_Y,
) -> None:
    group_slopes = slopes.set_index("Group")
    groups = ["HC", "ASD"]
    x_positions = np.arange(len(groups))
    estimates = np.array([group_slopes.loc[group, "slope_beta"] for group in groups])
    ci_low = np.array([group_slopes.loc[group, "ci_low"] for group in groups])
    ci_high = np.array([group_slopes.loc[group, "ci_high"] for group in groups])
    asymmetric_errors = np.vstack([estimates - ci_low, ci_high - estimates])

    for index, group in enumerate(groups):
        axis.errorbar(
            x_positions[index],
            estimates[index],
            yerr=asymmetric_errors[:, index : index + 1],
            fmt="o",
            color=GROUP_COLORS[group],
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor=GROUP_COLORS[group],
            markersize=10,
            elinewidth=1.8,
            capsize=6,
            capthick=1.8,
            zorder=3,
        )

    interval_minimum = float(ci_low.min())
    interval_maximum = float(ci_high.max())
    interval_span = max(interval_maximum - interval_minimum, 0.006)
    y_minimum = max(0.0, interval_minimum - 0.55 * interval_span)
    bracket_y = interval_maximum + 0.55 * interval_span
    bracket_height = 0.13 * interval_span
    y_maximum = bracket_y + 0.55 * interval_span

    axis.plot(
        [x_positions[0], x_positions[0], x_positions[1], x_positions[1]],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color="#555555",
        linewidth=1.5,
        clip_on=False,
    )

    interaction_p_value = float(group_slopes.loc["ASD_minus_HC", "p_value"])
    axis.text(
        np.mean(x_positions),
        bracket_y + bracket_height + 0.08 * interval_span,
        format_p_value(interaction_p_value),
        ha="center",
        va="bottom",
    )

    for index, group in enumerate(groups):
        axis.annotate(
            f"β = {estimates[index]:.4f}",
            xy=(x_positions[index], estimates[index]),
            xytext=(14, 0),
            textcoords="offset points",
            color=GROUP_COLORS[group],
            ha="left",
            va="center",
        )

    axis.axhline(0, color="#BDBDBD", linewidth=1.2, linestyle=(0, (2, 2)))
    axis.set_xlim(-0.45, 1.65)
    axis.set_ylim(y_minimum, y_maximum)
    axis.set_xticks(x_positions, groups)
    axis.set_xlabel("Group", labelpad=10)
    axis.set_ylabel("Estimated G*-early\nSEC slope", labelpad=10)
    axis.yaxis.set_label_coords(SLOPE_YLABEL_X, 0.5)
    add_axis_title(axis, model_title, title_x, title_y)
    axis.tick_params(direction="out", length=7, width=1.4, pad=6)


def add_panel_label(
    axis: plt.Axes,
    label: str,
    label_x: float = PANEL_LABEL_X,
    label_y: float = PANEL_HEADER_Y,
) -> None:
    axis.text(
        label_x,
        label_y,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def get_horizontal_content_bounds(figure: plt.Figure) -> tuple[int, int, int]:
    """返回画布中最左和最右非白色像素边界，仅检测水平方向。"""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # 背景为纯白色；任何非白色且非完全透明像素都视为图形内容。
    content_mask = np.any(rgb != 255, axis=2) & (alpha != 0)
    occupied_columns = np.flatnonzero(content_mask.any(axis=0))

    canvas_width = int(rgba.shape[1])
    if occupied_columns.size == 0:
        return 0, canvas_width, canvas_width

    x_left = int(occupied_columns[0])
    x_right_exclusive = int(occupied_columns[-1]) + 1
    return x_left, x_right_exclusive, canvas_width


def fit_axes_to_full_canvas_width(
    figure: plt.Figure,
    axes: np.ndarray,
) -> None:
    """
    将所有子图做统一水平仿射变换，使最左和最右内容像素
    分别落在 1800 px 画布的第 0 列和最后一列。

    该过程不使用 bbox_inches='tight'，也不裁剪上下边界。
    """
    flat_axes = list(np.asarray(axes).flat)

    for _ in range(HORIZONTAL_FIT_MAX_ITERATIONS):
        x_left, x_right, canvas_width = get_horizontal_content_bounds(figure)

        left_error = x_left
        right_error = canvas_width - x_right
        if (
            left_error <= HORIZONTAL_FIT_TOLERANCE_PX
            and right_error <= HORIZONTAL_FIT_TOLERANCE_PX
        ):
            break

        left_fraction = x_left / canvas_width
        right_fraction = x_right / canvas_width
        occupied_fraction = right_fraction - left_fraction

        if occupied_fraction <= 0:
            break

        for axis in flat_axes:
            position = axis.get_position()
            new_x0 = (position.x0 - left_fraction) / occupied_fraction
            new_width = position.width / occupied_fraction
            axis.set_position(
                [new_x0, position.y0, new_width, position.height]
            )

    # 最终绘制一次，确保保存时使用校正后的布局。
    figure.canvas.draw()


def create_figure() -> plt.Figure:
    unadjusted_results = load_model_results(MODEL_DIRECTORIES["unadjusted"])
    adjusted_results = load_model_results(MODEL_DIRECTORIES["adjusted"])

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        dpi=PNG_DPI,
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.11,
        right=0.992,
        bottom=0.075,
        top=0.90,
        wspace=0.38,
        hspace=0.48,
    )

    # a：未控制系统差异模型的联合散点与拟合线
    plot_groups_combined(
        axes[0, 0],
        unadjusted_results,
        "Combined scatter — System-unadjusted model\n(early_slope_scaled ~ G_star * Group + Age + Sex_bin + FIQ)",
    )
    # b：未控制系统差异模型的斜率估计比较
    plot_slope_summary(
        axes[0, 1],
        unadjusted_results["slopes"],
        "Slope comparison — System-unadjusted model\n(early_slope_scaled ~ G_star * Group + Age + Sex_bin + FIQ)",
    )

    # c：控制系统差异模型的联合散点与拟合线
    plot_groups_combined(
        axes[1, 0],
        adjusted_results,
        "Combined scatter — System-adjusted model\n(early_slope_scaled ~ G_star * Group + C(system)\n+ Age + Sex_bin + FIQ)",
        title_y=1.25,
    )
    # d：控制系统差异模型的斜率估计比较
    plot_slope_summary(
        axes[1, 1],
        adjusted_results["slopes"],
        "Slope comparison — System-adjusted model\n(early_slope_scaled ~ G_star * Group + C(system)\n+ Age + Sex_bin + FIQ)",
        title_y=1.25,
    )

    for panel_index, (panel_label, axis) in enumerate(zip("abcd", axes.flat)):
        label_x = -0.30 if panel_index % 2 else PANEL_LABEL_X
        label_y = 1.25 if panel_index >= 2 else PANEL_HEADER_Y
        add_panel_label(axis, panel_label, label_x, label_y)

    align_titles_with_y_axis_labels(figure, axes)

    # 只校正左右边界：最左/最右内容像素即为画布边界。
    # 不对顶部和底部进行任何额外裁剪。
    return figure


def main() -> None:
    configure_matplotlib()
    figure = create_figure()

    svg_path = SCRIPT_DIRECTORY / "figure-results3-14.svg"
    png_path = SCRIPT_DIRECTORY / "figure-results3-14.png"

    # 不使用 bbox_inches="tight"：保持 1800 px 固定画布，且不裁剪上下边界。
    figure.savefig(
        svg_path,
        format="svg",
        dpi=PNG_DPI,
        facecolor="white",
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=PNG_DPI,
        facecolor="white",
    )

    plt.close(figure)
    print(f"Saved editable SVG: {svg_path}")
    print(f"Saved high-resolution PNG: {png_path}")


if __name__ == "__main__":
    main()
