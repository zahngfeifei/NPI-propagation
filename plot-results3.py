from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FONT_SIZE = 10
FIGURE_WIDTH_INCHES = 9.8
FIGURE_HEIGHT_INCHES = 8.8
PNG_DPI = 600

SYSTEM_ORDER = ["H1_sensory", "H2_attention", "H3_control", "H4_DMN"]
SYSTEM_LABELS = {
    "H1_sensory": "H1",
    "H2_attention": "H2",
    "H3_control": "H3",
    "H4_DMN": "H4",
}
SYSTEM_COLORS = {
    "H1_sensory": "#3690C0",
    "H2_attention": "#18A689",
    "H3_control": "#D98BB8",
    "H4_DMN": "#E69F00",
}
GROUP_COLORS = {"HC": "#0072B2", "ASD": "#D55E00"}

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MODEL_DIRECTORIES = {
    "unadjusted": SCRIPT_DIRECTORY / "ABIDE2_结果2-不控制系统差异",
    "adjusted": SCRIPT_DIRECTORY / "ABIDE2_结果2-控制系统",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
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
    axis.set_ylim(-0.235, 0.245)
    axis.set_xticks(np.arange(-1.5, 1.6, 0.5))
    axis.set_yticks(np.arange(-0.2, 0.21, 0.1))
    axis.set_xlabel("Within-subject EC G1 (G*)", labelpad=1)
    axis.set_ylabel("Early SEC slope\n(scaled ×1,000)", labelpad=2)
    axis.tick_params(direction="out", length=3, width=0.7, pad=2)


def add_system_scatter(
    axis: plt.Axes,
    observations: pd.DataFrame,
    group: str,
) -> None:
    group_observations = observations.loc[observations["Group"] == group]
    for system_name in SYSTEM_ORDER:
        system_observations = group_observations.loc[
            group_observations["system"] == system_name
        ]
        axis.scatter(
            system_observations["G_star"],
            system_observations["early_slope_scaled"],
            s=8,
            marker="s",
            color=SYSTEM_COLORS[system_name],
            alpha=0.38,
            linewidths=0,
            rasterized=False,
        )


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
        linewidth=0.9,
        zorder=4,
    )


def plot_group_by_system(
    axis: plt.Axes,
    model_results: dict[str, pd.DataFrame],
    group: str,
    model_title: str,
) -> None:
    add_system_scatter(axis, model_results["observations"], group)
    line_style = "-" if group == "HC" else (0, (3, 2))
    add_prediction_line(axis, model_results["predictions"], group, line_style)
    style_scatter_axis(axis)
    axis.set_title(f"{group}, {model_title}", loc="left", pad=4)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=SYSTEM_COLORS[system_name],
            markeredgewidth=0,
            markersize=5.5,
            label=SYSTEM_LABELS[system_name],
        )
        for system_name in SYSTEM_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#222222",
            linewidth=0.9,
            linestyle=line_style,
            label=f"{group} fit",
        )
    )
    axis.legend(
        handles=legend_handles,
        loc="lower right",
        borderaxespad=0.3,
        handlelength=1.8,
        labelspacing=0.3,
    )


def plot_groups_combined(
    axis: plt.Axes,
    model_results: dict[str, pd.DataFrame],
    model_title: str,
) -> None:
    observations = model_results["observations"]
    for group in ["HC", "ASD"]:
        group_observations = observations.loc[observations["Group"] == group]
        axis.scatter(
            group_observations["G_star"],
            group_observations["early_slope_scaled"],
            s=7,
            marker="s",
            color=GROUP_COLORS[group],
            alpha=0.28,
            linewidths=0,
            rasterized=False,
        )
    add_prediction_line(axis, model_results["predictions"], "HC", "-")
    add_prediction_line(
        axis, model_results["predictions"], "ASD", (0, (3, 2))
    )
    style_scatter_axis(axis)
    axis.set_title(model_title, loc="left", pad=4)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=GROUP_COLORS["HC"],
                markeredgewidth=0,
                markersize=5.5,
                label="HC",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=GROUP_COLORS["ASD"],
                markeredgewidth=0,
                markersize=5.5,
                label="ASD",
            ),
            Line2D([0], [0], color="#222222", linewidth=0.9, label="HC fit"),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                label="ASD fit",
            ),
        ],
        loc="lower right",
        borderaxespad=0.3,
        handlelength=1.8,
        labelspacing=0.3,
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
            markersize=5.8,
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
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
        linewidth=0.8,
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
            xytext=(10, 0),
            textcoords="offset points",
            color=GROUP_COLORS[group],
            ha="left",
            va="center",
        )

    axis.axhline(0, color="#BDBDBD", linewidth=0.6, linestyle=(0, (2, 2)))
    axis.set_xlim(-0.45, 1.65)
    axis.set_ylim(y_minimum, y_maximum)
    axis.set_xticks(x_positions, groups)
    axis.set_xlabel("Group", labelpad=2)
    axis.set_ylabel("Estimated G*-early\nSEC slope", labelpad=2)
    axis.set_title(model_title, loc="left", pad=4)
    axis.tick_params(direction="out", length=3, width=0.7, pad=2)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def create_figure() -> plt.Figure:
    unadjusted_results = load_model_results(MODEL_DIRECTORIES["unadjusted"])
    adjusted_results = load_model_results(MODEL_DIRECTORIES["adjusted"])

    figure, axes = plt.subplots(
        4,
        2,
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        constrained_layout=False,
    )
    plt.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.065,
        top=0.985,
        wspace=0.32,
        hspace=0.48,
    )

    plot_group_by_system(
        axes[0, 0], unadjusted_results, "ASD", "System-unadjusted model"
    )
    plot_group_by_system(
        axes[0, 1], unadjusted_results, "HC", "System-unadjusted model"
    )
    plot_groups_combined(axes[1, 0], unadjusted_results, "System-unadjusted model")
    plot_slope_summary(axes[1, 1], unadjusted_results["slopes"], "System-unadjusted model")

    plot_group_by_system(
        axes[2, 0], adjusted_results, "ASD", "System-adjusted model"
    )
    plot_group_by_system(
        axes[2, 1], adjusted_results, "HC", "System-adjusted model"
    )
    plot_groups_combined(axes[3, 0], adjusted_results, "System-adjusted model")
    plot_slope_summary(axes[3, 1], adjusted_results["slopes"], "System-adjusted model")

    for panel_label, axis in zip("abcdefgh", axes.flat):
        add_panel_label(axis, panel_label)
    return figure


def main() -> None:
    configure_matplotlib()
    figure = create_figure()
    svg_path = SCRIPT_DIRECTORY / "figure-results3.svg"
    png_path = SCRIPT_DIRECTORY / "figure-results3.png"
    figure.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    figure.savefig(
        png_path,
        format="png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Saved editable SVG: {svg_path}")
    print(f"Saved high-resolution PNG: {png_path}")


if __name__ == "__main__":
    main()
