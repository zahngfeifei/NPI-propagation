from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


BASE_DIR = Path(__file__).resolve().parent
NEGATIVE_CONTROL_DIR = BASE_DIR / "补充_EC-G2 同源负对照" / "ABIDE1"
FC_CONTROL_DIR = BASE_DIR / "补充_控制经典 FC hierarchy" / "ABIDE1"
NEGATIVE_CONTROL_RESULTS_PATH = NEGATIVE_CONTROL_DIR / "ec-g2-model-parameters.csv"
JOINT_MODEL_RESULTS_PATH = FC_CONTROL_DIR / "fc-adjusted-model-parameters.csv"
SOURCE_DATA_PATH = BASE_DIR / "S7-source-data.csv"
SVG_OUTPUT_PATH = BASE_DIR / "S7-forest-plot.svg"
PNG_PREVIEW_PATH = BASE_DIR / "S7-forest-plot-preview.png"

FONT_SIZE_PT = 12
FIGURE_WIDTH_INCHES = 11.5
FIGURE_HEIGHT_INCHES = 2.75
X_AXIS_MIN = -0.03
X_AXIS_MAX = 0.03
X_AXIS_TICKS = [-0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03]

NEUTRAL_COLOR = "#6F6F6F"
EC_COLOR = "#0057B8"
FC_COLOR = "#F05A00"
AXIS_COLOR = "#333333"


def configure_figure_style() -> FontProperties:
    arial_font_path = Path("C:/Windows/Fonts/arial.ttf")
    if not arial_font_path.exists():
        raise FileNotFoundError(
            "Arial font was not found at C:/Windows/Fonts/arial.ttf."
        )

    arial_font = FontProperties(family="Arial", size=FONT_SIZE_PT)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
        }
    )
    return arial_font


def select_result_row(results: pd.DataFrame, term: str) -> pd.Series:
    matching_rows = results.loc[results["term"] == term]
    if len(matching_rows) != 1:
        raise ValueError(f"Expected one result for {term!r}, found {len(matching_rows)}.")
    return matching_rows.iloc[0]


def format_fixed(value: float, decimal_places: int = 4) -> str:
    return f"{value:.{decimal_places}f}"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return f"{value:.5f}".rstrip("0")
    if value < 0.01:
        return f"{value:.5f}".rstrip("0")
    return f"{value:.3f}"


def format_ci(result_row: pd.Series) -> str:
    return (
        f"[{format_fixed(result_row['ci95Low'])}, "
        f"{format_fixed(result_row['ci95High'])}]"
    )


def style_forest_axis(axis: plt.Axes) -> None:
    axis.set_xlim(X_AXIS_MIN, X_AXIS_MAX)
    axis.set_xticks(X_AXIS_TICKS)
    axis.set_ylim(-0.65, 1.65)
    axis.set_yticks([])
    axis.axvline(0, color=AXIS_COLOR, linestyle=(0, (4, 4)), linewidth=0.8, zorder=0)
    axis.spines["bottom"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="x", width=0.8, length=3, color=AXIS_COLOR, pad=2)
    axis.set_xlabel("Coefficient (b)", labelpad=3)


def draw_estimate(
    axis: plt.Axes,
    result_row: pd.Series,
    y_position: float,
    color: str,
) -> None:
    lower_error = result_row["beta"] - result_row["ci95Low"]
    upper_error = result_row["ci95High"] - result_row["beta"]
    axis.errorbar(
        result_row["beta"],
        y_position,
        xerr=[[lower_error], [upper_error]],
        fmt="o",
        markersize=5.2,
        markerfacecolor=color,
        markeredgecolor=color,
        ecolor=color,
        elinewidth=1.1,
        capsize=3,
        capthick=1.1,
        zorder=3,
    )


def draw_panel_a(
    figure: plt.Figure,
    panel_spec: mpl.gridspec.SubplotSpec,
    result_row: pd.Series,
    arial_font: FontProperties,
) -> None:
    panel_grid = panel_spec.subgridspec(2, 2, height_ratios=[0.27, 0.73], width_ratios=[1.1, 2.1])
    title_axis = figure.add_subplot(panel_grid[0, :])
    text_axis = figure.add_subplot(panel_grid[1, 0])
    forest_axis = figure.add_subplot(panel_grid[1, 1])

    title_axis.axis("off")
    title_axis.text(-0.03, 0.84, "a", fontproperties=arial_font, fontweight="bold", va="top")
    title_axis.text(
        0.06,
        0.84,
        "EC-G2 homologous negative-control analysis (ABIDE I)",
        fontproperties=arial_font,
        fontweight="bold",
        va="top",
    )


    text_axis.axis("off")
    text_axis.text(
        0.43,
        0.93,
        "EC-G2 × Group\ninteraction",
        fontproperties=arial_font,
        fontweight="bold",
        ha="center",
        va="top",
    )
    text_axis.text(
        0.43,
        0.55,
        f"Coefficient (b): {format_fixed(result_row['beta'])}",
        fontproperties=arial_font,
        ha="center",
    )
    text_axis.text(
        0.43,
        0.36,
        f"95% CI: {format_ci(result_row)}",
        fontproperties=arial_font,
        ha="center",
    )
    text_axis.text(
        0.43,
        0.17,
        f"P value: {format_p_value(result_row['pValue'])}",
        fontproperties=arial_font,
        ha="center",
    )

    style_forest_axis(forest_axis)
    draw_estimate(forest_axis, result_row, y_position=0.55, color=NEUTRAL_COLOR)


def draw_panel_b(
    figure: plt.Figure,
    panel_spec: mpl.gridspec.SubplotSpec,
    ec_result_row: pd.Series,
    fc_result_row: pd.Series,
    arial_font: FontProperties,
) -> None:
    panel_grid = panel_spec.subgridspec(
        2,
        3,
        height_ratios=[0.27, 0.73],
        width_ratios=[0.95, 1.55, 2.05],
        wspace=0.03,
    )
    title_axis = figure.add_subplot(panel_grid[0, :])
    label_axis = figure.add_subplot(panel_grid[1, 0])
    forest_axis = figure.add_subplot(panel_grid[1, 1])
    table_axis = figure.add_subplot(panel_grid[1, 2])

    title_axis.axis("off")
    title_axis.text(-0.01, 0.84, "b", fontproperties=arial_font, fontweight="bold", va="top")
    title_axis.text(
        0.08,
        0.84,
        "Joint model of EC-G1 and FC-G1 interactions (ABIDE I)",
        fontproperties=arial_font,
        fontweight="bold",
        va="top",
    )


    label_axis.set_xlim(0, 1)
    label_axis.set_ylim(-0.65, 1.65)
    label_axis.axis("off")
    label_axis.text(
        0.94,
        1.05,
        "EC-G1 × Group\ninteraction",
        fontproperties=arial_font,
        fontweight="bold",
        color=EC_COLOR,
        ha="right",
        va="center",
    )
    label_axis.text(
        0.94,
        0.05,
        "FC-G1 × Group\ninteraction",
        fontproperties=arial_font,
        fontweight="bold",
        color=FC_COLOR,
        ha="right",
        va="center",
    )

    style_forest_axis(forest_axis)
    draw_estimate(forest_axis, ec_result_row, y_position=1.05, color=EC_COLOR)
    draw_estimate(forest_axis, fc_result_row, y_position=0.05, color=FC_COLOR)

    table_axis.set_xlim(0, 1)
    table_axis.set_ylim(-0.65, 1.65)
    table_axis.axis("off")
    column_positions = [0.18, 0.56, 0.90]
    column_headers = ["Coefficient (b)", "95% CI", "P value"]
    for x_position, header in zip(column_positions, column_headers):
        table_axis.text(
            x_position,
            1.54,
            header,
            fontproperties=arial_font,
            fontweight="bold",
            ha="center",
            va="center",
        )

    for result_row, y_position, color in [
        (ec_result_row, 1.05, EC_COLOR),
        (fc_result_row, 0.05, FC_COLOR),
    ]:
        table_axis.text(
            column_positions[0],
            y_position,
            format_fixed(result_row["beta"]),
            fontproperties=arial_font,
            color=color,
            ha="center",
            va="center",
        )
        table_axis.text(
            column_positions[1],
            y_position,
            format_ci(result_row),
            fontproperties=arial_font,
            color=color,
            ha="center",
            va="center",
        )
        table_axis.text(
            column_positions[2],
            y_position,
            format_p_value(result_row["pValue"]),
            fontproperties=arial_font,
            color=color,
            ha="center",
            va="center",
        )


def build_source_data(
    negative_control_row: pd.Series,
    ec_result_row: pd.Series,
    fc_result_row: pd.Series,
) -> pd.DataFrame:
    source_rows = []
    for panel, label, result_row in [
        ("a", "EC-G2 × Group interaction", negative_control_row),
        ("b", "EC-G1 × Group interaction", ec_result_row),
        ("b", "FC-G1 × Group interaction", fc_result_row),
    ]:
        source_rows.append(
            {
                "panel": panel,
                "label": label,
                "coefficient_b": result_row["beta"],
                "ci95_low": result_row["ci95Low"],
                "ci95_high": result_row["ci95High"],
                "p_value": result_row["pValue"],
            }
        )
    return pd.DataFrame(source_rows)


def main() -> None:
    arial_font = configure_figure_style()
    negative_control_results = pd.read_csv(NEGATIVE_CONTROL_RESULTS_PATH)
    joint_model_results = pd.read_csv(JOINT_MODEL_RESULTS_PATH)

    negative_control_row = select_result_row(
        negative_control_results, "EC_G2_star:Group[T.ASD]"
    )
    ec_result_row = select_result_row(
        joint_model_results, "EC_G1_star:Group[T.ASD]"
    )
    fc_result_row = select_result_row(
        joint_model_results, "FC_G1_star:Group[T.ASD]"
    )

    source_data = build_source_data(
        negative_control_row, ec_result_row, fc_result_row
    )
    source_data.to_csv(SOURCE_DATA_PATH, index=False, encoding="utf-8-sig")

    figure = plt.figure(
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES), facecolor="white"
    )
    main_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[0.90, 1.35],
        left=0.025,
        right=0.985,
        top=0.965,
        bottom=0.17,
        wspace=0.08,
    )
    draw_panel_a(figure, main_grid[0], negative_control_row, arial_font)
    draw_panel_b(figure, main_grid[1], ec_result_row, fc_result_row, arial_font)

    figure.savefig(SVG_OUTPUT_PATH, format="svg", facecolor="white")
    figure.savefig(PNG_PREVIEW_PATH, dpi=300, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
