from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANVAS_WIDTH_PX = 1800
CANVAS_HEIGHT_PX = 1350
DISPLAY_DPI = 150
FONT_SIZE_PX = 25
FONT_SIZE_PT = FONT_SIZE_PX * 72 / DISPLAY_DPI
ZERO_LINE_COLOR = "#707070"
PRIMARY_COLOR = "#3B6FB6"
CONTROL_COLOR = "#A7A9AC"
ACCENT_COLOR = "#D66A4A"
REPLICATION_COLOR = "#3D8B74"


def configure_plot_style() -> None:
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
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
        }
    )


def add_panel_title(axis: plt.Axes, panelLetter: str, title: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold", pad=14)
    axis.panelLetterArtist = axis.text(
        0,
        1,
        panelLetter,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontweight="bold",
    )


def get_y_axis_text_left_edge(axis: plt.Axes, renderer) -> float:
    yAxisTextArtists = [label for label in axis.get_yticklabels() if label.get_visible()]
    if axis.yaxis.label.get_text():
        yAxisTextArtists.append(axis.yaxis.label)
    visibleBounds = [
        artist.get_window_extent(renderer=renderer)
        for artist in yAxisTextArtists
        if artist.get_text()
    ]
    return min(bounds.x0 for bounds in visibleBounds)


def align_headers_and_adjust_widths(figure: plt.Figure, axes: np.ndarray) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverseFigureTransform = figure.transFigure.inverted()

    for columnIndex in range(axes.shape[1]):
        columnAxes = axes[:, columnIndex]
        originalBounds = [axis.get_position() for axis in columnAxes]
        textLeftFigureXs = [
            inverseFigureTransform.transform((get_y_axis_text_left_edge(axis, renderer), 0))[0]
            for axis in columnAxes
        ]
        yTextInsets = [
            axisBounds.x0 - textLeftFigureX
            for axisBounds, textLeftFigureX in zip(originalBounds, textLeftFigureXs)
        ]
        sharedTextLeftFigureX = min(textLeftFigureXs)
        sharedRightFigureX = min(axisBounds.x1 for axisBounds in originalBounds)

        for axis, axisBounds, yTextInset in zip(columnAxes, originalBounds, yTextInsets):
            adjustedLeftFigureX = sharedTextLeftFigureX + yTextInset
            axis.set_position(
                [
                    adjustedLeftFigureX,
                    axisBounds.y0,
                    sharedRightFigureX - adjustedLeftFigureX,
                    axisBounds.height,
                ]
            )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    for axis in axes.ravel():
        axisBounds = axis.get_position()
        textLeftFigureX = inverseFigureTransform.transform(
            (get_y_axis_text_left_edge(axis, renderer), 0)
        )[0]
        titleAxisX = (textLeftFigureX - axisBounds.x0) / axisBounds.width
        leftTitle = axis._left_title
        leftTitle.set_position((titleAxisX, 1.0))
        leftTitle.set_horizontalalignment("left")
        panelGapAxes = 12 / axis.get_window_extent(renderer=renderer).width
        axis.panelLetterArtist.set_transform(leftTitle.get_transform())
        axis.panelLetterArtist.set_position((titleAxisX - panelGapAxes, 1.0))


def draw_forest_plot(
    axis: plt.Axes,
    estimates: np.ndarray,
    ciLow: np.ndarray,
    ciHigh: np.ndarray,
    labels: list[str],
    colors: list[str],
) -> None:
    yPositions = np.arange(len(labels))[::-1]
    axis.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.5, linestyle="--", zorder=0)
    for yPosition, estimate, lowerBound, upperBound, color in zip(
        yPositions, estimates, ciLow, ciHigh, colors
    ):
        axis.errorbar(
            estimate,
            yPosition,
            xerr=[[estimate - lowerBound], [upperBound - estimate]],
            fmt="o",
            markersize=11,
            color=color,
            ecolor=color,
            elinewidth=3,
            capsize=6,
            capthick=2,
            zorder=3,
        )
    axis.set_yticks(yPositions, labels)
    axis.set_xlabel(r"ASD−HC interaction, $\beta$ ($\times 10^{-3}$)")
    axis.grid(axis="x", color="#E6E6E6", linewidth=1, zorder=0)


def main() -> None:
    configure_plot_style()
    rootDirectory = Path(__file__).resolve().parent

    homologousControlDirectory = rootDirectory / "补充_EC-G2 同源负对照" / "ABIDE1"
    fcControlDirectory = rootDirectory / "补充_控制经典 FC hierarchy" / "ABIDE1"
    crossCohortDirectory = rootDirectory / "补充_跨队列固定模板同源检查"

    gradientComparison = pd.read_csv(
        homologousControlDirectory / "g1-g2-interaction-comparison.csv"
    )
    bootstrapDifference = pd.read_csv(
        homologousControlDirectory / "bootstrap-g1-minus-g2-distribution.csv"
    )
    bootstrapDifferenceSummary = pd.read_csv(
        homologousControlDirectory / "bootstrap-g1-minus-g2-summary.csv"
    ).iloc[0]
    fcAdjustmentComparison = pd.read_csv(
        fcControlDirectory / "interaction-adjustment-comparison.csv"
    )
    crossCohortSummary = pd.read_csv(crossCohortDirectory / "cross-direction-summary.csv")

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(CANVAS_WIDTH_PX / DISPLAY_DPI, CANVAS_HEIGHT_PX / DISPLAY_DPI),
        dpi=DISPLAY_DPI,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    axisA, axisB, axisC, axisD = axes.ravel()

    gradientScale = 1000
    draw_forest_plot(
        axisA,
        gradientComparison["beta"].to_numpy() * gradientScale,
        gradientComparison["ci95Low"].to_numpy() * gradientScale,
        gradientComparison["ci95High"].to_numpy() * gradientScale,
        ["EC-G1", "EC-G2"],
        [PRIMARY_COLOR, CONTROL_COLOR],
    )
    add_panel_title(axisA, "a", "Homologous gradient control")

    bootstrapValues = bootstrapDifference["deltaBetaG1MinusG2"].to_numpy() * gradientScale
    observedDifference = bootstrapDifferenceSummary["observedDeltaBeta"] * gradientScale
    axisB.hist(
        bootstrapValues,
        bins=42,
        color=PRIMARY_COLOR,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    axisB.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.5, linestyle="--")
    axisB.axvline(observedDifference, color=ACCENT_COLOR, linewidth=3)
    axisB.set_xlabel(r"$\beta_{G1}-\beta_{G2}$ ($\times 10^{-3}$)")
    axisB.set_ylabel("Bootstrap count")
    add_panel_title(axisB, "b", "G1−G2 bootstrap contrast")

    fcLabels = ["EC-G1 (raw)", "EC-G1 (adjusted)", "FC-G1 (adjusted)"]
    draw_forest_plot(
        axisC,
        fcAdjustmentComparison["beta"].to_numpy() * gradientScale,
        fcAdjustmentComparison["ci95Low"].to_numpy() * gradientScale,
        fcAdjustmentComparison["ci95High"].to_numpy() * gradientScale,
        fcLabels,
        [PRIMARY_COLOR, ACCENT_COLOR, CONTROL_COLOR],
    )
    add_panel_title(axisC, "c", "FC hierarchy adjustment")

    crossCohortLabels = ["A1 template → A2", "A2 template → A1"]
    draw_forest_plot(
        axisD,
        crossCohortSummary["primaryBeta"].to_numpy() * gradientScale,
        crossCohortSummary["bootstrapCi95Low"].to_numpy() * gradientScale,
        crossCohortSummary["bootstrapCi95High"].to_numpy() * gradientScale,
        crossCohortLabels,
        [REPLICATION_COLOR, REPLICATION_COLOR],
    )
    add_panel_title(axisD, "d", "Cross-cohort replication")

    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.94, wspace=0.40, hspace=0.48)
    align_headers_and_adjust_widths(figure, axes)

    outputStem = rootDirectory / "s8-robustness-evidence"
    figure.savefig(outputStem.with_suffix(".png"), dpi=DISPLAY_DPI, facecolor="white")
    figure.savefig(outputStem.with_suffix(".svg"), facecolor="white")
    figure.savefig(outputStem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
