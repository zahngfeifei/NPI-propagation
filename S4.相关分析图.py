from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
ABIDE_ROOTS = {
    "ABIDE I": SCRIPT_DIR / "ABIDE1_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-HC_ASD分组",
    "ABIDE II": SCRIPT_DIR / "ABIDE2_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-HC_ASD分组",
}
GROUP_SUMMARY_ROOTS = {
    "ABIDE I": PROJECT_ROOT / "2.梯度分析" / "功能梯度" / "ABIDE1_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平",
    "ABIDE II": PROJECT_ROOT / "2.梯度分析" / "功能梯度" / "ABIDE2_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-组水平",
}

OUTPUT_STEM = SCRIPT_DIR / "S4_ABIDE_EC_FC_subject_group_spin_and_group_mean_map"
OUTPUT_PNG = OUTPUT_STEM.with_suffix(".png")
OUTPUT_SVG = OUTPUT_STEM.with_suffix(".svg")
OUTPUT_PDF = OUTPUT_STEM.with_suffix(".pdf")
OUTPUT_VALUES = OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_values.csv")

FONT_FAMILY = "Arial"
FONT_SIZE_PX = 25
CANVAS_WIDTH_PX = 1800
EXPORT_DPI = 150
FONT_SIZE = FONT_SIZE_PX * 72 / EXPORT_DPI
CANVAS_WIDTH_IN = CANVAS_WIDTH_PX / EXPORT_DPI
CANVAS_HEIGHT_IN = 8.0
PLOT_AREA_RIGHT_SHIFT = 0.022

COMPARISON_ORDER = [
    "EC_G1_vs_FC_G1",
    "EC_G1_vs_FC_G2",
    "EC_G2_vs_FC_G1",
    "EC_G2_vs_FC_G2",
]
COMPARISON_LABELS = [
    "EC-G1 vs FC-G1",
    "EC-G1 vs FC-G2",
    "EC-G2 vs FC-G1",
    "EC-G2 vs FC-G2",
]
POINT_COLOR = "#2f76d2"
BOX_COLOR = "#cbdcf7"
BAR_COLOR = "#cbdcf7"
MEAN_COLOR = "#a40000"
MEDIAN_COLOR = "#0a3f98"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT_FAMILY],
    "font.size": FONT_SIZE,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def readCsv(csvPath):
    if not csvPath.is_file():
        raise FileNotFoundError(f"Required input file is missing: {csvPath}")
    return pd.read_csv(csvPath, encoding="utf-8-sig")


def twoSidedPermutationP(observedValue, nullValues):
    finiteNullValues = np.asarray(nullValues, dtype=float)
    finiteNullValues = finiteNullValues[np.isfinite(finiteNullValues)]
    if finiteNullValues.size == 0 or not np.isfinite(observedValue):
        return np.nan
    return (np.count_nonzero(np.abs(finiteNullValues) >= abs(observedValue)) + 1) / (
        finiteNullValues.size + 1
    )


def formatPValue(pValue):
    if not np.isfinite(pValue):
        return r"$p_{spin}=\mathrm{NA}$"
    if pValue < 0.001:
        return r"$p_{spin}<0.001$"
    return rf"$p_{{spin}}={pValue:.3f}$"


def loadDataset(datasetLabel, datasetRoot):
    allResultsDir = datasetRoot / "all_results"
    groupResultsDir = datasetRoot / "group_level_results"
    subjectResults = readCsv(
        allResultsDir / "ALL_HC_ASD_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test_long.csv"
    )
    groupSummary = readCsv(
        GROUP_SUMMARY_ROOTS[datasetLabel]
        / "group_level_results"
        / "group_level_subject_mean_wholebrain_EC-G1G2_vs_FC-G1G2_cross_axis_spin_test.csv"
    ).set_index("comparison")

    groupCounts = subjectResults[["match_id", "Group"]].drop_duplicates()["Group"].value_counts()
    subjectCount = int(groupCounts.sum())
    groupWeights = groupCounts / subjectCount

    meanValues = (
        subjectResults.groupby("comparison", sort=False)["spearman_r"]
        .mean()
        .reindex(COMPARISON_ORDER)
    )
    standardErrors = (
        subjectResults.groupby("comparison", sort=False)["spearman_r"]
        .sem()
        .reindex(COMPARISON_ORDER)
    )

    pooledSubjectNull = {}
    ecGroupMaps = []
    fcGroupMaps = []
    groupMapNullByComparison = {comparison: [] for comparison in COMPARISON_ORDER}

    for groupLabel, groupWeight in groupWeights.items():
        groupDirectory = groupResultsDir / str(groupLabel)
        subjectNull = readCsv(groupDirectory / "group_mean_subject_mean_wholebrain_spin_null.csv")
        nullColumn = "null_group_mean" if "null_group_mean" in subjectNull.columns else subjectNull.columns[-1]
        if "comparison" in subjectNull.columns:
            for comparison in COMPARISON_ORDER:
                comparisonNull = subjectNull.loc[
                    subjectNull["comparison"].astype(str) == comparison, nullColumn
                ].to_numpy(dtype=float)
                pooledSubjectNull.setdefault(comparison, []).append(groupWeight * comparisonNull)

        mapDirectory = groupDirectory / "group_mean_maps"
        ecMap = readCsv(mapDirectory / f"{groupLabel}_EC_group_mean_gradient_map.csv")
        fcMap = readCsv(mapDirectory / f"{groupLabel}_FC_group_mean_gradient_map.csv")
        ecGroupMaps.append((groupWeight, ecMap))
        fcGroupMaps.append((groupWeight, fcMap))

        for comparison in COMPARISON_ORDER:
            mapNull = readCsv(mapDirectory / f"group_mean_map_spin_null_{comparison}.csv")
            nullColumn = "null_rho" if "null_rho" in mapNull.columns else mapNull.columns[-1]
            groupMapNullByComparison[comparison].append(
                groupWeight * mapNull[nullColumn].to_numpy(dtype=float)
            )

    ecMap = ecGroupMaps[0][1].copy()
    fcMap = fcGroupMaps[0][1].copy()
    for gradientName in ("EC_G1", "EC_G2"):
        ecMap[gradientName] = sum(
            groupWeight * groupMap[gradientName].to_numpy(dtype=float)
            for groupWeight, groupMap in ecGroupMaps
        )
    for gradientName in ("FC_G1", "FC_G2"):
        fcMap[gradientName] = sum(
            groupWeight * groupMap[gradientName].to_numpy(dtype=float)
            for groupWeight, groupMap in fcGroupMaps
        )

    subjectPValues = {}
    mapStatistics = {}
    for comparison in COMPARISON_ORDER:
        subjectPValues[comparison] = float(groupSummary.loc[comparison, "group_p_spin"])

        ecGradient, fcGradient = comparison.split("_vs_")
        observedMapRho = pd.Series(ecMap[ecGradient]).corr(pd.Series(fcMap[fcGradient]), method="spearman")
        combinedMapNull = np.sum(np.vstack(groupMapNullByComparison[comparison]), axis=0)
        mapStatistics[comparison] = {
            "rho": float(observedMapRho),
            "p": twoSidedPermutationP(observedMapRho, combinedMapNull),
        }

    return {
        "label": datasetLabel,
        "subjectResults": subjectResults,
        "subjectCount": subjectCount,
        "means": meanValues,
        "standardErrors": standardErrors,
        "subjectPValues": subjectPValues,
        "ecMap": ecMap,
        "fcMap": fcMap,
        "mapStatistics": mapStatistics,
    }


def styleAxis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", length=4, width=1.0)


def setTitle(axis, title):
    axis.set_title(title, y=1.0, pad=0, fontweight="bold")


def addPanelLabel(figure, axis, panelLabel, titleY, renderer):
    yAxisLabel = axis.yaxis.label
    if yAxisLabel.get_text():
        labelLeftPixels = yAxisLabel.get_window_extent(renderer=renderer).x0
    else:
        visibleTickLabels = [tickLabel for tickLabel in axis.get_yticklabels() if tickLabel.get_visible()]
        labelLeftPixels = min(
            tickLabel.get_window_extent(renderer=renderer).x0
            for tickLabel in visibleTickLabels
        )
    labelLeft = labelLeftPixels / figure.bbox.width
    figure.text(
        labelLeft,
        titleY,
        panelLabel,
        ha="left",
        va="baseline",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )


def drawBoxplot(axis, dataset):
    values = [
        dataset["subjectResults"].loc[
            dataset["subjectResults"]["comparison"].astype(str) == comparison, "spearman_r"
        ].dropna().to_numpy(dtype=float)
        for comparison in COMPARISON_ORDER
    ]
    positions = np.arange(1, len(COMPARISON_ORDER) + 1)
    axis.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": MEDIAN_COLOR, "linewidth": 1.4},
        boxprops={"facecolor": BOX_COLOR, "color": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
    )
    randomGenerator = np.random.default_rng(42)
    for position, comparisonValues in zip(positions, values):
        jitter = randomGenerator.normal(0, 0.045, comparisonValues.size)
        axis.scatter(position + jitter, comparisonValues, s=5, alpha=0.38, color=POINT_COLOR, edgecolors="none")
    axis.scatter(
        positions,
        dataset["means"].to_numpy(dtype=float),
        marker="D",
        s=32,
        color=MEAN_COLOR,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        label="Mean",
    )
    axis.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axis.set_xticks(positions, COMPARISON_LABELS, rotation=38, ha="right")
    axis.set_ylabel(r"Subject-level Spearman $\rho$")
    setTitle(axis, f"{dataset['label']} | Subject $\\rho$ (n = {dataset['subjectCount']})")
    axis.legend(frameon=False, loc="upper right", handletextpad=0.3, borderpad=0.1)
    styleAxis(axis)


def drawMeanBarplot(axis, dataset):
    positions = np.arange(len(COMPARISON_ORDER))
    axis.bar(
        positions,
        dataset["means"].to_numpy(dtype=float),
        yerr=dataset["standardErrors"].to_numpy(dtype=float),
        width=0.72,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        error_kw={"elinewidth": 0.9, "capsize": 3, "capthick": 0.9},
    )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(positions, COMPARISON_LABELS, rotation=38, ha="right")
    axis.set_ylabel(r"Mean subject-level $\rho$")
    setTitle(axis, f"{dataset['label']} | Mean $\\rho$")
    styleAxis(axis)


def drawMapScatter(axis, dataset, ecGradient, fcGradient):
    comparison = f"{ecGradient}_vs_{fcGradient}"
    xValues = dataset["ecMap"][ecGradient].to_numpy(dtype=float)
    yValues = dataset["fcMap"][fcGradient].to_numpy(dtype=float)
    finiteMask = np.isfinite(xValues) & np.isfinite(yValues)
    xValues = xValues[finiteMask]
    yValues = yValues[finiteMask]
    axis.scatter(xValues, yValues, s=7, color=POINT_COLOR, alpha=0.9, edgecolors="none")
    coefficients = np.polyfit(xValues, yValues, 1)
    fittedX = np.linspace(xValues.min(), xValues.max(), 200)
    axis.plot(fittedX, coefficients[0] * fittedX + coefficients[1], color=POINT_COLOR, linewidth=1.2)
    displayEc = ecGradient.replace("_", "-")
    displayFc = fcGradient.replace("_", "-")
    mapStatistic = dataset["mapStatistics"][comparison]
    axis.text(
        0.04,
        0.96,
        "Group-average map\n"
        + rf"$\rho_{{map}}={mapStatistic['rho']:.4f}$"
        + "\n"
        + formatPValue(mapStatistic["p"]),
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontweight="normal",
    )
    axis.set_xlabel(f"Group-average {displayEc}")
    axis.set_ylabel(f"Group-average {displayFc}")
    setTitle(axis, f"{dataset['label']} | {displayEc} vs {displayFc}")
    styleAxis(axis)


def saveNumericValues(datasets):
    outputRows = []
    for dataset in datasets:
        for comparison in COMPARISON_ORDER:
            outputRows.append({
                "dataset": dataset["label"],
                "comparison": comparison,
                "n_subjects": dataset["subjectCount"],
                "mean_subject_spearman_r": dataset["means"][comparison],
                "subject_standard_error": dataset["standardErrors"][comparison],
                "group_p_spin": dataset["subjectPValues"][comparison],
                "group_mean_map_spearman_r": dataset["mapStatistics"][comparison]["rho"],
                "group_mean_map_p_spin": dataset["mapStatistics"][comparison]["p"],
            })
    pd.DataFrame(outputRows).to_csv(OUTPUT_VALUES, index=False, encoding="utf-8-sig")


def main():
    datasets = [loadDataset(label, root) for label, root in ABIDE_ROOTS.items()]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(CANVAS_WIDTH_IN, CANVAS_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )

    for rowIndex, dataset in enumerate(datasets):
        drawBoxplot(axes[rowIndex, 0], dataset)
        drawMeanBarplot(axes[rowIndex, 1], dataset)
        drawMapScatter(axes[rowIndex, 2], dataset, "EC_G2", "FC_G2")
    figure.subplots_adjust(
        left=0.072,
        right=0.985,
        top=0.955,
        bottom=0.120,
        wspace=0.24,
        hspace=0.72,
    )
    for meanAxis in axes[:, 1]:
        meanPosition = meanAxis.get_position()
        meanAxis.set_position([
            meanPosition.x0 + 0.012,
            meanPosition.y0,
            meanPosition.width - 0.012,
            meanPosition.height,
        ])
    titlePositions = []
    for rowIndex in range(2):
        commonTop = axes[rowIndex, 0].get_position().y1
        titleY = commonTop + 0.030
        for axis in axes[rowIndex]:
            axisPosition = axis.get_position()
            axis.title.set_transform(figure.transFigure)
            axis.title.set_ha("center")
            axis.title.set_position((axisPosition.x0 + axisPosition.width / 2, titleY))
            titlePositions.append((axis, titleY))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for (axis, titleY), panelLabel in zip(titlePositions, "abcdef"):
        addPanelLabel(figure, axis, panelLabel, titleY, renderer)

    # Panel labels remain fixed while the plotting areas use a rightward offset.
    for axis in axes.flat:
        axisPosition = axis.get_position()
        axis.set_position([
            axisPosition.x0 + PLOT_AREA_RIGHT_SHIFT,
            axisPosition.y0,
            axisPosition.width - PLOT_AREA_RIGHT_SHIFT,
            axisPosition.height,
        ])

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis, titleY in titlePositions:
        yAxisLabelLeft = axis.yaxis.label.get_window_extent(renderer=renderer).x0 / figure.bbox.width
        axis.title.set_ha("left")
        axis.title.set_position((yAxisLabelLeft, titleY))

    figure.savefig(OUTPUT_PNG, dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(OUTPUT_SVG, facecolor="white")
    figure.savefig(OUTPUT_PDF, facecolor="white")
    plt.close(figure)
    saveNumericValues(datasets)

    print(f"Saved {CANVAS_WIDTH_PX} px PNG: {OUTPUT_PNG}")
    print(f"Saved editable SVG: {OUTPUT_SVG}")
    print(f"Saved PDF: {OUTPUT_PDF}")
    print(f"Saved values: {OUTPUT_VALUES}")


if __name__ == "__main__":
    main()
