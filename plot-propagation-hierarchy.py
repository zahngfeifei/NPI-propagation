from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fontManager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import seaborn as sns

from brainspace.datasets import load_conte69, load_parcellation
from brainspace.plotting import plot_hemispheres


UNIFIED_FONT_SIZE = 12.0
FIGURE_WIDTH_INCHES = 16.0
FIGURE_HEIGHT_INCHES = 8.2
RANDOM_SEED = 20260722

COLORS = {
    "H1_sensory": "#2878B5",
    "H2_attention": "#F57C00",
    "H3_control": "#2CA02C",
    "H4_DMN": "#E52B2B",
    "HC": "#2878B5",
    "ASD": "#E52B2B",
    "neutral": "#3A3A3A",
    "zero": "#8A8A8A",
}

SYSTEM_LABELS = {
    "H1_sensory": "Vis and SMN",
    "H2_attention": "Attention",
    "H3_control": "Control",
    "H4_DMN": "DMN",
}

SYSTEM_NETWORKS = {
    "H1_sensory": {"Vis", "SomMot"},
    "H2_attention": {"DorsAttn", "SalVentAttn"},
    "H3_control": {"Cont"},
    "H4_DMN": {"Default"},
}

SYSTEM_LEVELS = {
    "H1_sensory": "L1",
    "H2_attention": "L2",
    "H3_control": "L3",
    "H4_DMN": "L4",
}


def configure_matplotlib() -> None:
    arialPath = Path(r"C:\Windows\Fonts\arial.ttf")
    arialBoldPath = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if not arialPath.exists() or not arialBoldPath.exists():
        raise FileNotFoundError("Arial font files were not found in C:\\Windows\\Fonts.")

    fontManager.fontManager.addfont(str(arialPath))
    fontManager.fontManager.addfont(str(arialBoldPath))
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": UNIFIED_FONT_SIZE,
            "axes.titlesize": UNIFIED_FONT_SIZE,
            "axes.labelsize": UNIFIED_FONT_SIZE,
            "xtick.labelsize": UNIFIED_FONT_SIZE,
            "ytick.labelsize": UNIFIED_FONT_SIZE,
            "legend.fontsize": UNIFIED_FONT_SIZE,
            "figure.titlesize": UNIFIED_FONT_SIZE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def locate_input_paths(figureDirectory: Path) -> dict[str, Path]:
    projectRoot = figureDirectory.parents[2]
    atlasCandidates = sorted(
        path
        for path in projectRoot.rglob(
            "Schaefer2018_400Parcels_7Networks_order.txt"
        )
        if "ABIDE2" in str(path)
    )
    if not atlasCandidates:
        raise FileNotFoundError("Schaefer-400 Yeo-7 label table was not found.")

    result4Directory = (
        figureDirectory
        / "ABIDE2_新结果4"
        / "GroupStats_H4minusH1_contrast_noFD_noSite"
    )
    return {
        "metrics": figureDirectory
        / "ABIDE2_新结果1_combat"
        / "EC_SEC_metrics_all_subjects_combat.csv",
        "descriptive": figureDirectory
        / "ABIDE2_新结果3"
        / "result3_first_arrival_descriptive_stats.csv",
        "friedman": figureDirectory
        / "ABIDE2_新结果3"
        / "result3_first_arrival_friedman_test.csv",
        "pairwise": figureDirectory
        / "ABIDE2_新结果3"
        / "result3_first_arrival_pairwise_wilcoxon.csv",
        "subjectLong": figureDirectory
        / "ABIDE2_新结果3"
        / "result3_first_arrival_subject_level_long_used.csv",
        "contrast": result4Directory
        / "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast.csv",
        "contrastData": result4Directory
        / "EC_SEC_GroupStats_ASD_vs_HC_H4_minus_H1_contrast_data_used.csv",
        "supplementary": result4Directory
        / "EC_SEC_GroupStats_ASD_vs_HC_single_metrics_supplementary.csv",
        "atlasLabels": atlasCandidates[0],
    }


def read_inputs(inputPaths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    missingPaths = [str(path) for path in inputPaths.values() if not path.exists()]
    if missingPaths:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missingPaths))

    return {
        key: pd.read_csv(path, sep=r"\s+", header=None)
        if key == "atlasLabels"
        else pd.read_csv(path)
        for key, path in inputPaths.items()
    }


def network_name(parcelName: str) -> str:
    nameParts = parcelName.split("_")
    if len(nameParts) < 3:
        raise ValueError(f"Unexpected Schaefer parcel name: {parcelName}")
    return nameParts[2]


def crop_brainspace_lateral_view(imagePath: Path) -> None:
    sourceImage = Image.open(imagePath).convert("RGB")
    lateralQuadrant = sourceImage.crop(
        (0, 0, sourceImage.width // 2, sourceImage.height // 2)
    )
    whiteBackground = Image.new("RGB", lateralQuadrant.size, "white")
    differenceImage = ImageChops.difference(
        lateralQuadrant, whiteBackground
    ).convert("L")
    visibleMask = differenceImage.point(
        lambda pixelValue: 255 if pixelValue > 4 else 0
    )
    visibleBounds = visibleMask.getbbox()
    if visibleBounds is None:
        raise RuntimeError(f"BrainSpace produced an empty image: {imagePath}")
    left, upper, right, lower = visibleBounds
    paddingPixels = 8
    cropBounds = (
        max(0, left - paddingPixels),
        max(0, upper - paddingPixels),
        min(lateralQuadrant.width, right + paddingPixels),
        min(lateralQuadrant.height, lower + paddingPixels),
    )
    lateralQuadrant.crop(cropBounds).save(imagePath)


def render_brainspace_networks(
    atlasLabels: pd.DataFrame,
    outputDirectory: Path,
) -> dict[str, Path]:
    outputDirectory.mkdir(parents=True, exist_ok=True)
    hemisphereLeft, hemisphereRight = load_conte69()
    parcelLeft, parcelRight = load_parcellation("schaefer", scale=400, join=False)

    parcelNetworkById = {
        int(row.iloc[0]): network_name(str(row.iloc[1]))
        for _, row in atlasLabels.iterrows()
    }
    renderedPaths: dict[str, Path] = {}

    for systemName, includedNetworks in SYSTEM_NETWORKS.items():
        screenshotPath = outputDirectory / f"brainspace-reference-{systemName}.png"
        if not screenshotPath.exists():
            selectedLeft = np.array(
                [
                    1.0
                    if parcelNetworkById.get(int(parcelId)) in includedNetworks
                    else 0.0
                    for parcelId in parcelLeft
                ]
            )
            selectedRight = np.array(
                [
                    1.0
                    if parcelNetworkById.get(int(parcelId)) in includedNetworks
                    else 0.0
                    for parcelId in parcelRight
                ]
            )
            brainColorMap = LinearSegmentedColormap.from_list(
                f"{systemName}Map", ["#E6E6E6", COLORS[systemName]], N=256
            )
            plot_hemispheres(
                hemisphereLeft,
                hemisphereRight,
                array_name=np.concatenate([selectedLeft, selectedRight]),
                color_bar=False,
                color_range=(0, 1),
                cmap=brainColorMap,
                nan_color=(0.78, 0.78, 0.78, 1.0),
                layout_style="grid",
                background=(1.0, 1.0, 1.0),
                size=(1000, 900),
                zoom=1.25,
                interactive=False,
                screenshot=True,
                filename=str(screenshotPath),
                transparent_bg=False,
                actor__ambient=0.72,
                actor__diffuse=0.28,
                actor__specular=0.0,
            )
            crop_brainspace_lateral_view(screenshotPath)
        renderedPaths[systemName] = screenshotPath

    overviewPath = outputDirectory / "brainspace-L1-L4-overview-reference.png"
    if not overviewPath.exists():
        networkLevelValue = {
            networkName: levelValue
            for levelValue, includedNetworks in enumerate(
                SYSTEM_NETWORKS.values(), start=1
            )
            for networkName in includedNetworks
        }
        overviewLeft = np.array(
            [
                networkLevelValue.get(parcelNetworkById.get(int(parcelId)), 0)
                for parcelId in parcelLeft
            ],
            dtype=float,
        )
        overviewRight = np.array(
            [
                networkLevelValue.get(parcelNetworkById.get(int(parcelId)), 0)
                for parcelId in parcelRight
            ],
            dtype=float,
        )
        overviewColorMap = ListedColormap(
            [
                "#E6E6E6",
                COLORS["H1_sensory"],
                COLORS["H2_attention"],
                COLORS["H3_control"],
                COLORS["H4_DMN"],
            ],
            name="L1L4HierarchyMap",
        )
        plot_hemispheres(
            hemisphereLeft,
            hemisphereRight,
            array_name=np.concatenate([overviewLeft, overviewRight]),
            color_bar=False,
            color_range=(0, 4),
            cmap=overviewColorMap,
            nan_color=(0.78, 0.78, 0.78, 1.0),
            layout_style="grid",
            background=(1.0, 1.0, 1.0),
            size=(1000, 900),
            zoom=1.25,
            interactive=False,
            screenshot=True,
            filename=str(overviewPath),
            transparent_bg=False,
            actor__ambient=0.72,
            actor__diffuse=0.28,
            actor__specular=0.0,
        )
        crop_brainspace_lateral_view(overviewPath)
    renderedPaths["hierarchyOverview"] = overviewPath

    return renderedPaths


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.06,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=UNIFIED_FONT_SIZE,
        clip_on=False,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.tick_params(width=0.8, length=3)
    axis.grid(False)


def add_significance_bracket(
    axis: plt.Axes,
    leftPosition: float,
    rightPosition: float,
    height: float,
    bracketHeight: float,
    label: str,
) -> None:
    axis.plot(
        [leftPosition, leftPosition, rightPosition, rightPosition],
        [height, height + bracketHeight, height + bracketHeight, height],
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    axis.text(
        (leftPosition + rightPosition) / 2,
        height + bracketHeight,
        label,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def significance_label(pValue: float) -> str:
    if pValue < 0.001:
        return "***"
    if pValue < 0.01:
        return "**"
    if pValue < 0.05:
        return "*"
    return "n.s."


def format_scientific_math(value: float, decimalPlaces: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"{coefficient:.{decimalPlaces}f} \times 10^{{{exponent}}}"


def draw_panel_a(
    axis: plt.Axes,
    descriptiveTable: pd.DataFrame,
    brainImagePaths: dict[str, Path],
) -> None:
    orderedSystems = list(SYSTEM_LABELS)
    summaryTable = descriptiveTable.set_index("system").loc[orderedSystems]
    hierarchyLevels = np.arange(1, 5)
    meanCentroids = summaryTable["mean"].to_numpy(float)

    overviewLateralView = plt.imread(brainImagePaths["hierarchyOverview"])
    overviewBox = OffsetImage(
        overviewLateralView,
        zoom=0.12,
        interpolation="lanczos",
        resample=True,
    )
    axis.add_artist(
        AnnotationBbox(
            overviewBox,
            (1.08, 21.31),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=1,
        )
    )
    for legendIndex, systemName in enumerate(orderedSystems):
        legendY = 21.40 - legendIndex * 0.075
        axis.scatter(
            1.33,
            legendY,
            marker="s",
            s=18,
            color=COLORS[systemName],
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
        axis.text(
            1.39,
            legendY,
            SYSTEM_LEVELS[systemName],
            ha="left",
            va="center",
            color="black",
        )

    axis.plot(hierarchyLevels, meanCentroids, color="black", linewidth=1.1, zorder=2)
    for level, systemName, meanCentroid in zip(
        hierarchyLevels, orderedSystems, meanCentroids
    ):
        axis.scatter(
            level,
            meanCentroid,
            s=34,
            color=COLORS[systemName],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.text(
            level,
            meanCentroid + 0.12,
            f"{meanCentroid:.2f}",
            color=COLORS[systemName],
            ha="center",
            va="bottom",
        )
        lateralView = plt.imread(brainImagePaths[systemName])
        imageBox = OffsetImage(
            lateralView,
            zoom=0.19,
            interpolation="lanczos",
            resample=True,
        )
        brainHorizontalPosition = level + (
            0.15 if systemName == "H1_sensory" else 0.0
        )
        brainAnnotation = AnnotationBbox(
            imageBox,
            (brainHorizontalPosition, meanCentroid - 0.27),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=1,
        )
        axis.add_artist(brainAnnotation)

    axis.set_title(
        "Propagation hierarchy across cortical levels (L1–L4)",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    axis.set_ylabel("Temporal centroid (step)")
    axis.set_xlabel("Hierarchy level")
    axis.set_xticks(hierarchyLevels)
    axis.set_xticklabels(
        [f"{SYSTEM_LEVELS[name]}\n{SYSTEM_LABELS[name]}" for name in orderedSystems]
    )
    for tickLabel, systemName in zip(axis.get_xticklabels(), orderedSystems):
        tickLabel.set_color(COLORS[systemName])
    axis.set_ylim(19.55, 21.45)
    axis.set_xlim(0.82, 4.10)
    axis.set_yticks([19.5, 20.0, 20.5, 21.0, 21.5])
    style_axis(axis)
    add_panel_label(axis, "a")


def draw_panel_b(
    axis: plt.Axes,
    subjectLongTable: pd.DataFrame,
    descriptiveTable: pd.DataFrame,
    pairwiseTable: pd.DataFrame,
) -> None:
    orderedSystems = list(SYSTEM_LABELS)
    randomGenerator = np.random.default_rng(RANDOM_SEED)
    verticalPositions = np.arange(4, 0, -1)
    summaryTable = descriptiveTable.set_index("system").loc[orderedSystems]

    for verticalPosition, systemName in zip(verticalPositions, orderedSystems):
        systemValues = subjectLongTable.loc[
            subjectLongTable["system"] == systemName, "metric_value"
        ].to_numpy(float)
        jitter = randomGenerator.normal(0, 0.065, size=systemValues.size)
        axis.scatter(
            systemValues,
            verticalPosition + jitter,
            s=4,
            color=COLORS[systemName],
            alpha=0.42,
            linewidth=0,
            rasterized=True,
        )
        meanValue = float(summaryTable.loc[systemName, "mean"])
        axis.scatter(
            meanValue,
            verticalPosition,
            s=30,
            color=COLORS[systemName],
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        axis.text(
            19.23,
            verticalPosition,
            f"{SYSTEM_LEVELS[systemName]}  {SYSTEM_LABELS[systemName]}",
            ha="right",
            va="center",
            color=COLORS[systemName],
        )
        axis.text(
            21.94,
            verticalPosition,
            f"{meanValue:.2f}",
            ha="left",
            va="center",
            color=COLORS[systemName],
        )

    axis.axvline(20.5, color="#BDBDBD", linewidth=0.7, zorder=0)
    axis.set_title(
        "Level propagation (Markov step centroids)",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    axis.set_xlabel("Diffusion centroid (step)")
    axis.set_yticks([])
    axis.set_xlim(19.25, 22.35)
    axis.set_xticks([19.3, 19.9, 20.5, 21.2, 21.8])
    axis.set_ylim(0.45, 4.75)
    axis.text(21.94, 4.52, "Mean", ha="left", va="center", fontweight="bold")
    l1VsL4Comparison = pairwiseTable.loc[
        (pairwiseTable["system_A"] == "H1_sensory")
        & (pairwiseTable["system_B"] == "H4_DMN")
    ]
    if len(l1VsL4Comparison) != 1:
        raise ValueError(
            "Expected exactly one H1_sensory vs H4_DMN pairwise comparison."
        )
    l1VsL4CorrectedPValue = float(l1VsL4Comparison["p_corrected"].iloc[0])
    bracketX = 22.20
    bracketCapLeft = 22.16
    axis.plot(
        [bracketCapLeft, bracketX, bracketX, bracketCapLeft],
        [4.0, 4.0, 1.0, 1.0],
        color="black",
        linewidth=0.8,
        clip_on=False,
    )
    axis.text(
        22.25,
        2.5,
        significance_label(l1VsL4CorrectedPValue),
        ha="left",
        va="center",
        fontweight="bold",
    )
    correctedPLabel = (
        "q FDR < 0.001"
        if l1VsL4CorrectedPValue < 0.001
        else f"q FDR = {l1VsL4CorrectedPValue:.3g}"
    )
    axis.text(
        0.995,
        0.025,
        f"{significance_label(l1VsL4CorrectedPValue)}  {correctedPLabel}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    style_axis(axis)
    add_panel_label(axis, "b")


def draw_panel_c(axis: plt.Axes, contrastTable: pd.DataFrame) -> None:
    result = contrastTable.iloc[0]
    betaValue = float(result["Beta_ASD_minus_HC"])
    confidenceLow = float(result["ci_low"])
    confidenceHigh = float(result["ci_high"])
    pValue = float(result["p_value"])
    tValue = float(result["t_value"])

    axis.errorbar(
        betaValue,
        0,
        xerr=[[betaValue - confidenceLow], [confidenceHigh - betaValue]],
        fmt="o",
        color="#B51635",
        ecolor="#B51635",
        elinewidth=1.2,
        capsize=4,
        markersize=5,
    )
    axis.axvline(0, color=COLORS["zero"], linestyle="--", linewidth=0.8)
    axis.text(
        -0.1e-5,
        0.14,
        rf"$\beta = {format_scientific_math(betaValue)}$"
        + "\n"
        + rf"$t$ = {tValue:.3f}, $p$ = {pValue:.3g}",
        ha="left",
        va="center",
    )
    axis.set_title(
        "Replicated ASD group effect on the\nL4–L1 early SEC-slope contrast (steps 1–10)",
        loc="left",
        fontweight="bold",
        pad=8,
    )
    axis.set_xlabel("ASD–HC effect on L4–L1 early SEC-slope contrast\n(β coefficient)")
    axis.set_yticks([])
    axis.set_ylim(-0.45, 0.45)
    axis.set_xlim(-3.0e-5, 1.0e-5)
    axis.set_xticks([-3.0e-5, -2.0e-5, -1.0e-5, 0.0])
    scientificFormatter = ScalarFormatter(useMathText=True)
    scientificFormatter.set_powerlimits((-5, -5))
    axis.xaxis.set_major_formatter(scientificFormatter)
    style_axis(axis)
    add_panel_label(axis, "c")


def draw_raincloud_panel(
    axis: plt.Axes,
    plotTable: pd.DataFrame,
    valueColumn: str,
    title: str,
    yLabel: str,
    pValue: float,
    panelLabel: str | None = None,
) -> None:
    groupOrder = ["HC", "ASD"]
    palette = {group: COLORS[group] for group in groupOrder}
    sns.violinplot(
        data=plotTable,
        x="Group",
        y=valueColumn,
        hue="Group",
        order=groupOrder,
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=0,
        alpha=0.16,
        width=0.72,
        ax=axis,
    )
    for violinBody in axis.collections:
        violinBody.set_alpha(0.16)

    sns.stripplot(
        data=plotTable,
        x="Group",
        y=valueColumn,
        hue="Group",
        order=groupOrder,
        palette=palette,
        legend=False,
        size=1.5,
        alpha=0.46,
        jitter=0.16,
        linewidth=0,
        ax=axis,
        rasterized=True,
    )
    sns.boxplot(
        data=plotTable,
        x="Group",
        y=valueColumn,
        order=groupOrder,
        width=0.18,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        ax=axis,
    )

    groupCounts = plotTable.groupby("Group")[valueColumn].count()
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        [f"HC\n(n = {groupCounts['HC']})", f"ASD\n(n = {groupCounts['ASD']})"]
    )
    axis.get_xticklabels()[0].set_color(COLORS["HC"])
    axis.get_xticklabels()[1].set_color(COLORS["ASD"])
    axis.set_title(title, loc="left", fontweight="bold", pad=8)
    axis.set_xlabel("")
    axis.set_ylabel(yLabel)
    valueRange = float(plotTable[valueColumn].max() - plotTable[valueColumn].min())
    currentTop = float(plotTable[valueColumn].max())
    bracketBase = currentTop + 0.04 * valueRange
    add_significance_bracket(
        axis,
        0,
        1,
        bracketBase,
        0.025 * valueRange,
        significance_label(pValue),
    )
    currentBottom = float(plotTable[valueColumn].min())
    axis.set_ylim(currentBottom - 0.05 * valueRange, currentTop + 0.18 * valueRange)
    style_axis(axis)
    if panelLabel is not None:
        add_panel_label(axis, panelLabel)


def create_figure(
    inputTables: dict[str, pd.DataFrame],
    brainImagePaths: dict[str, Path],
) -> plt.Figure:
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        constrained_layout=False,
        facecolor="white",
    )
    outerGrid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.35, 0.85],
        hspace=0.34,
        left=0.055,
        right=0.985,
        top=0.95,
        bottom=0.09,
    )
    upperGrid = outerGrid[0].subgridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.26)
    lowerGrid = outerGrid[1].subgridspec(
        1, 4, width_ratios=[2.45, 0.82, 0.82, 0.82], wspace=0.66
    )

    panelA = figure.add_subplot(upperGrid[0, 0])
    panelB = figure.add_subplot(upperGrid[0, 1])
    panelC = figure.add_subplot(lowerGrid[0, 0])
    panelD = figure.add_subplot(lowerGrid[0, 1])
    panelE1 = figure.add_subplot(lowerGrid[0, 2])
    panelE2 = figure.add_subplot(lowerGrid[0, 3])

    draw_panel_a(panelA, inputTables["descriptive"], brainImagePaths)
    draw_panel_b(
        panelB,
        inputTables["subjectLong"],
        inputTables["descriptive"],
        inputTables["pairwise"],
    )
    draw_panel_c(panelC, inputTables["contrast"])

    contrastPlotTable = inputTables["contrastData"].rename(
        columns={"Group_original": "Group"}
    )
    contrastPValue = float(inputTables["contrast"]["p_value"].iloc[0])
    draw_raincloud_panel(
        panelD,
        contrastPlotTable,
        "H4_minus_H1_early_slope_1_10",
        "L4–L1 early SEC-slope contrast",
        r"Early SEC-slope contrast ($\times 10^{-3}$)",
        contrastPValue,
        panelLabel="d",
    )
    panelD.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value * 1e3:.2f}"))

    metricsTable = inputTables["metrics"].copy()
    supplementaryTable = inputTables["supplementary"].set_index("Actual_column")
    h1PValue = float(
        supplementaryTable.loc[
            "H1_sensory_early_slope_1_10",
            "p_FDR__supplementary_early_slope_H1H2H3H4_all",
        ]
    )
    h4PValue = float(
        supplementaryTable.loc[
            "H4_DMN_early_slope_1_10",
            "p_FDR__supplementary_early_slope_H1H2H3H4_all",
        ]
    )
    draw_raincloud_panel(
        panelE1,
        metricsTable,
        "H1_sensory_early_slope_1_10",
        "L1 (sensory) early SEC-slope",
        r"Early SEC-slope ($\times 10^{-5}$)",
        h1PValue,
        panelLabel=None,
    )
    draw_raincloud_panel(
        panelE2,
        metricsTable,
        "H4_DMN_early_slope_1_10",
        "L4 (DMN) early SEC-slope",
        r"Early SEC-slope ($\times 10^{-5}$)",
        h4PValue,
    )
    for axis in [panelE1, panelE2]:
        axis.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _: f"{value * 1e5:.0f}")
        )

    return figure


def save_figure(figure: plt.Figure, outputStem: Path) -> None:
    figure.savefig(outputStem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(outputStem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    figure.savefig(outputStem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    figure.savefig(
        outputStem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def write_source_data(inputTables: dict[str, pd.DataFrame], outputPath: Path) -> None:
    sourceColumns = [
        "sub_id",
        "Group",
        "H1_sensory_temporal_centroid",
        "H2_attention_temporal_centroid",
        "H3_control_temporal_centroid",
        "H4_DMN_temporal_centroid",
        "H1_sensory_early_slope_1_10",
        "H4_DMN_early_slope_1_10",
    ]
    sourceData = inputTables["metrics"][sourceColumns].copy()
    contrastBySubject = inputTables["contrastData"].set_index("sub_id")
    sourceData["H4_minus_H1_early_slope_1_10"] = sourceData["sub_id"].map(
        contrastBySubject["H4_minus_H1_early_slope_1_10"]
    )
    sourceData = sourceData.rename(
        columns={
            "H1_sensory_temporal_centroid": "L1_sensory_temporal_centroid",
            "H2_attention_temporal_centroid": "L2_attention_temporal_centroid",
            "H3_control_temporal_centroid": "L3_control_temporal_centroid",
            "H4_DMN_temporal_centroid": "L4_DMN_temporal_centroid",
            "H1_sensory_early_slope_1_10": "L1_sensory_early_slope_1_10",
            "H4_DMN_early_slope_1_10": "L4_DMN_early_slope_1_10",
            "H4_minus_H1_early_slope_1_10": "L4_minus_L1_early_slope_1_10",
        }
    )
    sourceData.to_csv(outputPath, index=False)


def main() -> None:
    configure_matplotlib()
    figureDirectory = Path(__file__).resolve().parent
    outputDirectory = figureDirectory / "abide2-propagation-hierarchy-figure"
    outputDirectory.mkdir(parents=True, exist_ok=True)

    inputPaths = locate_input_paths(figureDirectory)
    inputTables = read_inputs(inputPaths)
    brainImagePaths = render_brainspace_networks(
        inputTables["atlasLabels"], outputDirectory / "brainspace-renders"
    )
    figure = create_figure(inputTables, brainImagePaths)
    outputStem = outputDirectory / "abide2-propagation-hierarchy-result"
    save_figure(figure, outputStem)
    write_source_data(
        inputTables, outputDirectory / "abide2-propagation-hierarchy-source-data.csv"
    )
    shutil.copy2(Path(__file__), outputDirectory / Path(__file__).name)
    plt.close(figure)
    print(f"Saved figure bundle to: {outputDirectory}")


if __name__ == "__main__":
    main()
