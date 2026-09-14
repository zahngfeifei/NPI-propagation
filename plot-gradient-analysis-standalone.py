"""Standalone gradient-analysis figure renderer.

This file bundles the panel A-G plotting implementations and SVG geometry
analysis used by the original modular workflow. It can be copied by itself;
it does not load or import sibling Python scripts.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import zscore


TARGET_WIDTH_POINTS = 1766.514
TARGET_HEIGHT_POINTS = 1735.416
TARGET_FONT_SIZE_POINTS = 25.0
TARGET_PANEL_LABEL_SIZE_POINTS = 25.0
TARGET_PANEL_TITLE_OFFSET_POINTS = 33.584
EXPORT_DPI = 600
ROW_TWO_PLOT_AXIS_HEIGHT_INCHES = 361.022 / 72
PANEL_G_BETA_COLOR_RANGE = (-0.05, 0.05)
PANEL_G_BETA_COLORBAR_TICKS = (-0.05, 0.0, 0.05)

ROW_ONE_SOURCE_HEIGHT_POINTS = 381.6
ROW_TWO_SOURCE_HEIGHT_POINTS = 399.6
ROW_THREE_SOURCE_HEIGHT_POINTS = 500.4

ROW_TWO_PANEL_D_EXTRA_SHIFT_POINTS = 116.40
ROW_TWO_PANEL_E_HEADING_EXTRA_SHIFT_POINTS = 74.15
PANEL_E_DISTRIBUTION_Y_AXIS_RIGHT_SHIFT_POINTS = 8.0

# Only panel labels a-g and the seven main panel titles remain bold.
# All other text objects are forced to regular weight after SVG calibration.
PANEL_LABEL_TEXTS = set("abcdefg")
MAIN_PANEL_TITLE_TEXTS = {
    "Yeo-7 organization along EC principal gradient",
    "EC gradient 1 (subject-wise whole-brain z-score)",
    "EC gradient space (400 parcels)",
    "FC–EC G1 correlation",
    "Default–Vis/SomMot G1 separation",
    "Reduced sensory–transmodal separation in ASD",
    "Yeo-7 network group main-effect β values",
}
@dataclass(frozen=True)
class RowGeometry:
    sourceHeightPoints: float
    scaleX: float
    scaleY: float
    translateXPoints: float
    translateYPoints: float
    sourceGridLeftPoints: float
    sourceGridRightPoints: float
    sourceGridTopPoints: float
    sourceGridBottomPoints: float
    gridSpacing: float

    def targetLeftFraction(self) -> float:
        return (
            self.scaleX * self.sourceGridLeftPoints
            + self.translateXPoints
        ) / TARGET_WIDTH_POINTS

    def targetRightFraction(self) -> float:
        return (
            self.scaleX * self.sourceGridRightPoints
            + self.translateXPoints
        ) / TARGET_WIDTH_POINTS

    def targetTopFraction(self) -> float:
        targetTopPoints = (
            self.scaleY * self.sourceGridTopPoints
            + self.translateYPoints
        )
        return 1.0 - targetTopPoints / TARGET_HEIGHT_POINTS

    def targetBottomFraction(self) -> float:
        sourceBottomFromTopPoints = (
            self.sourceHeightPoints - self.sourceGridBottomPoints
        )
        targetBottomFromTopPoints = (
            self.scaleY * sourceBottomFromTopPoints
            + self.translateYPoints
        )
        return 1.0 - targetBottomFromTopPoints / TARGET_HEIGHT_POINTS


ROW_ONE_GEOMETRY = RowGeometry(
    sourceHeightPoints=ROW_ONE_SOURCE_HEIGHT_POINTS,
    scaleX=1.46016,
    scaleY=1.46017,
    translateXPoints=-63.078,
    translateYPoints=-28.774,
    sourceGridLeftPoints=60 / 1800 * 1296,
    sourceGridRightPoints=(1 - 60 / 1800) * 1296,
    sourceGridTopPoints=35 / 568 * ROW_ONE_SOURCE_HEIGHT_POINTS,
    sourceGridBottomPoints=40 / 568 * ROW_ONE_SOURCE_HEIGHT_POINTS,
    gridSpacing=0.32,
)

ROW_TWO_GEOMETRY = RowGeometry(
    sourceHeightPoints=ROW_TWO_SOURCE_HEIGHT_POINTS,
    scaleX=1.392846,
    scaleY=1.392832,
    translateXPoints=-21.590,
    translateYPoints=530.400,
    sourceGridLeftPoints=25 / 1800 * 1296,
    sourceGridRightPoints=(1 - 25 / 1800) * 1296,
    sourceGridTopPoints=30 / 555 * ROW_TWO_SOURCE_HEIGHT_POINTS,
    sourceGridBottomPoints=22 / 555 * ROW_TWO_SOURCE_HEIGHT_POINTS,
    gridSpacing=0.46,
)

ROW_THREE_GEOMETRY = RowGeometry(
    sourceHeightPoints=ROW_THREE_SOURCE_HEIGHT_POINTS,
    scaleX=1.379796,
    scaleY=1.379794,
    translateXPoints=-22.190,
    translateYPoints=1096.280,
    sourceGridLeftPoints=25 / 1800 * 1296,
    sourceGridRightPoints=(1 - 25 / 1800) * 1296,
    sourceGridTopPoints=30 / 695 * ROW_THREE_SOURCE_HEIGHT_POINTS,
    sourceGridBottomPoints=55 / 695 * ROW_THREE_SOURCE_HEIGHT_POINTS,
    gridSpacing=0.38,
)


EMBEDDED_MODULE_SOURCES = {'plot-gradient-analysis-row-1-panels-ab.py': 'from __future__ import annotations\n\nimport math\n\nfrom pathlib import Path\n\nimport matplotlib\n\nmatplotlib.use("Agg")\n\nimport matplotlib.colors as matplotlibColors\n\nimport matplotlib.patheffects as pathEffects\n\nimport matplotlib.pyplot as plt\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom brainspace.datasets import load_conte69, load_parcellation\n\nfrom brainspace.plotting import plot_hemispheres, plot_surf\n\nfrom brainspace.utils.parcellation import map_to_labels\n\nfrom matplotlib.lines import Line2D\n\nfrom matplotlib.colors import to_rgba\n\nfrom matplotlib.patches import Patch\n\nfrom matplotlib.transforms import ScaledTranslation\n\nfrom scipy.stats import gaussian_kde, spearmanr, ttest_ind, zscore\n\nFONT_FAMILY = "Arial"\n\nFONT_SIZE = 18\n\nPANEL_LABEL_SIZE = 22\n\nPANEL_TITLE_OFFSET_POINTS = 23\n\nFIGURE_WIDTH_INCHES = 18.0\n\nFIGURE_HEIGHT_INCHES = 17.8\n\nAB_ROW_HEIGHT_INCHES = 5.68\n\nCD_ROW_HEIGHT_INCHES = 6.30\n\nEF_ROW_HEIGHT_INCHES = 5.68\n\nHISTOGRAM_BIN_WIDTH = 0.1\n\nPLOT_AXIS_HEIGHT_INCHES = 3.60\n\nD_MEAN_MAP_SHIFT_INCHES = 0.30\n\nD_MEAN_MAP_EXTRA_WIDTH_INCHES = 0.15\n\nEXPORT_DPI = 100\n\nRANDOM_SEED = 42\n\nYEO7_ATLAS_ZOOM = 1.44\n\nGRADIENT_BRAIN_ZOOM = 1.72\n\nIMAGE_CROP_PADDING_PIXELS = 12\n\nGRADIENT_BRAIN_ROW_GAP_PIXELS = 4\n\nGROUP_COLORBAR_BOUNDS = (0.12, 0.88, 0.76, 0.46)\n\nPANEL_B_GROUP_WSPACE = 0.02\n\nP_VALUE_CMAP = "viridis_r"\n\nP_VALUE_BRAIN_ZOOM = 1.70\n\nGROUP_COLORS = {\n    "HC": "#4C78A8",\n    "ASD": "#E07A5F",\n}\n\nNETWORK_ORDER = [\n    "Vis",\n    "SomMot",\n    "DorsAttn",\n    "SalVentAttn",\n    "Limbic",\n    "Cont",\n    "Default",\n]\n\nNETWORK_DISPLAY_NAMES = {\n    "Vis": "Vis",\n    "SomMot": "SMN",\n    "DorsAttn": "DAN",\n    "SalVentAttn": "VAN",\n    "Limbic": "Lim",\n    "Cont": "Cont",\n    "Default": "DMN",\n}\n\nNETWORK_COLORS = {\n    "Vis": "#6D56A5",\n    "SomMot": "#4B9BD3",\n    "DorsAttn": "#58A65C",\n    "SalVentAttn": "#E3A33B",\n    "Limbic": "#D9B23C",\n    "Cont": "#39A7A0",\n    "Default": "#D34B4B",\n}\n\nGRADIENT_CMAP = "RdBu_r"\n\nNEUTRAL_GRAY = "#5A5A5A"\n\nLIGHT_GRAY = "#D9D9D9"\n\nSENSORY_DISTRIBUTION_COLOR = "#5B5EA6"\n\nDMN_DISTRIBUTION_COLOR = "#C43C39"\n\ndef configureMatplotlib() -> None:\n    plt.rcParams.update(\n        {\n            "font.family": "sans-serif",\n            "font.sans-serif": [FONT_FAMILY],\n            "font.size": FONT_SIZE,\n            "axes.titlesize": FONT_SIZE,\n            "axes.labelsize": FONT_SIZE,\n            "axes.labelpad": 10,\n            "xtick.labelsize": FONT_SIZE,\n            "ytick.labelsize": FONT_SIZE,\n            "legend.fontsize": FONT_SIZE,\n            "figure.titlesize": FONT_SIZE,\n            "axes.spines.right": False,\n            "axes.spines.top": False,\n            "axes.linewidth": 1.6,\n            "legend.frameon": False,\n            "svg.fonttype": "none",\n            "pdf.fonttype": 42,\n            "ps.fonttype": 42,\n            "mathtext.fontset": "custom",\n            "mathtext.rm": FONT_FAMILY,\n            "mathtext.it": f"{FONT_FAMILY}:italic",\n            "mathtext.bf": f"{FONT_FAMILY}:bold",\n            "mathtext.default": "regular",\n            "axes.unicode_minus": False,\n        }\n    )\n\ndef findSingleFile(searchRoot: Path, fileName: str) -> Path:\n    matchingPaths = list(searchRoot.rglob(fileName))\n    if len(matchingPaths) != 1:\n        raise FileNotFoundError(\n            f"Expected exactly one {fileName!r}, found {len(matchingPaths)}."\n        )\n    return matchingPaths[0]\n\ndef findSingleDirectory(searchRoot: Path, requiredFileName: str) -> Path:\n    return findSingleFile(searchRoot, requiredFileName).parent\n\ndef addPanelHeading(\n    containerAxis: plt.Axes,\n    panelLabel: str,\n    panelTitle: str,\n) -> None:\n    containerAxis.set_axis_off()\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelLabel,\n        transform=containerAxis.transAxes,\n        ha="left",\n        va="bottom",\n        fontsize=PANEL_LABEL_SIZE,\n        fontweight="bold",\n    )\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelTitle,\n        transform=(\n            containerAxis.transAxes\n            + ScaledTranslation(\n                PANEL_TITLE_OFFSET_POINTS / 72,\n                0,\n                containerAxis.figure.dpi_scale_trans,\n            )\n        ),\n        ha="left",\n        va="bottom",\n        fontweight="bold",\n    )\n\ndef addGroupHeader(\n    headerAxis: plt.Axes,\n    groupName: str,\n) -> None:\n    headerAxis.set_axis_off()\n    headerAxis.text(\n        0.5,\n        0.69,\n        groupName,\n        transform=headerAxis.transAxes,\n        ha="center",\n        va="center",\n        fontweight="bold",\n    )\n\ndef stylePlotAxis(plotAxis: plt.Axes) -> None:\n    plotAxis.minorticks_off()\n    plotAxis.tick_params(\n        which="major",\n        direction="out",\n        length=7,\n        width=1.5,\n        pad=7,\n        colors=NEUTRAL_GRAY,\n    )\n    plotAxis.spines["left"].set_linewidth(1.6)\n    plotAxis.spines["bottom"].set_linewidth(1.6)\n    plotAxis.spines["left"].set_color(NEUTRAL_GRAY)\n    plotAxis.spines["bottom"].set_color(NEUTRAL_GRAY)\n\ndef makeNetworkColormap() -> matplotlibColors.ListedColormap:\n    return matplotlibColors.ListedColormap(\n        [NETWORK_COLORS[networkName] for networkName in NETWORK_ORDER],\n        name="yeo7",\n    )\n\ndef renderBrainSurfaceAssets(\n    outputDirectory: Path,\n    networkLabelFrame: pd.DataFrame,\n    ecGroupMeanZ: dict[str, dict[str, np.ndarray]],\n    gradientName: str = "G1",\n) -> dict[str, Path]:\n    assetDirectory = outputDirectory / "brainspace-assets"\n    assetDirectory.mkdir(parents=True, exist_ok=True)\n\n    surfaceLeft, surfaceRight = load_conte69()\n    parcelLabeling = load_parcellation("schaefer", scale=400, join=True)\n    parcelMask = parcelLabeling > 0\n    sourceLabels = np.arange(1, 401)\n\n    renderedPaths: dict[str, Path] = {}\n    networkNumberByName = {\n        networkName: networkIndex + 1\n        for networkIndex, networkName in enumerate(NETWORK_ORDER)\n    }\n    parcelNetworkValues = (\n        networkLabelFrame["Yeo7_network"]\n        .map(networkNumberByName)\n        .to_numpy(dtype=float)\n    )\n    vertexNetworkValues = map_to_labels(\n        parcelNetworkValues,\n        parcelLabeling,\n        mask=parcelMask,\n        fill=np.nan,\n        source_lab=sourceLabels,\n    )\n    networkArrayName = "yeo7_network"\n    surfaceLeft.append_array(\n        vertexNetworkValues[: surfaceLeft.n_points],\n        name=networkArrayName,\n        at="p",\n    )\n    networkBrainPath = assetDirectory / "yeo7-network-two-views.png"\n    plot_surf(\n        {"left": surfaceLeft},\n        np.array([["left"], ["left"]]),\n        array_name=networkArrayName,\n        view=np.array([["lateral"], ["medial"]]),\n        color_bar=False,\n        color_range=(1, 7),\n        cmap=makeNetworkColormap(),\n        nan_color=(1, 1, 1, 1),\n        size=(560, 960),\n        zoom=YEO7_ATLAS_ZOOM,\n        interactive=False,\n        screenshot=True,\n        filename=str(networkBrainPath),\n        transparent_bg=False,\n        background=(1, 1, 1),\n        scale=(2, 2),\n    )\n    renderedPaths["YEO7"] = networkBrainPath\n\n    allGradientValues = np.concatenate(\n        [\n            ecGroupMeanZ[groupName][gradientName]\n            for groupName in ("ASD", "HC")\n        ]\n    )\n    symmetricLimit = float(np.nanmax(np.abs(allGradientValues)))\n\n    for groupName in ("ASD", "HC"):\n        parcelValues = ecGroupMeanZ[groupName][gradientName]\n        vertexValues = map_to_labels(\n            parcelValues,\n            parcelLabeling,\n            mask=parcelMask,\n            fill=np.nan,\n            source_lab=sourceLabels,\n        )\n        brainPath = (\n            assetDirectory\n            / f"{groupName.lower()}-{gradientName.lower()}-ec-gradient.png"\n        )\n        plot_hemispheres(\n            surfaceLeft,\n            surfaceRight,\n            array_name=vertexValues,\n            color_bar=False,\n            color_range=(-symmetricLimit, symmetricLimit),\n            cmap=GRADIENT_CMAP,\n            nan_color=(0.95, 0.95, 0.95, 1),\n            layout_style="grid",\n            size=(900, 480),\n            zoom=GRADIENT_BRAIN_ZOOM,\n            interactive=False,\n            screenshot=True,\n            filename=str(brainPath),\n            transparent_bg=False,\n            background=(1, 1, 1),\n            scale=(2, 2),\n        )\n        renderedPaths[f"{groupName}_{gradientName}"] = brainPath\n\n    return renderedPaths\n\ndef readImageWithoutMargins(imagePath: Path) -> np.ndarray:\n    imageValues = plt.imread(imagePath)\n    if imageValues.ndim == 3 and imageValues.shape[2] == 4:\n        rgbValues = imageValues[:, :, :3]\n    else:\n        rgbValues = imageValues\n    nonWhiteMask = np.any(rgbValues < 0.985, axis=2)\n    rowIndices, columnIndices = np.where(nonWhiteMask)\n    if len(rowIndices) == 0:\n        return imageValues\n    rowStart = max(int(rowIndices.min()) - IMAGE_CROP_PADDING_PIXELS, 0)\n    rowStop = min(\n        int(rowIndices.max()) + IMAGE_CROP_PADDING_PIXELS + 1,\n        imageValues.shape[0],\n    )\n    columnStart = max(\n        int(columnIndices.min()) - IMAGE_CROP_PADDING_PIXELS,\n        0,\n    )\n    columnStop = min(\n        int(columnIndices.max()) + IMAGE_CROP_PADDING_PIXELS + 1,\n        imageValues.shape[1],\n    )\n    return imageValues[rowStart:rowStop, columnStart:columnStop]\n\ndef readGradientBrainImage(imagePath: Path) -> np.ndarray:\n    imageValues = readImageWithoutMargins(imagePath)\n    rgbValues = (\n        imageValues[:, :, :3]\n        if imageValues.ndim == 3 and imageValues.shape[2] == 4\n        else imageValues\n    )\n    rowHasBrainPixels = np.any(\n        np.any(rgbValues < 0.985, axis=2),\n        axis=1,\n    )\n    centerRow = imageValues.shape[0] // 2\n    gapStart = centerRow\n    while gapStart > 0 and not rowHasBrainPixels[gapStart - 1]:\n        gapStart -= 1\n    gapStop = centerRow\n    while (\n        gapStop < imageValues.shape[0]\n        and not rowHasBrainPixels[gapStop]\n    ):\n        gapStop += 1\n\n    detectedGapPixels = gapStop - gapStart\n    if detectedGapPixels <= GRADIENT_BRAIN_ROW_GAP_PIXELS:\n        return imageValues\n\n    retainedTopGapPixels = GRADIENT_BRAIN_ROW_GAP_PIXELS // 2\n    retainedBottomGapPixels = (\n        GRADIENT_BRAIN_ROW_GAP_PIXELS - retainedTopGapPixels\n    )\n    upperImage = imageValues[\n        : gapStart + retainedTopGapPixels\n    ]\n    lowerImage = imageValues[\n        gapStop - retainedBottomGapPixels :\n    ]\n    return np.concatenate([upperImage, lowerImage], axis=0)\n\ndef drawPanelA(\n    figure: plt.Figure,\n    panelSpec,\n    networkLongFrame: pd.DataFrame,\n    networkBrainPath: Path,\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "a",\n        "Yeo-7 organization along EC principal gradient",\n    )\n\n    nestedGrid = panelSpec.subgridspec(\n        4,\n        4,\n        height_ratios=[0.14, 0.58, 0.15, 0.13],\n        width_ratios=[0.34, 1.0, 1.18, 1.0],\n        hspace=0.04,\n        wspace=0.16,\n    )\n    randomGenerator = np.random.default_rng(RANDOM_SEED)\n    networkPositions = {\n        networkName: networkIndex\n        for networkIndex, networkName in enumerate(NETWORK_ORDER)\n    }\n    allNetworkValues = networkLongFrame["g1_net"].to_numpy(dtype=float)\n    xLimit = math.ceil(float(np.nanmax(np.abs(allNetworkValues))) * 2) / 2\n\n    distributionAxes: list[plt.Axes] = []\n    for columnIndex, groupName in ((1, "HC"), (3, "ASD")):\n        groupLabelAxis = figure.add_subplot(nestedGrid[0, columnIndex])\n        groupFrame = networkLongFrame.loc[\n            networkLongFrame["Group"].eq(groupName)\n        ].copy()\n        addGroupHeader(groupLabelAxis, groupName)\n\n        distributionAxis = figure.add_subplot(nestedGrid[1, columnIndex])\n        distributionAxes.append(distributionAxis)\n        for networkName in NETWORK_ORDER:\n            networkValues = groupFrame.loc[\n                groupFrame["network"].eq(networkName),\n                "g1_net",\n            ].to_numpy(dtype=float)\n            basePosition = networkPositions[networkName]\n            jitterValues = randomGenerator.normal(\n                loc=0.0,\n                scale=0.065,\n                size=len(networkValues),\n            )\n            distributionAxis.scatter(\n                networkValues,\n                basePosition + jitterValues,\n                s=11,\n                color=NETWORK_COLORS[networkName],\n                alpha=0.52,\n                edgecolors="none",\n                rasterized=True,\n            )\n            distributionAxis.plot(\n                [\n                    np.nanmedian(networkValues),\n                    np.nanmedian(networkValues),\n                ],\n                [basePosition - 0.25, basePosition + 0.25],\n                color=NETWORK_COLORS[networkName],\n                linewidth=1.8,\n                solid_capstyle="round",\n            )\n\n        distributionAxis.set_xlim(-xLimit, xLimit)\n        distributionAxis.set_ylim(-0.5, len(NETWORK_ORDER) - 0.5)\n        distributionAxis.set_yticks(range(len(NETWORK_ORDER)))\n        if columnIndex == 1:\n            distributionAxis.set_yticklabels(\n                [NETWORK_DISPLAY_NAMES[name] for name in NETWORK_ORDER]\n            )\n        else:\n            distributionAxis.set_yticklabels([])\n        distributionAxis.axvline(\n            0,\n            color=LIGHT_GRAY,\n            linewidth=1.1,\n            linestyle="--",\n            zorder=0,\n        )\n        stylePlotAxis(distributionAxis)\n\n        # Keep the left HC y-axis line and render its visible labels in black.\n        distributionAxis.tick_params(\n            axis="y",\n            which="both",\n            labelcolor="black",\n        )\n\n        # Remove only the right ASD y-axis line, tick marks, and tick labels.\n        if groupName == "ASD":\n            distributionAxis.spines["left"].set_visible(False)\n            distributionAxis.tick_params(\n                axis="y",\n                which="both",\n                left=False,\n                right=False,\n                labelleft=False,\n            )\n\n    # Use a single x-axis title centered beneath the complete HC/ASD block.\n    figure.canvas.draw()\n    leftDistributionPosition = distributionAxes[0].get_position(original=False)\n    rightDistributionPosition = distributionAxes[1].get_position(original=False)\n    sharedXAxisCenter = (\n        leftDistributionPosition.x0 + rightDistributionPosition.x1\n    ) / 2\n    sharedXAxisLabelAxis = figure.add_subplot(nestedGrid[2, 1:4])\n    sharedXAxisLabelAxis.set_axis_off()\n    sharedXAxisLabelPosition = sharedXAxisLabelAxis.get_position(original=False)\n    sharedXAxisCenterInLabelAxis = (\n        sharedXAxisCenter - sharedXAxisLabelPosition.x0\n    ) / sharedXAxisLabelPosition.width\n    sharedXAxisLabelAxis.text(\n        sharedXAxisCenterInLabelAxis,\n        0.34,\n        "EC principal gradient (z)",\n        transform=sharedXAxisLabelAxis.transAxes,\n        ha="center",\n        va="center",\n    )\n\n    brainTitleAxis = figure.add_subplot(nestedGrid[0, 2])\n    brainTitleAxis.set_axis_off()\n    brainTitleAxis.text(\n        0.5,\n        0.48,\n        "Yeo-7 atlas",\n        transform=brainTitleAxis.transAxes,\n        ha="center",\n        va="center",\n        fontweight="bold",\n    )\n    brainAxis = figure.add_subplot(nestedGrid[1, 2])\n    brainAxis.imshow(readImageWithoutMargins(networkBrainPath))\n    brainAxis.set_axis_off()\n\n    legendAxis = figure.add_subplot(nestedGrid[3, :])\n    legendAxis.set_axis_off()\n    legendHandles = [\n        Patch(\n            facecolor=NETWORK_COLORS[networkName],\n            edgecolor="none",\n            label=NETWORK_DISPLAY_NAMES[networkName],\n        )\n        for networkName in NETWORK_ORDER\n    ]\n    firstLegend = legendAxis.legend(\n        handles=legendHandles[:4],\n        loc="center",\n        bbox_to_anchor=(0.5, 0.78),\n        ncol=4,\n        handlelength=0.8,\n        handleheight=0.8,\n        handletextpad=0.4,\n        columnspacing=0.9,\n        borderaxespad=0,\n    )\n    legendAxis.add_artist(firstLegend)\n    legendAxis.legend(\n        handles=legendHandles[4:],\n        loc="center",\n        bbox_to_anchor=(0.5, 0.18),\n        ncol=3,\n        handlelength=0.8,\n        handleheight=0.8,\n        handletextpad=0.4,\n        columnspacing=1.0,\n        borderaxespad=0,\n    )\n\ndef drawPanelB(\n    figure: plt.Figure,\n    panelSpec,\n    brainPaths: dict[str, Path],\n    ecGroupMeanZ: dict[str, dict[str, np.ndarray]],\n    gradientName: str = "G1",\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "b",\n        f"EC gradient {gradientName.removeprefix(\'G\')} (subject-wise whole-brain z-score)",\n    )\n\n    nestedGrid = panelSpec.subgridspec(\n        4,\n        2,\n        height_ratios=[0.14, 0.71, 0.09, 0.06],\n        hspace=0.01,\n        wspace=PANEL_B_GROUP_WSPACE,\n    )\n    allGradientValues = np.concatenate(\n        [\n            ecGroupMeanZ[groupName][gradientName]\n            for groupName in ("ASD", "HC")\n        ]\n    )\n    sharedSymmetricLimit = float(np.nanmax(np.abs(allGradientValues)))\n    scalarMap = plt.cm.ScalarMappable(\n        norm=matplotlibColors.Normalize(\n            vmin=-sharedSymmetricLimit,\n            vmax=sharedSymmetricLimit,\n        ),\n        cmap=GRADIENT_CMAP,\n    )\n\n    for columnIndex, groupName in enumerate(("HC", "ASD")):\n        groupHeaderAxis = figure.add_subplot(nestedGrid[0, columnIndex])\n        addGroupHeader(groupHeaderAxis, groupName)\n\n        brainAxis = figure.add_subplot(nestedGrid[1, columnIndex])\n        brainAxis.imshow(\n            readGradientBrainImage(brainPaths[f"{groupName}_{gradientName}"])\n        )\n        brainAxis.set_axis_off()\n\n        groupColorbarContainerAxis = figure.add_subplot(\n            nestedGrid[2, columnIndex]\n        )\n        groupColorbarContainerAxis.set_axis_off()\n        groupColorbarAxis = groupColorbarContainerAxis.inset_axes(\n            GROUP_COLORBAR_BOUNDS\n        )\n        groupColorbar = figure.colorbar(\n            scalarMap,\n            cax=groupColorbarAxis,\n            orientation="horizontal",\n        )\n        groupColorbar.set_label(\n            "Group-mean z-scored gradient",\n            labelpad=5,\n        )\n        groupColorbar.ax.tick_params(\n            direction="out",\n            length=6,\n            width=1.5,\n            pad=5,\n        )\n        groupColorbar.outline.set_linewidth(0.7)\n\n\n\ndef saveRowFigure(figure: plt.Figure, outputBase: Path) -> list[Path]:\n    outputPaths = [outputBase.with_suffix(extension) for extension in (".svg", ".pdf", ".png", ".tiff")]\n    figure.savefig(outputPaths[0], facecolor="white")\n    figure.savefig(outputPaths[1], facecolor="white")\n    figure.savefig(outputPaths[2], dpi=EXPORT_DPI, facecolor="white")\n    figure.savefig(outputPaths[3], dpi=EXPORT_DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})\n    plt.close(figure)\n    return outputPaths\n\n\ndef loadRowOneInputs(analysisRoot: Path) -> dict[str, object]:\n    resultDirectory = findSingleDirectory(analysisRoot, "Yeo7_network_level_G1G2_long.csv")\n    networkLongFrame = pd.read_csv(resultDirectory / "Yeo7_network_level_G1G2_long.csv")\n    g1Matrix = np.load(resultDirectory / "G1_matrix.npy")\n    networkLabelPath = (\n        analysisRoot.parents[2]\n        / "3.步进分析"\n        / "Schaefer400_7Yeo_network_labels_with_ID.csv"\n    )\n    networkLabelFrame = pd.read_csv(networkLabelPath)\n    if g1Matrix.shape != (networkLongFrame["sub_id"].nunique(), 400): raise ValueError("G1 matrix shape is inconsistent with the source data.")\n    return {"networkLongFrame": networkLongFrame, "g1Matrix": g1Matrix, "networkLabelFrame": networkLabelFrame}\n\n\ndef createRowOneFigure(analysisRoot: Path, outputDirectory: Path) -> list[Path]:\n    configureMatplotlib(); outputDirectory.mkdir(parents=True, exist_ok=True)\n    inputs = loadRowOneInputs(analysisRoot)\n    subjectFrame = inputs["networkLongFrame"][["sub_id", "Group"]].drop_duplicates().reset_index(drop=True)\n    g1SubjectZ = zscore(inputs["g1Matrix"], axis=1, ddof=0, nan_policy="omit")\n    if not np.isfinite(g1SubjectZ).all(): raise ValueError("G1 z-scoring produced non-finite values.")\n    ecGroupMeanZ = {groupName: {"G1": np.mean(g1SubjectZ[subjectFrame["Group"].eq(groupName).to_numpy()], axis=0)} for groupName in ("ASD", "HC")}\n    brainPaths = renderBrainSurfaceAssets(outputDirectory, inputs["networkLabelFrame"], ecGroupMeanZ)\n    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 5.30), facecolor="white")\n    grid = figure.add_gridspec(1, 13, wspace=0.32, left=60/1800, right=1-60/1800, top=1-35/568, bottom=40/568)\n    drawPanelA(figure, grid[0, :5], inputs["networkLongFrame"], brainPaths["YEO7"])\n    drawPanelB(figure, grid[0, 5:], brainPaths, ecGroupMeanZ)\n    return saveRowFigure(figure, outputDirectory / "gradient-analysis-row-1-panels-ab")\n\n\ndef main() -> None:\n    analysisRoot = Path(__file__).resolve().parent.parent\n    for outputPath in createRowOneFigure(analysisRoot, analysisRoot): print(outputPath.name)\n\nif __name__ == "__main__": main()\n', 'plot-gradient-analysis-row-2-panels-cde.py': 'from __future__ import annotations\n\nimport math\n\nfrom pathlib import Path\n\nimport matplotlib\n\nmatplotlib.use("Agg")\n\nimport matplotlib.colors as matplotlibColors\n\nimport matplotlib.patheffects as pathEffects\n\nimport matplotlib.pyplot as plt\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom brainspace.datasets import load_conte69, load_parcellation\n\nfrom brainspace.plotting import plot_hemispheres, plot_surf\n\nfrom brainspace.utils.parcellation import map_to_labels\n\nfrom matplotlib.lines import Line2D\n\nfrom matplotlib.colors import to_rgba\n\nfrom matplotlib.patches import Patch\n\nfrom matplotlib.transforms import ScaledTranslation\n\nfrom scipy.stats import gaussian_kde, spearmanr, ttest_ind, zscore\n\nFONT_FAMILY = "Arial"\n\nFONT_SIZE = 18\n\nPANEL_LABEL_SIZE = 22\n\nPANEL_TITLE_OFFSET_POINTS = 23\n\nFIGURE_WIDTH_INCHES = 18.0\n\nFIGURE_HEIGHT_INCHES = 17.8\n\nAB_ROW_HEIGHT_INCHES = 5.68\n\nCD_ROW_HEIGHT_INCHES = 6.30\n\nEF_ROW_HEIGHT_INCHES = 5.68\n\nHISTOGRAM_BIN_WIDTH = 0.1\n\nPLOT_AXIS_HEIGHT_INCHES = 3.60\n\nPANEL_E_PLOT_WIDTH_SCALE = 0.70\n\nD_MEAN_MAP_SHIFT_INCHES = 0.30\n\nD_MEAN_MAP_EXTRA_WIDTH_INCHES = 0.15\n\nEXPORT_DPI = 100\n\nRANDOM_SEED = 42\n\nYEO7_ATLAS_ZOOM = 1.44\n\nGRADIENT_BRAIN_ZOOM = 1.72\n\nIMAGE_CROP_PADDING_PIXELS = 12\n\nGRADIENT_BRAIN_ROW_GAP_PIXELS = 4\n\nGROUP_COLORBAR_BOUNDS = (0.12, 0.88, 0.76, 0.46)\n\nPANEL_B_GROUP_WSPACE = 0.02\n\nP_VALUE_CMAP = "viridis_r"\n\nP_VALUE_BRAIN_ZOOM = 1.70\n\nGROUP_COLORS = {\n    "HC": "#4C78A8",\n    "ASD": "#E07A5F",\n}\n\nNETWORK_ORDER = [\n    "Vis",\n    "SomMot",\n    "DorsAttn",\n    "SalVentAttn",\n    "Limbic",\n    "Cont",\n    "Default",\n]\n\nNETWORK_DISPLAY_NAMES = {\n    "Vis": "Vis",\n    "SomMot": "SMN",\n    "DorsAttn": "DAN",\n    "SalVentAttn": "VAN",\n    "Limbic": "Lim",\n    "Cont": "Cont",\n    "Default": "DMN",\n}\n\nNETWORK_COLORS = {\n    "Vis": "#6D56A5",\n    "SomMot": "#4B9BD3",\n    "DorsAttn": "#58A65C",\n    "SalVentAttn": "#E3A33B",\n    "Limbic": "#D9B23C",\n    "Cont": "#39A7A0",\n    "Default": "#D34B4B",\n}\n\nGRADIENT_CMAP = "RdBu_r"\n\nNEUTRAL_GRAY = "#5A5A5A"\n\nLIGHT_GRAY = "#D9D9D9"\n\nSENSORY_DISTRIBUTION_COLOR = "#5B5EA6"\n\nDMN_DISTRIBUTION_COLOR = "#C43C39"\n\ndef configureMatplotlib() -> None:\n    plt.rcParams.update(\n        {\n            "font.family": "sans-serif",\n            "font.sans-serif": [FONT_FAMILY],\n            "font.size": FONT_SIZE,\n            "axes.titlesize": FONT_SIZE,\n            "axes.labelsize": FONT_SIZE,\n            "axes.labelpad": 10,\n            "xtick.labelsize": FONT_SIZE,\n            "ytick.labelsize": FONT_SIZE,\n            "legend.fontsize": FONT_SIZE,\n            "figure.titlesize": FONT_SIZE,\n            "axes.spines.right": False,\n            "axes.spines.top": False,\n            "axes.linewidth": 1.6,\n            "legend.frameon": False,\n            "svg.fonttype": "none",\n            "pdf.fonttype": 42,\n            "ps.fonttype": 42,\n            "mathtext.fontset": "custom",\n            "mathtext.rm": FONT_FAMILY,\n            "mathtext.it": f"{FONT_FAMILY}:italic",\n            "mathtext.bf": f"{FONT_FAMILY}:bold",\n            "mathtext.default": "regular",\n            "axes.unicode_minus": False,\n        }\n    )\n\ndef findSingleFile(searchRoot: Path, fileName: str) -> Path:\n    matchingPaths = list(searchRoot.rglob(fileName))\n    if len(matchingPaths) != 1:\n        raise FileNotFoundError(\n            f"Expected exactly one {fileName!r}, found {len(matchingPaths)}."\n        )\n    return matchingPaths[0]\n\ndef findSingleDirectory(searchRoot: Path, requiredFileName: str) -> Path:\n    return findSingleFile(searchRoot, requiredFileName).parent\n\ndef addPanelHeading(\n    containerAxis: plt.Axes,\n    panelLabel: str,\n    panelTitle: str,\n) -> None:\n    containerAxis.set_axis_off()\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelLabel,\n        transform=containerAxis.transAxes,\n        ha="left",\n        va="bottom",\n        fontsize=PANEL_LABEL_SIZE,\n        fontweight="bold",\n    )\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelTitle,\n        transform=(\n            containerAxis.transAxes\n            + ScaledTranslation(\n                PANEL_TITLE_OFFSET_POINTS / 72,\n                0,\n                containerAxis.figure.dpi_scale_trans,\n            )\n        ),\n        ha="left",\n        va="bottom",\n        fontweight="bold",\n    )\n\ndef stylePlotAxis(plotAxis: plt.Axes) -> None:\n    plotAxis.minorticks_off()\n    plotAxis.tick_params(\n        which="major",\n        direction="out",\n        length=7,\n        width=1.5,\n        pad=7,\n        colors=NEUTRAL_GRAY,\n    )\n    plotAxis.spines["left"].set_linewidth(1.6)\n    plotAxis.spines["bottom"].set_linewidth(1.6)\n    plotAxis.spines["left"].set_color(NEUTRAL_GRAY)\n    plotAxis.spines["bottom"].set_color(NEUTRAL_GRAY)\n\ndef addSubplotHeader(\n    headerAxis: plt.Axes,\n    title: str,\n) -> None:\n    headerAxis.set_axis_off()\n    headerAxis.text(\n        0.5,\n        0.50,\n        title,\n        transform=headerAxis.transAxes,\n        ha="center",\n        va="center",\n        fontweight="normal",\n    )\n\ndef setUnifiedPlotHeight(\n    plotAxis: plt.Axes,\n    figure: plt.Figure,\n) -> None:\n    """Give every plotting axis the same physical height."""\n    axisPosition = plotAxis.get_position()\n    axisWidthInches = axisPosition.width * figure.figure.get_figwidth()\n    if axisWidthInches <= 0:\n        raise ValueError("Plot axis width must be positive.")\n    plotAxis.set_box_aspect(PLOT_AXIS_HEIGHT_INCHES / axisWidthInches)\n    plotAxis.set_anchor("N")\n\n\ndef alignPlotAxesVertically(\n    plotAxes: list[plt.Axes],\n    referenceAxis: plt.Axes,\n) -> None:\n    """Align all main plots to the same exact top and bottom boundaries."""\n    referencePosition = referenceAxis.get_position(original=False)\n    for plotAxis in plotAxes:\n        currentPosition = plotAxis.get_position(original=False)\n        plotAxis.set_box_aspect(None)\n        plotAxis.set_position(\n            [\n                currentPosition.x0,\n                referencePosition.y0,\n                currentPosition.width,\n                referencePosition.height,\n            ]\n        )\n\n\ndef alignHeaderAxesVertically(\n    figure: plt.Figure,\n    headerTitles: set[str],\n    referenceTitle: str,\n) -> None:\n    """Align subplot header bands and their text centerlines."""\n    headerAxesByTitle = {\n        textArtist.get_text(): headerAxis\n        for headerAxis in figure.axes\n        for textArtist in headerAxis.texts\n        if textArtist.get_text() in headerTitles\n    }\n    referencePosition = headerAxesByTitle[referenceTitle].get_position(\n        original=False\n    )\n    for headerTitle, headerAxis in headerAxesByTitle.items():\n        currentPosition = headerAxis.get_position(original=False)\n        headerAxis.set_position(\n            [\n                currentPosition.x0,\n                referencePosition.y0,\n                currentPosition.width,\n                referencePosition.height,\n            ]\n        )\n        for textArtist in headerAxis.texts:\n            if textArtist.get_text() == headerTitle:\n                textArtist.set_y(0.5)\n                textArtist.set_va("center")\n\n\ndef scalePlotAxesWidths(\n    plotAxes: list[plt.Axes],\n    widthScale: float,\n) -> None:\n    """Scale widths without changing horizontal centers or vertical bounds."""\n    for plotAxis in plotAxes:\n        currentPosition = plotAxis.get_position(original=False)\n        scaledWidth = currentPosition.width * widthScale\n        horizontalCenter = currentPosition.x0 + currentPosition.width / 2\n        plotAxis.set_position(\n            [\n                horizontalCenter - scaledWidth / 2,\n                currentPosition.y0,\n                scaledWidth,\n                currentPosition.height,\n            ]\n        )\n\n\ndef drawNetworkCentroidLabels(\n    plotAxis: plt.Axes,\n    xValues: np.ndarray,\n    yValues: np.ndarray,\n    parcelNetworks: np.ndarray,\n) -> None:\n    labelOffsets = {\n        "Vis": (-2, 10),\n        "SomMot": (-23, -17),\n        "DorsAttn": (-24, 10),\n        "SalVentAttn": (15, -27),\n        "Limbic": (25, -7),\n        "Cont": (27, 5),\n        "Default": (23, 20),\n    }\n    for networkName in NETWORK_ORDER:\n        networkMask = parcelNetworks == networkName\n        xMedian = float(np.nanmedian(xValues[networkMask]))\n        yMedian = float(np.nanmedian(yValues[networkMask]))\n        labelArtist = plotAxis.annotate(\n            NETWORK_DISPLAY_NAMES[networkName],\n            xy=(xMedian, yMedian),\n            xytext=labelOffsets[networkName],\n            textcoords="offset points",\n            color=NETWORK_COLORS[networkName],\n            ha="center",\n            va="center",\n            fontweight="normal",\n        )\n        labelArtist.set_path_effects(\n            [\n                pathEffects.withStroke(linewidth=1.6, foreground="white"),\n            ]\n        )\n\ndef drawPanelC(\n    figure: plt.Figure,\n    panelSpec,\n    ecGroupMeanZ: dict[str, dict[str, np.ndarray]],\n    networkLabelFrame: pd.DataFrame,\n    subjectFrame: pd.DataFrame,\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "c",\n        "EC gradient space (400 parcels)",\n    )\n\n    nestedGrid = panelSpec.subgridspec(\n        2,\n        3,\n        height_ratios=[0.10, 0.90],\n        width_ratios=[0.28, 1.0, 1.0],\n        hspace=0.04,\n        wspace=0.30,\n    )\n    parcelNetworks = networkLabelFrame["Yeo7_network"].to_numpy()\n    combinedG1Values = np.concatenate(\n        [ecGroupMeanZ[groupName]["G1"] for groupName in ("HC", "ASD")]\n    )\n    combinedG2Values = np.concatenate(\n        [ecGroupMeanZ[groupName]["G2"] for groupName in ("HC", "ASD")]\n    )\n    xLimit = math.ceil(float(np.nanmax(np.abs(combinedG1Values))) * 2) / 2\n    yLimit = math.ceil(float(np.nanmax(np.abs(combinedG2Values))) * 2) / 2\n    randomGenerator = np.random.default_rng(RANDOM_SEED)\n    scatterAxes: list[plt.Axes] = []\n\n    for columnIndex, groupName in enumerate(("HC", "ASD")):\n        gridColumn = columnIndex + 1\n        groupHeaderAxis = figure.add_subplot(nestedGrid[0, gridColumn])\n        addSubplotHeader(groupHeaderAxis, groupName)\n\n        scatterAxis = figure.add_subplot(nestedGrid[1, gridColumn])\n        g1Values = ecGroupMeanZ[groupName]["G1"]\n        g2Values = ecGroupMeanZ[groupName]["G2"]\n        randomizedIndices = randomGenerator.permutation(len(g1Values))\n        pointColors = [\n            NETWORK_COLORS[networkName]\n            for networkName in parcelNetworks[randomizedIndices]\n        ]\n        scatterAxis.scatter(\n            g1Values[randomizedIndices],\n            g2Values[randomizedIndices],\n            s=16,\n            color=pointColors,\n            alpha=0.63,\n            edgecolors="none",\n            rasterized=True,\n        )\n        drawNetworkCentroidLabels(\n            scatterAxis,\n            g1Values,\n            g2Values,\n            parcelNetworks,\n        )\n        scatterAxis.axvline(\n            0,\n            color=LIGHT_GRAY,\n            linestyle="--",\n            linewidth=1.1,\n            zorder=0,\n        )\n        scatterAxis.axhline(\n            0,\n            color=LIGHT_GRAY,\n            linestyle="--",\n            linewidth=1.1,\n            zorder=0,\n        )\n        scatterAxis.set_xlim(-xLimit, xLimit)\n        scatterAxis.set_ylim(-yLimit, yLimit)\n        setUnifiedPlotHeight(scatterAxis, figure)\n        scatterAxis.set_xlabel("")\n        if columnIndex == 0:\n            scatterAxis.set_ylabel("EC gradient 2 (z)")\n        else:\n            scatterAxis.set_ylabel("")\n        stylePlotAxis(scatterAxis)\n        if columnIndex == 1:\n            scatterAxis.spines["left"].set_visible(False)\n            scatterAxis.tick_params(axis="y", left=False, length=0, labelleft=False)\n        scatterAxes.append(scatterAxis)\n\n    # Use the left scatter axis as the carrier for one shared x-axis label.\n    # Keeping it as a native axis label makes its vertical position identical\n    # to the x-axis labels in panel d, while its x coordinate is centered\n    # across both panel-c scatter plots.\n    figure.canvas.draw()\n    leftScatterPosition = scatterAxes[0].get_position(original=False)\n    rightScatterPosition = scatterAxes[1].get_position(original=False)\n    sharedXCenter = (leftScatterPosition.x0 + rightScatterPosition.x1) / 2\n    sharedXInLeftAxis = (\n        sharedXCenter - leftScatterPosition.x0\n    ) / leftScatterPosition.width\n    scatterAxes[0].set_xlabel(\n        "EC gradient 1 (z)",\n        x=sharedXInLeftAxis,\n        ha="center",\n    )\n\ndef drawPanelDSummary(\n    figure: plt.Figure,\n    panelSpec,\n    groupMeanMapFrame: pd.DataFrame,\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "d",\n        "FC–EC G1 correlation",\n    )\n    nestedGrid = panelSpec.subgridspec(\n        2,\n        1,\n        height_ratios=[0.12, 0.88],\n        hspace=0.04,\n    )\n    headerAxis = figure.add_subplot(nestedGrid[0, 0])\n    addSubplotHeader(headerAxis, "Group-average map")\n\n    summaryAxis = figure.add_subplot(nestedGrid[1, 0])\n    ecG1Values = groupMeanMapFrame["EC_G1"].to_numpy(dtype=float)\n    fcG1Values = groupMeanMapFrame["FC_G1"].to_numpy(dtype=float)\n    correlationCoefficient, _ = spearmanr(ecG1Values, fcG1Values)\n    summaryAxis.scatter(\n        ecG1Values,\n        fcG1Values,\n        s=11,\n        color="#2F80ED",\n        alpha=0.82,\n        edgecolors="none",\n        rasterized=True,\n    )\n    summaryAxis.text(\n        0.05,\n        0.95,\n        f"$\\\\rho_{{\\\\mathrm{{map}}}}$ = {correlationCoefficient:.4f}\\n"\n        r"$p_{\\mathrm{spin}}$ < 0.001",\n        transform=summaryAxis.transAxes,\n        ha="left",\n        va="top",\n    )\n    summaryAxis.set_xlabel("Group-average EC-G1")\n    summaryAxis.set_ylabel("Group-average FC-G1")\n    stylePlotAxis(summaryAxis)\n    setUnifiedPlotHeight(summaryAxis, figure)\n    return\n\n    groupOrder = ["ASD", "HC"]\n    groupValues = [\n        subjectCorrelationFrame.loc[\n            subjectCorrelationFrame["Group"].eq(groupName),\n            "EC_G1_vs_FC_G1_r",\n        ]\n        .dropna()\n        .to_numpy(dtype=float)\n        for groupName in groupOrder\n    ]\n    groupMeans = [float(np.mean(values)) for values in groupValues]\n    groupStandardDeviations = [\n        float(np.std(values, ddof=1)) for values in groupValues\n    ]\n    summaryAxis.bar(\n        np.arange(2),\n        groupMeans,\n        yerr=groupStandardDeviations,\n        width=0.72,\n        capsize=6,\n        color=[GROUP_COLORS[groupName] for groupName in groupOrder],\n        edgecolor=NEUTRAL_GRAY,\n        linewidth=1.0,\n        error_kw={"elinewidth": 1.4, "capthick": 1.4},\n    )\n    randomGenerator = np.random.default_rng(RANDOM_SEED)\n    for groupIndex, (groupName, values) in enumerate(zip(groupOrder, groupValues)):\n        pointJitter = randomGenerator.normal(0, 0.055, size=len(values))\n        summaryAxis.scatter(\n            groupIndex + pointJitter,\n            values,\n            s=10,\n            color=GROUP_COLORS[groupName],\n            alpha=0.27,\n            edgecolors="none",\n            rasterized=True,\n            zorder=3,\n        )\n    summaryAxis.set_ylabel("FC G1 vs EC G1 Spearman ρ")\n    summaryAxis.set_xticks(np.arange(2))\n    summaryAxis.set_xticklabels(groupOrder)\n    summaryAxis.set_ylim(0.40, 0.90)\n    stylePlotAxis(summaryAxis)\n    setUnifiedPlotHeight(summaryAxis, figure)\n\ndef drawViolinWithPoints(\n    violinAxis: plt.Axes,\n    groupValues: list[np.ndarray],\n    groupOrder: list[str],\n) -> None:\n    violinParts = violinAxis.violinplot(\n        groupValues,\n        positions=np.arange(len(groupOrder)),\n        widths=0.72,\n        showmeans=False,\n        showmedians=False,\n        showextrema=False,\n    )\n    for violinBody, groupName in zip(violinParts["bodies"], groupOrder):\n        violinBody.set_facecolor(GROUP_COLORS[groupName])\n        violinBody.set_edgecolor(GROUP_COLORS[groupName])\n        violinBody.set_alpha(0.28)\n        violinBody.set_linewidth(1.7)\n\n    violinAxis.boxplot(\n        groupValues,\n        positions=np.arange(len(groupOrder)),\n        widths=0.22,\n        patch_artist=True,\n        showfliers=False,\n        medianprops={"color": NEUTRAL_GRAY, "linewidth": 1.8},\n        boxprops={\n            "facecolor": "white",\n            "edgecolor": NEUTRAL_GRAY,\n            "linewidth": 1.2,\n        },\n        whiskerprops={"color": NEUTRAL_GRAY, "linewidth": 1.2},\n        capprops={"color": NEUTRAL_GRAY, "linewidth": 1.2},\n    )\n\n    randomGenerator = np.random.default_rng(RANDOM_SEED)\n    for groupIndex, (groupName, values) in enumerate(zip(groupOrder, groupValues)):\n        pointJitter = randomGenerator.normal(0, 0.055, size=len(values))\n        violinAxis.scatter(\n            groupIndex + pointJitter,\n            values,\n            s=9,\n            color=GROUP_COLORS[groupName],\n            alpha=0.20,\n            edgecolors="none",\n            rasterized=True,\n            zorder=3,\n        )\n\ndef createHistogramBinEdges(values: np.ndarray) -> np.ndarray:\n    minimumValue = float(np.nanmin(values))\n    maximumValue = float(np.nanmax(values))\n    binEdges = np.arange(\n        minimumValue,\n        maximumValue,\n        HISTOGRAM_BIN_WIDTH,\n        dtype=float,\n    )\n    if binEdges.size == 0 or not np.isclose(binEdges[0], minimumValue):\n        binEdges = np.insert(binEdges, 0, minimumValue)\n    if not np.isclose(binEdges[-1], maximumValue):\n        binEdges = np.append(binEdges, maximumValue)\n    return binEdges\n\ndef drawPanelE(\n    figure: plt.Figure,\n    panelSpec,\n    hierarchyFrame: pd.DataFrame,\n    hierarchyGlmFrame: pd.DataFrame,\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "e",\n        "Default–Vis/SomMot G1 separation",\n    )\n    nestedGrid = panelSpec.subgridspec(\n        3,\n        3,\n        height_ratios=[0.08, 0.80, 0.12],\n        width_ratios=[0.22, 0.88, 1.12],\n        hspace=0.04,\n        wspace=0.50,\n    )\n    groupOrder = ["HC", "ASD"]\n    separationValues = [\n        hierarchyFrame.loc[\n            hierarchyFrame["Group"].eq(groupName),\n            "G1_Default_dist_VisSomMot",\n        ]\n        .dropna()\n        .to_numpy(dtype=float)\n        for groupName in groupOrder\n    ]\n\n    violinAxis = figure.add_subplot(nestedGrid[1, 1])\n    drawViolinWithPoints(violinAxis, separationValues, groupOrder)\n    violinAxis.set_xticks(np.arange(2))\n    violinAxis.set_xticklabels(\n        [\n            rf"$\\bf{{HC}}$",\n            rf"$\\bf{{ASD}}$",\n        ]\n    )\n    violinAxis.set_ylabel("G1 separation (z)")\n    combinedValues = np.concatenate(separationValues)\n    valueRange = float(np.nanmax(combinedValues) - np.nanmin(combinedValues))\n    dataMaximum = float(np.nanmax(combinedValues))\n    violinAxis.set_ylim(\n        min(0.40, float(np.nanmin(combinedValues)) - valueRange * 0.04),\n        dataMaximum + valueRange * 0.16,\n    )\n\n    # Replace the beta and p-value text with a conventional significance mark.\n    bracketY = dataMaximum + valueRange * 0.075\n    bracketHeight = valueRange * 0.025\n    violinAxis.plot(\n        [0, 0, 1, 1],\n        [bracketY, bracketY + bracketHeight, bracketY + bracketHeight, bracketY],\n        color=NEUTRAL_GRAY,\n        linewidth=1.4,\n        clip_on=False,\n    )\n    violinAxis.text(\n        0.5,\n        bracketY + bracketHeight + valueRange * 0.012,\n        "***",\n        ha="center",\n        va="bottom",\n        fontweight="bold",\n    )\n\n    statisticsAxis = figure.add_subplot(nestedGrid[0, 1])\n    statisticsAxis.set_axis_off()\n    statisticsAxis.text(\n        0.5,\n        0.20,\n        "Group comparison",\n        transform=statisticsAxis.transAxes,\n        ha="center",\n        va="center",\n        fontweight="bold",\n    )\n    stylePlotAxis(violinAxis)\n\n    densityAxis = figure.add_subplot(nestedGrid[1, 2])\n    binEdges = createHistogramBinEdges(combinedValues)\n    for groupName, values in zip(groupOrder, separationValues):\n        percentageWeights = np.full(values.size, 100.0 / values.size)\n        densityAxis.hist(\n            values,\n            bins=binEdges,\n            weights=percentageWeights,\n            color=to_rgba(GROUP_COLORS[groupName], 0.20),\n            edgecolor="#6B6B6B",\n            linewidth=1.0,\n            histtype="bar",\n            rwidth=1.0,\n            label=groupName,\n        )\n        smoothXValues = np.linspace(binEdges[0], binEdges[-1], 400)\n        densityAxis.plot(\n            smoothXValues,\n            gaussian_kde(values)(smoothXValues)\n            * HISTOGRAM_BIN_WIDTH\n            * 100.0,\n            color=GROUP_COLORS[groupName],\n            linewidth=1.8,\n        )\n        densityAxis.axvline(\n            float(np.mean(values)),\n            color=GROUP_COLORS[groupName],\n            linestyle="--",\n            linewidth=1.1,\n            alpha=0.75,\n            ymax=0.92,\n        )\n    participantTitleAxis = figure.add_subplot(nestedGrid[0, 2])\n    participantTitleAxis.set_axis_off()\n    participantTitleAxis.text(\n        0.5,\n        0.20,\n        "Participant-level distribution",\n        transform=participantTitleAxis.transAxes,\n        ha="center",\n        va="center",\n        fontweight="bold",\n    )\n    densityAxis.set_xlabel("G1 separation (z)")\n    densityAxis.set_ylabel("Percentage (%)")\n    densityAxis.legend(\n        loc="upper left",\n        bbox_to_anchor=(0.02, 0.84),\n        handlelength=1.4,\n        labelspacing=0.5,\n        handletextpad=0.6,\n    )\n    stylePlotAxis(densityAxis)\n    setUnifiedPlotHeight(violinAxis, figure)\n    setUnifiedPlotHeight(densityAxis, figure)\n    violinAxis.tick_params(axis="x", pad=2)\n    densityAxis.xaxis.labelpad = 6\n    groupComparisonHorizontalShift = 20 / 1800\n    for axisToShift in (containerAxis, statisticsAxis, violinAxis):\n        originalPosition = axisToShift.get_position()\n        axisToShift.set_position(\n            [\n                originalPosition.x0 + groupComparisonHorizontalShift,\n                originalPosition.y0,\n                originalPosition.width,\n                originalPosition.height,\n                ]\n            )\n\n\n\ndef saveRowFigure(figure: plt.Figure, outputBase: Path) -> list[Path]:\n    outputPaths = [outputBase.with_suffix(extension) for extension in (".svg", ".pdf", ".png", ".tiff")]\n    figure.savefig(outputPaths[0], facecolor="white")\n    figure.savefig(outputPaths[1], facecolor="white")\n    figure.savefig(outputPaths[2], dpi=EXPORT_DPI, facecolor="white")\n    figure.savefig(outputPaths[3], dpi=EXPORT_DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})\n    plt.close(figure)\n    return outputPaths\n\n\ndef loadRowTwoInputs(analysisRoot: Path) -> dict[str, object]:\n    resultDirectory = findSingleDirectory(analysisRoot, "Yeo7_network_level_G1G2_long.csv")\n    correlationDirectory = (\n        analysisRoot.parents[1]\n        / "补充内容-最新"\n        / "S4"\n        / "ABIDE1_EC_FC_BrainSpace空间置换检验_combat_2x2_cross_axis-HC_ASD分组"\n        / "subject_mean_wholebrain_results"\n    )\n    networkLongFrame = pd.read_csv(resultDirectory / "Yeo7_network_level_G1G2_long.csv")\n    networkLabelPath = (\n        analysisRoot.parents[2]\n        / "3.步进分析"\n        / "Schaefer400_7Yeo_network_labels_with_ID.csv"\n    )\n    return {"networkLongFrame": networkLongFrame, "g1Matrix": np.load(resultDirectory / "G1_matrix.npy"), "g2Matrix": np.load(resultDirectory / "G2_matrix.npy"), "hierarchyFrame": pd.read_csv(resultDirectory / "Yeo7_hierarchy_distances_G1G2.csv"), "hierarchyGlmFrame": pd.read_csv(resultDirectory / "GLM_hierarchy_distance_G1_Default_dist_VisSomMot_main_effect_rawP.csv"), "subjectCorrelationFrame": pd.read_csv(correlationDirectory / "ALL_HC_ASD_subject_mean_wholebrain_correlation_across_4_cross_axis_comparisons.csv"), "networkLabelFrame": pd.read_csv(networkLabelPath)}\n\n\ndef createRowTwoFigure(analysisRoot: Path, outputDirectory: Path) -> list[Path]:\n    configureMatplotlib(); outputDirectory.mkdir(parents=True, exist_ok=True); inputs = loadRowTwoInputs(analysisRoot)\n    subjectFrame = inputs["networkLongFrame"][["sub_id", "Group"]].drop_duplicates().reset_index(drop=True)\n    if inputs["g1Matrix"].shape != (len(subjectFrame), 400): raise ValueError("G1 matrix shape is inconsistent with the source data.")\n    g1SubjectZ = zscore(inputs["g1Matrix"], axis=1, ddof=0, nan_policy="omit")\n    g2SubjectZ = zscore(inputs["g2Matrix"], axis=1, ddof=0, nan_policy="omit")\n    if not np.isfinite(g1SubjectZ).all() or not np.isfinite(g2SubjectZ).all(): raise ValueError("Gradient z-scoring produced non-finite values.")\n    ecGroupMeanZ = {groupName: {"G1": np.mean(g1SubjectZ[subjectFrame["Group"].eq(groupName).to_numpy()], axis=0), "G2": np.mean(g2SubjectZ[subjectFrame["Group"].eq(groupName).to_numpy()], axis=0)} for groupName in ("ASD", "HC")}\n    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 5.55), facecolor="white")\n    grid = figure.add_gridspec(1, 13, wspace=0.46, left=25/1800, right=1-25/1800, top=1-30/555, bottom=22/555)\n    existingAxisCount = len(figure.axes)\n    drawPanelC(figure, grid[0, :5], ecGroupMeanZ, inputs["networkLabelFrame"], subjectFrame)\n    panelCAxes = [\n        plotAxis\n        for plotAxis in figure.axes[existingAxisCount:]\n        if plotAxis.has_data()\n    ]\n    existingAxisCount = len(figure.axes)\n    drawPanelDSummary(figure, grid[0, 5:7], inputs["subjectCorrelationFrame"])\n    panelDAxes = [\n        plotAxis\n        for plotAxis in figure.axes[existingAxisCount:]\n        if plotAxis.has_data()\n    ]\n    existingAxisCount = len(figure.axes)\n    drawPanelE(figure, grid[0, 7:], inputs["hierarchyFrame"], inputs["hierarchyGlmFrame"])\n    panelEAxes = [\n        plotAxis\n        for plotAxis in figure.axes[existingAxisCount:]\n        if plotAxis.has_data()\n    ]\n    figure.canvas.draw()\n    alignPlotAxesVertically(\n        panelCAxes + panelDAxes + panelEAxes,\n        panelCAxes[0],\n    )\n    alignHeaderAxesVertically(\n        figure,\n        {\n            "HC",\n            "ASD",\n            "Subject-level r",\n            "Group comparison",\n            "Participant-level distribution",\n        },\n        "HC",\n    )\n    scalePlotAxesWidths(panelEAxes, PANEL_E_PLOT_WIDTH_SCALE)\n    return saveRowFigure(figure, outputDirectory / "gradient-analysis-row-2-panels-cde")\n\n\ndef main() -> None:\n    analysisRoot = Path(__file__).resolve().parent.parent\n    for outputPath in createRowTwoFigure(analysisRoot, analysisRoot): print(outputPath.name)\n\nif __name__ == "__main__": main()\n', 'plot-gradient-analysis-row-3-panels-fg.py': 'from __future__ import annotations\n\nimport math\n\nfrom pathlib import Path\n\nimport matplotlib\n\nmatplotlib.use("Agg")\n\nimport matplotlib.colors as matplotlibColors\n\nimport matplotlib.patheffects as pathEffects\n\nimport matplotlib.pyplot as plt\n\nimport numpy as np\n\nimport pandas as pd\n\nfrom brainspace.datasets import load_conte69, load_parcellation\n\nfrom brainspace.plotting import plot_hemispheres, plot_surf\n\nfrom brainspace.utils.parcellation import map_to_labels\n\nfrom matplotlib.lines import Line2D\n\nfrom matplotlib.colors import to_rgba\n\nfrom matplotlib.patches import Patch\n\nfrom matplotlib.transforms import ScaledTranslation\n\nfrom scipy.stats import gaussian_kde, spearmanr, ttest_ind, zscore\n\nFONT_FAMILY = "Arial"\n\nFONT_SIZE = 18\n\nPANEL_LABEL_SIZE = 22\n\nPANEL_TITLE_OFFSET_POINTS = 23\n\nFIGURE_WIDTH_INCHES = 18.0\n\nFIGURE_HEIGHT_INCHES = 17.8\n\nAB_ROW_HEIGHT_INCHES = 5.68\n\nCD_ROW_HEIGHT_INCHES = 6.30\n\nEF_ROW_HEIGHT_INCHES = 5.68\n\nHISTOGRAM_BIN_WIDTH = 0.1\n\nPLOT_AXIS_HEIGHT_INCHES = 3.60\n\nD_MEAN_MAP_SHIFT_INCHES = 0.30\n\nD_MEAN_MAP_EXTRA_WIDTH_INCHES = 0.15\n\nEXPORT_DPI = 100\n\nRANDOM_SEED = 42\n\nYEO7_ATLAS_ZOOM = 1.44\n\nGRADIENT_BRAIN_ZOOM = 1.72\n\nIMAGE_CROP_PADDING_PIXELS = 12\n\nGRADIENT_BRAIN_ROW_GAP_PIXELS = 4\n\nGROUP_COLORBAR_BOUNDS = (0.12, 0.88, 0.76, 0.46)\n\nPANEL_B_GROUP_WSPACE = 0.02\n\nP_VALUE_CMAP = "RdBu_r"\nP_VALUE_COLOR_RANGE = (0.0, 1.0)\nP_VALUE_COLORBAR_TICKS = (0.0, 0.5, 1.0)\n\nP_VALUE_BRAIN_ZOOM = 1.70\n\nGROUP_COLORS = {\n    "HC": "#4C78A8",\n    "ASD": "#E07A5F",\n}\n\nNETWORK_ORDER = [\n    "Vis",\n    "SomMot",\n    "DorsAttn",\n    "SalVentAttn",\n    "Limbic",\n    "Cont",\n    "Default",\n]\n\nNETWORK_DISPLAY_NAMES = {\n    "Vis": "Vis",\n    "SomMot": "SMN",\n    "DorsAttn": "DAN",\n    "SalVentAttn": "VAN",\n    "Limbic": "Lim",\n    "Cont": "Cont",\n    "Default": "DMN",\n}\n\nNETWORK_COLORS = {\n    "Vis": "#6D56A5",\n    "SomMot": "#4B9BD3",\n    "DorsAttn": "#58A65C",\n    "SalVentAttn": "#E3A33B",\n    "Limbic": "#D9B23C",\n    "Cont": "#39A7A0",\n    "Default": "#D34B4B",\n}\n\nGRADIENT_CMAP = "RdBu_r"\n\nNEUTRAL_GRAY = "#5A5A5A"\n\nLIGHT_GRAY = "#D9D9D9"\n\nSENSORY_DISTRIBUTION_COLOR = "#1F4E79"\n\nDMN_DISTRIBUTION_COLOR = "#8B1E2D"\n\ndef configureMatplotlib() -> None:\n    plt.rcParams.update(\n        {\n            "font.family": "sans-serif",\n            "font.sans-serif": [FONT_FAMILY],\n            "font.size": FONT_SIZE,\n            "axes.titlesize": FONT_SIZE,\n            "axes.labelsize": FONT_SIZE,\n            "axes.labelpad": 10,\n            "xtick.labelsize": FONT_SIZE,\n            "ytick.labelsize": FONT_SIZE,\n            "legend.fontsize": FONT_SIZE,\n            "figure.titlesize": FONT_SIZE,\n            "axes.spines.right": False,\n            "axes.spines.top": False,\n            "axes.linewidth": 1.6,\n            "legend.frameon": False,\n            "svg.fonttype": "none",\n            "pdf.fonttype": 42,\n            "ps.fonttype": 42,\n            "mathtext.fontset": "custom",\n            "mathtext.rm": FONT_FAMILY,\n            "mathtext.it": f"{FONT_FAMILY}:italic",\n            "mathtext.bf": f"{FONT_FAMILY}:bold",\n            "mathtext.default": "regular",\n            "axes.unicode_minus": False,\n        }\n    )\n\ndef findSingleFile(searchRoot: Path, fileName: str) -> Path:\n    matchingPaths = list(searchRoot.rglob(fileName))\n    if len(matchingPaths) != 1:\n        raise FileNotFoundError(\n            f"Expected exactly one {fileName!r}, found {len(matchingPaths)}."\n        )\n    return matchingPaths[0]\n\ndef findSingleDirectory(searchRoot: Path, requiredFileName: str) -> Path:\n    return findSingleFile(searchRoot, requiredFileName).parent\n\ndef addPanelHeading(\n    containerAxis: plt.Axes,\n    panelLabel: str,\n    panelTitle: str,\n) -> None:\n    containerAxis.set_axis_off()\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelLabel,\n        transform=containerAxis.transAxes,\n        ha="left",\n        va="bottom",\n        fontsize=PANEL_LABEL_SIZE,\n        fontweight="bold",\n    )\n    containerAxis.text(\n        0.0,\n        1.00,\n        panelTitle,\n        transform=(\n            containerAxis.transAxes\n            + ScaledTranslation(\n                PANEL_TITLE_OFFSET_POINTS / 72,\n                0,\n                containerAxis.figure.dpi_scale_trans,\n            )\n        ),\n        ha="left",\n        va="bottom",\n        fontweight="bold",\n    )\n\ndef stylePlotAxis(plotAxis: plt.Axes) -> None:\n    plotAxis.minorticks_off()\n    plotAxis.tick_params(\n        which="major",\n        direction="out",\n        length=7,\n        width=1.5,\n        pad=7,\n        colors=NEUTRAL_GRAY,\n    )\n    plotAxis.spines["left"].set_linewidth(1.6)\n    plotAxis.spines["bottom"].set_linewidth(1.6)\n    plotAxis.spines["left"].set_color(NEUTRAL_GRAY)\n    plotAxis.spines["bottom"].set_color(NEUTRAL_GRAY)\n\ndef readImageWithoutMargins(imagePath: Path) -> np.ndarray:\n    imageValues = plt.imread(imagePath)\n    if imageValues.ndim == 3 and imageValues.shape[2] == 4:\n        rgbValues = imageValues[:, :, :3]\n    else:\n        rgbValues = imageValues\n    nonWhiteMask = np.any(rgbValues < 0.985, axis=2)\n    rowIndices, columnIndices = np.where(nonWhiteMask)\n    if len(rowIndices) == 0:\n        return imageValues\n    rowStart = max(int(rowIndices.min()) - IMAGE_CROP_PADDING_PIXELS, 0)\n    rowStop = min(\n        int(rowIndices.max()) + IMAGE_CROP_PADDING_PIXELS + 1,\n        imageValues.shape[0],\n    )\n    columnStart = max(\n        int(columnIndices.min()) - IMAGE_CROP_PADDING_PIXELS,\n        0,\n    )\n    columnStop = min(\n        int(columnIndices.max()) + IMAGE_CROP_PADDING_PIXELS + 1,\n        imageValues.shape[1],\n    )\n    return imageValues[rowStart:rowStop, columnStart:columnStop]\n\ndef prepareNetworkDistributions(\n    networkLongFrame: pd.DataFrame,\n) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray]:\n    networkPivotFrame = (\n        networkLongFrame.pivot_table(\n            index=["sub_id", "Group"],\n            columns="network",\n            values="g1_net",\n            aggfunc="first",\n        )\n        .reset_index()\n    )\n    distributionsByGroup: dict[str, dict[str, np.ndarray]] = {}\n    combinedValues: list[np.ndarray] = []\n    for groupName in ("HC", "ASD"):\n        groupFrame = networkPivotFrame.loc[\n            networkPivotFrame["Group"].eq(groupName)\n        ]\n        sensoryValues = (\n            groupFrame["Vis"].to_numpy(dtype=float)\n            + groupFrame["SomMot"].to_numpy(dtype=float)\n        ) / 2.0\n        dmnValues = groupFrame["Default"].to_numpy(dtype=float)\n        distributionsByGroup[groupName] = {\n            "sensory": sensoryValues,\n            "dmn": dmnValues,\n        }\n        combinedValues.extend([sensoryValues, dmnValues])\n\n    allValues = np.concatenate(combinedValues)\n    binEdges = createHistogramBinEdges(allValues)\n    return networkPivotFrame, distributionsByGroup, binEdges\n\ndef createHistogramBinEdges(values: np.ndarray) -> np.ndarray:\n    minimumValue = float(np.nanmin(values))\n    maximumValue = float(np.nanmax(values))\n    binEdges = np.arange(\n        minimumValue,\n        maximumValue,\n        HISTOGRAM_BIN_WIDTH,\n        dtype=float,\n    )\n    if binEdges.size == 0 or not np.isclose(binEdges[0], minimumValue):\n        binEdges = np.insert(binEdges, 0, minimumValue)\n    if not np.isclose(binEdges[-1], maximumValue):\n        binEdges = np.append(binEdges, maximumValue)\n    return binEdges\n\ndef calculateHistogramPercentages(\n    values: np.ndarray,\n    binEdges: np.ndarray,\n) -> np.ndarray:\n    validValues = values[np.isfinite(values)]\n    binCounts, _ = np.histogram(validValues, bins=binEdges)\n    return binCounts.astype(float) / validValues.size * 100.0\n\ndef drawNetworkDistribution(\n    percentageAxis: plt.Axes,\n    groupDistributions: dict[str, np.ndarray],\n    binEdges: np.ndarray,\n    upperPercentageLimit: float,\n    showYLabel: bool,\n) -> None:\n    sensoryValues = groupDistributions["sensory"]\n    dmnValues = groupDistributions["dmn"]\n    sensoryWeights = np.full(\n        sensoryValues.size,\n        100.0 / sensoryValues.size,\n    )\n    dmnWeights = np.full(dmnValues.size, 100.0 / dmnValues.size)\n\n    percentageAxis.hist(\n        sensoryValues,\n        bins=binEdges,\n        weights=sensoryWeights,\n        color=to_rgba(SENSORY_DISTRIBUTION_COLOR, 0.20),\n        edgecolor="#6B6B6B",\n        linewidth=1.0,\n        histtype="bar",\n        rwidth=1.0,\n    )\n    percentageAxis.hist(\n        dmnValues,\n        bins=binEdges,\n        weights=dmnWeights,\n        color=to_rgba(DMN_DISTRIBUTION_COLOR, 0.20),\n        edgecolor="#6B6B6B",\n        linewidth=1.0,\n        histtype="bar",\n        rwidth=1.0,\n    )\n    smoothXValues = np.linspace(binEdges[0], binEdges[-1], 400)\n    percentageAxis.plot(\n        smoothXValues,\n        gaussian_kde(sensoryValues)(smoothXValues)\n        * HISTOGRAM_BIN_WIDTH\n        * 100.0,\n        color=SENSORY_DISTRIBUTION_COLOR,\n        linewidth=1.8,\n    )\n    percentageAxis.plot(\n        smoothXValues,\n        gaussian_kde(dmnValues)(smoothXValues)\n        * HISTOGRAM_BIN_WIDTH\n        * 100.0,\n        color=DMN_DISTRIBUTION_COLOR,\n        linewidth=1.8,\n    )\n\n    sensoryMean = float(np.mean(sensoryValues))\n    dmnMean = float(np.mean(dmnValues))\n    percentageAxis.axvline(\n        sensoryMean,\n        ymax=0.90,\n        color=SENSORY_DISTRIBUTION_COLOR,\n        linestyle="--",\n        linewidth=1.1,\n    )\n    percentageAxis.axvline(\n        dmnMean,\n        ymax=0.90,\n        color=DMN_DISTRIBUTION_COLOR,\n        linestyle="--",\n        linewidth=1.1,\n    )\n    percentageAxis.annotate(\n        "",\n        xy=(dmnMean, 0.94),\n        xytext=(sensoryMean, 0.94),\n        xycoords=("data", "axes fraction"),\n        textcoords=("data", "axes fraction"),\n        arrowprops={\n            "arrowstyle": "<->",\n            "color": NEUTRAL_GRAY,\n            "linewidth": 1.2,\n        },\n    )\n\n    percentageAxis.set_xlim(float(binEdges[0]), float(binEdges[-1]))\n    percentageAxis.set_ylim(0, upperPercentageLimit)\n    percentageAxis.set_xlabel("EC G1 score (z)")\n    if showYLabel:\n        percentageAxis.set_ylabel("Percentage (%)", labelpad=4)\n        percentageAxis.yaxis.set_label_coords(-0.17, 0.5)\n    else:\n        percentageAxis.set_yticks([])\n    stylePlotAxis(percentageAxis)\n    percentageAxis.spines["left"].set_visible(showYLabel)\n\ndef drawPanelF(\n    figure: plt.Figure,\n    panelSpec,\n    networkLongFrame: pd.DataFrame,\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "f",\n        "Reduced sensory–transmodal separation in ASD",\n    )\n\n    networkPivotFrame, distributionsByGroup, binEdges = prepareNetworkDistributions(\n        networkLongFrame\n    )\n    percentageMaximums = [\n        float(\n            np.max(\n                calculateHistogramPercentages(\n                    groupDistributions[networkName],\n                    binEdges,\n                )\n            )\n        )\n        for groupDistributions in distributionsByGroup.values()\n        for networkName in ("sensory", "dmn")\n    ]\n    upperPercentageLimit = math.ceil(max(percentageMaximums) / 5.0) * 5.0\n    upperPercentageLimit *= 1.08\n\n    # Use exactly the same title row, plotting row, and bottom row as panel e.\n    # The density range is tightened in drawNetworkDistribution so the HC/ASD\n    # curves grow upward within this matched plotting height.\n    # The plotting row is enlarged from 0.76 to 0.80 so the complete HC/ASD\n    # density plots, rather than only their text, are vertically adjusted.\n    nestedGrid = panelSpec.subgridspec(\n        3,\n        2,\n        height_ratios=[0.08, 0.80, 0.12],\n        hspace=0.04,\n        wspace=0.6,\n    )\n    for columnIndex, groupName in enumerate(("HC", "ASD")):\n        groupDistributions = distributionsByGroup[groupName]\n        separationValue = float(\n            np.mean(groupDistributions["dmn"])\n            - np.mean(groupDistributions["sensory"])\n        )\n\n        titleAxis = figure.add_subplot(nestedGrid[0, columnIndex])\n        titleAxis.set_axis_off()\n        titleAxis.text(\n            0.5,\n            0.20,\n            f"{groupName} separation = {separationValue:.2f} z",\n            transform=titleAxis.transAxes,\n            ha="center",\n            va="center",\n        )\n\n        densityAxis = figure.add_subplot(nestedGrid[1, columnIndex])\n        drawNetworkDistribution(\n            densityAxis,\n            groupDistributions,\n            binEdges,\n            upperPercentageLimit,\n            columnIndex == 0,\n        )\n        if columnIndex == 1:\n            densityAxis.legend(\n                handles=[\n                    Patch(\n                        facecolor=to_rgba(SENSORY_DISTRIBUTION_COLOR, 0.20),\n                        edgecolor=SENSORY_DISTRIBUTION_COLOR,\n                        label="Sensory (Vis/SMN)",\n                    ),\n                    Patch(\n                        facecolor=to_rgba(DMN_DISTRIBUTION_COLOR, 0.20),\n                        edgecolor=DMN_DISTRIBUTION_COLOR,\n                        label="DMN",\n                    ),\n                ],\n                loc="upper right",\n                fontsize=FONT_SIZE - 2,\n                handlelength=1.2,\n                handletextpad=0.4,\n                borderaxespad=0.5,\n            )\n        if columnIndex == 0:\n            horizontalShift = 104 / 1800\n            for axisToShift in (titleAxis, densityAxis):\n                originalPosition = axisToShift.get_position()\n                axisToShift.set_position(\n                    [\n                        originalPosition.x0 + horizontalShift,\n                        originalPosition.y0,\n                        originalPosition.width,\n                        originalPosition.height,\n                    ]\n                )\n\ndef renderPanelGBrainAssets(\n    outputDirectory: Path,\n    networkLabelFrame: pd.DataFrame,\n    networkMainEffectFrame: pd.DataFrame,\n) -> dict[str, Path]:\n    assetDirectory = outputDirectory / "brainspace-assets"\n    assetDirectory.mkdir(parents=True, exist_ok=True)\n\n    surfaceLeft, _ = load_conte69()\n    parcelLabeling = load_parcellation("schaefer", scale=400, join=True)\n    parcelMask = parcelLabeling > 0\n    sourceLabels = np.arange(1, 401)\n    parcelNetworkNames = networkLabelFrame["Yeo7_network"].to_numpy()\n    correctedPValueByNetwork = (\n        networkMainEffectFrame.set_index("item")["p_FDR"].astype(float).to_dict()\n    )\n    allNetworkPValues = np.array(\n        [\n            correctedPValueByNetwork[networkName]\n            for networkName in parcelNetworkNames\n        ],\n        dtype=float,\n    )\n\n    panelGBrainPaths: dict[str, Path] = {}\n    mapNames = NETWORK_ORDER + ["All"]\n    for mapIndex, mapName in enumerate(mapNames):\n        if mapName == "All":\n            parcelPValues = allNetworkPValues\n        else:\n            parcelPValues = np.where(\n                parcelNetworkNames == mapName,\n                allNetworkPValues,\n                np.nan,\n            )\n        vertexPValues = map_to_labels(\n            parcelPValues,\n            parcelLabeling,\n            mask=parcelMask,\n            fill=np.nan,\n            source_lab=sourceLabels,\n        )\n        arrayName = f"panel_g_p_value_{mapIndex}"\n        surfaceLeft.append_array(\n            vertexPValues[: surfaceLeft.n_points],\n            name=arrayName,\n            at="p",\n        )\n        for viewName in ("lateral", "medial"):\n            brainPath = (\n                assetDirectory\n                / f"panel-g-{mapName.lower()}-{viewName}-p-value.png"\n            )\n            plot_surf(\n                {"left": surfaceLeft},\n                np.array([["left"]]),\n                array_name=arrayName,\n                view=np.array([[viewName]]),\n                color_bar=False,\n                color_range=P_VALUE_COLOR_RANGE,\n                cmap=P_VALUE_CMAP,\n                nan_color=(0.94, 0.94, 0.94, 1),\n                size=(500, 300),\n                zoom=P_VALUE_BRAIN_ZOOM,\n                interactive=False,\n                screenshot=True,\n                filename=str(brainPath),\n                transparent_bg=False,\n                background=(1, 1, 1),\n                scale=(2, 2),\n            )\n            panelGBrainPaths[f"{mapName}_{viewName}"] = brainPath\n\n    return panelGBrainPaths\n\ndef drawPanelG(\n    figure: plt.Figure,\n    panelSpec,\n    panelGBrainPaths: dict[str, Path],\n) -> None:\n    containerAxis = figure.add_subplot(panelSpec)\n    addPanelHeading(\n        containerAxis,\n        "g",\n        "Yeo-7 network group main-effect p values",\n    )\n    nestedGrid = panelSpec.subgridspec(\n        7,\n        4,\n        height_ratios=[0.08, 0.185, 0.185, 0.185, 0.185, 0.03, 0.12],\n        hspace=0.02,\n        wspace=0.05,\n    )\n    mapNames = NETWORK_ORDER + ["All"]\n    for mapIndex, mapName in enumerate(mapNames):\n        mapRowIndex, columnIndex = divmod(mapIndex, 4)\n        firstViewRowIndex = mapRowIndex * 2\n        for viewOffset, viewName in enumerate(("lateral", "medial")):\n            brainAxis = figure.add_subplot(\n                nestedGrid[firstViewRowIndex + viewOffset + 1, columnIndex]\n            )\n            brainAxis.imshow(\n                readImageWithoutMargins(\n                    panelGBrainPaths[f"{mapName}_{viewName}"]\n                )\n            )\n            if viewOffset == 0:\n                brainAxis.set_title(\n                    "All networks"\n                    if mapName == "All"\n                    else NETWORK_DISPLAY_NAMES[mapName],\n                    pad=2,\n                    fontweight="bold",\n                )\n            brainAxis.set_axis_off()\n\n    colorbarContainerAxis = figure.add_subplot(nestedGrid[6, 1:3])\n    colorbarContainerAxis.set_axis_off()\n    colorbarAxis = colorbarContainerAxis.inset_axes((0.04, 0.335, 0.92, 0.25))\n    scalarMap = plt.cm.ScalarMappable(\n        norm=matplotlibColors.Normalize(\n            vmin=P_VALUE_COLOR_RANGE[0],\n            vmax=P_VALUE_COLOR_RANGE[1],\n        ),\n        cmap=P_VALUE_CMAP,\n    )\n    colorbar = figure.colorbar(\n        scalarMap,\n        cax=colorbarAxis,\n        orientation="horizontal",\n    )\n    colorbar.set_ticks(P_VALUE_COLORBAR_TICKS)\n    colorbar.set_label("FDR-corrected p value", labelpad=4)\n    colorbar.ax.tick_params(direction="out", length=5, width=1.2, pad=4)\n    colorbar.outline.set_linewidth(0.7)\n\n\n\ndef saveRowFigure(figure: plt.Figure, outputBase: Path) -> list[Path]:\n    outputPaths = [outputBase.with_suffix(extension) for extension in (".svg", ".pdf", ".png", ".tiff")]\n    figure.savefig(outputPaths[0], facecolor="white")\n    figure.savefig(outputPaths[1], facecolor="white")\n    figure.savefig(outputPaths[2], dpi=EXPORT_DPI, facecolor="white")\n    figure.savefig(outputPaths[3], dpi=EXPORT_DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})\n    plt.close(figure)\n    return outputPaths\n\n\ndef loadRowThreeInputs(analysisRoot: Path) -> dict[str, object]:\n    resultDirectory = (\n        analysisRoot\n        / "ABIDE1_结果6_integrated_G1G2_absDistance_DMN_VisSMN_noDistanceFDR"\n    )\n    mainEffectPath = (\n        resultDirectory / "GLM_Yeo7_network_G1G2_main_effect_splitFDR.csv"\n    )\n    allGradientMainEffectFrame = pd.read_csv(mainEffectPath)\n    networkMainEffectFrame = allGradientMainEffectFrame.loc[\n        allGradientMainEffectFrame["dv"].eq("G1_network")\n    ].copy()\n    if set(networkMainEffectFrame["item"]) != set(NETWORK_ORDER):\n        raise ValueError(\n            "The G1 network main-effect table must contain all seven "\n            "Yeo networks."\n        )\n\n    networkLabelPath = (\n        analysisRoot.parents[2]\n        / "3.步进分析"\n        / "Schaefer400_7Yeo_network_labels_with_ID.csv"\n    )\n    return {\n        "networkLongFrame": pd.read_csv(\n            resultDirectory / "Yeo7_network_level_G1G2_long.csv"\n        ),\n        "networkMainEffectFrame": networkMainEffectFrame,\n        "networkLabelFrame": pd.read_csv(networkLabelPath),\n    }\n\n\ndef createRowThreeFigure(analysisRoot: Path, outputDirectory: Path) -> list[Path]:\n    configureMatplotlib(); outputDirectory.mkdir(parents=True, exist_ok=True); inputs = loadRowThreeInputs(analysisRoot)\n    brainPaths = renderPanelGBrainAssets(outputDirectory, inputs["networkLabelFrame"], inputs["networkMainEffectFrame"])\n    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 6.95), facecolor="white")\n    grid = figure.add_gridspec(1, 13, wspace=0.38, left=25/1800, right=1-25/1800, top=1-30/695, bottom=55/695)\n    drawPanelF(figure, grid[0, :7], inputs["networkLongFrame"])\n    drawPanelG(figure, grid[0, 7:], brainPaths)\n    return saveRowFigure(figure, outputDirectory / "gradient-analysis-row-3-panels-fg")\n\n\ndef main() -> None:\n    analysisRoot = Path(__file__).resolve().parent.parent\n    for outputPath in createRowThreeFigure(analysisRoot, analysisRoot): print(outputPath.name)\n\nif __name__ == "__main__": main()\n', 'analyze-svg-geometry.py': 'from __future__ import annotations\n\nimport argparse\nimport math\nimport re\nimport xml.etree.ElementTree as ElementTree\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\n\n\nNUMBER_PATTERN = re.compile(r"[-+]?(?:\\d*\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?")\nTRANSFORM_PATTERN = re.compile(r"([A-Za-z]+)\\s*\\(([^)]*)\\)")\n\n\n@dataclass(frozen=True)\nclass AffineTransform:\n    a: float = 1.0\n    b: float = 0.0\n    c: float = 0.0\n    d: float = 1.0\n    e: float = 0.0\n    f: float = 0.0\n\n    def compose(self, childTransform: "AffineTransform") -> "AffineTransform":\n        return AffineTransform(\n            a=self.a * childTransform.a + self.c * childTransform.b,\n            b=self.b * childTransform.a + self.d * childTransform.b,\n            c=self.a * childTransform.c + self.c * childTransform.d,\n            d=self.b * childTransform.c + self.d * childTransform.d,\n            e=self.a * childTransform.e + self.c * childTransform.f + self.e,\n            f=self.b * childTransform.e + self.d * childTransform.f + self.f,\n        )\n\n    def apply(self, xCoordinate: float, yCoordinate: float) -> tuple[float, float]:\n        return (\n            self.a * xCoordinate + self.c * yCoordinate + self.e,\n            self.b * xCoordinate + self.d * yCoordinate + self.f,\n        )\n\n\ndef parseTransform(transformText: str | None) -> AffineTransform:\n    combinedTransform = AffineTransform()\n    if not transformText:\n        return combinedTransform\n    for transformName, argumentText in TRANSFORM_PATTERN.findall(transformText):\n        values = [float(value) for value in NUMBER_PATTERN.findall(argumentText)]\n        normalizedName = transformName.lower()\n        if normalizedName == "matrix" and len(values) == 6:\n            nextTransform = AffineTransform(*values)\n        elif normalizedName == "translate":\n            nextTransform = AffineTransform(\n                e=values[0],\n                f=values[1] if len(values) > 1 else 0.0,\n            )\n        elif normalizedName == "scale":\n            nextTransform = AffineTransform(\n                a=values[0],\n                d=values[1] if len(values) > 1 else values[0],\n            )\n        elif normalizedName == "rotate":\n            angleRadians = math.radians(values[0])\n            rotationTransform = AffineTransform(\n                a=math.cos(angleRadians),\n                b=math.sin(angleRadians),\n                c=-math.sin(angleRadians),\n                d=math.cos(angleRadians),\n            )\n            if len(values) == 3:\n                centerX, centerY = values[1:]\n                nextTransform = (\n                    AffineTransform(e=centerX, f=centerY)\n                    .compose(rotationTransform)\n                    .compose(AffineTransform(e=-centerX, f=-centerY))\n                )\n            else:\n                nextTransform = rotationTransform\n        else:\n            continue\n        combinedTransform = combinedTransform.compose(nextTransform)\n    return combinedTransform\n\n\ndef parseCoordinate(coordinateText: str | None) -> float:\n    if coordinateText is None:\n        return 0.0\n    matchedValue = NUMBER_PATTERN.search(coordinateText)\n    return float(matchedValue.group()) if matchedValue else 0.0\n\n\ndef normalizeText(textValue: str) -> str:\n    return " ".join(textValue.replace("\\u2212", "-").split())\n\n\ndef collectTextAnchors(svgPath: Path) -> list[dict[str, object]]:\n    svgRoot = ElementTree.parse(svgPath).getroot()\n    anchors: list[dict[str, object]] = []\n\n    def visit(\n        element: ElementTree.Element,\n        parentTransform: AffineTransform,\n    ) -> None:\n        elementTransform = parentTransform.compose(\n            parseTransform(element.attrib.get("transform"))\n        )\n        tagName = element.tag.split("}")[-1]\n        if tagName == "text":\n            textValue = normalizeText("".join(element.itertext()))\n            if textValue:\n                textX = parseCoordinate(element.attrib.get("x"))\n                textY = parseCoordinate(element.attrib.get("y"))\n                if textX == 0.0 and textY == 0.0:\n                    firstTspan = next(\n                        (\n                            child\n                            for child in element.iter()\n                            if child.tag.split("}")[-1] == "tspan"\n                        ),\n                        None,\n                    )\n                    if firstTspan is not None:\n                        textX = parseCoordinate(firstTspan.attrib.get("x"))\n                        textY = parseCoordinate(firstTspan.attrib.get("y"))\n                anchorX, anchorY = elementTransform.apply(textX, textY)\n                anchors.append(\n                    {\n                        "text": textValue,\n                        "x": anchorX,\n                        "y": anchorY,\n                        "fontSize": element.attrib.get("font-size"),\n                        "transform": elementTransform,\n                    }\n                )\n        for child in element:\n            visit(child, elementTransform)\n\n    visit(svgRoot, AffineTransform())\n    return anchors\n\n\ndef collectRectangleBounds(svgPath: Path) -> list[tuple[float, float, float, float]]:\n    svgRoot = ElementTree.parse(svgPath).getroot()\n    rectangleBounds: list[tuple[float, float, float, float]] = []\n\n    def visit(\n        element: ElementTree.Element,\n        parentTransform: AffineTransform,\n    ) -> None:\n        elementTransform = parentTransform.compose(\n            parseTransform(element.attrib.get("transform"))\n        )\n        if element.tag.split("}")[-1] == "rect":\n            rectangleX = parseCoordinate(element.attrib.get("x"))\n            rectangleY = parseCoordinate(element.attrib.get("y"))\n            rectangleWidth = parseCoordinate(element.attrib.get("width"))\n            rectangleHeight = parseCoordinate(element.attrib.get("height"))\n            corners = [\n                elementTransform.apply(rectangleX, rectangleY),\n                elementTransform.apply(rectangleX + rectangleWidth, rectangleY),\n                elementTransform.apply(rectangleX, rectangleY + rectangleHeight),\n                elementTransform.apply(\n                    rectangleX + rectangleWidth,\n                    rectangleY + rectangleHeight,\n                ),\n            ]\n            xCoordinates = [corner[0] for corner in corners]\n            yCoordinates = [corner[1] for corner in corners]\n            rectangleBounds.append(\n                (\n                    min(xCoordinates),\n                    min(yCoordinates),\n                    max(xCoordinates) - min(xCoordinates),\n                    max(yCoordinates) - min(yCoordinates),\n                )\n            )\n        for child in element:\n            visit(child, elementTransform)\n\n    visit(svgRoot, AffineTransform())\n    return rectangleBounds\n\n\ndef printMatchingAnchors(referencePath: Path, sourcePaths: list[Path]) -> None:\n    referenceAnchors = collectTextAnchors(referencePath)\n    referenceByText: dict[str, list[dict[str, object]]] = {}\n    for anchor in referenceAnchors:\n        referenceByText.setdefault(str(anchor["text"]), []).append(anchor)\n\n    for sourcePath in sourcePaths:\n        print(f"\\nSOURCE {sourcePath.name}")\n        sourceAnchors = collectTextAnchors(sourcePath)\n        sourceByText: dict[str, list[dict[str, object]]] = {}\n        for anchor in sourceAnchors:\n            sourceByText.setdefault(str(anchor["text"]), []).append(anchor)\n        for sourceAnchor in sourceAnchors:\n            textValue = str(sourceAnchor["text"])\n            matchingReferenceAnchors = referenceByText.get(textValue, [])\n            if len(matchingReferenceAnchors) != 1:\n                continue\n            referenceAnchor = matchingReferenceAnchors[0]\n            print(\n                f"{textValue[:72]!r}\\t"\n                f"source=({sourceAnchor[\'x\']:.3f},{sourceAnchor[\'y\']:.3f})\\t"\n                f"reference=({referenceAnchor[\'x\']:.3f},{referenceAnchor[\'y\']:.3f})"\n            )\n        print("ALL MATCHED TEXT RESIDUALS")\n        for textValue, matchingSourceAnchors in sourceByText.items():\n            matchingReferenceAnchors = referenceByText.get(textValue, [])\n            if len(matchingSourceAnchors) != len(matchingReferenceAnchors):\n                continue\n            sourceCoordinates = np.array(\n                [\n                    [float(anchor["x"]), float(anchor["y"])]\n                    for anchor in matchingSourceAnchors\n                ]\n            )\n            referenceCoordinates = np.array(\n                [\n                    [float(anchor["x"]), float(anchor["y"])]\n                    for anchor in matchingReferenceAnchors\n                ]\n            )\n            distanceMatrix = np.linalg.norm(\n                sourceCoordinates[:, None, :]\n                - referenceCoordinates[None, :, :],\n                axis=2,\n            )\n            sourceIndices, referenceIndices = linear_sum_assignment(\n                distanceMatrix\n            )\n            for sourceIndex, referenceIndex in zip(\n                sourceIndices,\n                referenceIndices,\n            ):\n                deltaX, deltaY = (\n                    referenceCoordinates[referenceIndex]\n                    - sourceCoordinates[sourceIndex]\n                )\n                if abs(deltaX) <= 0.01 and abs(deltaY) <= 0.01:\n                    continue\n                print(\n                    f"{textValue[:48]!r}\\t"\n                    f"source=({sourceCoordinates[sourceIndex, 0]:.3f},"\n                    f"{sourceCoordinates[sourceIndex, 1]:.3f})\\t"\n                    f"reference=({referenceCoordinates[referenceIndex, 0]:.3f},"\n                    f"{referenceCoordinates[referenceIndex, 1]:.3f})\\t"\n                    f"delta=({deltaX:.3f},{deltaY:.3f})"\n                )\n        print("LARGE RECTANGLES")\n        for rectangleBounds in collectRectangleBounds(sourcePath):\n            if rectangleBounds[2] >= 50 and rectangleBounds[3] >= 40:\n                print("sourceRect", *(f"{value:.3f}" for value in rectangleBounds))\n\n    print("\\nREFERENCE LARGE RECTANGLES")\n    for rectangleBounds in collectRectangleBounds(referencePath):\n        if rectangleBounds[2] >= 50 and rectangleBounds[3] >= 40:\n            print("referenceRect", *(f"{value:.3f}" for value in rectangleBounds))\n\n\ndef main() -> None:\n    argumentParser = argparse.ArgumentParser()\n    argumentParser.add_argument("referenceSvg", type=Path)\n    argumentParser.add_argument("sourceSvgs", nargs="+", type=Path)\n    arguments = argumentParser.parse_args()\n    printMatchingAnchors(arguments.referenceSvg, arguments.sourceSvgs)\n\n\nif __name__ == "__main__":\n    main()\n'}


def loadSourceModule(moduleName: str, scriptPath: Path) -> ModuleType:
    """Load a bundled plotting helper without referencing another file."""
    embeddedSource = EMBEDDED_MODULE_SOURCES.get(scriptPath.name)
    if embeddedSource is None:
        raise ImportError(f"Bundled plotting module is unavailable: {scriptPath.name}")
    embeddedSource = embeddedSource.replace("ABIDE1_", "ABIDE2_")
    embeddedSource = embeddedSource.replace(
        'resultDirectory = findSingleDirectory(analysisRoot, "Yeo7_network_level_G1G2_long.csv")',
        'resultDirectory = (\n'
        '        analysisRoot\n'
        '        / "ABIDE2_结果6_integrated_G1G2_absDistance_DMN_VisSMN_noDistanceFDR"\n'
        '    )',
    )
    if scriptPath.name == "plot-gradient-analysis-row-2-panels-cde.py":
        correlationDirectoryStart = embeddedSource.find(
            "    correlationDirectory = ("
        )
        correlationDirectoryEnd = embeddedSource.find(
            "    networkLongFrame =",
            correlationDirectoryStart,
        )
        if correlationDirectoryStart < 0 or correlationDirectoryEnd < 0:
            raise ImportError("Unable to configure the panel-d input directory.")
        embeddedSource = (
            embeddedSource[:correlationDirectoryStart]
            + "    correlationDirectory = findSingleDirectory(\n"
            + "        analysisRoot,\n"
            + "        \"ALL_HC_ASD_subject_mean_wholebrain_correlation_across_4_cross_axis_comparisons.csv\",\n"
            + "    )\n"
            + embeddedSource[correlationDirectoryEnd:]
        )
    if scriptPath.name == "plot-gradient-analysis-row-3-panels-fg.py":
        sourceReplacements = {
            '["p_FDR"]': '["beta"]',
            'P_VALUE_CMAP = "viridis_r"': 'P_VALUE_CMAP = "RdBu_r"',
            'panel_g_p_value_': 'panel_g_beta_',
            '-p-value.png': '-beta.png',
            'Yeo-7 network group main-effect p values': (
                'Yeo-7 network group main-effect β values'
            ),
            'FDR-corrected p value': 'Group main-effect β',
            'correctedPValueByNetwork': 'betaByNetwork',
            'allNetworkPValues': 'allNetworkBetaValues',
            'parcelPValues': 'parcelBetaValues',
            'vertexPValues': 'vertexBetaValues',
            'P_VALUE_': 'BETA_',
        }
        for oldSourceText, newSourceText in sourceReplacements.items():
            embeddedSource = embeddedSource.replace(
                oldSourceText,
                newSourceText,
            )
    sourceModule = ModuleType(moduleName)
    sourceModule.__file__ = str(scriptPath)
    sys.modules[moduleName] = sourceModule
    exec(compile(embeddedSource, str(scriptPath), "exec"), sourceModule.__dict__)
    return sourceModule


def configureSourceModule(sourceModule: ModuleType) -> None:
    sourceModule.FONT_SIZE = TARGET_FONT_SIZE_POINTS
    sourceModule.PANEL_LABEL_SIZE = TARGET_PANEL_LABEL_SIZE_POINTS
    sourceModule.PANEL_TITLE_OFFSET_POINTS = TARGET_PANEL_TITLE_OFFSET_POINTS
    sourceModule.EXPORT_DPI = EXPORT_DPI
    sourceModule.configureMatplotlib()


def setRowTwoGeometryConstants(rowTwoModule: ModuleType) -> None:
    rowTwoModule.PLOT_AXIS_HEIGHT_INCHES = ROW_TWO_PLOT_AXIS_HEIGHT_INCHES


def addRowGrid(
    figure: plt.Figure,
    rowGeometry: RowGeometry,
) -> matplotlib.gridspec.GridSpec:
    return figure.add_gridspec(
        1,
        13,
        left=rowGeometry.targetLeftFraction(),
        right=rowGeometry.targetRightFraction(),
        top=rowGeometry.targetTopFraction(),
        bottom=rowGeometry.targetBottomFraction(),
        wspace=rowGeometry.gridSpacing,
    )


def calculateEcGroupMeans(
    networkLongFrame,
    g1Matrix: np.ndarray,
    g2Matrix: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    subjectFrame = (
        networkLongFrame[["sub_id", "Group"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if g1Matrix.shape != (len(subjectFrame), 400):
        raise ValueError("G1 matrix shape is inconsistent with the source data.")
    g1SubjectZ = zscore(g1Matrix, axis=1, ddof=0, nan_policy="omit")
    if not np.isfinite(g1SubjectZ).all():
        raise ValueError("G1 z-scoring produced non-finite values.")

    g2SubjectZ = None
    if g2Matrix is not None:
        if g2Matrix.shape != (len(subjectFrame), 400):
            raise ValueError(
                "G2 matrix shape is inconsistent with the source data."
            )
        g2SubjectZ = zscore(g2Matrix, axis=1, ddof=0, nan_policy="omit")
        if not np.isfinite(g2SubjectZ).all():
            raise ValueError("G2 z-scoring produced non-finite values.")

    groupMeans: dict[str, dict[str, np.ndarray]] = {}
    for groupName in ("ASD", "HC"):
        groupMask = subjectFrame["Group"].eq(groupName).to_numpy()
        groupMeans[groupName] = {
            "G1": np.mean(g1SubjectZ[groupMask], axis=0),
        }
        if g2SubjectZ is not None:
            groupMeans[groupName]["G2"] = np.mean(
                g2SubjectZ[groupMask],
                axis=0,
            )
    return groupMeans


def loadCombinedEcFcGroupMeanMap(
    analysisRoot: Path,
    rowTwoModule: ModuleType,
):
    """Build the panel-d map table from the HC and ASD mean-map exports."""
    ecMapPaths = sorted(
        analysisRoot.rglob("*_EC_group_mean_gradient_map.csv")
    )
    fcMapPaths = sorted(
        analysisRoot.rglob("*_FC_group_mean_gradient_map.csv")
    )
    if len(ecMapPaths) != 2 or len(fcMapPaths) != 2:
        raise FileNotFoundError(
            "Panel d requires exactly one HC and one ASD EC/FC mean-map CSV."
        )

    pandasModule = rowTwoModule.pd
    ecValues = [
        pandasModule.read_csv(mapPath)["EC_G1"].to_numpy(dtype=float)
        for mapPath in ecMapPaths
    ]
    fcValues = [
        pandasModule.read_csv(mapPath)["FC_G1"].to_numpy(dtype=float)
        for mapPath in fcMapPaths
    ]
    return pandasModule.DataFrame(
        {
            "EC_G1": np.mean(ecValues, axis=0),
            "FC_G1": np.mean(fcValues, axis=0),
        }
    )


def getRowOneBrainPaths(
    rowOneModule: ModuleType,
    outputDirectory: Path,
    rowOneInputs: dict[str, object],
    ecGroupMeanZ: dict[str, dict[str, np.ndarray]],
    refreshAssets: bool,
) -> dict[str, Path]:
    assetDirectory = outputDirectory / "brainspace-assets"
    existingPaths = {
        "YEO7": assetDirectory / "yeo7-network-two-views.png",
        "ASD_G1": assetDirectory / "asd-g1-ec-gradient.png",
        "HC_G1": assetDirectory / "hc-g1-ec-gradient.png",
    }
    if refreshAssets or not all(path.exists() for path in existingPaths.values()):
        return rowOneModule.renderBrainSurfaceAssets(
            outputDirectory,
            rowOneInputs["networkLabelFrame"],
            ecGroupMeanZ,
        )
    return existingPaths


def getRowThreeBrainPaths(
    rowThreeModule: ModuleType,
    outputDirectory: Path,
    rowThreeInputs: dict[str, object],
    refreshAssets: bool,
) -> dict[str, Path]:
    assetDirectory = outputDirectory / "brainspace-assets"
    existingPaths = {
        f"{mapName}_{viewName}": (
            assetDirectory
            / f"panel-g-{mapName.lower()}-{viewName}-beta.png"
        )
        for mapName in rowThreeModule.NETWORK_ORDER + ["All"]
        for viewName in ("lateral", "medial")
    }
    if refreshAssets or not all(path.exists() for path in existingPaths.values()):
        return rowThreeModule.renderPanelGBrainAssets(
            outputDirectory,
            rowThreeInputs["networkLabelFrame"],
            rowThreeInputs["networkMainEffectFrame"],
        )
    return existingPaths


def shiftAxesHorizontally(
    axes: list[plt.Axes],
    horizontalShiftPoints: float,
) -> None:
    horizontalShiftFraction = horizontalShiftPoints / TARGET_WIDTH_POINTS
    for plotAxis in axes:
        originalPosition = plotAxis.get_position()
        plotAxis.set_position(
            [
                originalPosition.x0 + horizontalShiftFraction,
                originalPosition.y0,
                originalPosition.width,
                originalPosition.height,
            ]
        )


def shiftPanelHeading(
    panelAxes: list[plt.Axes],
    horizontalShiftPoints: float,
) -> None:
    headingAxes = [
        plotAxis
        for plotAxis in panelAxes
        if len(plotAxis.texts) >= 2
        and {textObject.get_text() for textObject in plotAxis.texts}
        >= {"e", "Default–Vis/SomMot G1 separation"}
    ]
    if len(headingAxes) != 1:
        raise RuntimeError("Unable to identify the panel e heading axis.")
    shiftAxesHorizontally(headingAxes, horizontalShiftPoints)


def shiftTextBySvgPoints(
    textObject: matplotlib.text.Text,
    horizontalShiftPoints: float,
    verticalShiftPoints: float,
) -> None:
    textObject.set_transform(
        textObject.get_transform()
        + matplotlib.transforms.ScaledTranslation(
            horizontalShiftPoints / 72,
            -verticalShiftPoints / 72,
            textObject.get_figure().dpi_scale_trans,
        )
    )


def shiftUniqueText(
    figure: plt.Figure,
    textValue: str,
    horizontalShiftPoints: float,
    verticalShiftPoints: float,
) -> None:
    matchingTextObjects = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == textValue
    ]
    if not matchingTextObjects:
        return
    if len(matchingTextObjects) != 1:
        raise RuntimeError(
            f"Expected one text object for {textValue!r}; "
            f"found {len(matchingTextObjects)}."
        )
    shiftTextBySvgPoints(
        matchingTextObjects[0],
        horizontalShiftPoints,
        verticalShiftPoints,
    )


def setPanelGImageRows(
    panelGAxes: list[plt.Axes],
) -> None:
    imageAxes = [plotAxis for plotAxis in panelGAxes if plotAxis.images]
    if len(imageAxes) != 16:
        raise RuntimeError("Panel g must contain sixteen brain image axes.")
    imageAxesByRow: dict[int, list[plt.Axes]] = {}
    for imageAxis in imageAxes:
        sourceTopPoints = (
            1.0 - imageAxis.get_position().y1
        ) * TARGET_HEIGHT_POINTS
        rowIndex = min(
            range(4),
            key=lambda candidateIndex: abs(
                sourceTopPoints
                - (1176.923 + 115.333 * candidateIndex)
            ),
        )
        imageAxesByRow.setdefault(rowIndex, []).append(imageAxis)

    targetTopPointsByRow = {
        0: 1175.328,
        1: 1283.803,
        2: 1420.824,
        3: 1535.633,
    }
    horizontalShiftPointsByRow = {
        0: -0.001,
        1: 0.667,
        2: 0.667,
        3: -0.348,
    }
    for rowIndex, rowAxes in imageAxesByRow.items():
        for imageAxis in rowAxes:
            originalPosition = imageAxis.get_position()
            targetTopFraction = (
                1.0
                - targetTopPointsByRow[rowIndex] / TARGET_HEIGHT_POINTS
            )
            imageAxis.set_position(
                [
                    originalPosition.x0
                    + horizontalShiftPointsByRow[rowIndex]
                    / TARGET_WIDTH_POINTS,
                    targetTopFraction - originalPosition.height,
                    originalPosition.width,
                    originalPosition.height,
                ]
            )


def shiftYAxisTickLabels(
    plotAxis: plt.Axes,
    horizontalShiftPoints: float,
    verticalShiftPoints: float = 0.0,
) -> None:
    for tickLabel in plotAxis.get_yticklabels():
        shiftTextBySvgPoints(
            tickLabel,
            horizontalShiftPoints,
            verticalShiftPoints,
        )


def setAllAxisElementsBlack(figure: plt.Figure) -> None:
    """Set every visible coordinate-axis element to solid black."""
    allAxes = figure.findobj(
        match=lambda figureObject: isinstance(
            figureObject,
            matplotlib.axes.Axes,
        )
    )
    for plotAxis in allAxes:
        for spine in plotAxis.spines.values():
            spine.set_color("black")
        plotAxis.tick_params(
            axis="both",
            which="both",
            color="black",
            labelcolor="black",
        )
        plotAxis.xaxis.label.set_color("black")
        plotAxis.yaxis.label.set_color("black")
        plotAxis.xaxis.get_offset_text().set_color("black")
        plotAxis.yaxis.get_offset_text().set_color("black")


def movePanelEDistributionYAxisRight(
    figure: plt.Figure,
    horizontalShiftPoints: float,
) -> None:
    """Move only panel e's distribution y-axis inward without moving its axes box."""
    matchingAxes = [
        plotAxis
        for plotAxis in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.axes.Axes,
            )
        )
        if plotAxis.get_xlabel() == "G1 separation (z)"
        and plotAxis.get_ylabel() == "Percentage (%)"
    ]
    if len(matchingAxes) != 1:
        raise RuntimeError(
            "Unable to identify the participant-level distribution axis "
            f"in panel e; found {len(matchingAxes)} candidates."
        )

    distributionAxis = matchingAxes[0]
    distributionAxis.spines["left"].set_position(
        ("outward", -horizontalShiftPoints)
    )


def setTextFontWeight(
    textObject: matplotlib.text.Text,
    fontWeight: str,
) -> None:
    """Force a Text object to use a real regular or bold font face.

    Some source axes use FontProperties created from an explicit font-file
    path. Copying that object preserves the original file (for example an
    Arial Bold file), so changing only the weight metadata does not change
    the rendered glyphs. Rebuild FontProperties without a fixed font file
    so Matplotlib selects the correct face from the family and weight.
    """
    currentProperties = textObject.get_fontproperties()
    fontFamilies = currentProperties.get_family() or ["Arial"]

    replacementProperties = matplotlib.font_manager.FontProperties(
        family=fontFamilies,
        style=currentProperties.get_style(),
        variant=currentProperties.get_variant(),
        weight=fontWeight,
        stretch=currentProperties.get_stretch(),
        size=currentProperties.get_size_in_points(),
    )
    replacementProperties.set_file(None)

    textObject.set_fontproperties(replacementProperties)
    textObject.set_fontweight(fontWeight)


def setFigureFontWeights(figure: plt.Figure) -> None:
    """Keep only panel labels a-g and main panel titles bold."""
    allTextObjects = figure.findobj(
        match=lambda figureObject: isinstance(
            figureObject,
            matplotlib.text.Text,
        )
    )

    # First force every visible text object to regular weight. This also
    # overrides explicit bold settings inherited from the source modules.
    for textObject in allTextObjects:
        if textObject.get_text().strip():
            setTextFontWeight(textObject, "normal")

    # Tick-label objects can be regenerated during a canvas draw. Set both
    # label1 and label2 explicitly so HC/ASD and every other tick stay regular.
    allAxes = figure.findobj(
        match=lambda figureObject: isinstance(
            figureObject,
            matplotlib.axes.Axes,
        )
    )
    for plotAxis in allAxes:
        for axisObject in (plotAxis.xaxis, plotAxis.yaxis):
            for tickObject in (
                list(axisObject.get_major_ticks())
                + list(axisObject.get_minor_ticks())
            ):
                setTextFontWeight(tickObject.label1, "normal")
                setTextFontWeight(tickObject.label2, "normal")

        setTextFontWeight(plotAxis.xaxis.label, "normal")
        setTextFontWeight(plotAxis.yaxis.label, "normal")
        setTextFontWeight(plotAxis.xaxis.get_offset_text(), "normal")
        setTextFontWeight(plotAxis.yaxis.get_offset_text(), "normal")

    # Reapply bold only to the panel labels and the seven main panel titles.
    for textObject in allTextObjects:
        textValue = textObject.get_text().strip()
        if (
            textValue in PANEL_LABEL_TEXTS
            or textValue in MAIN_PANEL_TITLE_TEXTS
        ):
            setTextFontWeight(textObject, "bold")


def applyFinalFigureAppearance(figure: plt.Figure) -> None:
    """Apply final axis and font edits after SVG text calibration."""
    setAllAxisElementsBlack(figure)
    movePanelEDistributionYAxisRight(
        figure,
        PANEL_E_DISTRIBUTION_Y_AXIS_RIGHT_SHIFT_POINTS,
    )

    # Draw once before and once after applying font properties. The second
    # pass catches tick labels that Matplotlib creates during the first draw.
    figure.canvas.draw()
    setFigureFontWeights(figure)
    figure.canvas.draw()
    setFigureFontWeights(figure)


def normalizeTextForMatching(textValue: str) -> str:
    return " ".join(textValue.replace("\u2212", "-").split())


def calibrateTextUsingRenderedSvg(
    figure: plt.Figure,
    referenceSvgPath: Path,
    renderedSvgPath: Path,
    geometryAnalyzerModule: ModuleType,
) -> None:
    figure.canvas.draw()
    referenceAnchors = geometryAnalyzerModule.collectTextAnchors(
        referenceSvgPath
    )
    renderedAnchors = geometryAnalyzerModule.collectTextAnchors(
        renderedSvgPath
    )
    referenceByText: dict[str, list[dict[str, object]]] = {}
    for referenceAnchor in referenceAnchors:
        normalizedText = normalizeTextForMatching(
            str(referenceAnchor["text"])
        )
        referenceByText.setdefault(normalizedText, []).append(referenceAnchor)
    renderedByText: dict[str, list[dict[str, object]]] = {}
    for renderedAnchor in renderedAnchors:
        normalizedText = normalizeTextForMatching(
            str(renderedAnchor["text"])
        )
        renderedByText.setdefault(normalizedText, []).append(renderedAnchor)

    sourceTextObjects = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_visible()
        and normalizeTextForMatching(textObject.get_text())
    ]
    sourceByText: dict[str, list[matplotlib.text.Text]] = {}
    for textObject in sourceTextObjects:
        normalizedText = normalizeTextForMatching(textObject.get_text())
        sourceByText.setdefault(normalizedText, []).append(textObject)

    for normalizedText, matchingTextObjects in sourceByText.items():
        matchingReferenceAnchors = referenceByText.get(normalizedText, [])
        matchingRenderedAnchors = renderedByText.get(normalizedText, [])
        if (
            not matchingRenderedAnchors
            or
            len(matchingReferenceAnchors) != len(matchingRenderedAnchors)
            or len(matchingTextObjects) < len(matchingRenderedAnchors)
        ):
            continue

        textObjectCoordinates: list[tuple[float, float]] = []
        for textObject in matchingTextObjects:
            displayX, displayY = textObject.get_transform().transform(
                textObject.get_position()
            )
            textObjectCoordinates.append(
                (
                    displayX * 72 / figure.dpi,
                    TARGET_HEIGHT_POINTS - displayY * 72 / figure.dpi,
                )
            )
        referenceCoordinates = [
            (
                float(referenceAnchor["x"]),
                float(referenceAnchor["y"]),
            )
            for referenceAnchor in matchingReferenceAnchors
        ]
        renderedCoordinates = [
            (
                float(renderedAnchor["x"]),
                float(renderedAnchor["y"]),
            )
            for renderedAnchor in matchingRenderedAnchors
        ]
        textObjectCoordinateArray = np.asarray(textObjectCoordinates)
        referenceCoordinateArray = np.asarray(referenceCoordinates)
        renderedCoordinateArray = np.asarray(renderedCoordinates)

        objectToRenderedDistanceMatrix = np.linalg.norm(
            textObjectCoordinateArray[:, None, :]
            - renderedCoordinateArray[None, :, :],
            axis=2,
        )
        objectIndices, renderedIndicesForObjects = linear_sum_assignment(
            objectToRenderedDistanceMatrix
        )
        renderedToObjectIndex = {
            renderedIndex: objectIndex
            for objectIndex, renderedIndex in zip(
                objectIndices,
                renderedIndicesForObjects,
            )
        }

        renderedToReferenceDistanceMatrix = np.linalg.norm(
            renderedCoordinateArray[:, None, :]
            - referenceCoordinateArray[None, :, :],
            axis=2,
        )
        renderedIndices, referenceIndices = linear_sum_assignment(
            renderedToReferenceDistanceMatrix
        )
        for renderedIndex, referenceIndex in zip(
            renderedIndices,
            referenceIndices,
        ):
            matchedTextObject = matchingTextObjects[
                renderedToObjectIndex[renderedIndex]
            ]
            matchedTextObject.set_horizontalalignment("left")
            deltaX, deltaY = (
                referenceCoordinateArray[referenceIndex]
                - renderedCoordinateArray[renderedIndex]
            )
            shiftTextBySvgPoints(
                matchedTextObject,
                float(deltaX),
                float(deltaY),
            )


def applyPanelAFinalLayout(figure: plt.Figure) -> None:
    """Apply the requested final spacing adjustments to panel a."""
    panelABrainAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.images and plotAxis.get_position().x0 < 0.30
    ]
    if len(panelABrainAxes) != 1:
        raise RuntimeError("Unable to identify panel a's Yeo-7 brain axis.")
    brainAxis = panelABrainAxes[0]
    originalBrainPosition = brainAxis.get_position()
    brainScale = 0.76
    scaledBrainWidth = originalBrainPosition.width * brainScale
    scaledBrainHeight = originalBrainPosition.height * brainScale
    brainAxis.set_position(
        [
            originalBrainPosition.x0
            + (originalBrainPosition.width - scaledBrainWidth) / 2,
            originalBrainPosition.y0
            + (originalBrainPosition.height - scaledBrainHeight) / 2,
            scaledBrainWidth,
            scaledBrainHeight,
        ]
    )

    hcScatterAxes = [
        plotAxis
        for plotAxis in figure.axes
        if "DMN" in [tickLabel.get_text() for tickLabel in plotAxis.get_yticklabels()]
        and plotAxis.get_position().y0 > 0.75
    ]
    if len(hcScatterAxes) != 1:
        raise RuntimeError("Unable to identify panel a's HC scatter axis.")
    shiftAxesHorizontally(hcScatterAxes, 24.0)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    hcHeaderTexts = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == "HC"
        and textObject.get_window_extent(renderer).y0 > 0.75 * figure.bbox.height
        and textObject.get_window_extent(renderer).x0 < 0.30 * figure.bbox.width
    ]
    if len(hcHeaderTexts) != 1:
        raise RuntimeError("Unable to identify panel a's HC header.")
    shiftTextBySvgPoints(hcHeaderTexts[0], 24.0, 0.0)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panelATitleTexts = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text()
        == "Yeo-7 organization along EC principal gradient"
    ]
    if len(panelATitleTexts) != 1:
        raise RuntimeError("Unable to identify panel a's title.")
    targetLeftPixels = panelATitleTexts[0].get_window_extent(renderer).x0
    for tickLabel in hcScatterAxes[0].get_yticklabels():
        currentLeftPixels = tickLabel.get_window_extent(renderer).x0
        shiftTextBySvgPoints(
            tickLabel,
            (targetLeftPixels - currentLeftPixels) * 72 / figure.dpi,
            0.0,
        )

    yeoAtlasTitles = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == "Yeo-7 atlas"
    ]
    if len(yeoAtlasTitles) != 1:
        raise RuntimeError("Unable to identify the Yeo-7 atlas header.")
    targetHeaderBottomPixels = yeoAtlasTitles[0].get_window_extent(renderer).y0
    groupHeaderTexts = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() in {"HC", "ASD"}
        and textObject.get_window_extent(renderer).y0 > 0.90 * figure.bbox.height
    ]
    if len(groupHeaderTexts) != 4:
        raise RuntimeError("Unable to identify the panel a/b group headers.")
    for headerText in groupHeaderTexts:
        currentHeaderBottomPixels = headerText.get_window_extent(renderer).y0
        shiftTextBySvgPoints(
            headerText,
            0.0,
            (currentHeaderBottomPixels - targetHeaderBottomPixels)
            * 72
            / figure.dpi,
        )

    panelAXAxisTitles = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == "EC principal gradient (z)"
    ]
    if len(panelAXAxisTitles) != 1:
        raise RuntimeError("Unable to identify panel a's x-axis title.")
    shiftTextBySvgPoints(panelAXAxisTitles[0], 24.0, 0.0)

    # Panel a creates its legend axis immediately after its eight content
    # axes; subsequent panels are appended after this stable first-row order.
    panelALegendAxis = figure.axes[8]
    shiftAxesHorizontally([panelALegendAxis], 24.0)


def applyPanelDFinalLayout(figure: plt.Figure) -> None:
    """Align panel d's vertical label and use equal, regular tick spacing."""
    panelDPlotAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.get_ylabel() == "Group-average FC-G1"
    ]
    if len(panelDPlotAxes) != 1:
        raise RuntimeError("Unable to identify panel d's correlation axis.")
    panelDPlotAxis = panelDPlotAxes[0]
    panelDPlotAxis.set_xticks(np.arange(-1.0, 1.1, 1.0))
    panelDPlotAxis.set_yticks(np.arange(-1.0, 1.1, 1.0))
    panelDPlotAxis.set_aspect("auto")
    panelDPlotAxis.yaxis.set_label_coords(-0.40, 0.5)
    panelDPlotAxis.tick_params(axis="y", pad=6)
    panelCAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.get_ylabel() == "EC gradient 2 (z)"
    ]
    if len(panelCAxes) != 1:
        raise RuntimeError("Unable to identify panel c's reference axis.")
    panelCPosition = panelCAxes[0].get_position()
    panelDPosition = panelDPlotAxis.get_position()
    panelDPlotAxis.set_position(
        [
            panelDPosition.x0 - 24.0 / TARGET_WIDTH_POINTS,
            panelCPosition.y0,
            panelDPosition.width,
            panelCPosition.height,
        ]
    )

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panelDTitles = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if "G1 correlation" in textObject.get_text()
    ]
    if len(panelDTitles) != 1:
        raise RuntimeError("Unable to identify panel d's title.")
    panelDSubtitles = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == "Group-average map"
    ]
    if len(panelDSubtitles) != 1:
        raise RuntimeError("Unable to identify panel d's plot title.")
    targetCenterPixels = (
        panelDPlotAxis.get_window_extent(renderer).x0
        + panelDPlotAxis.get_window_extent(renderer).width / 2
    )
    currentCenterPixels = (
        panelDSubtitles[0].get_window_extent(renderer).x0
        + panelDSubtitles[0].get_window_extent(renderer).width / 2
    )
    shiftTextBySvgPoints(
        panelDSubtitles[0],
        (targetCenterPixels - currentCenterPixels) * 72 / figure.dpi,
        0.0,
    )

    # 1. Lock the y-axis title to the left edge of panel d's main title.
    targetYAxisTitleLeftPixels = panelDTitles[0].get_window_extent(renderer).x0
    currentYAxisTitleLeftPixels = (
        panelDPlotAxis.yaxis.label.get_window_extent(renderer).x0
    )
    shiftTextBySvgPoints(
        panelDPlotAxis.yaxis.label,
        (targetYAxisTitleLeftPixels - currentYAxisTitleLeftPixels)
        * 72 / figure.dpi,
        0.0,
    )

    # The y-axis title remains fixed; tick labels use a 5 px gap.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    visibleYTickLabels = [
        tickLabel
        for tickLabel in panelDPlotAxis.get_yticklabels()
        if tickLabel.get_text()
    ]
    yAxisLabelRightPixels = panelDPlotAxis.yaxis.label.get_window_extent(
        renderer
    ).x1
    yTickLabelLeftPixels = min(
        tickLabel.get_window_extent(renderer).x0
        for tickLabel in visibleYTickLabels
    )
    tickLabelShiftPoints = (
        yAxisLabelRightPixels + 5.0 - yTickLabelLeftPixels
    ) * 72 / figure.dpi
    for tickLabel in visibleYTickLabels:
        shiftTextBySvgPoints(tickLabel, tickLabelShiftPoints, 0.0)

    # Tick marks use 3 px; the locked title remains unchanged.
    panelDPlotAxis.tick_params(
        axis="y",
        which="major",
        length=3.0 * 72.0 / figure.dpi,
    )


def applyUnifiedYAxisSpacing(figure: plt.Figure) -> None:
    """Measure and enforce clean 3 px/5 px y-axis spacing."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for plotAxis in figure.axes:
        if not plotAxis.get_ylabel() or not plotAxis.yaxis.get_visible():
            continue
        # Panel d has a title-aligned y label with an explicitly measured
        # 5 px gap, so do not overwrite that final calibration here.
        if plotAxis.get_ylabel() == "Group-average FC-G1":
            continue
        visibleYTickLabels = [
            tickLabel
            for tickLabel in plotAxis.get_yticklabels()
            if tickLabel.get_text() and tickLabel.get_visible()
        ]
        if not visibleYTickLabels:
            continue
        tickTransform, verticalAlignment, horizontalAlignment = (
            plotAxis.get_yaxis_text1_transform(
                3.0 * 72.0 / figure.dpi
            )
        )
        for tickLabel in visibleYTickLabels:
            tickLabel.set_transform(tickTransform)
            tickLabel.set_verticalalignment(verticalAlignment)
            tickLabel.set_horizontalalignment(horizontalAlignment)

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        leftmostTickPixel = min(
            tickLabel.get_window_extent(renderer).x0
            for tickLabel in visibleYTickLabels
        )
        yAxisLabelRightPixel = plotAxis.yaxis.label.get_window_extent(
            renderer
        ).x1
        shiftTextBySvgPoints(
            plotAxis.yaxis.label,
            (leftmostTickPixel - 5.0 - yAxisLabelRightPixel)
            * 72
            / figure.dpi,
            0.0,
        )
        plotAxis.tick_params(
            axis="y",
            which="major",
            length=3.0 * 72.0 / figure.dpi,
        )


def applyPanelCFinalLayout(figure: plt.Figure) -> None:
    """Move panel c left as one unit and align its y title to its main title."""
    panelCLeftAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.get_ylabel() == "EC gradient 2 (z)"
    ]
    if len(panelCLeftAxes) != 1:
        raise RuntimeError("Unable to identify panel c's left scatter axis.")
    panelCLeftAxis = panelCLeftAxes[0]
    panelCLeftPosition = panelCLeftAxis.get_position()
    panelCScatterAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.has_data()
        and np.isclose(plotAxis.get_position().y0, panelCLeftPosition.y0)
        and np.isclose(plotAxis.get_position().height, panelCLeftPosition.height)
        and plotAxis.get_position().x0 < 0.40
    ]
    if len(panelCScatterAxes) != 2:
        raise RuntimeError("Unable to identify both panel c scatter axes.")

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panelCTitles = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if "EC gradient space" in textObject.get_text()
    ]
    if len(panelCTitles) != 1:
        raise RuntimeError("Unable to identify panel c's main title.")

    targetLeftPixels = panelCTitles[0].get_window_extent(renderer).x0
    currentLeftPixels = panelCLeftAxis.yaxis.label.get_window_extent(
        renderer
    ).x0
    horizontalShiftPoints = (
        targetLeftPixels - currentLeftPixels
    ) * 72 / figure.dpi
    shiftAxesHorizontally(panelCScatterAxes, horizontalShiftPoints)

    panelCAxisTopPixels = max(
        plotAxis.get_window_extent(renderer).y1
        for plotAxis in panelCScatterAxes
    )
    panelCHeaderTexts = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() in {"HC", "ASD"}
        and panelCAxisTopPixels
        < textObject.get_window_extent(renderer).y0
        < panelCAxisTopPixels + 80
        and textObject.get_window_extent(renderer).x0 < 0.40 * figure.bbox.width
    ]
    if len(panelCHeaderTexts) != 2:
        raise RuntimeError("Unable to identify panel c's HC/ASD headers.")
    for headerText in panelCHeaderTexts:
        shiftTextBySvgPoints(headerText, horizontalShiftPoints, 0.0)


def applyPanelFFinalLayout(figure: plt.Figure) -> None:
    """Move panel f's complete HC distribution block to the right."""
    panelFDensityAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.get_xlabel() == "EC G1 score (z)"
        and plotAxis.get_position().y0 < 0.35
    ]
    if len(panelFDensityAxes) != 2:
        raise RuntimeError("Unable to identify both panel f distribution axes.")
    hcDensityAxis = min(
        panelFDensityAxes,
        key=lambda plotAxis: plotAxis.get_position().x0,
    )
    horizontalShiftPoints = 24.0
    shiftAxesHorizontally([hcDensityAxis], horizontalShiftPoints)

    hcHeaderTexts = [
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text().startswith("HC separation =")
    ]
    if len(hcHeaderTexts) != 1:
        raise RuntimeError("Unable to identify panel f's HC separation title.")
    shiftTextBySvgPoints(hcHeaderTexts[0], horizontalShiftPoints, 0.0)


def drawCombinedFigure(
    analysisRoot: Path,
    outputDirectory: Path,
    refreshAssets: bool,
) -> plt.Figure:
    scriptDirectory = Path(__file__).resolve().parent
    rowOneModule = loadSourceModule(
        "gradientAnalysisRowOne",
        scriptDirectory / "plot-gradient-analysis-row-1-panels-ab.py",
    )
    rowTwoModule = loadSourceModule(
        "gradientAnalysisRowTwo",
        scriptDirectory / "plot-gradient-analysis-row-2-panels-cde.py",
    )
    rowThreeModule = loadSourceModule(
        "gradientAnalysisRowThree",
        scriptDirectory / "plot-gradient-analysis-row-3-panels-fg.py",
    )
    for sourceModule in (rowOneModule, rowTwoModule, rowThreeModule):
        configureSourceModule(sourceModule)
    setRowTwoGeometryConstants(rowTwoModule)
    rowThreeModule.BETA_COLOR_RANGE = PANEL_G_BETA_COLOR_RANGE
    rowThreeModule.BETA_COLORBAR_TICKS = PANEL_G_BETA_COLORBAR_TICKS

    rowOneInputs = rowOneModule.loadRowOneInputs(analysisRoot)
    rowTwoInputs = rowTwoModule.loadRowTwoInputs(analysisRoot)
    rowThreeInputs = rowThreeModule.loadRowThreeInputs(analysisRoot)

    rowOneGroupMeans = calculateEcGroupMeans(
        rowOneInputs["networkLongFrame"],
        rowOneInputs["g1Matrix"],
    )
    rowTwoGroupMeans = calculateEcGroupMeans(
        rowTwoInputs["networkLongFrame"],
        rowTwoInputs["g1Matrix"],
        rowTwoInputs["g2Matrix"],
    )
    rowOneBrainPaths = getRowOneBrainPaths(
        rowOneModule,
        outputDirectory,
        rowOneInputs,
        rowOneGroupMeans,
        refreshAssets,
    )
    rowThreeBrainPaths = getRowThreeBrainPaths(
        rowThreeModule,
        outputDirectory,
        rowThreeInputs,
        refreshAssets,
    )

    figure = plt.figure(
        figsize=(
            TARGET_WIDTH_POINTS / 72,
            TARGET_HEIGHT_POINTS / 72,
        ),
        facecolor="white",
    )

    rowOneGrid = addRowGrid(figure, ROW_ONE_GEOMETRY)
    rowOneAxisStart = len(figure.axes)
    rowOneModule.drawPanelA(
        figure,
        rowOneGrid[0, :5],
        rowOneInputs["networkLongFrame"],
        rowOneBrainPaths["YEO7"],
    )
    rowOneModule.drawPanelB(
        figure,
        rowOneGrid[0, 5:],
        rowOneBrainPaths,
        rowOneGroupMeans,
    )
    rowOneAxes = figure.axes[rowOneAxisStart:]

    rowTwoGrid = addRowGrid(figure, ROW_TWO_GEOMETRY)
    rowTwoAxisStart = len(figure.axes)
    rowTwoModule.drawPanelC(
        figure,
        rowTwoGrid[0, :5],
        rowTwoGroupMeans,
        rowTwoInputs["networkLabelFrame"],
        (
            rowTwoInputs["networkLongFrame"][["sub_id", "Group"]]
            .drop_duplicates()
            .reset_index(drop=True)
        ),
    )

    panelDAxisStart = len(figure.axes)
    rowTwoModule.drawPanelDSummary(
        figure,
        rowTwoGrid[0, 5:7],
        loadCombinedEcFcGroupMeanMap(analysisRoot, rowTwoModule),
    )
    panelDAxes = figure.axes[panelDAxisStart:]

    panelEAxisStart = len(figure.axes)
    rowTwoModule.drawPanelE(
        figure,
        rowTwoGrid[0, 7:],
        rowTwoInputs["hierarchyFrame"],
        rowTwoInputs["hierarchyGlmFrame"],
    )
    panelEAxes = figure.axes[panelEAxisStart:]

    figure.canvas.draw()
    rowTwoPlotAxes = [
        plotAxis
        for plotAxis in figure.axes
        if plotAxis.has_data()
        and ROW_TWO_GEOMETRY.targetBottomFraction()
        <= plotAxis.get_position().y0
        <= ROW_TWO_GEOMETRY.targetTopFraction()
    ]
    rowTwoModule.alignPlotAxesVertically(
        rowTwoPlotAxes,
        rowTwoPlotAxes[0],
    )
    rowTwoModule.alignHeaderAxesVertically(
        figure,
        {
            "HC",
            "ASD",
            "Subject-level r",
            "Group comparison",
            "Participant-level distribution",
        },
        "HC",
    )
    rowTwoModule.scalePlotAxesWidths(
        [plotAxis for plotAxis in panelEAxes if plotAxis.has_data()],
        rowTwoModule.PANEL_E_PLOT_WIDTH_SCALE,
    )
    shiftAxesHorizontally(
        [plotAxis for plotAxis in panelDAxes if plotAxis.has_data()],
        ROW_TWO_PANEL_D_EXTRA_SHIFT_POINTS,
    )
    shiftPanelHeading(
        panelEAxes,
        ROW_TWO_PANEL_E_HEADING_EXTRA_SHIFT_POINTS,
    )
    rowTwoAxes = figure.axes[rowTwoAxisStart:]

    rowThreeGrid = addRowGrid(figure, ROW_THREE_GEOMETRY)
    rowThreeAxisStart = len(figure.axes)
    rowThreeModule.drawPanelF(
        figure,
        rowThreeGrid[0, :7],
        rowThreeInputs["networkLongFrame"],
    )
    rowThreeModule.drawPanelG(
        figure,
        rowThreeGrid[0, 7:],
        rowThreeBrainPaths,
    )
    rowThreeAxes = figure.axes[rowThreeAxisStart:]
    panelGAxes = [
        plotAxis
        for plotAxis in rowThreeAxes
        if plotAxis.get_position().x0
        >= 950 / TARGET_WIDTH_POINTS
    ]
    setPanelGImageRows(panelGAxes)

    # Illustrator retained the calibrated plot boxes but moved several
    # headings and labels independently. These are point-space corrections,
    # equivalent to title offsets and label-coordinate adjustments.
    for plotAxis in rowOneAxes:
        if {textObject.get_text() for textObject in plotAxis.texts} & {"a", "b"}:
            for textObject in plotAxis.texts:
                shiftTextBySvgPoints(textObject, 0.0, 21.41)
    for plotAxis in rowTwoAxes:
        if {textObject.get_text() for textObject in plotAxis.texts} & {
            "c",
            "d",
            "e",
        }:
            for textObject in plotAxis.texts:
                shiftTextBySvgPoints(textObject, 0.0, -1.108)
    for plotAxis in rowThreeAxes:
        if {textObject.get_text() for textObject in plotAxis.texts} & {"f", "g"}:
            for textObject in plotAxis.texts:
                shiftTextBySvgPoints(textObject, 0.0, -1.075)

    figure.canvas.draw()
    shiftUniqueText(
        figure,
        "EC principal gradient (z)",
        -140.962,
        0.322,
    )
    shiftUniqueText(figure, "Yeo-7 atlas", -68.663, 8.754)
    shiftUniqueText(figure, "HC separation = 1.45 z", -126.108, -0.044)
    shiftUniqueText(figure, "ASD separation = 1.38 z", -134.958, -0.044)
    shiftUniqueText(
        figure,
        "Group main-effect β",
        -108.027,
        1.371,
    )

    # Panel g's colorbar uses an upward offset with unchanged horizontal scale.
    colorbarLabel = next(
        textObject
        for textObject in figure.findobj(
            match=lambda figureObject: isinstance(
                figureObject,
                matplotlib.text.Text,
            )
        )
        if textObject.get_text() == "Group main-effect β"
    )
    colorbarAxes = [
        childAxis
        for parentAxis in panelGAxes
        for childAxis in parentAxis.child_axes
    ]
    if len(colorbarAxes) != 1:
        raise RuntimeError("Unable to identify the panel g colorbar axis.")
    colorbarAxis = colorbarAxes[0]
    originalColorbarPosition = colorbarAxis.get_position()
    colorbarAxis.set_axes_locator(None)
    colorbarAxis.set_position(
        [
            originalColorbarPosition.x0,
            originalColorbarPosition.y0
            + 35.228 / TARGET_HEIGHT_POINTS,
            originalColorbarPosition.width,
            originalColorbarPosition.height,
        ]
    )
    for tickLabel in colorbarAxis.get_xticklabels():
        shiftTextBySvgPoints(tickLabel, -0.375, 0.0)

    for panelDPlotAxis in (
        plotAxis for plotAxis in panelDAxes if plotAxis.has_data()
    ):
        shiftYAxisTickLabels(panelDPlotAxis, -40.367, 0.031)
    for panelEPlotAxis in (
        plotAxis for plotAxis in panelEAxes if plotAxis.has_data()
    ):
        shiftYAxisTickLabels(panelEPlotAxis, -39.949, 0.031)
    for rowThreeAxis in rowThreeAxes:
        if rowThreeAxis.yaxis.label.get_text() == "Percentage (%)":
            shiftYAxisTickLabels(rowThreeAxis, -31.693, -0.058)

    uniqueTextCorrections = {
        "a": (-0.001, -0.589),
        "b": (0.005, -0.589),
        "Yeo-7 organization along EC principal gradient": (-0.001, 0.547),
        "EC gradient 1 (subject-wise whole-brain z-score)": (0.005, 0.547),
        "c": (-0.013, 0.0),
        "EC gradient space (400 parcels)": (-1.562, 1.100),
        "d": (-0.020, 0.0),
        "FC–EC G1 correlation": (-1.569, 1.099),
        "e": (-0.499, 0.0),
        "Default–Vis/SomMot G1 separation": (-2.049, 1.099),
        "f": (-0.003, 0.017),
        "Reduced sensory–transmodal separation in ASD": (-1.852, 1.108),
        "g": (-0.001, -0.054),
        "Yeo-7 network group main-effect β values": (-1.848, 1.108),
        "EC gradient 1 (z)": (-94.768, 9.551),
        "EC gradient 2 (z)": (-9.520, 94.755),
        "FC G1 vs EC G1 Spearman ρ": (30.810, 165.971),
        "Subject-level r": (13.605, 0.194),
        "Group comparison": (-111.732, 0.025),
        "Participant-level distribution": (-169.284, 0.025),
        "***": (-14.233, -0.008),
        "All networks": (-73.834, 0.096),
    }
    for textValue, (
        horizontalShiftPoints,
        verticalShiftPoints,
    ) in uniqueTextCorrections.items():
        shiftUniqueText(
            figure,
            textValue,
            horizontalShiftPoints,
            verticalShiftPoints,
        )
    return figure


def saveCombinedFigure(
    figure: plt.Figure,
    outputBase: Path,
) -> list[Path]:
    outputPaths = [
        outputBase.with_suffix(fileExtension)
        for fileExtension in (".svg", ".pdf", ".png", ".tiff")
    ]
    setFigureFontWeights(figure)
    figure.savefig(outputPaths[0], facecolor="white")
    setFigureFontWeights(figure)
    figure.savefig(outputPaths[1], facecolor="white")
    targetRasterBounds = matplotlib.transforms.Bbox.from_bounds(
        0,
        0,
        14720 / EXPORT_DPI,
        14462.5 / EXPORT_DPI,
    )
    setFigureFontWeights(figure)
    figure.savefig(
        outputPaths[2],
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches=targetRasterBounds,
        pad_inches=0,
    )
    setFigureFontWeights(figure)
    figure.savefig(
        outputPaths[3],
        dpi=EXPORT_DPI,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return outputPaths


def parseArguments() -> argparse.Namespace:
    argumentParser = argparse.ArgumentParser(
        description=(
            "Rebuild the complete gradient-analysis figure using geometry "
            "calibrated from the Illustrator-adjusted SVG."
        )
    )
    defaultAnalysisRoot = Path(__file__).resolve().parent
    argumentParser.add_argument(
        "--analysis-root",
        type=Path,
        default=defaultAnalysisRoot,
    )
    argumentParser.add_argument(
        "--output-directory",
        type=Path,
        default=defaultAnalysisRoot,
    )
    argumentParser.add_argument(
        "--refresh-assets",
        action="store_true",
        help="Regenerate BrainSpace raster assets instead of reusing them.",
    )
    return argumentParser.parse_args()


def main() -> None:
    arguments = parseArguments()
    analysisRoot = arguments.analysis_root.resolve()
    outputDirectory = arguments.output_directory.resolve()
    outputDirectory.mkdir(parents=True, exist_ok=True)
    combinedFigure = drawCombinedFigure(
        analysisRoot,
        outputDirectory,
        arguments.refresh_assets,
    )
    outputBase = outputDirectory / "gradient-analysis-combined"
    calibrationSvgPath = outputBase.with_suffix(".svg")
    geometryAnalyzerModule = loadSourceModule(
        "svgGeometryAnalyzer",
        Path(__file__).resolve().parent / "analyze-svg-geometry.py",
    )
    combinedFigure.savefig(calibrationSvgPath, facecolor="white")
    for _ in range(2):
        calibrateTextUsingRenderedSvg(
            combinedFigure,
            analysisRoot / "S5.svg",
            calibrationSvgPath,
            geometryAnalyzerModule,
        )
        combinedFigure.savefig(calibrationSvgPath, facecolor="white")

    # Apply these edits after text calibration so the calibrated SVG cannot
    # restore the former gray axis styling, y-axis position, or font weights.
    applyPanelAFinalLayout(combinedFigure)
    applyPanelDFinalLayout(combinedFigure)
    applyUnifiedYAxisSpacing(combinedFigure)
    applyPanelCFinalLayout(combinedFigure)
    applyPanelFFinalLayout(combinedFigure)
    applyFinalFigureAppearance(combinedFigure)

    outputPaths = saveCombinedFigure(
        combinedFigure,
        outputBase,
    )
    for outputPath in outputPaths:
        print(outputPath.name)


if __name__ == "__main__":
    main()
