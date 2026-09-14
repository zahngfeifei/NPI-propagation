from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.datasets import load_conte69, load_parcellation
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from PIL import Image, ImageChops
from scipy.stats import gaussian_kde, spearmanr


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["image.composite_image"] = False


FIGURE_DIRECTORY = Path(__file__).resolve().parent
MAIN_ANALYSIS_DIRECTORY = FIGURE_DIRECTORY / "ABIDE2_6.基因分析_2mm"
SUPPLEMENT_ANALYSIS_DIRECTORY = FIGURE_DIRECTORY / "ABIDE2补充分析"
OUTPUT_DIRECTORY = FIGURE_DIRECTORY / "outputs"
BRAINSPACE_RENDER_DIRECTORY = OUTPUT_DIRECTORY / "brainspace-renders"
SOURCE_DATA_DIRECTORY = FIGURE_DIRECTORY / "source-data"

PHENOTYPE_PATH = (
    MAIN_ANALYSIS_DIRECTORY
    / "ABIDE2_主流程结果6_AHBA表型"
    / "ABIDE2_H1H4_phenotypes.parquet"
)
GENE_SCORE_PATH = (
    MAIN_ANALYSIS_DIRECTORY
    / "ABIDE2_主流程结果9_基因集表达评分"
    / "gene-set-scores-selected-374.csv"
)
PRIMARY_RESULT_PATH = (
    SUPPLEMENT_ANALYSIS_DIRECTORY
    / "ABIDE2_补充结果3_汇总统计"
    / "primary-result.csv"
)
SPATIAL_NULL_PATH = (
    MAIN_ANALYSIS_DIRECTORY
    / "ABIDE2_主流程结果12_空间相关"
    / "spatial-null-correlations.parquet"
)
MATCHED_NULL_PATH = (
    SUPPLEMENT_ANALYSIS_DIRECTORY
    / "ABIDE2_补充结果1_匹配基因集零模型"
    / "matched-gene-set-null-correlations.parquet"
)
MATCHING_QC_SUMMARY_PATH = (
    SUPPLEMENT_ANALYSIS_DIRECTORY
    / "ABIDE2_补充结果1_匹配基因集零模型"
    / "matching-qc-summary.csv"
)

PRIMARY_PHENOTYPE = "system_residual"
PRIMARY_GENE_SET = "SFARI_high_confidence_nonsyndromic"
PRIMARY_SCORE_METHOD = "mean_z"
EXPECTED_SELECTED_ROI_COUNT = 374
EXPECTED_FULL_ROI_COUNT = 400
EXPECTED_NULL_COUNT = 10_000
BRAIN_MAP_LIMIT = 2.5
BRAIN_COLORBAR_GAP_PX = 2.0
BRAIN_COLORBAR_HEIGHT_PX = 18.0
BRAIN_COLORBAR_BOTTOM_PADDING_PX = 16.0
SCATTER_HEIGHT_EXTENSION_SCALE = 0.5
NULL_X_LIMIT = 0.32
P_VALUE_Y_MIN = 1e-4
P_VALUE_Y_MAX = 1e-1
BOTTOM_HEADER_GAP_PX = 20.0
CANVAS_WIDTH_PX = 1800
CANVAS_HEIGHT_PX = 1333
EXPORT_DPI = 100
FIGURE_SIZE_INCHES = (
    CANVAS_WIDTH_PX / EXPORT_DPI,
    CANVAS_HEIGHT_PX / EXPORT_DPI,
)
PANEL_WIDTH_SCALE = 0.90
FONT_FAMILY = "Arial"
FONT_SIZE_PX = 25.0
FONT_SIZE_PT = FONT_SIZE_PX * 72.0 / EXPORT_DPI
FIGURE_FONT_SIZE = FONT_SIZE_PT
LEGEND_FONT_SIZE = FONT_SIZE_PT

COLOR_BLUE = "#3775BA"
COLOR_ORANGE = "#E58A2B"
COLOR_PURPLE = "#8D67B5"
COLOR_RED = "#000000"
COLOR_NEUTRAL_DARK = "#4D4D4D"
COLOR_NEUTRAL_MID = "#8C8C8C"
COLOR_NEUTRAL_LIGHT = "#D8D8D8"
COLOR_PANEL_BORDER = "#B7B7B7"
COLOR_EXCLUDED_PARCEL = (0.84, 0.84, 0.84, 1.0)
COLOR_ANNOTATION = "#000000"
MAIN_FIGURE_TITLE = "Empirical null distributions and conjunction analysis"


def configurePublicationStyle() -> None:
    """Apply compact publication defaults while keeping SVG text editable."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "svg.fonttype": "none",
            "image.composite_image": False,
            "pdf.fonttype": 42,
            "font.size": FIGURE_FONT_SIZE,
            "figure.titlesize": FIGURE_FONT_SIZE,
            "figure.labelsize": FIGURE_FONT_SIZE,
            "axes.titlesize": FIGURE_FONT_SIZE,
            "axes.labelsize": FIGURE_FONT_SIZE,
            "xtick.labelsize": FIGURE_FONT_SIZE,
            "ytick.labelsize": FIGURE_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def requireColumns(frame: pd.DataFrame, requiredColumns: set[str], frameName: str) -> None:
    """Fail early when a frozen result table no longer matches its contract."""
    missingColumns = sorted(requiredColumns.difference(frame.columns))
    if missingColumns:
        raise ValueError(f"{frameName} is missing required columns: {missingColumns}")


def calculateSampleZScore(values: np.ndarray) -> np.ndarray:
    """Return a sample-standardized vector using the same ddof=1 convention."""
    numericValues = np.asarray(values, dtype=float)
    sampleStandardDeviation = numericValues.std(ddof=1)
    if not np.isfinite(sampleStandardDeviation) or sampleStandardDeviation <= 0:
        raise ValueError("Cannot standardize a constant or non-finite vector.")
    return (numericValues - numericValues.mean()) / sampleStandardDeviation


def calculateSystemResidual(values: np.ndarray, systemLabels: pd.Series) -> np.ndarray:
    """Residualize parcel values against intercept plus H1–H4 system indicators."""
    numericValues = np.asarray(values, dtype=float)
    systemIndicatorFrame = pd.get_dummies(
        systemLabels.astype(str),
        drop_first=True,
        dtype=float,
    )
    designMatrix = np.column_stack(
        [
            np.ones(numericValues.size, dtype=float),
            systemIndicatorFrame.to_numpy(dtype=float),
        ]
    )
    return numericValues - designMatrix @ (np.linalg.pinv(designMatrix) @ numericValues)


def calculateEmpiricalTwoSidedP(nullValues: np.ndarray, observedValue: float) -> float:
    """Reproduce the frozen finite-sample empirical two-sided P formula."""
    numericNullValues = np.asarray(nullValues, dtype=float)
    extremeCount = int(np.sum(np.abs(numericNullValues) >= abs(observedValue)))
    return (extremeCount + 1) / (numericNullValues.size + 1)


def loadPrimaryFigureData() -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Load, filter, merge and validate the ABIDE1 primary-result inputs."""
    sourcePhenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    sourceGeneScoreFrame = pd.read_csv(GENE_SCORE_PATH)
    primaryResultFrame = pd.read_csv(PRIMARY_RESULT_PATH)
    spatialNullFrame = pd.read_parquet(SPATIAL_NULL_PATH)
    matchedNullFrame = pd.read_parquet(MATCHED_NULL_PATH)
    matchingQcFrame = pd.read_csv(MATCHING_QC_SUMMARY_PATH)

    requireColumns(
        sourcePhenotypeFrame,
        {"ROI_ID_1based", "system", "hemisphere", "system_residual", "system_residual_z"},
        "phenotype table",
    )
    requireColumns(
        sourceGeneScoreFrame,
        {"ROI_ID_1based", "geneSetName", "scoreMethod", "score"},
        "gene-score table",
    )
    requireColumns(
        primaryResultFrame,
        {
            "phenotypeName",
            "geneSetName",
            "scoreMethod",
            "spearmanR",
            "pSpatial",
            "pGeneSet",
            "pConjunction",
            "spatialNullCount",
            "matchedNullCount",
        },
        "primary-result table",
    )
    requireColumns(
        matchingQcFrame,
        {"geneSetName", "trueGeneCount", "matchingQcPassed"},
        "matching-QC table",
    )

    primaryGeneScoreFrame = sourceGeneScoreFrame.loc[
        (sourceGeneScoreFrame["geneSetName"] == PRIMARY_GENE_SET)
        & (sourceGeneScoreFrame["scoreMethod"] == PRIMARY_SCORE_METHOD),
        ["ROI_ID_1based", "score"],
    ].copy()
    primaryGeneScoreFrame = primaryGeneScoreFrame.rename(
        columns={"score": "riskGeneExpressionMeanZ"}
    )

    parcelFrame = sourcePhenotypeFrame.merge(
        primaryGeneScoreFrame,
        on="ROI_ID_1based",
        how="inner",
        validate="one_to_one",
    ).sort_values("ROI_ID_1based")
    parcelFrame["riskGeneExpressionSystemResidual"] = calculateSystemResidual(
        parcelFrame["riskGeneExpressionMeanZ"].to_numpy(dtype=float),
        parcelFrame["system"],
    )
    parcelFrame["riskGeneExpressionSystemResidualZ"] = calculateSampleZScore(
        parcelFrame["riskGeneExpressionSystemResidual"].to_numpy(dtype=float)
    )
    parcelFrame["imagingPhenotypeZ"] = parcelFrame["system_residual_z"].astype(float)

    primaryRows = primaryResultFrame.loc[
        (primaryResultFrame["phenotypeName"] == PRIMARY_PHENOTYPE)
        & (primaryResultFrame["geneSetName"] == PRIMARY_GENE_SET)
        & (primaryResultFrame["scoreMethod"] == PRIMARY_SCORE_METHOD)
    ]
    if len(primaryRows) != 1:
        raise ValueError(f"Expected exactly one primary result row, found {len(primaryRows)}.")
    primaryResult = primaryRows.iloc[0].copy()
    matchingQcRows = matchingQcFrame.loc[
        matchingQcFrame["geneSetName"] == PRIMARY_GENE_SET
    ]
    if len(matchingQcRows) != 1:
        raise ValueError(
            f"Expected exactly one matching-QC row, found {len(matchingQcRows)}."
        )
    if not bool(matchingQcRows.iloc[0]["matchingQcPassed"]):
        raise ValueError("Matched gene-set quality control did not pass.")
    primaryResult["effectiveGeneCount"] = int(
        matchingQcRows.iloc[0]["trueGeneCount"]
    )

    primarySpatialNull = spatialNullFrame.loc[
        (spatialNullFrame["phenotypeName"] == PRIMARY_PHENOTYPE)
        & (spatialNullFrame["geneSetName"] == PRIMARY_GENE_SET)
        & (spatialNullFrame["scoreMethod"] == PRIMARY_SCORE_METHOD),
        "nullSpearmanR",
    ].to_numpy(dtype=float)
    primaryMatchedNull = matchedNullFrame.loc[
        (matchedNullFrame["phenotypeName"] == PRIMARY_PHENOTYPE)
        & (matchedNullFrame["geneSetName"] == PRIMARY_GENE_SET),
        "nullSpearmanR",
    ].to_numpy(dtype=float)

    validatePrimaryFigureData(
        parcelFrame,
        primaryResult,
        primarySpatialNull,
        primaryMatchedNull,
    )
    return parcelFrame, primaryResult, primarySpatialNull, primaryMatchedNull


def validatePrimaryFigureData(
    parcelFrame: pd.DataFrame,
    primaryResult: pd.Series,
    spatialNullValues: np.ndarray,
    matchedNullValues: np.ndarray,
) -> None:
    """Verify parcel mapping, effect size and both empirical P values."""
    if len(parcelFrame) != EXPECTED_SELECTED_ROI_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SELECTED_ROI_COUNT} selected parcels, found {len(parcelFrame)}."
        )
    if not parcelFrame["ROI_ID_1based"].is_unique:
        raise ValueError("ROI identifiers must be unique before BrainSpace mapping.")
    validatedRoiIds = parcelFrame["ROI_ID_1based"].to_numpy(dtype=int)
    if validatedRoiIds.min() < 1 or validatedRoiIds.max() > EXPECTED_FULL_ROI_COUNT:
        raise ValueError("ROI identifiers fall outside the Schaefer-400 label range.")
    if spatialNullValues.size != EXPECTED_NULL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NULL_COUNT} spatial nulls, found {spatialNullValues.size}."
        )
    if matchedNullValues.size != EXPECTED_NULL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NULL_COUNT} matched nulls, found {matchedNullValues.size}."
        )

    observedSpearman = float(
        spearmanr(
            parcelFrame["riskGeneExpressionSystemResidualZ"],
            parcelFrame["imagingPhenotypeZ"],
        ).statistic
    )
    if not np.isclose(observedSpearman, float(primaryResult["spearmanR"]), atol=1e-12):
        raise ValueError("Recomputed Spearman correlation does not match the frozen result.")

    recomputedSpatialP = calculateEmpiricalTwoSidedP(spatialNullValues, observedSpearman)
    recomputedMatchedP = calculateEmpiricalTwoSidedP(matchedNullValues, observedSpearman)
    if not np.isclose(recomputedSpatialP, float(primaryResult["pSpatial"]), atol=1e-12):
        raise ValueError("Recomputed spatial-null P does not match the frozen result.")
    if not np.isclose(recomputedMatchedP, float(primaryResult["pGeneSet"]), atol=1e-12):
        raise ValueError("Recomputed matched-gene-set P does not match the frozen result.")
    recomputedConjunctionP = max(recomputedSpatialP, recomputedMatchedP)
    if not np.isclose(
        recomputedConjunctionP,
        float(primaryResult["pConjunction"]),
        atol=1e-12,
    ):
        raise ValueError("Recomputed conjunction P does not match the frozen result.")


def buildFullParcelMap(parcelFrame: pd.DataFrame, valueColumn: str) -> np.ndarray:
    """Place 374 selected values into the full Schaefer-400 label vector."""
    fullParcelValues = np.full(EXPECTED_FULL_ROI_COUNT, np.nan, dtype=float)
    zeroBasedRoiIds = parcelFrame["ROI_ID_1based"].to_numpy(dtype=int) - 1
    fullParcelValues[zeroBasedRoiIds] = parcelFrame[valueColumn].to_numpy(dtype=float)
    return fullParcelValues


def cropWhiteMargins(imagePath: Path, paddingPixels: int = 10) -> None:
    """Crop only uniform outer whitespace from a BrainSpace screenshot."""
    renderedImage = Image.open(imagePath).convert("RGB")
    whiteBackground = Image.new("RGB", renderedImage.size, (255, 255, 255))
    differenceImage = ImageChops.difference(renderedImage, whiteBackground)
    differenceBox = differenceImage.getbbox()
    if differenceBox is None:
        return
    left, upper, right, lower = differenceBox
    paddedBox = (
        max(0, left - paddingPixels),
        max(0, upper - paddingPixels),
        min(renderedImage.width, right + paddingPixels),
        min(renderedImage.height, lower + paddingPixels),
    )
    renderedImage.crop(paddedBox).save(imagePath)


def renderBrainSpaceMap(
    fullParcelValues: np.ndarray,
    outputPath: Path,
    surfaceLeft: object,
    surfaceRight: object,
    schaeferLabels: np.ndarray,
) -> None:
    """Render Schaefer-400 parcel values on Conte69 surfaces in four views."""
    vertexValues = map_to_labels(
        fullParcelValues,
        schaeferLabels,
        mask=0,
        fill=np.nan,
    )
    plot_hemispheres(
        surfaceLeft,
        surfaceRight,
        array_name=vertexValues,
        color_bar=False,
        color_range=(-BRAIN_MAP_LIMIT, BRAIN_MAP_LIMIT),
        layout_style="grid",
        cmap=mpl.colormaps["RdBu_r"],
        nan_color=COLOR_EXCLUDED_PARCEL,
        zoom=1.46,
        background=(1, 1, 1),
        size=(980, 680),
        interactive=False,
        screenshot=True,
        filename=str(outputPath),
        scale=(2, 2),
        transparent_bg=False,
        suppress_warnings=True,
    )
    cropWhiteMargins(outputPath)


def addPanelLabel(axis: plt.Axes, label: str, xPosition: float = -0.10) -> None:
    """Add a compact Nature-style lowercase panel label."""
    axis.text(
        xPosition,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def addLeftAlignedTitle(
    axis: plt.Axes,
    title: str,
    xPosition: float = 0.0,
    yPosition: float = 1.0,
) -> None:
    """Place a panel title using an explicit left anchor instead of centered title placement."""
    axis.text(
        xPosition,
        yPosition,
        title,
        transform=axis.transAxes,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def getAxisOuterLeftFigureCoordinate(axis: plt.Axes) -> float:
    """Return the left edge of the full decorated axis (ticks and y-label included)."""
    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tightBoundingBox = axis.get_tightbbox(renderer)
    return tightBoundingBox.transformed(figure.transFigure.inverted()).x0


def getAxisPlotLeftFigureCoordinate(axis: plt.Axes) -> float:
    """Return the left edge of the plotting area itself."""
    return axis.get_position().x0


def convertPixelLengthToAxisFraction(axis: plt.Axes, pixelLength: float) -> float:
    """Convert a vertical pixel length to the equivalent axis-fraction height."""
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    axisBoundingBox = axis.get_window_extent(renderer=renderer)
    if axisBoundingBox.height <= 0:
        raise ValueError("Axis height must be positive before converting pixel lengths.")
    return float(pixelLength) / float(axisBoundingBox.height)


def addFigureLevelPanelHeader(
    figure: plt.Figure,
    titleX: float,
    rowY: float,
    panelLabel: str,
    panelTitle: str,
    labelOffset: float = 0.028,
) -> tuple[mpl.text.Text, mpl.text.Text]:
    """Draw a panel label to the left of the title on the same top-aligned row."""
    labelX = max(0.005, titleX - labelOffset)
    labelText = figure.text(
        labelX,
        rowY,
        panelLabel,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    titleText = figure.text(
        titleX,
        rowY,
        panelTitle,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    return labelText, titleText


def moveFigureLevelHeaderToGapAboveAxis(
    figure: plt.Figure,
    headerArtists: tuple[mpl.text.Text, mpl.text.Text],
    referenceAxis: plt.Axes,
    gapPixels: float = BOTTOM_HEADER_GAP_PX,
    iterationCount: int = 2,
) -> None:
    """Move one figure-level header so the title sits gapPixels above a panel."""
    labelText, titleText = headerArtists
    for _ in range(iterationCount):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        titleBoundingBox = titleText.get_window_extent(renderer=renderer)
        axisBoundingBox = referenceAxis.get_window_extent(renderer=renderer)
        desiredTitleBottom = float(axisBoundingBox.y1) + float(gapPixels)
        pixelShift = desiredTitleBottom - float(titleBoundingBox.y0)
        if abs(pixelShift) < 0.5:
            break
        figureShift = pixelShift / float(figure.bbox.height)
        labelX, labelY = labelText.get_position()
        titleX, titleY = titleText.get_position()
        labelText.set_position((labelX, labelY + figureShift))
        titleText.set_position((titleX, titleY + figureShift))


def addBrainMapPanel(
    axis: plt.Axes,
    brainImagePath: Path,
    title: str,
) -> plt.Axes:
    """Place a BrainSpace render, view labels and a shared-style color bar."""
    brainImage = Image.open(brainImagePath)
    brainImageAxis = axis.inset_axes([0, 0, 1, 1], zorder=0)
    brainImageAxis.imshow(
        brainImage,
        interpolation="none",
        rasterized=False,
    )
    # imshow preserves the image aspect ratio. Without an explicit anchor,
    # the brain image is vertically centered in the taller panel, creating
    # a large apparent gap above the color bar.
    brainImageAxis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.text(
        0.27,
        1.0,
        "Left hemisphere",
        transform=axis.transAxes,
        fontsize=FIGURE_FONT_SIZE,
        ha="center",
        va="top",
    )
    axis.text(
        0.74,
        1.0,
        "Right hemisphere",
        transform=axis.transAxes,
        fontsize=FIGURE_FONT_SIZE,
        ha="center",
        va="top",
    )
    colorBarHeightFraction = convertPixelLengthToAxisFraction(axis, BRAIN_COLORBAR_HEIGHT_PX)
    colorBarBottomPaddingFraction = convertPixelLengthToAxisFraction(
        axis,
        BRAIN_COLORBAR_BOTTOM_PADDING_PX,
    )
    colorBarAxis = axis.inset_axes(
        [0.20, colorBarBottomPaddingFraction, 0.60, colorBarHeightFraction]
    )
    colorNormalization = mpl.colors.Normalize(
        vmin=-BRAIN_MAP_LIMIT,
        vmax=BRAIN_MAP_LIMIT,
    )
    colorBar = mpl.colorbar.ColorbarBase(
        colorBarAxis,
        cmap=mpl.colormaps["RdBu_r"],
        norm=colorNormalization,
        orientation="horizontal",
    )
    colorBar.set_ticks([-BRAIN_MAP_LIMIT, 0, BRAIN_MAP_LIMIT])
    colorBar.set_ticklabels([f"−{BRAIN_MAP_LIMIT:g}", "0", f"{BRAIN_MAP_LIMIT:g}"])
    colorBar.set_label("Parcel z score", fontsize=FIGURE_FONT_SIZE, labelpad=1)
    colorBar.ax.tick_params(labelsize=FIGURE_FONT_SIZE, length=2, pad=1)
    colorBar.outline.set_linewidth(0.5)
    return colorBarAxis


def scalePanelWidthsAboutCenters(
    panelAxes: list[plt.Axes],
    widthScale: float = PANEL_WIDTH_SCALE,
) -> None:
    """Shrink every main panel horizontally while keeping its center fixed."""
    if not 0 < widthScale <= 1:
        raise ValueError("widthScale must be in the interval (0, 1].")

    for panelAxis in panelAxes:
        originalPosition = panelAxis.get_position()
        scaledWidth = originalPosition.width * widthScale
        scaledLeft = originalPosition.x0 + (originalPosition.width - scaledWidth) / 2
        panelAxis.set_position(
            [
                scaledLeft,
                originalPosition.y0,
                scaledWidth,
                originalPosition.height,
            ]
        )


def extendAxisDownward(axis: plt.Axes, extraHeight: float) -> None:
    """Increase one axis downward while preserving its top edge position."""
    if extraHeight <= 0:
        return
    axisPosition = axis.get_position()
    axis.set_position(
        [
            axisPosition.x0,
            axisPosition.y0 - extraHeight,
            axisPosition.width,
            axisPosition.height + extraHeight,
        ]
    )


def validateMainPanelTopAlignment(panelAxes: list[plt.Axes]) -> None:
    """Require every main top-row subplot to share the same top boundary."""
    if not panelAxes:
        raise ValueError("At least one panel axis is required for top-alignment validation.")
    panelAxes[0].figure.canvas.draw()
    panelTopEdges = np.array(
        [panelAxis.get_position().y1 for panelAxis in panelAxes],
        dtype=float,
    )
    if float(np.ptp(panelTopEdges)) > 1e-6:
        raise ValueError(
            "Main subplot top edges are inconsistent: "
            f"{[round(float(panelTopEdge), 6) for panelTopEdge in panelTopEdges]}"
        )


def adjustAxisHeightToAlignXAxisLabelBottom(
    targetAxis: plt.Axes,
    referenceAxis: plt.Axes,
    iterationCount: int = 2,
) -> None:
    """Adjust axis height so the target x-axis label bottom aligns with the reference label."""
    figure = targetAxis.figure
    for _ in range(iterationCount):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        targetBoundingBox = targetAxis.xaxis.get_label().get_window_extent(renderer=renderer)
        referenceBoundingBox = referenceAxis.xaxis.get_label().get_window_extent(renderer=renderer)
        pixelDelta = float(targetBoundingBox.y0 - referenceBoundingBox.y0)
        if abs(pixelDelta) < 0.5:
            break
        figureDelta = pixelDelta / float(figure.bbox.height)
        axisPosition = targetAxis.get_position()
        newBottom = axisPosition.y0 - figureDelta
        newHeight = axisPosition.height + figureDelta
        minimumHeight = 0.05
        if newHeight < minimumHeight:
            newBottom = axisPosition.y1 - minimumHeight
            newHeight = minimumHeight
        targetAxis.set_position(
            [
                axisPosition.x0,
                newBottom,
                axisPosition.width,
                newHeight,
            ]
        )


def validateAxesInsideFigure(figure: plt.Figure) -> None:
    """Fail before SVG export if any axes extend outside the fixed figure canvas."""
    figure.canvas.draw()
    invalidAxes = []
    for axisIndex, axis in enumerate(figure.axes):
        position = axis.get_position()
        if (
            position.x0 < -1e-9
            or position.y0 < -1e-9
            or position.x1 > 1.0 + 1e-9
            or position.y1 > 1.0 + 1e-9
        ):
            invalidAxes.append(
                (
                    axisIndex,
                    round(float(position.x0), 6),
                    round(float(position.y0), 6),
                    round(float(position.x1), 6),
                    round(float(position.y1), 6),
                )
            )
    if invalidAxes:
        raise ValueError(
            "One or more axes extend outside the SVG canvas, which can cause "
            f"Illustrator paste errors: {invalidAxes}"
        )


def evaluateDensity(values: np.ndarray, xGrid: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian kernel density with finite-value validation."""
    numericValues = np.asarray(values, dtype=float)
    finiteValues = numericValues[np.isfinite(numericValues)]
    if finiteValues.size < 3:
        raise ValueError("At least three finite values are required for density estimation.")
    return gaussian_kde(finiteValues)(xGrid)


def calculateRegressionBand(
    xValues: np.ndarray,
    yValues: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an OLS line and pointwise 95% confidence band for display."""
    xNumeric = np.asarray(xValues, dtype=float)
    yNumeric = np.asarray(yValues, dtype=float)
    regressionSlope, regressionIntercept = np.polyfit(xNumeric, yNumeric, 1)
    xGrid = np.linspace(xNumeric.min(), xNumeric.max(), 240)
    fittedGrid = regressionIntercept + regressionSlope * xGrid
    fittedObserved = regressionIntercept + regressionSlope * xNumeric
    residualSumSquares = np.sum((yNumeric - fittedObserved) ** 2)
    residualVariance = residualSumSquares / (xNumeric.size - 2)
    centeredSumSquares = np.sum((xNumeric - xNumeric.mean()) ** 2)
    standardError = np.sqrt(
        residualVariance
        * (1 / xNumeric.size + (xGrid - xNumeric.mean()) ** 2 / centeredSumSquares)
    )
    confidenceDelta = 1.96 * standardError
    return xGrid, fittedGrid, fittedGrid - confidenceDelta, fittedGrid + confidenceDelta


def addScatterPanel(
    scatterAxis: plt.Axes,
    parcelFrame: pd.DataFrame,
    primaryResult: pd.Series,
) -> None:
    """Draw the parcelwise spatial correspondence without marginal densities."""
    xValues = parcelFrame["riskGeneExpressionSystemResidualZ"].to_numpy(dtype=float)
    yValues = parcelFrame["imagingPhenotypeZ"].to_numpy(dtype=float)
    scatterAxis.scatter(
        xValues,
        yValues,
        s=11,
        color="#2F2F2F",
        alpha=0.48,
        linewidths=0,
        rasterized=False,
    )
    xGrid, fittedGrid, lowerBand, upperBand = calculateRegressionBand(xValues, yValues)
    scatterAxis.fill_between(
        xGrid,
        lowerBand,
        upperBand,
        color=COLOR_NEUTRAL_MID,
        alpha=0.22,
        linewidth=0,
    )
    scatterAxis.plot(xGrid, fittedGrid, color=COLOR_NEUTRAL_DARK, linewidth=1.5)
    scatterAxis.axhline(0, color=COLOR_NEUTRAL_LIGHT, linewidth=0.7, zorder=0)
    scatterAxis.axvline(0, color=COLOR_NEUTRAL_LIGHT, linewidth=0.7, zorder=0)
    scatterAxis.set_xlabel("Risk-gene expression (z score)")
    scatterAxis.set_ylabel("Attenuated coupling in ASD (z score)")
    scatterAxis.set_xlim(-3.15, 3.45)
    scatterAxis.set_ylim(-4.95, 3.05)
    scatterAxis.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    scatterAxis.set_yticks([-4, -2, 0, 2])
    scatterAxis.text(
        0.98,
        0.98,
        (
            f"Spearman ρ = {float(primaryResult['spearmanR']):.4f}\n"
            f"P$_{{spatial}}$ = {float(primaryResult['pSpatial']):.4f}\n"
            f"N = {len(parcelFrame)} parcels"
        ),
        transform=scatterAxis.transAxes,
        ha="right",
        va="top",
        fontsize=FIGURE_FONT_SIZE,
    )

def addNullDistributionPanel(
    axis: plt.Axes,
    nullValues: np.ndarray,
    observedSpearman: float,
    empiricalP: float,
    panelTitle: str,
    pLabel: str,
    surrogateLabel: str,
) -> None:
    """Draw one empirical null distribution as histogram bars plus a density line."""
    xGrid = np.linspace(-NULL_X_LIMIT, NULL_X_LIMIT, 500)
    densityValues = evaluateDensity(nullValues, xGrid)
    histogramEdges = np.linspace(-NULL_X_LIMIT, NULL_X_LIMIT, 37)
    axis.hist(
        nullValues,
        bins=histogramEdges,
        density=True,
        color=COLOR_NEUTRAL_LIGHT,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
    )
    axis.plot(xGrid, densityValues, color=COLOR_NEUTRAL_DARK, linewidth=1.15)
    axis.axvline(
        float(np.median(nullValues)),
        color=COLOR_NEUTRAL_DARK,
        linestyle="--",
        linewidth=0.9,
    )
    axis.axvline(observedSpearman, color=COLOR_RED, linewidth=1.35)
    axis.set_xlim(-NULL_X_LIMIT, NULL_X_LIMIT)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Spearman ρ")
    axis.set_ylabel("Density")
    axis.text(
        0.03,
        0.94,
        f"Observed ρ = {observedSpearman:.4f}\n{pLabel} = {empiricalP:.4f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FIGURE_FONT_SIZE,
        color=COLOR_ANNOTATION,
    )

def addConjunctionPanel(
    panelAxis: plt.Axes,
    densityAxis: plt.Axes,
    pValueAxis: plt.Axes,
    spatialNullValues: np.ndarray,
    matchedNullValues: np.ndarray,
    primaryResult: pd.Series,
) -> None:
    """Show both empirical null families and their conjunction P values."""
    densityAxis.set_zorder(2)
    densityAxis.patch.set_alpha(0)
    pValueAxis.set_zorder(1)
    observedSpearman = float(primaryResult["spearmanR"])
    xGrid = np.linspace(-NULL_X_LIMIT, NULL_X_LIMIT, 500)
    spatialDensity = evaluateDensity(spatialNullValues, xGrid)
    matchedDensity = evaluateDensity(matchedNullValues, xGrid)

    histogramEdges = np.linspace(-NULL_X_LIMIT, NULL_X_LIMIT, 33)
    densityAxis.hist(
        spatialNullValues,
        bins=histogramEdges,
        density=True,
        color=COLOR_BLUE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.22,
    )
    densityAxis.hist(
        matchedNullValues,
        bins=histogramEdges,
        density=True,
        color=COLOR_ORANGE,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.22,
    )
    densityAxis.plot(
        xGrid,
        spatialDensity,
        color=COLOR_BLUE,
        linewidth=1.3,
        label="Spatial null",
    )
    densityAxis.plot(
        xGrid,
        matchedDensity,
        color=COLOR_ORANGE,
        linewidth=1.3,
        label="Gene-set null",
    )
    densityAxis.axvline(observedSpearman, color=COLOR_RED, linewidth=1.2)
    densityAxis.axvline(0, color=COLOR_NEUTRAL_MID, linewidth=0.7, linestyle=":")
    densityAxis.set_xlim(-NULL_X_LIMIT, NULL_X_LIMIT)
    densityAxis.set_ylim(bottom=0)
    densityAxis.set_xlabel("Spearman ρ")
    densityAxis.set_ylabel("Density")
    densityAxis.legend(
        loc="upper center",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    densityAxis.text(
        0.03,
        0.58,
        f"Observed\nρ = {observedSpearman:.4f}",
        transform=densityAxis.transAxes,
        color=COLOR_ANNOTATION,
        fontsize=FIGURE_FONT_SIZE,
        ha="left",
        va="center",
    )

    pValues = np.array(
        [
            float(primaryResult["pSpatial"]),
            float(primaryResult["pGeneSet"]),
            float(primaryResult["pConjunction"]),
        ]
    )
    pValueColors = [COLOR_BLUE, COLOR_ORANGE, COLOR_PURPLE]
    pValueLabels = ["Spatial", "Matched", "Conjunction"]
    barPositions = np.arange(pValues.size)
    # The Illustrator-specific SVG removes clip paths. On a logarithmic axis,
    # bars drawn from the default baseline y=0 transform to extremely large
    # negative SVG coordinates and therefore extend far below the artboard once
    # clipping is removed. Start every bar at the visible logarithmic lower
    # limit instead, so the SVG geometry itself remains inside the panel.
    pValueBarHeights = pValues - P_VALUE_Y_MIN
    if np.any(pValueBarHeights <= 0):
        raise ValueError("All empirical P values must exceed P_VALUE_Y_MIN.")
    pValueAxis.bar(
        barPositions,
        pValueBarHeights,
        bottom=P_VALUE_Y_MIN,
        width=0.62,
        color=pValueColors,
        edgecolor="white",
        linewidth=0.6,
    )
    pValueAxis.set_yscale("log")
    pValueAxis.set_ylim(P_VALUE_Y_MIN, P_VALUE_Y_MAX)
    pValueAxis.set_xticks(barPositions)
    pValueAxis.set_xticklabels(
        pValueLabels,
        rotation=40,
        ha="right",
        fontsize=FIGURE_FONT_SIZE,
    )
    pValueAxis.set_ylabel("Empirical P")
    pValueAxis.yaxis.tick_right()
    pValueAxis.yaxis.set_label_position("right")
    pValueAxis.spines["left"].set_visible(False)
    pValueAxis.spines["right"].set_visible(True)
    pValueAxis.spines["right"].set_color(COLOR_PANEL_BORDER)


def addPanelBorders(axes: list[plt.Axes]) -> None:
    """Use quiet separators to bind the multi-panel page without boxes."""
    for axis in axes:
        axis.spines["left"].set_color(COLOR_PANEL_BORDER)
        axis.spines["bottom"].set_color(COLOR_PANEL_BORDER)


def writeSourceData(
    parcelFrame: pd.DataFrame,
    primaryResult: pd.Series,
    spatialNullValues: np.ndarray,
    matchedNullValues: np.ndarray,
) -> None:
    """Save the plotted parcel values and concise statistical provenance."""
    sourceParcelFrame = parcelFrame.loc[
        :,
        [
            "ROI_ID_1based",
            "hemisphere",
            "system",
            "imagingPhenotypeZ",
            "riskGeneExpressionMeanZ",
            "riskGeneExpressionSystemResidual",
            "riskGeneExpressionSystemResidualZ",
        ],
    ].rename(
        columns={
            "ROI_ID_1based": "roi_id_1based",
            "imagingPhenotypeZ": "imaging_phenotype_z",
            "riskGeneExpressionMeanZ": "risk_gene_expression_mean_z",
            "riskGeneExpressionSystemResidual": (
                "risk_gene_expression_system_residual"
            ),
            "riskGeneExpressionSystemResidualZ": (
                "risk_gene_expression_system_residual_z"
            ),
        }
    )
    sourceParcelFrame.to_csv(
        SOURCE_DATA_DIRECTORY / "abide1-primary-parcel-data.csv",
        index=False,
    )

    excludedRoiIds = sorted(
        set(range(1, EXPECTED_FULL_ROI_COUNT + 1))
        .difference(sourceParcelFrame["roi_id_1based"].astype(int))
    )
    figureStatistics = {
        "dataset": "ABIDE1",
        "atlas": "Schaefer-400 on Conte69 surfaces",
        "fontFamily": FONT_FAMILY,
        "canvasWidthPx": CANVAS_WIDTH_PX,
        "canvasHeightPx": CANVAS_HEIGHT_PX,
        "exportDpi": EXPORT_DPI,
        "fontSizePx": FONT_SIZE_PX,
        "annotationFontSizePx": FONT_SIZE_PX,
        "legendFontSizePx": FONT_SIZE_PX,
        "fontSizePt": FIGURE_FONT_SIZE,
        "annotationFontSizePt": FIGURE_FONT_SIZE,
        "legendFontSizePt": LEGEND_FONT_SIZE,
        "selectedParcelCount": int(len(parcelFrame)),
        "excludedParcelCount": int(len(excludedRoiIds)),
        "excludedRoiIds1Based": excludedRoiIds,
        "phenotype": PRIMARY_PHENOTYPE,
        "geneSet": PRIMARY_GENE_SET,
        "scoreMethod": PRIMARY_SCORE_METHOD,
        "effectiveGeneCount": int(primaryResult["effectiveGeneCount"]),
        "spearmanR": float(primaryResult["spearmanR"]),
        "spatialP": float(primaryResult["pSpatial"]),
        "matchedGeneSetP": float(primaryResult["pGeneSet"]),
        "conjunctionP": float(primaryResult["pConjunction"]),
        "spatialNullCount": int(spatialNullValues.size),
        "matchedNullCount": int(matchedNullValues.size),
        "pValueFormula": "(extremeCount + 1) / (nullCount + 1), two-sided",
        "conjunctionFormula": "max(spatialP, matchedGeneSetP)",
        "interpretationBoundary": (
            "Regional correspondence between normal-adult AHBA expression and a "
            "group-level ABIDE1 phenotype; not an individual-level or causal result."
        ),
    }
    with (SOURCE_DATA_DIRECTORY / "figure-statistics.json").open(
        "w",
        encoding="utf-8",
    ) as statisticsFile:
        json.dump(figureStatistics, statisticsFile, ensure_ascii=False, indent=2)


def buildFigure(
    parcelFrame: pd.DataFrame,
    primaryResult: pd.Series,
    spatialNullValues: np.ndarray,
    matchedNullValues: np.ndarray,
    phenotypeBrainPath: Path,
    geneBrainPath: Path,
) -> plt.Figure:
    """Assemble the full publication-style ABIDE1 result figure."""
    figure = plt.figure(
        figsize=FIGURE_SIZE_INCHES,
        dpi=EXPORT_DPI,
        facecolor="white",
    )
    outerGrid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.8],
        hspace=0.34,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.095,
    )

    topGrid = outerGrid[0, 0].subgridspec(
        1,
        4,
        width_ratios=[1.3, 1.3, 0.10, 1.0],
        wspace=0.02,
    )
    bottomGrid = outerGrid[1, 0].subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.16,
    )

    phenotypeBrainAxis = figure.add_subplot(topGrid[0, 0])
    geneBrainAxis = figure.add_subplot(topGrid[0, 1])
    scatterAxis = figure.add_subplot(topGrid[0, 3])

    spatialNullAxis = figure.add_subplot(bottomGrid[0, 0])
    matchedNullAxis = figure.add_subplot(bottomGrid[0, 1])
    conjunctionPanelAxis = figure.add_subplot(bottomGrid[0, 2])

    # The GridSpec layout and all panel centers remain unchanged.
    # All six main panels use the same proportional width reduction.
    scalePanelWidthsAboutCenters(
        [
            phenotypeBrainAxis,
            geneBrainAxis,
            scatterAxis,
            spatialNullAxis,
            matchedNullAxis,
            conjunctionPanelAxis,
        ]
    )

    colorBarOverhangHeight = max(
        0.0,
        (
            BRAIN_COLORBAR_GAP_PX + BRAIN_COLORBAR_HEIGHT_PX
            - BRAIN_COLORBAR_BOTTOM_PADDING_PX
        ) / CANVAS_HEIGHT_PX,
    )
    extendAxisDownward(scatterAxis, colorBarOverhangHeight * SCATTER_HEIGHT_EXTENSION_SCALE)

    conjunctionPanelAxis.set_axis_off()
    conjunctionDensityAxis = conjunctionPanelAxis.inset_axes([0.0, 0.0, 0.55, 1.0])
    conjunctionPValueAxis = conjunctionPanelAxis.inset_axes([0.55, 0.0, 0.45, 1.0])

    phenotypeColorBarAxis = addBrainMapPanel(
        phenotypeBrainAxis,
        phenotypeBrainPath,
        "Level-adjusted imaging phenotype\nHC–ASD coupling contribution",
    )
    geneColorBarAxis = addBrainMapPanel(
        geneBrainAxis,
        geneBrainPath,
        "High-confidence nonsyndromic ASD-risk gene expression (118 genes)\nLevel-adjusted cortical expression",
    )
    addScatterPanel(
        scatterAxis,
        parcelFrame,
        primaryResult,
    )
    adjustAxisHeightToAlignXAxisLabelBottom(scatterAxis, phenotypeColorBarAxis)

    observedSpearman = float(primaryResult["spearmanR"])
    addNullDistributionPanel(
        spatialNullAxis,
        spatialNullValues,
        observedSpearman,
        float(primaryResult["pSpatial"]),
        "Spatial-autocorrelation null\ndistribution",
        "P$_{spatial}$",
        "BrainSMASH surrogates",
    )
    addNullDistributionPanel(
        matchedNullAxis,
        matchedNullValues,
        observedSpearman,
        float(primaryResult["pGeneSet"]),
        "Expression-property-matched\ngene-set null distribution",
        "P$_{matched}$",
        "matched gene sets",
    )
    addConjunctionPanel(
        conjunctionPanelAxis,
        conjunctionDensityAxis,
        conjunctionPValueAxis,
        spatialNullValues,
        matchedNullValues,
        primaryResult,
    )

    figure.canvas.draw()
    topRowY = 0.972
    bottomRowY = 0.432

    topRowHeaders = [
        (
            getAxisPlotLeftFigureCoordinate(phenotypeBrainAxis),
            "a",
            "Level-adjusted imaging phenotype\nHC–ASD coupling contribution",
        ),
        (
            getAxisPlotLeftFigureCoordinate(geneBrainAxis),
            "b",
            "High-confidence nonsyndromic\nASD-risk gene expression (118 genes)\nLevel-adjusted cortical expression",
        ),
        (
            getAxisOuterLeftFigureCoordinate(scatterAxis),
            "c",
            "Parcelwise imaging–gene\nexpression correspondence",
        ),
    ]
    bottomRowHeaders = [
        (
            getAxisOuterLeftFigureCoordinate(spatialNullAxis),
            "d",
            "Spatial-autocorrelation null\ndistribution",
        ),
        (
            getAxisOuterLeftFigureCoordinate(matchedNullAxis),
            "e",
            "Expression-property-matched\ngene-set null distribution",
        ),
        (
            getAxisOuterLeftFigureCoordinate(conjunctionDensityAxis),
            "f",
            "Intersection–union conjunction\nanalysis",
        ),
    ]

    for xPosition, panelLabel, panelTitle in topRowHeaders:
        addFigureLevelPanelHeader(
            figure,
            xPosition,
            topRowY,
            panelLabel,
            panelTitle,
        )

    bottomHeaderArtists = [
        addFigureLevelPanelHeader(
            figure,
            xPosition,
            bottomRowY,
            panelLabel,
            panelTitle,
        )
        for xPosition, panelLabel, panelTitle in bottomRowHeaders
    ]
    bottomHeaderReferenceAxes = [
        spatialNullAxis,
        matchedNullAxis,
        conjunctionDensityAxis,
    ]
    for headerArtists, referenceAxis in zip(
        bottomHeaderArtists,
        bottomHeaderReferenceAxes,
        strict=True,
    ):
        moveFigureLevelHeaderToGapAboveAxis(
            figure,
            headerArtists,
            referenceAxis,
            gapPixels=BOTTOM_HEADER_GAP_PX,
        )

    addPanelBorders(
        [
            scatterAxis,
            spatialNullAxis,
            matchedNullAxis,
            conjunctionDensityAxis,
        ]
    )
    validateMainPanelTopAlignment(
        [
            phenotypeBrainAxis,
            geneBrainAxis,
            scatterAxis,
        ]
    )
    validateAxesInsideFigure(figure)

    return figure


def validateFigureTypography(figure: plt.Figure) -> None:
    """Require one visible font family and one point size before export."""
    figure.canvas.draw()
    visibleTextArtists = [
        textArtist
        for textArtist in figure.findobj(match=mpl.text.Text)
        if textArtist.get_text().strip()
    ]
    observedFontSizes = {
        round(float(textArtist.get_fontsize()), 6)
        for textArtist in visibleTextArtists
    }
    expectedFontSizes = {FIGURE_FONT_SIZE, LEGEND_FONT_SIZE}
    if not observedFontSizes.issubset(expectedFontSizes):
        raise ValueError(
            f"Figure contains inconsistent font sizes: {sorted(observedFontSizes)}"
        )
    resolvedFontNames = {
        textArtist.get_fontproperties().get_name()
        for textArtist in visibleTextArtists
    }
    if resolvedFontNames != {FONT_FAMILY}:
        raise ValueError(
            f"Figure contains non-{FONT_FAMILY} fonts: {sorted(resolvedFontNames)}"
        )


def sanitizeSvgForIllustrator(sourcePath: Path, outputPath: Path) -> None:
    """Write a copy of the Matplotlib SVG without clipping structures that can
    leave hidden off-canvas geometry in Illustrator.

    Matplotlib commonly stores the complete geometry and relies on SVG
    clipPath elements to hide everything outside each axes rectangle. Illustrator
    may still include that hidden geometry when calculating copy/paste bounds.
    This sanitizer removes clip-path references and their definitions while
    preserving text as text and embedded brain-map images as images.
    """
    svgNamespace = "http://www.w3.org/2000/svg"
    xlinkNamespace = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", svgNamespace)
    ET.register_namespace("xlink", xlinkNamespace)

    tree = ET.parse(sourcePath)
    root = tree.getroot()

    clipPathAttribute = "clip-path"
    styleClipPattern = re.compile(r"(?:^|;)\s*clip-path\s*:\s*url\([^)]*\)\s*;?")

    for element in root.iter():
        element.attrib.pop(clipPathAttribute, None)

        styleValue = element.attrib.get("style")
        if styleValue and "clip-path" in styleValue:
            cleanedStyle = styleClipPattern.sub(";", styleValue)
            cleanedStyle = re.sub(r";{2,}", ";", cleanedStyle).strip("; ")
            if cleanedStyle:
                element.set("style", cleanedStyle)
            else:
                element.attrib.pop("style", None)

    clipPathTag = f"{{{svgNamespace}}}clipPath"
    for parent in root.iter():
        for child in list(parent):
            if child.tag == clipPathTag:
                parent.remove(child)

    tree.write(
        outputPath,
        encoding="utf-8",
        xml_declaration=True,
    )


def saveFigure(figure: plt.Figure) -> list[Path]:
    """Export the fixed-canvas figure plus an Illustrator-safe SVG copy."""
    outputBase = OUTPUT_DIRECTORY / "abide1-sfari-spatial-correspondence"
    standardSvgPath = outputBase.with_suffix(".svg")
    illustratorSvgPath = OUTPUT_DIRECTORY / "abide1-sfari-spatial-correspondence-illustrator.svg"
    pdfPath = outputBase.with_suffix(".pdf")
    pngPath = outputBase.with_suffix(".png")

    # bbox_inches="tight" is disabled because tight cropping changes the final canvas size.
    figure.savefig(
        standardSvgPath,
        format="svg",
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )

    # Illustrator can count geometry hidden behind SVG clip paths when copying
    # imported artwork.  Produce a second SVG with those clipping structures
    # removed.  Text remains editable because svg.fonttype is still "none".
    sanitizeSvgForIllustrator(standardSvgPath, illustratorSvgPath)

    figure.savefig(
        pdfPath,
        format="pdf",
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    figure.savefig(
        pngPath,
        dpi=EXPORT_DPI,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )

    with Image.open(pngPath) as exportedPng:
        if exportedPng.size != (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX):
            raise ValueError(
                "PNG canvas size is inconsistent: "
                f"expected {(CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX)}, "
                f"found {exportedPng.size}."
            )

    return [standardSvgPath, illustratorSvgPath, pdfPath, pngPath]


def main() -> None:
    """Generate all BrainSpace renders, source data and final figure exports."""
    configurePublicationStyle()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    BRAINSPACE_RENDER_DIRECTORY.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    parcelFrame, primaryResult, spatialNullValues, matchedNullValues = (
        loadPrimaryFigureData()
    )
    phenotypeParcelMap = buildFullParcelMap(parcelFrame, "imagingPhenotypeZ")
    geneExpressionParcelMap = buildFullParcelMap(
        parcelFrame,
        "riskGeneExpressionSystemResidualZ",
    )

    surfaceLeft, surfaceRight = load_conte69()
    schaeferLabels = load_parcellation("schaefer", scale=400, join=True)
    phenotypeBrainPath = (
        BRAINSPACE_RENDER_DIRECTORY / "abide1-system-adjusted-phenotype.png"
    )
    geneBrainPath = BRAINSPACE_RENDER_DIRECTORY / "sfari-risk-gene-expression.png"
    renderBrainSpaceMap(
        phenotypeParcelMap,
        phenotypeBrainPath,
        surfaceLeft,
        surfaceRight,
        schaeferLabels,
    )
    renderBrainSpaceMap(
        geneExpressionParcelMap,
        geneBrainPath,
        surfaceLeft,
        surfaceRight,
        schaeferLabels,
    )

    writeSourceData(
        parcelFrame,
        primaryResult,
        spatialNullValues,
        matchedNullValues,
    )
    figure = buildFigure(
        parcelFrame,
        primaryResult,
        spatialNullValues,
        matchedNullValues,
        phenotypeBrainPath,
        geneBrainPath,
    )
    validateFigureTypography(figure)
    outputPaths = saveFigure(figure)
    plt.close(figure)

    print("Validated ABIDE1 primary result:")
    print(f"  parcels: {len(parcelFrame)}")
    print(f"  Spearman rho: {float(primaryResult['spearmanR']):.6f}")
    print(f"  spatial P: {float(primaryResult['pSpatial']):.6f}")
    print(f"  matched-gene-set P: {float(primaryResult['pGeneSet']):.6f}")
    print(f"  conjunction P: {float(primaryResult['pConjunction']):.6f}")
    print(
        f"  typography: {FONT_FAMILY}; {FONT_SIZE_PX:g} px "
        f"({FIGURE_FONT_SIZE:g} pt at {EXPORT_DPI} dpi) for all text"
    )
    print(
        f"  canvas: {CANVAS_WIDTH_PX} x {CANVAS_HEIGHT_PX} px "
        f"at {EXPORT_DPI} dpi"
    )
    print("Saved figure files:")
    for outputPath in outputPaths:
        print(f"  {outputPath}")
    print("Use the *-illustrator.svg file for Illustrator copy/paste.")


if __name__ == "__main__":
    main()
