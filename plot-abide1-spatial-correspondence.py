from __future__ import annotations

import json
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
NULL_X_LIMIT = 0.32
FIGURE_SIZE_INCHES = (13.5, 10.0)
FIGURE_FONT_SIZE = 12.0
LEGEND_FONT_SIZE = 8.0

COLOR_BLUE = "#3775BA"
COLOR_ORANGE = "#E58A2B"
COLOR_PURPLE = "#8D67B5"
COLOR_RED = "#C43C39"
COLOR_NEUTRAL_DARK = "#4D4D4D"
COLOR_NEUTRAL_MID = "#8C8C8C"
COLOR_NEUTRAL_LIGHT = "#D8D8D8"
COLOR_PANEL_BORDER = "#B7B7B7"
COLOR_EXCLUDED_PARCEL = (0.84, 0.84, 0.84, 1.0)


def configurePublicationStyle() -> None:
    """Apply compact publication defaults while keeping SVG text editable."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "svg.fonttype": "none",
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
    """Load, filter, merge and validate the ABIDE2 primary-result inputs."""
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


def addPanelLabel(axis: plt.Axes, label: str, xPosition: float = -0.08) -> None:
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


def addBrainMapPanel(
    axis: plt.Axes,
    brainImagePath: Path,
    title: str,
) -> None:
    """Place a BrainSpace render, view labels and a shared-style color bar."""
    brainImage = Image.open(brainImagePath)
    brainImageAxis = axis.inset_axes([0, 0, 1, 1], zorder=0)
    brainImageAxis.imshow(brainImage)
    brainImageAxis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_axis_off()
    axis.set_title(
        title,
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        y=1.075,
        pad=0,
    )
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
    colorBarAxis = axis.inset_axes([0.20, -0.035, 0.60, 0.034])
    colorNormalization = mpl.colors.Normalize(
        vmin=-BRAIN_MAP_LIMIT,
        vmax=BRAIN_MAP_LIMIT,
    )
    colorBar = mpl.colorbar.ColorbarBase(
        colorBarAxis,
        cmap=mpl.colormaps["RdBu_r"],
        norm=colorNormalization,
        orientation="horizontal",
        extend="both",
    )
    colorBar.set_ticks([-BRAIN_MAP_LIMIT, 0, BRAIN_MAP_LIMIT])
    colorBar.set_ticklabels([f"−{BRAIN_MAP_LIMIT:g}", "0", f"{BRAIN_MAP_LIMIT:g}"])
    colorBar.set_label("Parcel z score", fontsize=FIGURE_FONT_SIZE, labelpad=1)
    colorBar.ax.tick_params(labelsize=FIGURE_FONT_SIZE, length=2, pad=1)
    colorBar.outline.set_linewidth(0.5)


def validateMainPanelHeights(panelAxes: list[plt.Axes]) -> None:
    """Require every main subplot to retain the same rendered axis height."""
    if not panelAxes:
        raise ValueError("At least one panel axis is required for height validation.")
    panelAxes[0].figure.canvas.draw()
    panelHeights = np.array(
        [panelAxis.get_position().height for panelAxis in panelAxes],
        dtype=float,
    )
    if float(np.ptp(panelHeights)) > 1e-6:
        raise ValueError(
            "Main subplot heights are inconsistent: "
            f"{[round(float(panelHeight), 6) for panelHeight in panelHeights]}"
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
        rasterized=True,
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
    scatterAxis.set_xlabel("System-adjusted risk-gene expression (z score)")
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
    scatterAxis.set_title("Parcelwise spatial correspondence", fontweight="bold", pad=5)

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
    axis.set_title(panelTitle, fontweight="bold", pad=6)
    axis.text(
        0.03,
        0.94,
        f"Observed ρ = {observedSpearman:.4f}\n{pLabel} = {empiricalP:.4f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FIGURE_FONT_SIZE,
        color=COLOR_RED,
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
    panelAxis.set_title(
        "Intersection–union conjunction",
        fontweight="bold",
        pad=6,
    )
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
        label="Spatial autocorrelation null",
    )
    densityAxis.plot(
        xGrid,
        matchedDensity,
        color=COLOR_ORANGE,
        linewidth=1.3,
        label="Matched gene-set null",
    )
    densityAxis.axvline(observedSpearman, color=COLOR_RED, linewidth=1.2)
    densityAxis.axvline(0, color=COLOR_NEUTRAL_MID, linewidth=0.7, linestyle=":")
    densityAxis.set_xlim(-NULL_X_LIMIT, NULL_X_LIMIT)
    densityAxis.set_ylim(bottom=0)
    densityAxis.set_xlabel("Spearman ρ")
    densityAxis.set_ylabel("Density")
    densityAxis.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    densityAxis.text(
        0.03,
        0.58,
        f"Observed\nρ = {observedSpearman:.4f}",
        transform=densityAxis.transAxes,
        color=COLOR_RED,
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
    pValueAxis.bar(
        barPositions,
        pValues,
        width=0.62,
        color=pValueColors,
        edgecolor="white",
        linewidth=0.6,
    )
    pValueAxis.set_yscale("log")
    pValueAxis.set_ylim(1e-4, max(1.0, float(pValues.max()) * 1.5))
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
    for barPosition, pValue, pColor in zip(barPositions, pValues, pValueColors):
        pValueAxis.text(
            barPosition,
            pValue * 1.18,
            f"{pValue:.4f}",
            color=pColor,
            fontsize=FIGURE_FONT_SIZE,
            fontweight="bold",
            ha="center",
            va="bottom",
        )


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
        SOURCE_DATA_DIRECTORY / "abide2-primary-parcel-data.csv",
        index=False,
    )

    excludedRoiIds = sorted(
        set(range(1, EXPECTED_FULL_ROI_COUNT + 1))
        .difference(sourceParcelFrame["roi_id_1based"].astype(int))
    )
    figureStatistics = {
        "dataset": "ABIDE2",
        "atlas": "Schaefer-400 on Conte69 surfaces",
        "fontFamily": "Arial",
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
            "group-level ABIDE2 phenotype; not an individual-level or causal result."
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
    """Assemble the full publication-style ABIDE2 result figure."""
    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, facecolor="white")
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
    conjunctionPanelAxis.set_axis_off()
    conjunctionDensityAxis = conjunctionPanelAxis.inset_axes([0.0, 0.0, 0.55, 1.0])
    conjunctionPValueAxis = conjunctionPanelAxis.inset_axes([0.55, 0.0, 0.45, 1.0])

    addBrainMapPanel(
        phenotypeBrainAxis,
        phenotypeBrainPath,
        "ABIDE2 system-adjusted imaging phenotype\nAttenuated coupling in ASD (HC − ASD)",
    )
    addBrainMapPanel(
        geneBrainAxis,
        geneBrainPath,
        "High-confidence nonsyndromic ASD-risk genes\nSystem-adjusted cortical expression (118 genes)",
    )
    addScatterPanel(
        scatterAxis,
        parcelFrame,
        primaryResult,
    )

    observedSpearman = float(primaryResult["spearmanR"])
    addNullDistributionPanel(
        spatialNullAxis,
        spatialNullValues,
        observedSpearman,
        float(primaryResult["pSpatial"]),
        "Null model 1: spatial autocorrelation",
        "P$_{spatial}$",
        "BrainSMASH surrogates",
    )
    addNullDistributionPanel(
        matchedNullAxis,
        matchedNullValues,
        observedSpearman,
        float(primaryResult["pGeneSet"]),
        "Null model 2: matched gene sets",
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

    addPanelLabel(phenotypeBrainAxis, "a", xPosition=-0.07)
    addPanelLabel(geneBrainAxis, "b", xPosition=-0.055)
    addPanelLabel(scatterAxis, "c", xPosition=-0.12)
    addPanelLabel(spatialNullAxis, "d")
    addPanelLabel(matchedNullAxis, "e")
    addPanelLabel(conjunctionPanelAxis, "f", xPosition=-0.08)
    addPanelBorders(
        [
            scatterAxis,
            spatialNullAxis,
            matchedNullAxis,
            conjunctionDensityAxis,
        ]
    )
    validateMainPanelHeights(
        [
            phenotypeBrainAxis,
            geneBrainAxis,
            scatterAxis,
        ]
    )

    if False:  # Overall title intentionally omitted from the exported figure.
        figure.suptitle(
        "Spatial correspondence between attenuated gradient–propagation coupling "
        "and ASD risk-gene expression in ABIDE2",
        fontsize=FIGURE_FONT_SIZE,
        fontweight="bold",
        y=0.968,
    )
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
    if resolvedFontNames != {"Arial"}:
        raise ValueError(
            f"Figure contains non-Arial fonts: {sorted(resolvedFontNames)}"
        )


def saveFigure(figure: plt.Figure) -> list[Path]:
    """Export editable-vector and high-resolution raster deliverables."""
    outputBase = OUTPUT_DIRECTORY / "abide2-sfari-spatial-correspondence"
    outputPaths = [
        outputBase.with_suffix(".svg"),
        outputBase.with_suffix(".pdf"),
        outputBase.with_suffix(".png"),
    ]
    figure.savefig(outputPaths[0], bbox_inches="tight", facecolor="white")
    figure.savefig(outputPaths[1], bbox_inches="tight", facecolor="white")
    figure.savefig(
        outputPaths[2],
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    return outputPaths


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
        BRAINSPACE_RENDER_DIRECTORY / "abide2-system-adjusted-phenotype.png"
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

    print("Validated ABIDE2 primary result:")
    print(f"  parcels: {len(parcelFrame)}")
    print(f"  Spearman rho: {float(primaryResult['spearmanR']):.6f}")
    print(f"  spatial P: {float(primaryResult['pSpatial']):.6f}")
    print(f"  matched-gene-set P: {float(primaryResult['pGeneSet']):.6f}")
    print(f"  conjunction P: {float(primaryResult['pConjunction']):.6f}")
    print(
        f"  typography: Arial; {FIGURE_FONT_SIZE:g} pt main text; "
        f"{LEGEND_FONT_SIZE:g} pt legend text"
    )
    print("Saved figure files:")
    for outputPath in outputPaths:
        print(f"  {outputPath}")


if __name__ == "__main__":
    main()
