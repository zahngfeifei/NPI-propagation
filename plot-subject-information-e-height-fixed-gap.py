from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"


SVG_FONT_FAMILY = "Arial"
SVG_FONT_SIZE_PX = 25
MATPLOTLIB_FONT_SIZE = SVG_FONT_SIZE_PX
SVG_CANVAS_WIDTH_PX = 1800
BASE_SVG_CANVAS_HEIGHT_PX = round(SVG_CANVAS_WIDTH_PX * 11.1 / 7.2)
SVG_COORDINATE_DPI = 72
CANVAS_CONTENT_PADDING_PX = 10
LEFT_COLUMN_CONTENT_SHIFT_PX = 148
PANEL_HEADER_Y = 1.06

# Panel e scaling. Increase this single value to make panel e taller while
# keeping the vertical gap between neighboring site bars unchanged.
SITE_PANEL_SCALE = 1.5
BASE_SITE_BAR_HEIGHT = 0.85
BASE_GRID_HEIGHT_RATIOS = (1.0, 1.05, 1.95, 1.25)
GRID_HEIGHT_RATIOS = (
    BASE_GRID_HEIGHT_RATIOS[0],
    BASE_GRID_HEIGHT_RATIOS[1],
    BASE_GRID_HEIGHT_RATIOS[2] * SITE_PANEL_SCALE,
    BASE_GRID_HEIGHT_RATIOS[3],
)
# With a fixed y-range, this preserves the original inter-bar gap in pixels
# when panel e is enlarged by SITE_PANEL_SCALE.
SITE_BAR_HEIGHT = 1.0 - (1.0 - BASE_SITE_BAR_HEIGHT) / SITE_PANEL_SCALE
# Increase only the canvas height needed by the enlarged panel e; width stays 1800 px.
SVG_CANVAS_HEIGHT_PX = round(
    BASE_SVG_CANVAS_HEIGHT_PX
    * sum(GRID_HEIGHT_RATIOS)
    / sum(BASE_GRID_HEIGHT_RATIOS)
)
FIGURE_SIZE_INCHES = (
    SVG_CANVAS_WIDTH_PX / SVG_COORDINATE_DPI,
    SVG_CANVAS_HEIGHT_PX / SVG_COORDINATE_DPI,
)
RASTER_OUTPUT_DPI = SVG_COORDINATE_DPI
OUTPUT_DIR = Path(__file__).resolve().parent
FULL_DATA_FILE = OUTPUT_DIR / "ABIDEⅠ_Ⅱ_合并_完整量表.csv"
INCLUDED_DATA_FILE = OUTPUT_DIR / "ABIDE_Ⅰ_Ⅱ_仅完成因果梯度被试_完整量表.csv"
EXCLUDED_DATA_FILE = OUTPUT_DIR / "ABIDEⅠ_Ⅱ_合并_缺失因果梯度结果被试.csv"
OUTPUT_BASE = OUTPUT_DIR / "subject-information-figure"
SOURCE_DATA_FILE = OUTPUT_DIR / "subject-information-figure-source-data.csv"

PALETTE = {
    "asd": "#0F4D92",
    "hc": "#9FB7D9",
    "male": "#4D4D4D",
    "female": "#B8A3C9",
    "included": "#2E6F9E",
    "excluded": "#C9C9C9",
    "abideI": "#4F8A8B",
    "abideII": "#C77C4A",
    "axis": "#272727",
    "grid": "#D9D9D9",
    "text": "#272727",
}


def applyPublicationStyle():
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": MATPLOTLIB_FONT_SIZE,
            "axes.titlesize": MATPLOTLIB_FONT_SIZE,
            "axes.titleweight": "bold",
            "axes.titley": 1.0,
            "axes.labelsize": MATPLOTLIB_FONT_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": MATPLOTLIB_FONT_SIZE,
            "ytick.labelsize": MATPLOTLIB_FONT_SIZE,
            "legend.fontsize": MATPLOTLIB_FONT_SIZE,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def readCsvTable(filePath):
    return pd.read_csv(filePath, encoding="utf-8-sig", low_memory=False)


def matchingColumns(dataFrame, requestedColumns):
    normalizedLookup = {}
    for columnName in dataFrame.columns:
        normalizedLookup.setdefault(str(columnName).strip(), []).append(columnName)

    foundColumns = []
    for requestedColumn in requestedColumns:
        foundColumns.extend(normalizedLookup.get(requestedColumn, []))
    return foundColumns


def coalesceNumericColumns(dataFrame, requestedColumns):
    outputSeries = pd.Series(np.nan, index=dataFrame.index, dtype="float64")
    for columnName in matchingColumns(dataFrame, requestedColumns):
        numericSeries = pd.to_numeric(dataFrame[columnName], errors="coerce")
        outputSeries = outputSeries.fillna(numericSeries)
    return outputSeries


def normalizeDatasetName(value):
    datasetText = str(value).strip().upper()
    if "Ⅱ" in datasetText or "II" in datasetText:
        return "ABIDE II"
    if "Ⅰ" in datasetText or datasetText.endswith("I") or datasetText == "ABIDEI":
        return "ABIDE I"
    return "Unknown"


def normalizeDiagnosis(value):
    if pd.isna(value):
        return "Unknown"
    try:
        diagnosisCode = int(float(value))
    except ValueError:
        return "Unknown"
    if diagnosisCode == 1:
        return "ASD"
    if diagnosisCode == 2:
        return "HC"
    return "Unknown"


def normalizeSex(value):
    if pd.isna(value):
        return "Unknown"
    try:
        sexCode = int(float(value))
    except ValueError:
        return "Unknown"
    if sexCode == 1:
        return "Male"
    if sexCode == 2:
        return "Female"
    return "Unknown"


def prepareSubjectTable(dataFrame):
    preparedTable = pd.DataFrame(
        {
            "dataset": dataFrame["DATASET"].map(normalizeDatasetName),
            "siteId": dataFrame["SITE_ID"].astype(str).str.strip(),
            "diagnosis": dataFrame["DX_GROUP"].map(normalizeDiagnosis),
            "sex": dataFrame["SEX"].map(normalizeSex),
            "age": coalesceNumericColumns(dataFrame, ["AGE_AT_SCAN"]),
        }
    )
    return preparedTable


def hasAnyMeasureValue(dataFrame, requestedColumns):
    foundColumns = matchingColumns(dataFrame, requestedColumns)
    if not foundColumns:
        return pd.Series(False, index=dataFrame.index)
    measureValues = dataFrame[foundColumns].replace(r"^\s*$", np.nan, regex=True)
    return measureValues.notna().any(axis=1)


def buildMeasureAvailability(includedRawData, includedSubjectData):
    measureColumns = {
        "FIQ": ["FIQ"],
        "VIQ": ["VIQ"],
        "PIQ": ["PIQ"],
        "ADOS": ["ADOS_TOTAL", "ADOS_G_TOTAL", "ADOS_GOTHAM_TOTAL", "ADOS_2_TOTAL"],
        "ADI-R": [
            "ADI_R_SOCIAL_TOTAL_A",
            "ADI_R_VERBAL_TOTAL_BV",
            "ADI_R_NONVERBAL_TOTAL_BV",
            "ADI_RRB_TOTAL_C",
            "ADI_R_RRB_TOTAL_C",
        ],
        "SRS": ["SRS_RAW_TOTAL", "SRS_TOTAL_RAW", "SRS_TOTAL_T"],
        "SCQ": ["SCQ_TOTAL"],
        "AQ": ["AQ_TOTAL"],
        "Vineland": [
            "VINELAND_COMMUNICATION_STANDARD",
            "VINELAND_DAILYLVNG_STANDARD",
            "VINELAND_DAILYLIVING_STANDARD",
            "VINELAND_SOCIAL_STANDARD",
            "VINELAND_ABC_STANDARD",
            "VINELAND_ABC_Standard",
        ],
        "RBS-R": ["RBSR_6SUBSCALE_TOTAL", "RBSR_5SUBSCALE_TOTAL"],
        "CBCL": ["CBCL_6-18_TOTAL_PROBLEM_T", "CBCL_1.5-5_TOTAL_T"],
        "BRIEF": ["BRIEF_GEC_T"],
    }
    columnOrder = ["ABIDE I ASD", "ABIDE I HC", "ABIDE II ASD", "ABIDE II HC"]
    availabilityRows = []
    for measureName, requestedColumns in measureColumns.items():
        hasMeasure = hasAnyMeasureValue(includedRawData, requestedColumns)
        rowValues = []
        for datasetName in ["ABIDE I", "ABIDE II"]:
            for diagnosisName in ["ASD", "HC"]:
                groupMask = (includedSubjectData["dataset"] == datasetName) & (
                    includedSubjectData["diagnosis"] == diagnosisName
                )
                denominator = int(groupMask.sum())
                numerator = int(hasMeasure[groupMask].sum())
                percentage = (100 * numerator / denominator) if denominator else np.nan
                rowValues.append(percentage)
                availabilityRows.append(
                    {
                        "panel": "f",
                        "measure": measureName,
                        "group": f"{datasetName} {diagnosisName}",
                        "count": numerator,
                        "denominator": denominator,
                        "value": percentage,
                    }
                )
    availabilityMatrix = pd.DataFrame(
        np.array([row["value"] for row in availabilityRows]).reshape(len(measureColumns), 4),
        index=list(measureColumns.keys()),
        columns=columnOrder,
    )
    return availabilityMatrix, pd.DataFrame(availabilityRows)


def addPanelLabel(ax, label, alignToCanvasLeft=False):
    xPosition = -0.08
    if alignToCanvasLeft:
        axisPosition = ax.get_position()
        panelLabelAnchorPx = CANVAS_CONTENT_PADDING_PX - 1
        xPosition = (
            panelLabelAnchorPx / SVG_CANVAS_WIDTH_PX - axisPosition.x0
        ) / axisPosition.width
    return ax.text(
        xPosition,
        PANEL_HEADER_Y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=MATPLOTLIB_FONT_SIZE,
        fontweight="bold",
        color=PALETTE["text"],
    )


def alignPanelHeaderTops(figure, axes, panelLabelArtists, titleArtists):
    for ax, panelLabelArtist, titleArtist in zip(axes, panelLabelArtists, titleArtists):
        panelLabelArtist.set_fontweight("bold")
        titleArtist.set_fontweight("bold")
        legend = ax.get_legend()
        if legend is not None:
            for legendText in legend.get_texts():
                legendText.set_fontweight("bold")
            for legendHandle in legend.legend_handles:
                if hasattr(legendHandle, "set_linewidth"):
                    legendHandle.set_linewidth(1.2)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for ax, panelLabelArtist, titleArtist in zip(axes, panelLabelArtists, titleArtists):
        targetTop = panelLabelArtist.get_window_extent(renderer).y1

        titleBounds = titleArtist.get_window_extent(renderer)
        titleOffset = targetTop - titleBounds.y1
        titleDisplayPosition = titleArtist.get_transform().transform(titleArtist.get_position())
        alignedTitlePosition = titleArtist.get_transform().inverted().transform(
            (titleDisplayPosition[0], titleDisplayPosition[1] + titleOffset)
        )
        titleArtist.set_position(alignedTitlePosition)

        legend = ax.get_legend()
        if legend is not None:
            legendBounds = legend.get_window_extent(renderer)
            legendOffset = targetTop - legendBounds.y1
            anchorBounds = legend.get_bbox_to_anchor()
            anchorPosition = ax.transAxes.inverted().transform((anchorBounds.x1, anchorBounds.y1))
            legend.set_bbox_to_anchor(
                (
                    anchorPosition[0],
                    anchorPosition[1] + legendOffset / ax.bbox.height,
                ),
                transform=ax.transAxes,
            )
    figure.canvas.draw()


def expandAxisToCanvasLeft(ax, shiftPx):
    axisPosition = ax.get_position()
    normalizedShift = shiftPx / SVG_CANVAS_WIDTH_PX
    ax.set_position(
        [
            axisPosition.x0 - normalizedShift,
            axisPosition.y0,
            axisPosition.width + normalizedShift,
            axisPosition.height,
        ]
    )


def alignAnnotationWithYAxisLabels(figure, annotationArtist, ax):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    visibleLabelBounds = [
        label.get_window_extent(renderer)
        for label in ax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    if not visibleLabelBounds:
        raise RuntimeError("No visible y-axis labels were found for annotation alignment.")

    targetLeft = min(labelBounds.x0 for labelBounds in visibleLabelBounds) - 1
    annotationBounds = annotationArtist.get_window_extent(renderer)
    annotationDisplayPosition = annotationArtist.get_transform().transform(
        annotationArtist.get_position()
    )
    alignedAnnotationPosition = annotationArtist.get_transform().inverted().transform(
        (
            annotationDisplayPosition[0] + targetLeft - annotationBounds.x0,
            annotationDisplayPosition[1],
        )
    )
    annotationArtist.set_position(alignedAnnotationPosition)
    figure.canvas.draw()


def cleanAxis(ax):
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["axis"], length=2.5)
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])


def annotateBars(ax, bars, yOffset=8, fontsize=MATPLOTLIB_FONT_SIZE):
    for bar in bars:
        barHeight = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            barHeight + yOffset,
            f"{int(barHeight)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PALETTE["text"],
        )


def plotInclusionFlow(ax, fullSubjectData, includedSubjectData, excludedSubjectData):
    ax.axis("off")
    fullCount = len(fullSubjectData)
    includedCount = len(includedSubjectData)
    excludedCount = len(excludedSubjectData)
    includedFraction = includedCount / fullCount
    excludedFraction = excludedCount / fullCount

    boxSpecs = [
        (0.04, 0.63, 0.34, 0.22, "Full phenotypic table", fullCount, PALETTE["axis"]),
        (0.58, 0.72, 0.34, 0.18, "Retained for analysis", includedCount, PALETTE["included"]),
        (0.58, 0.44, 0.34, 0.18, "No gradient result", excludedCount, PALETTE["excluded"]),
    ]
    for xValue, yValue, width, height, title, count, color in boxSpecs:
        textColor = "white" if color in [PALETTE["axis"], PALETTE["included"]] else PALETTE["text"]
        patch = FancyBboxPatch(
            (xValue, yValue),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.7,
            edgecolor=PALETTE["axis"],
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            xValue + width / 2,
            yValue + height * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=MATPLOTLIB_FONT_SIZE,
            color=textColor,
            transform=ax.transAxes,
        )
        ax.text(
            xValue + width / 2,
            yValue + height * 0.34,
            f"n = {count:,}",
            ha="center",
            va="center",
            fontsize=MATPLOTLIB_FONT_SIZE,
            fontweight="bold",
            color=textColor,
            transform=ax.transAxes,
        )

    ax.annotate(
        "",
        xy=(0.56, 0.80),
        xytext=(0.40, 0.74),
        arrowprops=dict(arrowstyle="->", color=PALETTE["axis"], lw=0.8),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
    )
    ax.annotate(
        "",
        xy=(0.56, 0.53),
        xytext=(0.40, 0.71),
        arrowprops=dict(arrowstyle="->", color=PALETTE["axis"], lw=0.8),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
    )
    ax.text(
        0.58,
        0.66,
        f"{includedFraction:.1%} retained",
        ha="left",
        va="center",
        fontsize=MATPLOTLIB_FONT_SIZE,
        color=PALETTE["included"],
        transform=ax.transAxes,
    )
    ax.text(
        0.58,
        0.38,
        f"{excludedFraction:.1%} excluded",
        ha="left",
        va="center",
        fontsize=MATPLOTLIB_FONT_SIZE,
        color=PALETTE["axis"],
        transform=ax.transAxes,
    )

    datasetCounts = includedSubjectData["dataset"].value_counts().reindex(["ABIDE I", "ABIDE II"])
    leftValue = 0.04
    totalWidth = 0.88
    for datasetName, color in [("ABIDE I", PALETTE["abideI"]), ("ABIDE II", PALETTE["abideII"])]:
        count = int(datasetCounts.loc[datasetName])
        width = totalWidth * count / includedCount
        ax.add_patch(
            Rectangle(
                (leftValue, 0.13),
                width,
                0.12,
                transform=ax.transAxes,
                color=color,
                ec=PALETTE["axis"],
                lw=0.4,
            )
        )
        ax.text(
            leftValue + width / 2,
            0.19,
            f"{datasetName}\n{count}",
            ha="center",
            va="center",
            fontsize=MATPLOTLIB_FONT_SIZE,
            color="white",
            transform=ax.transAxes,
        )
        leftValue += width
    ax.text(
        0.04,
        0.30,
        "Retained subjects by dataset",
        fontsize=MATPLOTLIB_FONT_SIZE,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.set_title("Cohort inclusion", loc="left", pad=3)


def plotDatasetDiagnosis(ax, includedSubjectData):
    countTable = (
        includedSubjectData.groupby(["dataset", "diagnosis"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["ABIDE I", "ABIDE II"], columns=["ASD", "HC"])
    )
    xValues = np.arange(len(countTable.index))
    bottomValues = np.zeros(len(countTable.index))
    for diagnosisName, color in [("ASD", PALETTE["asd"]), ("HC", PALETTE["hc"])]:
        bars = ax.bar(
            xValues,
            countTable[diagnosisName].values,
            bottom=bottomValues,
            color=color,
            edgecolor=PALETTE["axis"],
            linewidth=0.5,
            width=0.62,
            label=diagnosisName,
        )
        for bar, value, bottomValue in zip(bars, countTable[diagnosisName].values, bottomValues):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottomValue + value / 2,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=MATPLOTLIB_FONT_SIZE,
                color="white" if diagnosisName == "ASD" else PALETTE["text"],
            )
        bottomValues += countTable[diagnosisName].values
    ax.set_xticks(xValues)
    ax.set_xticklabels(countTable.index)
    ax.set_ylabel("Subjects, n")
    ax.set_title("Diagnosis composition", loc="left", pad=3)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.09),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.35)
    cleanAxis(ax)


def plotSexDiagnosis(ax, includedSubjectData):
    countTable = (
        includedSubjectData.groupby(["diagnosis", "sex"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["ASD", "HC"], columns=["Male", "Female"])
    )
    xValues = np.arange(len(countTable.index))
    bottomValues = np.zeros(len(countTable.index))
    for sexName, color in [("Male", PALETTE["male"]), ("Female", PALETTE["female"])]:
        bars = ax.bar(
            xValues,
            countTable[sexName].values,
            bottom=bottomValues,
            color=color,
            edgecolor=PALETTE["axis"],
            linewidth=0.5,
            width=0.62,
            label=sexName,
        )
        for bar, value, bottomValue in zip(bars, countTable[sexName].values, bottomValues):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottomValue + value / 2,
                str(int(value)),
                ha="center",
                va="center",
                fontsize=MATPLOTLIB_FONT_SIZE,
                color="white" if sexName == "Male" else PALETTE["text"],
            )
        bottomValues += countTable[sexName].values
    ax.set_xticks(xValues)
    ax.set_xticklabels(countTable.index)
    ax.set_ylabel("Subjects, n")
    ax.set_title("Sex distribution", loc="left", pad=3)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.08),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.35)
    cleanAxis(ax)


def plotAgeDistribution(ax, includedSubjectData):
    groupSpecs = [
        ("ABIDE I\nASD", (includedSubjectData["dataset"] == "ABIDE I") & (includedSubjectData["diagnosis"] == "ASD"), PALETTE["asd"]),
        ("ABIDE I\nHC", (includedSubjectData["dataset"] == "ABIDE I") & (includedSubjectData["diagnosis"] == "HC"), PALETTE["hc"]),
        ("ABIDE II\nASD", (includedSubjectData["dataset"] == "ABIDE II") & (includedSubjectData["diagnosis"] == "ASD"), PALETTE["asd"]),
        ("ABIDE II\nHC", (includedSubjectData["dataset"] == "ABIDE II") & (includedSubjectData["diagnosis"] == "HC"), PALETTE["hc"]),
    ]
    ageArrays = [includedSubjectData.loc[groupMask, "age"].dropna().values for _, groupMask, _ in groupSpecs]
    boxPlot = ax.boxplot(
        ageArrays,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": PALETTE["axis"], "linewidth": 0.8},
        whiskerprops={"color": PALETTE["axis"], "linewidth": 0.7},
        capprops={"color": PALETTE["axis"], "linewidth": 0.7},
    )
    for patch, (_, _, color) in zip(boxPlot["boxes"], groupSpecs):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(PALETTE["axis"])
        patch.set_linewidth(0.6)
    rng = np.random.default_rng(7)
    for index, ageValues in enumerate(ageArrays, start=1):
        if len(ageValues) > 0:
            sampledValues = ageValues if len(ageValues) <= 160 else rng.choice(ageValues, 160, replace=False)
            jitterValues = rng.normal(index, 0.035, len(sampledValues))
            ax.scatter(
                jitterValues,
                sampledValues,
                s=3,
                color=PALETTE["axis"],
                alpha=0.25,
                linewidth=0,
                zorder=1,
            )
    ax.set_xticks(np.arange(1, len(groupSpecs) + 1))
    ax.set_xticklabels([label for label, _, _ in groupSpecs])
    ax.set_ylabel("Age at scan (years)")
    ax.set_title("Age distribution", loc="left", pad=3)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.35)
    cleanAxis(ax)


def plotSiteComposition(ax, includedSubjectData):
    siteCounts = (
        includedSubjectData.groupby(["siteId", "diagnosis"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["ASD", "HC"], fill_value=0)
    )
    siteCounts["total"] = siteCounts.sum(axis=1)
    siteCounts = siteCounts.sort_values(["total", "ASD", "HC"], ascending=[True, True, True])
    yValues = np.arange(len(siteCounts))
    leftValues = np.zeros(len(siteCounts))
    for diagnosisName, color in [("ASD", PALETTE["asd"]), ("HC", PALETTE["hc"])]:
        ax.barh(
            yValues,
            siteCounts[diagnosisName].values,
            left=leftValues,
            color=color,
            edgecolor=PALETTE["axis"],
            linewidth=0.35,
            height=SITE_BAR_HEIGHT,
            label=diagnosisName,
        )
        leftValues += siteCounts[diagnosisName].values
    for yValue, totalCount in zip(yValues, siteCounts["total"].values):
        ax.text(
            totalCount + 3,
            yValue,
            f"{int(totalCount)}",
            ha="left",
            va="center",
            fontsize=MATPLOTLIB_FONT_SIZE,
            color=PALETTE["text"],
        )
    ax.set_yticks(yValues)
    ax.set_yticklabels(siteCounts.index, fontsize=MATPLOTLIB_FONT_SIZE)
    ax.set_xlabel("Subjects, n")
    ax.set_title(f"Site composition across all retained sites (n = {len(siteCounts)})", loc="left", pad=3)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.07),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.set_xlim(0, siteCounts["total"].max() * 1.09)
    # Keep the site-center spacing fixed at one data unit so that the
    # SITE_BAR_HEIGHT formula above preserves the original gap in pixels.
    ax.set_ylim(-0.5, len(siteCounts) - 0.5)
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.35)
    cleanAxis(ax)


def plotMeasureAvailability(ax, availabilityMatrix):
    matrixValues = availabilityMatrix.values.astype(float)
    image = ax.imshow(matrixValues, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(availabilityMatrix.shape[1]))
    ax.set_xticklabels(availabilityMatrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(availabilityMatrix.shape[0]))
    ax.set_yticklabels(availabilityMatrix.index)
    for rowIndex in range(matrixValues.shape[0]):
        for columnIndex in range(matrixValues.shape[1]):
            value = matrixValues[rowIndex, columnIndex]
            textColor = "white" if value >= 55 else PALETTE["text"]
            ax.text(
                columnIndex,
                rowIndex,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=MATPLOTLIB_FONT_SIZE,
                color=textColor,
            )
    ax.set_title("Scale availability", loc="left", pad=3)
    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("% available")
    colorbar.ax.tick_params(labelsize=MATPLOTLIB_FONT_SIZE, width=0.5, length=2)


def collectSourceData(fullSubjectData, includedSubjectData, excludedSubjectData, availabilityData):
    sourceRows = [
        {"panel": "a", "measure": "full phenotypic table", "group": "all", "count": len(fullSubjectData), "denominator": len(fullSubjectData), "value": len(fullSubjectData)},
        {"panel": "a", "measure": "retained for analysis", "group": "all", "count": len(includedSubjectData), "denominator": len(fullSubjectData), "value": len(includedSubjectData)},
        {"panel": "a", "measure": "no gradient result", "group": "all", "count": len(excludedSubjectData), "denominator": len(fullSubjectData), "value": len(excludedSubjectData)},
    ]
    for (datasetName, diagnosisName), count in includedSubjectData.groupby(["dataset", "diagnosis"]).size().items():
        sourceRows.append({"panel": "b", "measure": "diagnosis count", "group": f"{datasetName} {diagnosisName}", "count": int(count), "denominator": len(includedSubjectData), "value": int(count)})
    for (diagnosisName, sexName), count in includedSubjectData.groupby(["diagnosis", "sex"]).size().items():
        sourceRows.append({"panel": "c", "measure": "sex count", "group": f"{diagnosisName} {sexName}", "count": int(count), "denominator": len(includedSubjectData), "value": int(count)})
    for (datasetName, diagnosisName), groupData in includedSubjectData.groupby(["dataset", "diagnosis"]):
        sourceRows.append(
            {
                "panel": "d",
                "measure": "age median",
                "group": f"{datasetName} {diagnosisName}",
                "count": int(groupData["age"].notna().sum()),
                "denominator": int(len(groupData)),
                "value": float(groupData["age"].median()),
            }
        )
    for (siteId, diagnosisName), count in includedSubjectData.groupby(["siteId", "diagnosis"]).size().items():
        sourceRows.append({"panel": "e", "measure": "site count", "group": f"{siteId} {diagnosisName}", "count": int(count), "denominator": len(includedSubjectData), "value": int(count)})
    return pd.concat([pd.DataFrame(sourceRows), availabilityData], ignore_index=True)


def finalizeSvgOutput(svgFilePath):
    rawSvgText = svgFilePath.read_text(encoding="utf-8")
    updatedSvgText, fontSizeReplacementCount = re.subn(
        r"font-size:\s*[0-9.]+(?:px|pt)?",
        f"font-size: {SVG_FONT_SIZE_PX}px",
        rawSvgText,
    )
    updatedSvgText, fontFamilyReplacementCount = re.subn(
        r"(?:font-family:\s*[^;]+;\s*)+",
        f"font-family: {SVG_FONT_FAMILY}; ",
        updatedSvgText,
    )
    updatedSvgText, widthReplacementCount = re.subn(
        r'(<svg\b[^>]*\bwidth=")[^"]+(")',
        rf'\g<1>{SVG_CANVAS_WIDTH_PX}px\2',
        updatedSvgText,
        count=1,
    )
    updatedSvgText, heightReplacementCount = re.subn(
        r'(<svg\b[^>]*\bheight=")[^"]+(")',
        rf'\g<1>{SVG_CANVAS_HEIGHT_PX}px\2',
        updatedSvgText,
        count=1,
    )
    updatedSvgText, viewBoxReplacementCount = re.subn(
        r'(<svg\b[^>]*\bviewBox=")[^"]+(")',
        rf'\g<1>0 0 {SVG_CANVAS_WIDTH_PX} {SVG_CANVAS_HEIGHT_PX}\2',
        updatedSvgText,
        count=1,
    )
    if fontSizeReplacementCount == 0:
        raise RuntimeError("No SVG font-size declarations were found.")
    if fontFamilyReplacementCount == 0:
        raise RuntimeError("No SVG font-family declarations were found.")
    if widthReplacementCount != 1 or heightReplacementCount != 1 or viewBoxReplacementCount != 1:
        raise RuntimeError("The SVG canvas dimensions could not be finalized.")
    svgFilePath.write_text(updatedSvgText, encoding="utf-8")


def makeSubjectInformationFigure():
    applyPublicationStyle()
    fullRawData = readCsvTable(FULL_DATA_FILE)
    includedRawData = readCsvTable(INCLUDED_DATA_FILE)
    excludedRawData = readCsvTable(EXCLUDED_DATA_FILE)

    fullSubjectData = prepareSubjectTable(fullRawData)
    includedSubjectData = prepareSubjectTable(includedRawData)
    excludedSubjectData = prepareSubjectTable(excludedRawData)
    availabilityMatrix, availabilityData = buildMeasureAvailability(includedRawData, includedSubjectData)

    sourceData = collectSourceData(fullSubjectData, includedSubjectData, excludedSubjectData, availabilityData)
    sourceData.to_csv(SOURCE_DATA_FILE, index=False, encoding="utf-8-sig")

    figure = plt.figure(figsize=FIGURE_SIZE_INCHES, constrained_layout=False)
    gridSpec = GridSpec(
        4,
        2,
        figure=figure,
        height_ratios=GRID_HEIGHT_RATIOS,
        hspace=0.58,
        wspace=0.36,
    )
    axes = [
        figure.add_subplot(gridSpec[0, 0]),
        figure.add_subplot(gridSpec[0, 1]),
        figure.add_subplot(gridSpec[1, 0]),
        figure.add_subplot(gridSpec[1, 1]),
        figure.add_subplot(gridSpec[2, :]),
        figure.add_subplot(gridSpec[3, :]),
    ]
    figure.subplots_adjust(left=0.13, right=0.9841, top=0.981, bottom=0.085)
    expandAxisToCanvasLeft(axes[0], LEFT_COLUMN_CONTENT_SHIFT_PX)
    expandAxisToCanvasLeft(axes[2], LEFT_COLUMN_CONTENT_SHIFT_PX)

    plotInclusionFlow(axes[0], fullSubjectData, includedSubjectData, excludedSubjectData)
    plotDatasetDiagnosis(axes[1], includedSubjectData)
    plotSexDiagnosis(axes[2], includedSubjectData)
    plotAgeDistribution(axes[3], includedSubjectData)
    plotSiteComposition(axes[4], includedSubjectData)
    plotMeasureAvailability(axes[5], availabilityMatrix)
    panelLabelArtists = [
        addPanelLabel(ax, label, alignToCanvasLeft=label in {"a", "c"})
        for label, ax in zip(list("abcdef"), axes)
    ]
    titleArtists = [
        ax.set_title(ax.get_title(loc="left"), loc="left", pad=3, y=1.0)
        for ax in axes
    ]
    alignPanelHeaderTops(figure, axes, panelLabelArtists, titleArtists)

    sourceAnnotationArtist = figure.text(
        (CANVAS_CONTENT_PADDING_PX - 1) / SVG_CANVAS_WIDTH_PX,
        15 / SVG_CANVAS_HEIGHT_PX,
        "Source data: merged ABIDE phenotypic table filtered to subjects with completed causal-gradient results.\n"
        "Sex coding: 1 = male, 2 = female; diagnosis coding: 1 = ASD, 2 = HC.",
        fontsize=MATPLOTLIB_FONT_SIZE,
        color=PALETTE["text"],
        linespacing=1.25,
    )
    alignAnnotationWithYAxisLabels(figure, sourceAnnotationArtist, axes[5])
    svgOutputFile = OUTPUT_BASE.with_suffix(".svg")
    figure.savefig(svgOutputFile)
    finalizeSvgOutput(svgOutputFile)
    figure.savefig(f"{OUTPUT_BASE}.pdf")
    figure.savefig(f"{OUTPUT_BASE}.tiff", dpi=RASTER_OUTPUT_DPI)
    figure.savefig(f"{OUTPUT_BASE}.png", dpi=RASTER_OUTPUT_DPI)
    plt.close(figure)


if __name__ == "__main__":
    makeSubjectInformationFigure()
