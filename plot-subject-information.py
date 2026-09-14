from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUTPUT_DIR = Path(r"C:\Users\deng\Desktop\绘图\补充内容\S15")
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
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.8,
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


def addPanelLabel(ax, label):
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PALETTE["text"],
    )


def cleanAxis(ax):
    ax.spines["left"].set_color(PALETTE["axis"])
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["axis"], length=2.5)
    ax.title.set_color(PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])


def annotateBars(ax, bars, yOffset=8, fontsize=5.6):
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
            fontsize=6.2,
            color=textColor,
            transform=ax.transAxes,
        )
        ax.text(
            xValue + width / 2,
            yValue + height * 0.34,
            f"n = {count:,}",
            ha="center",
            va="center",
            fontsize=7,
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
        fontsize=6,
        color=PALETTE["included"],
        transform=ax.transAxes,
    )
    ax.text(
        0.58,
        0.38,
        f"{excludedFraction:.1%} excluded",
        ha="left",
        va="center",
        fontsize=6,
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
                (leftValue, 0.16),
                width,
                0.08,
                transform=ax.transAxes,
                color=color,
                ec=PALETTE["axis"],
                lw=0.4,
            )
        )
        ax.text(
            leftValue + width / 2,
            0.20,
            f"{datasetName}\n{count}",
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            transform=ax.transAxes,
        )
        leftValue += width
    ax.text(0.04, 0.28, "Retained subjects by dataset", fontsize=6, transform=ax.transAxes)
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
                fontsize=5.8,
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
                fontsize=5.8,
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
            height=0.66,
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
            fontsize=4.8,
            color=PALETTE["text"],
        )
    ax.set_yticks(yValues)
    ax.set_yticklabels(siteCounts.index, fontsize=4.8)
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
                fontsize=5.2,
                color=textColor,
            )
    ax.set_title("Scale availability", loc="left", pad=3)
    ax.set_xlabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("% available")
    colorbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)


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

    figure = plt.figure(figsize=(7.2, 11.1), constrained_layout=False)
    gridSpec = GridSpec(
        4,
        2,
        figure=figure,
        height_ratios=[1.0, 1.05, 1.95, 1.25],
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

    plotInclusionFlow(axes[0], fullSubjectData, includedSubjectData, excludedSubjectData)
    plotDatasetDiagnosis(axes[1], includedSubjectData)
    plotSexDiagnosis(axes[2], includedSubjectData)
    plotAgeDistribution(axes[3], includedSubjectData)
    plotSiteComposition(axes[4], includedSubjectData)
    plotMeasureAvailability(axes[5], availabilityMatrix)
    for label, ax in zip(list("abcdef"), axes):
        addPanelLabel(ax, label)

    figure.text(
        0.015,
        0.012,
        "Source data: merged ABIDE phenotypic table filtered to subjects with completed causal-gradient results. "
        "Sex coding: 1 = male, 2 = female; diagnosis coding: 1 = ASD, 2 = HC.",
        fontsize=5.2,
        color=PALETTE["text"],
    )
    figure.subplots_adjust(left=0.13, right=0.94, top=0.965, bottom=0.055)
    figure.savefig(f"{OUTPUT_BASE}.svg", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_BASE}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT_BASE}.tiff", dpi=600, bbox_inches="tight")
    figure.savefig(f"{OUTPUT_BASE}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    makeSubjectInformationFigure()
