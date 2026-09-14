from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIGURE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = FIGURE_ROOT / "supplementary-figure-s4-output"
OUTPUT_BASENAME = "supplementary-figure-s4"

FONT_FAMILY = "Arial"
FONT_SIZE = 9
FIGURE_WIDTH_INCHES = 7.20
FIGURE_HEIGHT_INCHES = 5.45

ABIDE_I_COLOR = "#3B6FB6"
ABIDE_II_COLOR = "#D56B3D"
REFERENCE_COLOR = "#4D4D4D"
GRID_COLOR = "#D9D9D9"

NETWORK_ORDER = [
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
]

NETWORK_LABELS = {
    "Vis": "Vis",
    "SomMot": "SMN",
    "DorsAttn": "DAN",
    "SalVentAttn": "VAN",
    "Limbic": "Lim",
    "Cont": "Cont",
    "Default": "DMN",
}


def configureFigureStyle():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def findDatasetDirectory(datasetPrefix, directorySuffix):
    matchingDirectories = sorted(
        directory
        for directory in FIGURE_ROOT.iterdir()
        if directory.is_dir()
        and directory.name.startswith(datasetPrefix)
        and directory.name.endswith(directorySuffix)
    )
    if len(matchingDirectories) != 1:
        raise FileNotFoundError(
            f"Expected one directory for {datasetPrefix} / {directorySuffix}, "
            f"found {len(matchingDirectories)}"
        )
    return matchingDirectories[0]


def findSingleFile(directory, filenamePattern):
    matchingFiles = sorted(directory.glob(filenamePattern))
    if len(matchingFiles) != 1:
        raise FileNotFoundError(
            f"Expected one file matching {filenamePattern} in {directory.name}, "
            f"found {len(matchingFiles)}"
        )
    return matchingFiles[0]


def requireColumns(frame, requiredColumns, sourceLabel):
    missingColumns = sorted(set(requiredColumns) - set(frame.columns))
    if missingColumns:
        raise ValueError(f"{sourceLabel} is missing columns: {missingColumns}")


def loadNetworkEffects():
    allEffects = []
    for datasetPrefix, datasetLabel in [
        ("ABIDE1_", "ABIDE I"),
        ("ABIDE2_", "ABIDE II"),
    ]:
        datasetDirectory = findDatasetDirectory(datasetPrefix, "noDistanceFDR")
        sourcePath = datasetDirectory / "GLM_Yeo7_network_G1G2_main_effect_splitFDR.csv"
        sourceFrame = pd.read_csv(sourcePath)
        requireColumns(
            sourceFrame,
            ["term", "dv", "item", "beta", "ci_low", "ci_high", "p_FDR"],
            datasetLabel,
        )
        selectedEffects = sourceFrame[
            sourceFrame["term"].astype(str).str.contains("T.ASD", regex=False, na=False)
            & sourceFrame["dv"].astype(str).eq("G1_network")
            & sourceFrame["item"].isin(NETWORK_ORDER)
        ].copy()
        if len(selectedEffects) != len(NETWORK_ORDER):
            raise ValueError(
                f"{datasetLabel}: expected {len(NETWORK_ORDER)} network effects, "
                f"found {len(selectedEffects)}"
            )
        selectedEffects["dataset"] = datasetLabel
        allEffects.append(selectedEffects)

    networkEffects = pd.concat(allEffects, ignore_index=True)
    networkEffects["item"] = pd.Categorical(
        networkEffects["item"], categories=NETWORK_ORDER, ordered=True
    )
    networkEffects["isSignificant"] = networkEffects["p_FDR"] < 0.05
    return networkEffects.sort_values(["item", "dataset"]).reset_index(drop=True)


def selectHierarchyEffect(sourcePath, datasetLabel, connectionLabel):
    sourceFrame = pd.read_csv(sourcePath)
    requireColumns(
        sourceFrame,
        ["term", "dv", "item", "beta", "ci_low", "ci_high", "p"],
        f"{datasetLabel} {connectionLabel}",
    )
    selectedEffects = sourceFrame[
        sourceFrame["term"].astype(str).str.contains("T.ASD", regex=False, na=False)
        & sourceFrame["dv"].astype(str).str.contains("G1", case=False, na=False)
        & sourceFrame["item"].astype(str).str.contains("Default", case=False, na=False)
        & sourceFrame["item"].astype(str).str.contains("Vis", case=False, na=False)
        & sourceFrame["item"].astype(str).str.contains(
            "Mot|SMN", case=False, regex=True, na=False
        )
    ].copy()
    if len(selectedEffects) != 1:
        raise ValueError(
            f"{datasetLabel} {connectionLabel}: expected one hierarchy effect, "
            f"found {len(selectedEffects)}"
        )

    selectedEffect = selectedEffects.iloc[0].copy()
    selectedEffect["dataset"] = datasetLabel
    selectedEffect["connectionType"] = connectionLabel
    if "p_FDR" in selectedEffects.columns and pd.notna(selectedEffect["p_FDR"]):
        selectedEffect["reportedPValue"] = float(selectedEffect["p_FDR"])
        selectedEffect["pValueLabel"] = "q"
    else:
        selectedEffect["reportedPValue"] = float(selectedEffect["p"])
        selectedEffect["pValueLabel"] = "P"
    return selectedEffect


def transformAbsoluteInterval(beta, ciLow, ciHigh):
    absoluteBeta = abs(float(beta))
    if ciLow <= 0 <= ciHigh:
        absoluteCiLow = 0.0
        absoluteCiHigh = max(abs(float(ciLow)), abs(float(ciHigh)))
    else:
        absoluteCiLow = min(abs(float(ciLow)), abs(float(ciHigh)))
        absoluteCiHigh = max(abs(float(ciLow)), abs(float(ciHigh)))
    return absoluteBeta, min(absoluteCiLow, absoluteBeta), max(absoluteCiHigh, absoluteBeta)


def loadSensitivityEffects():
    positiveDirectory = findDatasetDirectory("ABIDE2_", "noDistanceFDR")
    positivePath = positiveDirectory / (
        "GLM_hierarchy_distance_G1_Default_dist_VisSomMot_main_effect_rawP.csv"
    )

    negativeAbideIDirectory = findDatasetDirectory("ABIDE1_", "\u8d1f\u5411")
    negativeAbideIIDirectory = findDatasetDirectory("ABIDE2_", "\u8d1f\u5411")
    negativeFilenamePattern = "*GLM_hierarchy_distances_G1G2_main_effect_splitFDR.csv"

    selectedRows = [
        selectHierarchyEffect(positivePath, "ABIDE II", "Positive EC"),
        selectHierarchyEffect(
            findSingleFile(negativeAbideIDirectory, negativeFilenamePattern),
            "ABIDE I",
            "|Negative EC|",
        ),
        selectHierarchyEffect(
            findSingleFile(negativeAbideIIDirectory, negativeFilenamePattern),
            "ABIDE II",
            "|Negative EC|",
        ),
    ]
    sensitivityEffects = pd.DataFrame(selectedRows).reset_index(drop=True)
    transformedIntervals = sensitivityEffects.apply(
        lambda row: transformAbsoluteInterval(
            row["beta"], row["ci_low"], row["ci_high"]
        ),
        axis=1,
        result_type="expand",
    )
    sensitivityEffects[["absoluteBeta", "absoluteCiLow", "absoluteCiHigh"]] = (
        transformedIntervals
    )
    sensitivityEffects["isSignificant"] = sensitivityEffects["reportedPValue"] < 0.05
    return sensitivityEffects


def addPanelLabel(axis, panelLabel, labelX=-0.12):
    axis.text(
        labelX,
        1.08,
        panelLabel,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )


def drawNetworkForest(axis, networkEffects):
    datasetOffsets = {"ABIDE I": -0.15, "ABIDE II": 0.15}
    datasetColors = {"ABIDE I": ABIDE_I_COLOR, "ABIDE II": ABIDE_II_COLOR}
    yPositions = np.arange(len(NETWORK_ORDER))

    for datasetLabel in ["ABIDE I", "ABIDE II"]:
        datasetEffects = networkEffects[networkEffects["dataset"] == datasetLabel]
        for _, row in datasetEffects.iterrows():
            networkIndex = NETWORK_ORDER.index(str(row["item"]))
            effectSize = float(row["beta"])
            ciLow = float(row["ci_low"])
            ciHigh = float(row["ci_high"])
            color = datasetColors[datasetLabel]
            axis.errorbar(
                effectSize,
                networkIndex + datasetOffsets[datasetLabel],
                xerr=[[effectSize - ciLow], [ciHigh - effectSize]],
                fmt="o",
                markersize=4.8,
                markerfacecolor=color if row["isSignificant"] else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.2,
                zorder=3,
            )

    axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axis.set_yticks(yPositions)
    axis.set_yticklabels([NETWORK_LABELS[network] for network in NETWORK_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("ASD–HC centroid difference (β)")
    axis.set_title("Network centroid group effects", loc="left", pad=7)
    legendHandles = [
        Line2D([0], [0], marker="o", color=ABIDE_I_COLOR, markersize=4.8, label="ABIDE I"),
        Line2D([0], [0], marker="o", color=ABIDE_II_COLOR, markersize=4.8, label="ABIDE II"),
    ]
    axis.legend(handles=legendHandles, loc="lower right", handlelength=1.4)
    addPanelLabel(axis, "a")


def drawReplicationScatter(axis, networkEffects):
    wideEffects = networkEffects.pivot_table(
        index="item",
        columns="dataset",
        values=["beta", "p_FDR"],
        aggfunc="first",
        observed=False,
    )
    xValues = wideEffects[("beta", "ABIDE I")].astype(float)
    yValues = wideEffects[("beta", "ABIDE II")].astype(float)
    limit = max(float(np.nanmax(np.abs(pd.concat([xValues, yValues])))) * 1.28, 0.06)

    axis.axhline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axis.plot([-limit, limit], [-limit, limit], color="#969696", linewidth=0.8, linestyle=":")

    for network in NETWORK_ORDER:
        effectX = float(wideEffects.loc[network, ("beta", "ABIDE I")])
        effectY = float(wideEffects.loc[network, ("beta", "ABIDE II")])
        significantInAbideI = float(wideEffects.loc[network, ("p_FDR", "ABIDE I")]) < 0.05
        significantInAbideII = float(wideEffects.loc[network, ("p_FDR", "ABIDE II")]) < 0.05
        if significantInAbideI and significantInAbideII:
            markerColor = REFERENCE_COLOR
        elif significantInAbideI:
            markerColor = ABIDE_I_COLOR
        elif significantInAbideII:
            markerColor = ABIDE_II_COLOR
        else:
            markerColor = "white"
        axis.scatter(
            effectX,
            effectY,
            s=30,
            facecolor=markerColor,
            edgecolor=REFERENCE_COLOR,
            linewidth=0.8,
            zorder=3,
        )
        axis.annotate(
            NETWORK_LABELS[network],
            (effectX, effectY),
            xytext=(3, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    # Fill the complete right-hand grid cell so that panels a+b have the
    # same outer width as the full-width panel c below.
    axis.set_aspect("auto")
    axis.set_xlabel("ABIDE I β")
    axis.set_ylabel("ABIDE II β")
    axis.set_title("Effect-size replication", loc="left", pad=7)
    addPanelLabel(axis, "b")


def drawSensitivityForest(axis, sensitivityEffects):
    colorByConnection = {
        "Positive EC": ABIDE_I_COLOR,
        "|Negative EC|": ABIDE_II_COLOR,
    }
    rowLabels = [
        f"{row.connectionType}\n{row.dataset}"
        for row in sensitivityEffects.itertuples(index=False)
    ]
    yPositions = np.arange(len(sensitivityEffects))
    maximumCi = float(sensitivityEffects["absoluteCiHigh"].max())
    statisticX = maximumCi + 0.020
    xMaximum = statisticX + 0.115

    for rowIndex, row in sensitivityEffects.iterrows():
        effectSize = float(row["absoluteBeta"])
        ciLow = float(row["absoluteCiLow"])
        ciHigh = float(row["absoluteCiHigh"])
        color = colorByConnection[row["connectionType"]]
        axis.errorbar(
            effectSize,
            rowIndex,
            xerr=[[effectSize - ciLow], [ciHigh - effectSize]],
            fmt="o",
            markersize=5.2,
            markerfacecolor=color if row["isSignificant"] else "white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.2,
            zorder=3,
        )
        statisticText = (
            f"|β| = {effectSize:.3f}; 95% |CI| [{ciLow:.3f}, {ciHigh:.3f}]; "
            f"{row['pValueLabel']} = {row['reportedPValue']:.3g}"
        )
        axis.text(statisticX, rowIndex, statisticText, ha="left", va="center")

    axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.6, linestyle=":")
    axis.set_yticks(yPositions)
    axis.set_yticklabels(rowLabels)
    axis.invert_yaxis()
    axis.set_xlim(0, xMaximum)
    axis.set_xlabel("ASD–HC difference in |Default–VisSomMot G1 separation| (|β|)")
    axis.set_title(
        "G1 separation sensitivity analysis for positive and negative EC",
        loc="left",
        pad=7,
    )
    legendHandles = [
        Line2D([0], [0], marker="o", color=ABIDE_I_COLOR, markersize=4.8, label="Positive EC"),
        Line2D([0], [0], marker="o", color=ABIDE_II_COLOR, markersize=4.8, label="|Negative EC|"),
        Line2D(
            [0], [0], marker="o", color=REFERENCE_COLOR, markerfacecolor=REFERENCE_COLOR,
            linewidth=0, markersize=4.8, label="P or q < 0.05",
        ),
        Line2D(
            [0], [0], marker="o", color=REFERENCE_COLOR, markerfacecolor="white",
            linewidth=0, markersize=4.8, label="P or q ≥ 0.05",
        ),
    ]
    axis.legend(
        handles=legendHandles,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.35),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    addPanelLabel(axis, "c")


def buildFigure(networkEffects, sensitivityEffects):
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES))
    figureGrid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.48, 1.0],
        width_ratios=[1.10, 0.90],
        left=0.10,
        right=0.98,
        top=0.95,
        bottom=0.14,
        wspace=0.42,
        hspace=0.60,
    )
    forestAxis = figure.add_subplot(figureGrid[0, 0])
    replicationAxis = figure.add_subplot(figureGrid[0, 1])
    sensitivityAxis = figure.add_subplot(figureGrid[1, :])

    drawNetworkForest(forestAxis, networkEffects)
    drawReplicationScatter(replicationAxis, networkEffects)
    drawSensitivityForest(sensitivityAxis, sensitivityEffects)
    return figure


def saveFigure(figure):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputStem = OUTPUT_DIRECTORY / OUTPUT_BASENAME
    figure.savefig(outputStem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(outputStem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(outputStem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(
        outputStem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return outputStem


def main():
    configureFigureStyle()
    networkEffects = loadNetworkEffects()
    sensitivityEffects = loadSensitivityEffects()
    figure = buildFigure(networkEffects, sensitivityEffects)
    outputStem = saveFigure(figure)
    plt.close(figure)
    print(f"Saved figure files with stem: {outputStem}")


if __name__ == "__main__":
    main()
