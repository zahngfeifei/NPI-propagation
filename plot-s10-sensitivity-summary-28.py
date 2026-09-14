from __future__ import annotations

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


ROOT_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT_DIRECTORY / "outputs"
RENDER_DIRECTORY = OUTPUT_DIRECTORY / "brainspace-renders"
CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX, EXPORT_DPI = 1800, 1275, 100
FONT_SIZE_PX = 25
FONT_SIZE_PT = FONT_SIZE_PX * 72 / EXPORT_DPI
FONT_FAMILY = "Arial"
BLUE, GREEN, ORANGE, BLACK, GREY = "#1F4EAD", "#1C7C36", "#F0641E", "#111111", "#777777"


def findSingleFile(fileName: str) -> Path:
    matchedPaths = list(ROOT_DIRECTORY.rglob(fileName))
    if len(matchedPaths) != 1:
        raise FileNotFoundError(f"Expected one {fileName}; found {len(matchedPaths)}.")
    return matchedPaths[0]


def getResultRow(frame: pd.DataFrame, phenotypeName: str, geneSetName: str = "SFARI_high_confidence_nonsyndromic") -> pd.Series:
    selectedRows = frame.loc[(frame["phenotypeName"] == phenotypeName) & (frame["geneSetName"] == geneSetName)]
    if len(selectedRows) != 1:
        raise ValueError(f"Expected one frozen result for {phenotypeName}; found {len(selectedRows)}.")
    return selectedRows.iloc[0]


def cropWhiteMargins(imagePath: Path) -> None:
    image = Image.open(imagePath).convert("RGB")
    bounds = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if bounds:
        left, top, right, bottom = bounds
        image.crop((max(0, left - 8), max(0, top - 8), min(image.width, right + 8), min(image.height, bottom + 8))).save(imagePath)


def renderBrainMap(parcelValues: np.ndarray, outputPath: Path, colorRange: tuple[float, float]) -> None:
    leftSurface, rightSurface = load_conte69()
    schaeferLabels = load_parcellation("schaefer", scale=400, join=True)
    vertexValues = map_to_labels(parcelValues, schaeferLabels, mask=0, fill=np.nan)
    plot_hemispheres(leftSurface, rightSurface, array_name=vertexValues, color_bar=False,
        color_range=colorRange, layout_style="grid", cmap=mpl.colormaps["RdBu_r"],
        nan_color=(0.84, 0.84, 0.84, 1), zoom=1.46, background=(1, 1, 1),
        size=(980, 680), interactive=False, screenshot=True, filename=str(outputPath),
        scale=(2, 2), transparent_bg=False, suppress_warnings=True)
    cropWhiteMargins(outputPath)


def drawForest(
    axis: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    title: str,
    color: str,
    xLimits: tuple[float, float],
    titleYShiftFig: float = 0.0,
) -> mpl.text.Text:
    yPositions = np.arange(len(values))[::-1]
    axis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    axis.scatter(values, yPositions, s=55, color=color, zorder=3)
    axis.set(yticks=yPositions, yticklabels=labels, xlim=xLimits, ylim=(-0.8, len(values) + 0.9), xlabel="Spearman ρ")
    titleArtist = addAlignedTitle(axis, title, x=0.0, ha="left", yShiftFig=titleYShiftFig)
    axis.tick_params(axis="y", length=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    return titleArtist


def addBrainPanel(axis: plt.Axes, imagePath: Path, title: str, colorRange: tuple[float, float], colorbarLabel: str, anchor: str = "N") -> mpl.text.Text:
    axis.imshow(Image.open(imagePath), interpolation="none")
    # Preserve the BrainSpace render's native geometry; do not stretch either axis.
    axis.set_aspect("equal", adjustable="box")
    # Horizontal alignment is controlled by the anchor:
    # "NW" keeps the left brain image flush left, "NE" keeps the right brain image flush right.
    axis.set_anchor(anchor)
    axis.set_axis_off()
    # The brain-map small title remains centered.
    titleArtist = addAlignedTitle(axis, title, x=0.5, ha="center")
    colorAxis = axis.inset_axes([0.18, -0.11, 0.64, 0.045])
    mpl.colorbar.ColorbarBase(colorAxis, cmap=mpl.colormaps["RdBu_r"], norm=mpl.colors.Normalize(*colorRange), orientation="horizontal")
    colorAxis.set_xlabel(colorbarLabel, labelpad=3)
    return titleArtist


TITLE_PLOT_GAP_PX = 10
TITLE_PLOT_GAP_FIG = TITLE_PLOT_GAP_PX / CANVAS_HEIGHT_PX

# Exact horizontal gap between panel label (a-d) and the corresponding main title.
PANEL_LABEL_TITLE_GAP_PX = 30
PANEL_LABEL_TITLE_GAP_FIG = PANEL_LABEL_TITLE_GAP_PX / CANVAS_WIDTH_PX

# Automatically aligned panels remain at least 10 px inside the canvas.
LEFT_CANVAS_MARGIN_PX = 10
LEFT_CANVAS_MARGIN_FIG = LEFT_CANVAS_MARGIN_PX / CANVAS_WIDTH_PX

# Panels b and c apply a downward offset to the small titles and plotting content
# to create a new hierarchy level for a group title.
BC_SHIFT_DOWN_PX = 45
BC_SHIFT_DOWN_FIG = BC_SHIFT_DOWN_PX / CANVAS_HEIGHT_PX

PANEL_B_GROUP_TITLE = "Donor sensitivity analyses"
PANEL_C_GROUP_TITLE = "Spatial maps"
GROUP_TITLE_FONT_SCALE = 1.15

# Reduce the blank interval between the top row (a/b) and bottom row (c/d)
# by shifting the COMPLETE c/d panels upward as a unit.
ROW_GAP_REDUCTION_PX = 120
ROW_GAP_REDUCTION_FIG = ROW_GAP_REDUCTION_PX / CANVAS_HEIGHT_PX

# Top-row layout: reduce panel a / b widths and increase the blank interval
# between them.
TOP_A_X = 0.09
TOP_A_WIDTH = 0.36

TOP_B_LEFT_X = 0.61
TOP_B_RIGHT_X = 0.835
TOP_B_WIDTH = 0.145

# Panel d layout.
# Match panel d width to the TOTAL width of panel b.
# This must be defined AFTER the TOP_B_* constants.
D_PANEL_WIDTH = (TOP_B_RIGHT_X + TOP_B_WIDTH) - TOP_B_LEFT_X


def addAlignedTitle(
    axis: plt.Axes,
    title: str,
    x: float = 0.0,
    ha: str = "left",
    yShiftFig: float = 0.0,
) -> mpl.text.Text:
    """
    Draw titles in figure coordinates using the ORIGINAL axes position.

    This avoids vertical shifts caused by imshow(...)+set_aspect("equal"),
    which changes the active axes box of the brain-map panels.
    """
    originalBox = axis.get_position(original=True)
    titleX = originalBox.x0 + x * originalBox.width
    # The bottom edge of the title is exactly 10 px above the plotting area.
    titleY = originalBox.y1 + TITLE_PLOT_GAP_FIG + yShiftFig
    return axis.figure.text(
        titleX, titleY, title,
        transform=axis.figure.transFigure,
        ha=ha, va="bottom",
        fontweight="bold",
        clip_on=False,
    )


def addPanelLabel(
    axis: plt.Axes,
    label: str,
    yShiftFig: float = 0.0,
) -> mpl.text.Text:
    """
    Draw panel labels in figure coordinates using the ORIGINAL axes position.

    The label and title share exactly the same figure-level y coordinate.
    """
    originalBox = axis.get_position(original=True)
    titleX = originalBox.x0
    labelX = titleX - PANEL_LABEL_TITLE_GAP_FIG
    # The bottom edge of the panel label is also exactly 10 px above the plotting area,
    # so panel labels and titles remain vertically aligned.
    labelY = originalBox.y1 + TITLE_PLOT_GAP_FIG + yShiftFig
    return axis.figure.text(
        labelX, labelY, label,
        transform=axis.figure.transFigure,
        ha="right", va="bottom",
        fontweight="bold",
        fontsize=FONT_SIZE_PT * 1.1,
        clip_on=False,
    )


def addGroupTitle(
    figure: plt.Figure,
    leftAxis: plt.Axes,
    rightAxis: plt.Axes,
    label: str,
    title: str,
    restoredTitleY: float,
) -> tuple[mpl.text.Text, mpl.text.Text]:
    """
    Add one large title spanning a two-axis panel group.

    restoredTitleY is the vertical level formerly occupied by the small titles
    before panels b/c were shifted downward.
    """
    leftBox = leftAxis.get_position(original=True)
    groupTitleX = leftBox.x0
    groupLabelX = groupTitleX - PANEL_LABEL_TITLE_GAP_FIG

    titleArtist = figure.text(
        groupTitleX, restoredTitleY, title,
        transform=figure.transFigure,
        ha="left", va="bottom",
        fontweight="bold",
        fontsize=FONT_SIZE_PT * GROUP_TITLE_FONT_SCALE,
        clip_on=False,
    )
    labelArtist = figure.text(
        groupLabelX, restoredTitleY, label,
        transform=figure.transFigure,
        ha="right", va="bottom",
        fontweight="bold",
        fontsize=FONT_SIZE_PT * 1.1,
        clip_on=False,
    )
    return titleArtist, labelArtist


def alignBrainPanelsToSummaryLabelLeft(
    figure: plt.Figure,
    summaryAxis: plt.Axes,
    leftBrainAxis: plt.Axes,
    rightBrainAxis: plt.Axes,
) -> None:
    """
    Shift the two panel-c brain-map axes horizontally as one block so that the
    left boundary of the left brain image aligns with the left boundary of the
    y-axis label block in panel a.

    The alignment target is computed from the rendered y-tick-label extents of
    summaryAxis, so the result follows the actual text geometry rather than an
    approximate hard-coded offset.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    visibleTickLabels = [
        tickLabel
        for tickLabel in summaryAxis.get_yticklabels()
        if tickLabel.get_text().strip()
    ]
    if not visibleTickLabels:
        return

    labelLeftPx = min(
        tickLabel.get_window_extent(renderer=renderer).x0
        for tickLabel in visibleTickLabels
    )
    labelLeftFig = figure.transFigure.inverted().transform((labelLeftPx, 0))[0]

    # Align to panel a's y-label boundary when possible, but never allow the
    # panel-c block to cross the 10 px safe margin from the canvas edge.
    targetLeftFig = max(labelLeftFig, LEFT_CANVAS_MARGIN_FIG)

    leftBox = leftBrainAxis.get_position(original=True)
    rightBox = rightBrainAxis.get_position(original=True)
    deltaX = targetLeftFig - leftBox.x0

    leftBrainAxis.set_position([
        leftBox.x0 + deltaX, leftBox.y0, leftBox.width, leftBox.height
    ])
    rightBrainAxis.set_position([
        rightBox.x0 + deltaX, rightBox.y0, rightBox.width, rightBox.height
    ])


def alignMainTitleToRenderedLeft(
    figure: plt.Figure,
    axes: list[plt.Axes],
    titleArtist: mpl.text.Text,
    labelArtist: mpl.text.Text,
    *,
    useActiveAxisLeft: bool = False,
) -> None:
    """
    Align the MAIN panel title to the actual rendered left boundary of the panel.

    Parameters
    ----------
    axes
        Axes belonging to the panel.
    titleArtist
        Main title artist, e.g. "Overall sensitivity summary",
        "Donor sensitivity analyses", "Spatial maps",
        or "Effect of level adjustment".
    labelArtist
        Panel label artist (a-d). Its right edge remains exactly 30 px
        to the left of the main-title anchor.
    useActiveAxisLeft
        False: use the left edge of the axes tight bounding box, including
        y tick labels / y-axis title. This is used for panels a, b and d.
        True: use the active rendered axes box itself. This is used for panel c,
        so "Spatial maps" aligns to the actual left edge of the brain image area.

    Small titles are intentionally unchanged.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    if useActiveAxisLeft:
        # For the brain maps, use the real active axes box after aspect-ratio
        # adjustment, which corresponds to the displayed image boundary.
        leftPx = min(axis.get_window_extent(renderer=renderer).x0 for axis in axes)
    else:
        # For Cartesian panels, include tick labels and the y-axis label.
        tightBoxes = [
            axis.get_tightbbox(renderer)
            for axis in axes
            if axis.get_tightbbox(renderer) is not None
        ]
        if not tightBoxes:
            return
        leftPx = min(box.x0 for box in tightBoxes)

    titleXFig = figure.transFigure.inverted().transform((leftPx, 0))[0]

    # Horizontal movement preserves each title's existing y coordinate.
    _, titleY = titleArtist.get_position()
    titleArtist.set_position((titleXFig, titleY))
    titleArtist.set_ha("left")

    # The panel label remains exactly 30 px to the left of the title anchor.
    _, labelY = labelArtist.get_position()
    labelArtist.set_position(
        (titleXFig - PANEL_LABEL_TITLE_GAP_FIG, labelY)
    )
    labelArtist.set_ha("right")


def _shiftAxisGroup(axes: list[plt.Axes], dxFig: float = 0.0, dyFig: float = 0.0) -> None:
    for axis in axes:
        box = axis.get_position(original=True)
        axis.set_position([box.x0 + dxFig, box.y0 + dyFig, box.width, box.height])


def _shiftTextGroup(textArtists: list[mpl.text.Text], dxFig: float = 0.0, dyFig: float = 0.0) -> None:
    for artist in textArtists:
        x, y = artist.get_position()
        artist.set_position((x + dxFig, y + dyFig))


def pinPanelToCanvasMargin(
    figure: plt.Figure,
    axes: list[plt.Axes],
    textArtists: list[mpl.text.Text],
    *,
    marginPx: int = 10,
    pinLeft: bool = False,
    pinRight: bool = False,
    pinTop: bool = False,
    pinBottom: bool = False,
) -> None:
    """
    Treat one subplot (including title, panel label, axes labels, tick labels, etc.)
    as one complete layout panel, and pin the requested outer edge(s) to an EXACT
    marginPx distance from the canvas boundary.

    Example:
    - top-left panel:  pinLeft=True,  pinTop=True
    - top-right panel: pinRight=True, pinTop=True
    - bottom-left panel:  pinLeft=True,  pinBottom=True
    - bottom-right panel: pinRight=True, pinBottom=True
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvasWidthPx = figure.bbox.width
    canvasHeightPx = figure.bbox.height

    bboxes = []
    for axis in axes:
        bbox = axis.get_tightbbox(renderer)
        if bbox is not None:
            bboxes.append(bbox)
    for artist in textArtists:
        bbox = artist.get_window_extent(renderer=renderer)
        if bbox is not None:
            bboxes.append(bbox)

    if not bboxes:
        return

    union = bboxes[0]
    for bbox in bboxes[1:]:
        union = mpl.transforms.Bbox.union([union, bbox])

    dxPx = 0.0
    dyPx = 0.0

    if pinLeft and pinRight:
        raise ValueError("A panel cannot be pinned to both left and right simultaneously.")
    if pinTop and pinBottom:
        raise ValueError("A panel cannot be pinned to both top and bottom simultaneously.")

    if pinLeft:
        dxPx = marginPx - union.x0
    elif pinRight:
        dxPx = (canvasWidthPx - marginPx) - union.x1

    if pinBottom:
        dyPx = marginPx - union.y0
    elif pinTop:
        dyPx = (canvasHeightPx - marginPx) - union.y1

    if dxPx == 0 and dyPx == 0:
        return

    dxFig = dxPx / canvasWidthPx
    dyFig = dyPx / canvasHeightPx
    _shiftAxisGroup(axes, dxFig=dxFig, dyFig=dyFig)
    _shiftTextGroup(textArtists, dxFig=dxFig, dyFig=dyFig)


def main() -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.size": FONT_SIZE_PT, "svg.fonttype": "none", "image.composite_image": False, "axes.linewidth": 1.0})
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    RENDER_DIRECTORY.mkdir(exist_ok=True)
    primaryResults = pd.read_csv(findSingleFile("primary-result.csv"))
    baselineResults = pd.read_csv(findSingleFile("baseline-result.csv"))
    donorResults = pd.read_csv(findSingleFile("single-donor-correlations.csv"))
    lodoResults = pd.read_csv(findSingleFile("leave-one-donor-out.csv"))
    hemisphereResults = pd.read_csv(findSingleFile("hemisphere-sensitivity.csv"))
    phenotypeFrame = pd.read_parquet(findSingleFile("ABIDE1_H1H4_phenotypes.parquet"))
    geneScores = pd.read_csv(findSingleFile("gene-set-scores-selected-374.csv"))
    primary = getResultRow(primaryResults, "system_residual")
    baseline = getResultRow(baselineResults, "raw_beta")

    # Maps use frozen parcelwise outcome tables only; no statistical calculation is performed here.
    geneRows = geneScores.loc[(geneScores["geneSetName"] == "SFARI_high_confidence_nonsyndromic") & (geneScores["scoreMethod"] == "mean_z"), ["ROI_ID_1based", "score"]]
    mapFrame = phenotypeFrame.merge(geneRows, on="ROI_ID_1based", how="inner", validate="one_to_one")
    phenotypeMap = np.full(400, np.nan)
    geneMap = np.full(400, np.nan)
    roiIndices = mapFrame["ROI_ID_1based"].to_numpy(int) - 1
    phenotypeMap[roiIndices] = mapFrame["raw_beta_z"].to_numpy(float)
    geneMap[roiIndices] = mapFrame["score"].to_numpy(float)
    phenotypePath, genePath = RENDER_DIRECTORY / "raw-beta.png", RENDER_DIRECTORY / "risk-gene-expression.png"
    renderBrainMap(phenotypeMap, phenotypePath, (-2.5, 2.5))
    renderBrainMap(geneMap, genePath, (-0.7, 0.7))

    figure = plt.figure(figsize=(CANVAS_WIDTH_PX / EXPORT_DPI, CANVAS_HEIGHT_PX / EXPORT_DPI), dpi=EXPORT_DPI, facecolor="white")
    # Create the axes first, then place them manually so that:
    # - panel a and panel c are flush to the left,
    # - panel b and panel d are flush to the right,
    # - the whole top row sits slightly lower to prevent top-edge clipping.
    grid = figure.add_gridspec(2, 2)
    summaryAxis = figure.add_subplot(grid[0, 0])
    donorPanel = grid[0, 1].subgridspec(1, 2)
    donorAxis, lodoAxis = figure.add_subplot(donorPanel[0, 0]), figure.add_subplot(donorPanel[0, 1])
    brainPanel = grid[1, 0].subgridspec(1, 2)
    leftBrainAxis, rightBrainAxis = figure.add_subplot(brainPanel[0, 0]), figure.add_subplot(brainPanel[0, 1])
    adjustmentAxis = figure.add_subplot(grid[1, 1])

    # Manual alignment:
    # a/c left aligned, b/d right aligned.
    topRowY = 0.585
    topRowHeight = 0.305

    # Panels a and b have exactly the same plotting-area y-position and height.
    # Therefore their top and bottom borders are strictly aligned.
    # Their widths are reduced here to create a larger blank interval between a and b.
    summaryAxis.set_position([TOP_A_X, topRowY, TOP_A_WIDTH, topRowHeight])
    donorAxis.set_position([TOP_B_LEFT_X, topRowY, TOP_B_WIDTH, topRowHeight])
    lodoAxis.set_position([TOP_B_RIGHT_X, topRowY, TOP_B_WIDTH, topRowHeight])

    lowerY = 0.105
    lowerHeight = 0.285
    lowerPlotY = lowerY - BC_SHIFT_DOWN_FIG

    # Panels c and d keep their original plotting-area height.
    # The vertical row gap is reduced later by shifting the COMPLETE panels upward,
    # including titles, panel labels, plots, annotations, and colorbars.
    leftBrainAxis.set_position([0.09, lowerPlotY, 0.235, lowerHeight])
    rightBrainAxis.set_position([0.34, lowerPlotY, 0.235, lowerHeight])
    # Panel d width is matched to the total width of panel b.
    adjustmentAxis.set_position([0.60, lowerPlotY, D_PANEL_WIDTH, lowerHeight])

    donorValues = donorResults["spearmanR"].to_numpy(float)
    lodoValues = lodoResults["spearmanR"].to_numpy(float)
    donorLabels = [f"Donor {index}" for index in range(1, len(donorValues) + 1)]
    # The two b-panel small titles remain 45 px below the large b-panel title level,
    # while the actual plotting areas stay aligned with panel a.
    donorSmallTitle = drawForest(
        donorAxis, donorValues, donorLabels,
        "Single donor analyses", BLUE, (-0.33, 0.02),
        titleYShiftFig=-BC_SHIFT_DOWN_FIG,
    )
    lodoSmallTitle = drawForest(
        lodoAxis, lodoValues, donorLabels,
        "Leave one donor out", BLUE, (-0.33, 0.02),
        titleYShiftFig=-BC_SHIFT_DOWN_FIG,
    )

    sensitivityEntries = [
        {
            "label": "Primary level\nadjusted analysis",
            "value": float(primary["spearmanR"]),
            "rangeMinimum": None,
            "rangeMaximum": None,
            "color": BLACK,
        },
        {
            "label": "Single donor\nrange",
            "value": float(donorValues.mean()),
            "rangeMinimum": float(donorValues.min()),
            "rangeMaximum": float(donorValues.max()),
            "color": BLUE,
        },
        {
            "label": "Leave one donor\nout range",
            "value": float(lodoValues.mean()),
            "rangeMinimum": float(lodoValues.min()),
            "rangeMaximum": float(lodoValues.max()),
            "color": BLUE,
        },
    ]
    for hemisphere, displayName, color in [("lh", "Left", GREEN), ("rh", "Right", GREEN)]:
        rows = hemisphereResults.loc[hemisphereResults["hemisphere"].astype(str).str.lower() == hemisphere]
        if len(rows):
            sensitivityEntries.append(
                {
                    "label": f"{displayName}\nhemisphere",
                    "value": float(rows.iloc[0]["spearmanR"]),
                    "rangeMinimum": None,
                    "rangeMaximum": None,
                    "color": color,
                }
            )
    sensitivityEntries.append(
        {
            "label": "Unadjusted HC ASD\nphenotype",
            "value": float(baseline["spearmanR"]),
            "rangeMinimum": None,
            "rangeMaximum": None,
            "color": ORANGE,
        }
    )
    labels = [entry["label"] for entry in sensitivityEntries]
    values = [entry["value"] for entry in sensitivityEntries]
    positions = np.arange(len(values))[::-1]
    summaryAxis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    for position, entry in zip(positions, sensitivityEntries):
        value = entry["value"]
        rangeMinimum = entry["rangeMinimum"]
        rangeMaximum = entry["rangeMaximum"]
        color = entry["color"]
        if rangeMinimum is None or rangeMaximum is None:
            summaryAxis.scatter(value, position, s=64, color=color, zorder=3)
            continue
        summaryAxis.errorbar(
            value,
            position,
            xerr=[[value - rangeMinimum], [rangeMaximum - value]],
            fmt="o",
            color=color,
            capsize=4,
            markersize=8,
            linewidth=1.5,
        )
    summaryAxis.set(yticks=positions, yticklabels=labels, xlim=(-0.34, 0.12), ylim=(-0.8, len(values) - 0.2), xlabel="Spearman ρ")
    summaryTitle = addAlignedTitle(summaryAxis, "Overall sensitivity summary", x=0.0, ha="left")
    summaryAxis.tick_params(axis="y", length=0)
    summaryAxis.spines[["top", "right", "left"]].set_visible(False)
    for pos, value in zip(positions, values): summaryAxis.text(0.112, pos, f"{value:.4f}", ha="right", va="center")

    leftBrainTitle = addBrainPanel(leftBrainAxis, phenotypePath, "Phenotype map", (-2.5, 2.5), "Standardized value", anchor="NW")
    rightBrainTitle = addBrainPanel(rightBrainAxis, genePath, "Gene expression map", (-0.7, 0.7), "Mean z expression", anchor="NE")
    hemiText = hemisphereResults.loc[:, ["hemisphere", "spearmanR"]].drop_duplicates().sort_values("hemisphere")

    adjustmentAxis.axhline(0, color=BLACK, linewidth=1)
    adjustmentAxis.axvline(0, color="#999999", linestyle="--", linewidth=1)
    adjustmentAxis.scatter([0.30, 0.72], [float(primary["spearmanR"]), float(baseline["spearmanR"])], s=110, c=[BLACK, ORANGE], zorder=3)
    adjustmentAxis.vlines([0.30, 0.72], [0, 0], [float(primary["spearmanR"]), float(baseline["spearmanR"])], colors=[BLACK, ORANGE], linewidth=2)
    adjustmentAxis.text(0.30, 0.04, "System residual\nphenotype", ha="center", fontweight="bold")
    adjustmentAxis.text(0.72, 0.04, "Raw beta\nphenotype", ha="center", color=ORANGE, fontweight="bold")
    adjustmentAxis.text(0.30, float(primary["spearmanR"]) - 0.025, f"ρ = {float(primary['spearmanR']):.4f}\nSpatial P = {float(primary['pSpatial']):.4f}\nMatched P = {float(primary['pGeneSet']):.4f}", ha="center", va="top")
    adjustmentAxis.text(0.72, float(baseline["spearmanR"]) - 0.025, f"ρ = {float(baseline['spearmanR']):.4f}\nSpatial P = {float(baseline['pSpatial']):.4f}\nMatched P = {float(baseline['pGeneSet']):.4f}", ha="center", va="top", color=ORANGE)
    adjustmentAxis.set(xlim=(0.05, 0.97), ylim=(-0.30, 0.12), xticks=[], ylabel="Spearman ρ")
    # Panel d's main title uses the same upward offset as panel c's large group title.
    adjustmentTitle = addAlignedTitle(
        adjustmentAxis,
        "Effect of level adjustment",
        x=0.0,
        ha="left",
        yShiftFig=BC_SHIFT_DOWN_FIG,
    )
    adjustmentAxis.spines[["top", "right", "bottom"]].set_visible(False)
    # a and d retain the existing single-level title hierarchy.
    panelALabel = addPanelLabel(summaryAxis, "a")
    panelDLabel = addPanelLabel(
        adjustmentAxis,
        "d",
        yShiftFig=BC_SHIFT_DOWN_FIG,
    )

    # b and c receive a new large group title at the ORIGINAL small-title height.
    # Their existing small titles and all plotting content have been shifted downward.
    donorGroupTitleY = donorAxis.get_position(original=True).y1 + TITLE_PLOT_GAP_FIG
    brainGroupTitleY = leftBrainAxis.get_position(original=True).y1 + BC_SHIFT_DOWN_FIG + TITLE_PLOT_GAP_FIG

    bGroupTitle, bGroupLabel = addGroupTitle(
        figure, donorAxis, lodoAxis, "b",
        PANEL_B_GROUP_TITLE, donorGroupTitleY,
    )
    cGroupTitle, cGroupLabel = addGroupTitle(
        figure, leftBrainAxis, rightBrainAxis, "c",
        PANEL_C_GROUP_TITLE, brainGroupTitleY,
    )

    # --------------------------------------------------------------
    # Main-title horizontal alignment.
    # Movement is limited to the main titles and panel labels.
    # Small titles remain exactly where they are.
    #
    # a: align to the left edge of the rendered y tick-label block.
    # b: align to the left edge of the rendered donor-label block.
    # c: align to the actual left edge of the displayed brain-map area.
    # d: align to the left edge of the rendered y-axis / tick-label block.
    # --------------------------------------------------------------
    alignMainTitleToRenderedLeft(
        figure,
        [summaryAxis],
        summaryTitle,
        panelALabel,
        useActiveAxisLeft=False,
    )
    alignMainTitleToRenderedLeft(
        figure,
        [donorAxis, lodoAxis],
        bGroupTitle,
        bGroupLabel,
        useActiveAxisLeft=False,
    )
    alignMainTitleToRenderedLeft(
        figure,
        [leftBrainAxis, rightBrainAxis],
        cGroupTitle,
        cGroupLabel,
        useActiveAxisLeft=True,
    )
    alignMainTitleToRenderedLeft(
        figure,
        [adjustmentAxis],
        adjustmentTitle,
        panelDLabel,
        useActiveAxisLeft=False,
    )

    # Treat each subplot as one complete layout panel (including title/panel label)
    # and pin the OUTER edge(s) to an EXACT 10 px canvas margin.
    pinPanelToCanvasMargin(
        figure,
        [summaryAxis],
        [summaryTitle, panelALabel],
        marginPx=10,
        pinLeft=True,
        pinTop=True,
    )
    pinPanelToCanvasMargin(
        figure,
        [donorAxis, lodoAxis],
        [donorSmallTitle, lodoSmallTitle, bGroupTitle, bGroupLabel],
        marginPx=10,
        pinRight=True,
        pinTop=True,
    )
    # For the lower row, pin only the horizontal outer edges to 10 px.
    # The bottom edge remains unpinned so the later upward shift is preserved.
    pinPanelToCanvasMargin(
        figure,
        [leftBrainAxis, rightBrainAxis],
        [leftBrainTitle, rightBrainTitle, cGroupTitle, cGroupLabel],
        marginPx=10,
        pinLeft=True,
    )
    pinPanelToCanvasMargin(
        figure,
        [adjustmentAxis],
        [adjustmentTitle, panelDLabel],
        marginPx=10,
        pinRight=True,
    )

    # First lock c and d main-title / panel-label baselines to the same height.
    _, cTitleY = cGroupTitle.get_position()
    dTitleX, _ = adjustmentTitle.get_position()
    dLabelX, _ = panelDLabel.get_position()
    adjustmentTitle.set_position((dTitleX, cTitleY))
    panelDLabel.set_position((dLabelX, cTitleY))

    # The complete c and d panels use a fixed upward offset.
    # Axis-attached content (brain maps, colorbars, annotations, d-panel data)
    # follows the axes automatically; figure-level titles/labels are shifted explicitly.
    _shiftAxisGroup(
        [leftBrainAxis, rightBrainAxis],
        dyFig=ROW_GAP_REDUCTION_FIG,
    )
    _shiftTextGroup(
        [leftBrainTitle, rightBrainTitle, cGroupTitle, cGroupLabel],
        dyFig=ROW_GAP_REDUCTION_FIG,
    )

    _shiftAxisGroup(
        [adjustmentAxis],
        dyFig=ROW_GAP_REDUCTION_FIG,
    )
    _shiftTextGroup(
        [adjustmentTitle, panelDLabel],
        dyFig=ROW_GAP_REDUCTION_FIG,
    )

    outputBase = OUTPUT_DIRECTORY / "s10-sensitivity-summary-28"
    figure.savefig(outputBase.with_suffix(".png"), dpi=EXPORT_DPI, facecolor="white")
    figure.savefig(outputBase.with_suffix(".svg"), format="svg", facecolor="white")
    with Image.open(outputBase.with_suffix(".png")) as image:
        if image.size != (CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX): raise ValueError(f"Unexpected PNG canvas: {image.size}")
    print(outputBase.with_suffix(".png"))
    print(outputBase.with_suffix(".svg"))


if __name__ == "__main__":
    main()
