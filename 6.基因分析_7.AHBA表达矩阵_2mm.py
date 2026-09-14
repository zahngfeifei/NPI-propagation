from __future__ import annotations

import json
import importlib.util
from itertools import combinations
from pathlib import Path
import sys

try:
    import pkg_resources  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    from pip._vendor import pkg_resources as vendoredPkgResources

    sys.modules["pkg_resources"] = vendoredPkgResources

import abagen
import numpy as np
import pandas as pd

COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
commonModuleSpec = importlib.util.spec_from_file_location(
    "geneAnalysisCommon", COMMON_SCRIPT_PATH
)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

AHBA_DATA_DIRECTORY = geneAnalysisCommon.AHBA_DATA_DIRECTORY
AGGREGATE_EXPRESSION_PATH = geneAnalysisCommon.AGGREGATE_EXPRESSION_PATH
DONOR_EXPRESSION_DIRECTORY = geneAnalysisCommon.DONOR_EXPRESSION_DIRECTORY
DONOR_EXPRESSION_MANIFEST_PATH = geneAnalysisCommon.DONOR_EXPRESSION_MANIFEST_PATH
EXPRESSION_OUTPUT_DIRECTORY = geneAnalysisCommon.EXPRESSION_OUTPUT_DIRECTORY
GENE_DIFFERENTIAL_STABILITY_PATH = geneAnalysisCommon.GENE_DIFFERENTIAL_STABILITY_PATH
SCHAEFER_ATLAS_LABEL_PATH = geneAnalysisCommon.SCHAEFER_ATLAS_LABEL_PATH
SCHAEFER_ATLAS_PATH = geneAnalysisCommon.SCHAEFER_ATLAS_PATH
loadConfig = geneAnalysisCommon.loadConfig
sha256File = geneAnalysisCommon.sha256File
writeJson = geneAnalysisCommon.writeJson


def applyAbagenPandasCompatibility() -> None:
    """Provide pandas 1.x call signatures required by abagen 0.1.3."""
    originalSetAxis = pd.DataFrame.set_axis

    def compatibleSetAxis(self, labels, axis=0, inplace=None, copy=None):
        updatedFrame = originalSetAxis(self, labels, axis=axis, copy=copy)
        if inplace:
            self._mgr = updatedFrame._mgr
            return None
        return updatedFrame

    pd.DataFrame.set_axis = compatibleSetAxis
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **kwargs: pd.concat(
            [self, other], ignore_index=kwargs.get("ignore_index", False), sort=kwargs.get("sort", False)
        )
    if not hasattr(pd.Series, "append"):
        pd.Series.append = lambda self, other, **kwargs: pd.concat(
            [self, other], ignore_index=kwargs.get("ignore_index", False)
        )


def buildAtlasInfo(atlasLabels: list[str]) -> pd.DataFrame:
    parcelLabels = [str(label) for label in atlasLabels[1:]]
    return pd.DataFrame(
        {
            "id": np.arange(1, len(parcelLabels) + 1, dtype=int),
            "hemisphere": ["L" if "_LH_" in label else "R" for label in parcelLabels],
            "structure": ["cortex"] * len(parcelLabels),
        }
    )


def loadAtlasLabels() -> list[str]:
    atlasLabelFrame = pd.read_csv(
        SCHAEFER_ATLAS_LABEL_PATH,
        sep="\t",
        header=None,
        usecols=[0, 1],
        names=["roiId", "roiLabel"],
    )
    expectedRoiIds = np.arange(1, 401, dtype=int)
    observedRoiIds = atlasLabelFrame["roiId"].to_numpy(dtype=int)
    if not np.array_equal(observedRoiIds, expectedRoiIds):
        raise RuntimeError("Schaefer-400 标签文件的 ROI 顺序不完整或不连续。")
    return ["Background", *atlasLabelFrame["roiLabel"].astype(str).tolist()]


def calculateDifferentialStability(donorFrames: dict[str, pd.DataFrame]) -> pd.Series:
    donorKeys = sorted(donorFrames)
    commonGenes = donorFrames[donorKeys[0]].columns
    standardizedDonors: dict[str, np.ndarray] = {}
    for donorKey in donorKeys:
        donorValues = donorFrames[donorKey].loc[:, commonGenes].to_numpy(dtype=float)
        donorMeans = np.nanmean(donorValues, axis=0, keepdims=True)
        donorStandardDeviations = np.nanstd(donorValues, axis=0, ddof=1, keepdims=True)
        donorStandardDeviations[donorStandardDeviations == 0] = np.nan
        standardizedDonors[donorKey] = (donorValues - donorMeans) / donorStandardDeviations

    pairwiseCorrelations = []
    for firstDonorKey, secondDonorKey in combinations(donorKeys, 2):
        firstValues = standardizedDonors[firstDonorKey]
        secondValues = standardizedDonors[secondDonorKey]
        pairwiseCorrelation = np.nanmean(firstValues * secondValues, axis=0)
        pairwiseCorrelations.append(pairwiseCorrelation)
    stabilityValues = np.nanmean(np.vstack(pairwiseCorrelations), axis=0)
    return pd.Series(stabilityValues, index=commonGenes, name="differentialStability")


def standardizeDonorIndex(donorFrame: pd.DataFrame) -> pd.DataFrame:
    standardizedFrame = donorFrame.copy()
    standardizedFrame.index = standardizedFrame.index.astype(int)
    standardizedFrame.index.name = "ROI_ID_1based"
    return standardizedFrame.sort_index()


def sanitizeExistingOutputs(aggregateOutputPath, donorManifestPath) -> None:
    aggregateFrame = pd.read_parquet(aggregateOutputPath)
    geneColumns = [columnName for columnName in aggregateFrame.columns if columnName != "ROI_ID_1based"]
    finiteGeneMask = np.isfinite(aggregateFrame[geneColumns].to_numpy(dtype=float)).all(axis=0)
    validGeneColumns = [geneName for geneName, isFinite in zip(geneColumns, finiteGeneMask) if isFinite]
    excludedGeneColumns = [geneName for geneName, isFinite in zip(geneColumns, finiteGeneMask) if not isFinite]
    if excludedGeneColumns:
        aggregateFrame = aggregateFrame[["ROI_ID_1based", *validGeneColumns]]
        aggregateFrame.to_parquet(aggregateOutputPath, index=False)
        aggregateFrame.to_csv(
            EXPRESSION_OUTPUT_DIRECTORY / "ahba-expression-aggregate-2mm.csv.gz",
            index=False,
            compression="gzip",
        )
        donorManifestFrame = pd.read_csv(donorManifestPath)
        for rowIndex, donorRow in donorManifestFrame.iterrows():
            donorOutputPath = donorRow["outputPath"]
            donorFrame = pd.read_parquet(donorOutputPath)
            donorFrame = donorFrame[["ROI_ID_1based", *validGeneColumns]]
            donorFrame.to_parquet(donorOutputPath, index=False)
            donorManifestFrame.loc[rowIndex, "geneCount"] = len(validGeneColumns)
            donorManifestFrame.loc[rowIndex, "nonFiniteCellCount"] = int(
                (~np.isfinite(donorFrame[validGeneColumns].to_numpy(dtype=float))).sum()
            )
        donorManifestFrame.to_csv(donorManifestPath, index=False, encoding="utf-8-sig")
        qcPath = EXPRESSION_OUTPUT_DIRECTORY / "expression-qc.json"
        with qcPath.open("r", encoding="utf-8") as qcFile:
            qcPayload = json.load(qcFile)
        qcPayload["retainedGeneCount"] = len(validGeneColumns)
        qcPayload["aggregateNonFiniteCellCount"] = 0
        qcPayload["excludedNonFiniteAggregateGenes"] = excludedGeneColumns
        writeJson(qcPath, qcPayload)


def main() -> None:
    applyAbagenPandasCompatibility()
    config = loadConfig()
    expressionConfig = config["expression"]
    aggregateOutputPath = AGGREGATE_EXPRESSION_PATH
    donorManifestPath = DONOR_EXPRESSION_MANIFEST_PATH
    if aggregateOutputPath.exists() and donorManifestPath.exists():
        sanitizeExistingOutputs(aggregateOutputPath, donorManifestPath)
        print("已验证 2 mm AHBA 表达输出，跳过重复构建。")
        return

    if int(expressionConfig["atlasResolutionMm"]) != 2:
        raise RuntimeError("本分析锁定为 Schaefer-2018 400 分区、7 网络、2 mm。")
    atlasInfo = buildAtlasInfo(loadAtlasLabels())
    donorExpressionResult, abagenReport = abagen.get_expression_data(
        str(SCHAEFER_ATLAS_PATH),
        atlas_info=atlasInfo,
        ibf_threshold=float(expressionConfig["intensityBasedFilteringThreshold"]),
        probe_selection=str(expressionConfig["probeSelection"]),
        exact=False,
        missing=str(expressionConfig["missingParcelMethod"]),
        tolerance=int(expressionConfig["toleranceMm"]),
        sample_norm=str(expressionConfig["sampleNormalization"]),
        gene_norm=str(expressionConfig["geneNormalization"]),
        norm_matched=True,
        norm_structures=False,
        region_agg="donors",
        agg_metric="mean",
        corrected_mni=True,
        reannotated=True,
        return_donors=True,
        return_report=True,
        donors="all",
        data_dir=str(AHBA_DATA_DIRECTORY),
        verbose=1,
        n_proc=int(expressionConfig["nProcesses"]),
    )
    donorFrames = {
        str(donorKey): standardizeDonorIndex(donorFrame)
        for donorKey, donorFrame in donorExpressionResult.items()
    }
    differentialStability = calculateDifferentialStability(donorFrames)
    stabilityThreshold = float(expressionConfig["differentialStabilityThreshold"])
    retainedGenes = differentialStability.index[differentialStability.ge(stabilityThreshold)].tolist()
    if len(retainedGenes) < 1000:
        raise ValueError(f"Only {len(retainedGenes)} genes passed the differential-stability threshold.")

    donorOutputDirectory = DONOR_EXPRESSION_DIRECTORY
    donorOutputDirectory.mkdir(parents=True, exist_ok=True)
    donorManifestRows = []
    retainedDonorFrames: list[pd.DataFrame] = []
    for donorKey, donorFrame in donorFrames.items():
        retainedFrame = donorFrame.loc[:, retainedGenes].copy()
        nonFiniteCount = int((~np.isfinite(retainedFrame.to_numpy(dtype=float))).sum())
        donorOutputPath = donorOutputDirectory / f"donor-{donorKey}-expression.parquet"
        retainedFrame.reset_index().to_parquet(donorOutputPath, index=False)
        retainedDonorFrames.append(retainedFrame)
        donorManifestRows.append(
            {
                "donorKey": donorKey,
                "outputPath": str(donorOutputPath),
                "roiCount": len(retainedFrame),
                "geneCount": retainedFrame.shape[1],
                "nonFiniteCellCount": nonFiniteCount,
            }
        )

    donorValueStack = np.stack(
        [donorFrame.to_numpy(dtype=float) for donorFrame in retainedDonorFrames], axis=0
    )
    aggregateFrame = pd.DataFrame(
        np.nanmean(donorValueStack, axis=0),
        index=retainedDonorFrames[0].index,
        columns=retainedDonorFrames[0].columns,
    )
    aggregateFrame.index.name = "ROI_ID_1based"
    aggregateFrame.reset_index().to_parquet(aggregateOutputPath, index=False)
    aggregateFrame.reset_index().to_csv(
        EXPRESSION_OUTPUT_DIRECTORY / "ahba-expression-aggregate-2mm.csv.gz",
        index=False,
        compression="gzip",
    )
    differentialStability.sort_values(ascending=False).rename_axis("geneSymbol").reset_index().to_csv(
        GENE_DIFFERENTIAL_STABILITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    donorManifestFrame = pd.DataFrame(donorManifestRows)
    donorManifestPath.parent.mkdir(parents=True, exist_ok=True)
    donorManifestFrame.to_csv(donorManifestPath, index=False, encoding="utf-8-sig")
    (EXPRESSION_OUTPUT_DIRECTORY / "abagen-report.txt").write_text(
        abagenReport, encoding="utf-8"
    )
    writeJson(
        EXPRESSION_OUTPUT_DIRECTORY / "expression-qc.json",
        {
            "abagenVersion": abagen.__version__,
            "donorCount": len(donorFrames),
            "atlasRoiCount": len(aggregateFrame),
            "unfilteredGeneCount": len(differentialStability),
            "retainedGeneCount": len(retainedGenes),
            "differentialStabilityThreshold": stabilityThreshold,
            "aggregateNonFiniteCellCount": int((~np.isfinite(aggregateFrame.to_numpy(dtype=float))).sum()),
            "parameters": expressionConfig,
            "atlasPath": str(SCHAEFER_ATLAS_PATH),
            "atlasSha256": sha256File(SCHAEFER_ATLAS_PATH),
            "ahbaDataDirectory": r"%USERPROFILE%\abagen-data",
        },
    )
    sanitizeExistingOutputs(aggregateOutputPath, donorManifestPath)


if __name__ == "__main__":
    main()
