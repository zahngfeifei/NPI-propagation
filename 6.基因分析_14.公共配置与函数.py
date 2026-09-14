from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
from scipy import stats
from scipy import sparse
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程必要输入"
EXPRESSION_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果7_AHBA表达矩阵_2mm"
GENE_SET_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果8_SFARI基因集"
SCORE_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果9_基因集表达评分"
CENTROID_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果10_Schaefer400球面质心"
SPATIAL_NULL_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果11_空间零模型"
SPATIAL_CORRELATION_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果12_空间相关"
VALIDATION_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程结果13_主流程验证"
RUN_LOG_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程运行日志"

# 以下目录仅供“扩展分析_暂不运行”中的旧脚本兼容导入，本轮主流程不创建或使用。
MATCHED_NULL_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_扩展结果_匹配基因集零模型"
SENSITIVITY_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_扩展结果_供体与半球敏感性"
SUMMARY_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_扩展结果_汇总统计"
RANKED_GSEA_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_扩展结果_排名基因GSEA"
REPORT_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_扩展结果_分析摘要"

PHENOTYPE_PATH = (
    PROJECT_ROOT
    / "ABIDE2_主流程结果6_AHBA表型"
    / "ABIDE2_H1H4_phenotypes.parquet"
)
ROI_CENTROID_PATH = CENTROID_OUTPUT_DIRECTORY / "ABIDE2_selected_roi_centroids.csv"
SFARI_GENE_SCORING_PATH = INPUT_DIRECTORY / "sfari-gene-scoring-2026-q2-original.csv"
SCHAEFER_ATLAS_PATH = (
    INPUT_DIRECTORY
    / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)
SCHAEFER_ATLAS_LABEL_PATH = (
    INPUT_DIRECTORY / "Schaefer2018_400Parcels_7Networks_order.txt"
)
AHBA_DATA_DIRECTORY = Path(
    os.environ.get("ABIDE2_AHBA_DATA_DIRECTORY", Path.home() / "abagen-data")
)
NILEARN_DATA_DIRECTORY = Path(
    os.environ.get("ABIDE2_NILEARN_DATA_DIRECTORY", Path.home() / "nilearn_data")
)

AGGREGATE_EXPRESSION_PATH = (
    EXPRESSION_OUTPUT_DIRECTORY / "ahba-expression-aggregate-2mm.parquet"
)
DONOR_EXPRESSION_DIRECTORY = EXPRESSION_OUTPUT_DIRECTORY / "donor-expression"
DONOR_EXPRESSION_MANIFEST_PATH = (
    EXPRESSION_OUTPUT_DIRECTORY / "donor-expression-manifest.csv"
)
GENE_DIFFERENTIAL_STABILITY_PATH = (
    EXPRESSION_OUTPUT_DIRECTORY / "gene-differential-stability.csv"
)

ANALYSIS_CONFIG = {
    "project": {
        "seed": 42,
        "expectedSelectedRois": 374,
        "expectedFullRois": 400,
        "minimumGenesPerSet": 10,
    },
    "expression": {
        "atlasParcels": 400,
        "atlasNetworks": 7,
        "atlasResolutionMm": 2,
        "intensityBasedFilteringThreshold": 0.5,
        "probeSelection": "diff_stability",
        "sampleNormalization": "srs",
        "geneNormalization": "srs",
        "missingParcelMethod": "interpolate",
        "toleranceMm": 2,
        "differentialStabilityThreshold": 0.2,
        "nProcesses": 1,
    },
    "statistics": {
        "spatialNullCount": 10000,
        "spatialNullQcSampleCount": 200,
        "spatialVariogramMinimumCorrelation": 0.8,
        "spatialVariogramMaximumNormalizedRmse": 0.5,
        "spatialVariogramMinimumEnvelopeCoverage": 0.6,
        "spatialVariogramPercentileCandidates": [25, 40, 60, 75],
        "matchedGeneSetNullCount": 10000,
        "matchedReferenceGeneCount": 512,
        "matchedNeighborPoolSize": 256,
        "matchedMaxAttemptsMultiplier": 30,
        "matchedMaxAbsoluteStandardizedMeanDifference": 0.1,
        "rankedGenePermutationCount": 1000,
        "nJobs": -1,
    },
    "geneSets": {
        "SFARI_high_confidence_nonsyndromic": {
            "expectedRawGeneCount": 124,
            "selectionRule": "gene-score equals 1 and syndromic equals 0",
        },
        "SFARI_syndromic": {
            "expectedRawGeneCount": 218,
            "selectionRule": "syndromic equals 1 and gene-score is non-null",
        },
    },
}
PHENOTYPE_SPECS = (
    {
        "phenotypeName": "system_residual",
        "scoreAdjustment": "system",
        "surrogateKey": "system_residual",
        "surrogateAdjustment": "none",
        "phenotypeRole": "primary",
    },
    {
        "phenotypeName": "raw_beta",
        "scoreAdjustment": "none",
        "surrogateKey": "raw_beta",
        "surrogateAdjustment": "none",
        "phenotypeRole": "baseline",
    },
)
GENE_SET_NAMES = (
    "SFARI_high_confidence_nonsyndromic",
    "SFARI_syndromic",
)
PRIMARY_HYPOTHESIS = {
    "phenotypeName": "system_residual",
    "geneSetName": "SFARI_high_confidence_nonsyndromic",
    "scoreMethod": "mean_z",
}
TEST_TAIL = "two-sided"
EMPIRICAL_P_FORMULA = "(extremeCount + 1)/(nullCount + 1)"
ALPHA = 0.05


def loadConfig() -> dict[str, Any]:
    return json.loads(json.dumps(ANALYSIS_CONFIG))


def writeJson(outputPath: Path, payload: Any) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def sha256File(inputPath: Path, blockSize: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with inputPath.open("rb") as inputFile:
        while block := inputFile.read(blockSize):
            digest.update(block)
    return digest.hexdigest()


def stableSeed(baseSeed: int, label: str) -> int:
    labelDigest = hashlib.sha256(label.encode("utf-8")).digest()
    return (baseSeed + int.from_bytes(labelDigest[:4], "little")) % (2**32 - 1)


def normalizeExpressionColumns(expressionFrame: pd.DataFrame) -> pd.DataFrame:
    normalizedFrame = expressionFrame.copy()
    normalizedFrame.columns = normalizedFrame.columns.astype(str).str.upper()
    if normalizedFrame.columns.duplicated().any():
        normalizedFrame = normalizedFrame.T.groupby(level=0, sort=False).mean().T
    return normalizedFrame


def zscoreColumns(expressionFrame: pd.DataFrame) -> pd.DataFrame:
    columnMeans = expressionFrame.mean(axis=0)
    columnStandardDeviations = expressionFrame.std(axis=0, ddof=1)
    validColumns = columnStandardDeviations.gt(0) & columnStandardDeviations.notna()
    standardizedFrame = expressionFrame.loc[:, validColumns].subtract(
        columnMeans[validColumns], axis=1
    )
    return standardizedFrame.divide(columnStandardDeviations[validColumns], axis=1)


def validateSystemResidualDefinition(phenotypeFrame: pd.DataFrame) -> None:
    expectedResidual = residualizeValues(
        phenotypeFrame["raw_beta"].to_numpy(dtype=float),
        phenotypeFrame,
        "system",
    )
    observedResidual = phenotypeFrame["system_residual"].to_numpy(dtype=float)
    if not np.allclose(expectedResidual, observedResidual, atol=1e-10, rtol=1e-8):
        raise RuntimeError("system_residual does not match the locked residual model.")


def buildCovariateDesign(phenotypeFrame: pd.DataFrame, adjustment: str) -> np.ndarray | None:
    if adjustment == "none":
        return None
    if adjustment != "system":
        raise ValueError(f"Unsupported adjustment: {adjustment}")
    return np.asarray(
        patsy.dmatrix("1 + C(system)", phenotypeFrame, return_type="dataframe"),
        dtype=float,
    )


def residualizeValues(
    values: np.ndarray, phenotypeFrame: pd.DataFrame, adjustment: str
) -> np.ndarray:
    valueArray = np.asarray(values, dtype=float)
    designMatrix = buildCovariateDesign(phenotypeFrame, adjustment)
    if designMatrix is None:
        return valueArray.copy()
    return valueArray - designMatrix @ (np.linalg.pinv(designMatrix) @ valueArray)


def residualizeRows(
    valueMatrix: np.ndarray, phenotypeFrame: pd.DataFrame, adjustment: str
) -> np.ndarray:
    rowMatrix = np.asarray(valueMatrix, dtype=float)
    designMatrix = buildCovariateDesign(phenotypeFrame, adjustment)
    if designMatrix is None:
        return rowMatrix.copy()
    residualMaker = np.eye(len(phenotypeFrame)) - designMatrix @ np.linalg.pinv(designMatrix)
    return rowMatrix @ residualMaker.T


def spearmanCorrelation(firstValues: np.ndarray, secondValues: np.ndarray) -> float:
    return float(stats.spearmanr(firstValues, secondValues, nan_policy="raise").statistic)


def empiricalTwoSidedP(observedStatistic: float, nullStatistics: np.ndarray) -> float:
    finiteNullStatistics = np.asarray(nullStatistics, dtype=float)
    finiteNullStatistics = finiteNullStatistics[np.isfinite(finiteNullStatistics)]
    extremeCount = np.count_nonzero(np.abs(finiteNullStatistics) >= abs(observedStatistic))
    return float((extremeCount + 1) / (finiteNullStatistics.size + 1))


def adjustBenjaminiHochberg(rawPValues: pd.Series) -> pd.Series:
    adjustedValues = pd.Series(np.nan, index=rawPValues.index, dtype=float)
    validMask = rawPValues.notna()
    if validMask.any():
        adjustedValues.loc[validMask] = multipletests(
            rawPValues.loc[validMask].astype(float), method="fdr_bh"
        )[1]
    return adjustedValues


def standardizedMeanDifference(firstValues: np.ndarray, secondValues: np.ndarray) -> float:
    firstArray = np.asarray(firstValues, dtype=float)
    secondArray = np.asarray(secondValues, dtype=float)
    pooledVariance = (np.var(firstArray, ddof=1) + np.var(secondArray, ddof=1)) / 2
    if pooledVariance <= 0:
        return 0.0 if np.mean(firstArray) == np.mean(secondArray) else float("inf")
    return float((np.mean(firstArray) - np.mean(secondArray)) / np.sqrt(pooledVariance))


def vectorizedRowCorrelations(referenceValues: np.ndarray, rowValues: np.ndarray) -> np.ndarray:
    referenceRanks = stats.rankdata(referenceValues, method="average").astype(float)
    rowRanks = stats.rankdata(rowValues, axis=1, method="average").astype(float)
    referenceCentered = referenceRanks - referenceRanks.mean()
    rowCentered = rowRanks - rowRanks.mean(axis=1, keepdims=True)
    numerator = rowCentered @ referenceCentered
    denominator = np.sqrt(
        np.sum(rowCentered**2, axis=1) * np.sum(referenceCentered**2)
    )
    return numerator / denominator


def partialSpearmanCorrelation(
    firstValues: np.ndarray,
    secondValues: np.ndarray,
    phenotypeFrame: pd.DataFrame,
    adjustment: str,
) -> float:
    firstRanks = stats.rankdata(firstValues, method="average").astype(float)
    secondRanks = stats.rankdata(secondValues, method="average").astype(float)
    adjustedFirstRanks = residualizeValues(firstRanks, phenotypeFrame, adjustment)
    adjustedSecondRanks = residualizeValues(secondRanks, phenotypeFrame, adjustment)
    return float(stats.pearsonr(adjustedFirstRanks, adjustedSecondRanks).statistic)


def sphericalDistanceMatrix(unitSphereCoordinates: np.ndarray) -> np.ndarray:
    coordinateValues = np.asarray(unitSphereCoordinates, dtype=float)
    cosineSimilarity = np.clip(coordinateValues @ coordinateValues.T, -1.0, 1.0)
    angularDistances = np.arccos(cosineSimilarity)
    np.fill_diagonal(angularDistances, 0.0)
    return angularDistances


def buildHemisphereSpatialWeights(
    unitSphereCoordinates: np.ndarray,
    hemispheres: np.ndarray,
) -> sparse.csr_matrix:
    angularDistances = sphericalDistanceMatrix(unitSphereCoordinates)
    hemisphereValues = np.asarray(hemispheres, dtype=str)
    sameHemisphere = hemisphereValues[:, None] == hemisphereValues[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        weightMatrix = np.where(
            sameHemisphere & (angularDistances > 0),
            1.0 / angularDistances,
            0.0,
        )
    np.fill_diagonal(weightMatrix, 0.0)
    rowSums = weightMatrix.sum(axis=1, keepdims=True)
    weightMatrix = np.divide(
        weightMatrix,
        rowSums,
        out=np.zeros_like(weightMatrix),
        where=rowSums > 0,
    )
    return sparse.csr_matrix(weightMatrix)


def calculateMoranI(values: np.ndarray, spatialWeights: sparse.csr_matrix) -> float:
    valueArray = np.asarray(values, dtype=float)
    centeredValues = valueArray - np.mean(valueArray)
    denominator = float(centeredValues @ centeredValues)
    if denominator <= 0:
        return np.nan
    numerator = float(centeredValues @ spatialWeights.dot(centeredValues))
    return float(
        (len(centeredValues) / float(spatialWeights.sum())) * numerator / denominator
    )


def summarizeNullDistribution(
    observedStatistic: float, nullStatistics: np.ndarray, prefix: str
) -> dict[str, float]:
    nullValues = np.asarray(nullStatistics, dtype=float)
    nullValues = nullValues[np.isfinite(nullValues)]
    nullStandardDeviation = float(nullValues.std(ddof=1))
    absolutePercentile = float(
        (np.count_nonzero(np.abs(nullValues) <= abs(observedStatistic)) + 1)
        / (len(nullValues) + 1)
    )
    return {
        f"{prefix}Median": float(np.median(nullValues)),
        f"{prefix}Sd": nullStandardDeviation,
        f"{prefix}Q025": float(np.quantile(nullValues, 0.025)),
        f"{prefix}Q975": float(np.quantile(nullValues, 0.975)),
        f"{prefix}AbsolutePercentile": absolutePercentile,
        f"{prefix}Z": float(
            (observedStatistic - nullValues.mean()) / nullStandardDeviation
        )
        if nullStandardDeviation > 0
        else np.nan,
    }
