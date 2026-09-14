from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
commonModuleSpec = importlib.util.spec_from_file_location(
    "analysis_common", COMMON_SCRIPT_PATH
)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (  # noqa: E402
    AGGREGATE_EXPRESSION_PATH,
    ANALYSIS_CONFIG,
    GENE_SET_NAMES,
    GENE_SET_OUTPUT_DIRECTORY,
    PHENOTYPE_PATH,
    PRIMARY_HYPOTHESIS,
    ROI_CENTROID_PATH,
    SCORE_OUTPUT_DIRECTORY,
    SPATIAL_CORRELATION_OUTPUT_DIRECTORY,
    SPATIAL_NULL_OUTPUT_DIRECTORY,
    VALIDATION_OUTPUT_DIRECTORY,
    writeJson,
)


def main() -> None:
    expectedSelectedRois = int(
        ANALYSIS_CONFIG["project"]["expectedSelectedRois"]
    )
    expectedFullRois = int(ANALYSIS_CONFIG["project"]["expectedFullRois"])
    expectedNullCount = int(
        ANALYSIS_CONFIG["statistics"]["spatialNullCount"]
    )
    validationRows: list[dict[str, object]] = []

    def addCheck(checkName: str, passed: bool, detail: str) -> None:
        validationRows.append(
            {"checkName": checkName, "passed": bool(passed), "detail": detail}
        )

    requiredPaths = {
        "phenotype": PHENOTYPE_PATH,
        "aggregateExpression": AGGREGATE_EXPRESSION_PATH,
        "geneSetSummary": GENE_SET_OUTPUT_DIRECTORY / "gene-set-summary.csv",
        "selectedScores": SCORE_OUTPUT_DIRECTORY
        / "gene-set-scores-selected-374.parquet",
        "centroids": ROI_CENTROID_PATH,
        "spatialNullQc": SPATIAL_NULL_OUTPUT_DIRECTORY / "spatial-null-qc.csv",
        "spatialSurrogates": SPATIAL_NULL_OUTPUT_DIRECTORY
        / "phenotype-spatial-surrogates.npz",
        "spatialResults": SPATIAL_CORRELATION_OUTPUT_DIRECTORY
        / "spatial-correlation-results.csv",
    }
    for inputName, inputPath in requiredPaths.items():
        addCheck(
            f"requiredPath::{inputName}",
            inputPath.exists(),
            str(inputPath),
        )
    if not all(inputPath.exists() for inputPath in requiredPaths.values()):
        VALIDATION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        validationFrame = pd.DataFrame(validationRows)
        validationFrame.to_csv(
            VALIDATION_OUTPUT_DIRECTORY / "mainline-validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        raise FileNotFoundError("主流程输出不完整，详见验证表。")

    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    phenotypeRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).to_numpy()
    addCheck(
        "phenotypeRoiCount",
        len(phenotypeFrame) == expectedSelectedRois,
        f"observed={len(phenotypeFrame)}, expected={expectedSelectedRois}",
    )
    addCheck(
        "phenotypeRoiIdsUnique",
        len(np.unique(phenotypeRoiIds)) == len(phenotypeRoiIds),
        f"unique={len(np.unique(phenotypeRoiIds))}",
    )
    maximumSystemResidualMean = float(
        phenotypeFrame.groupby("system", observed=True)["system_residual"]
        .mean()
        .abs()
        .max()
    )
    addCheck(
        "systemResidualCentered",
        maximumSystemResidualMean < 1e-10,
        f"maximumAbsoluteSystemMean={maximumSystemResidualMean:.3e}",
    )

    expressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH)
    expressionRoiIds = expressionFrame["ROI_ID_1based"].astype(int).to_numpy()
    addCheck(
        "expressionRoiCoverage",
        np.array_equal(np.sort(expressionRoiIds), np.arange(1, expectedFullRois + 1)),
        f"rowCount={len(expressionFrame)}, expected={expectedFullRois}",
    )
    addCheck(
        "expressionGeneCountPositive",
        len(expressionFrame.columns) > 1,
        f"geneCount={len(expressionFrame.columns) - 1}",
    )

    geneSetSummaryFrame = pd.read_csv(requiredPaths["geneSetSummary"])
    observedGeneSetNames = set(geneSetSummaryFrame["geneSetName"].astype(str))
    addCheck(
        "geneSetNames",
        observedGeneSetNames == set(GENE_SET_NAMES),
        f"observed={sorted(observedGeneSetNames)}",
    )

    scoreFrame = pd.read_parquet(requiredPaths["selectedScores"])
    expectedScoreRows = expectedSelectedRois * len(GENE_SET_NAMES)
    addCheck(
        "selectedScoreRowCount",
        len(scoreFrame) == expectedScoreRows,
        f"observed={len(scoreFrame)}, expected={expectedScoreRows}",
    )
    addCheck(
        "selectedScoresFinite",
        bool(np.isfinite(scoreFrame["score"].to_numpy(dtype=float)).all()),
        "all selected mean-z scores must be finite",
    )

    centroidFrame = pd.read_csv(ROI_CENTROID_PATH)
    centroidRoiIds = centroidFrame["ROI_ID_1based"].astype(int).to_numpy()
    coordinateValues = centroidFrame[["sphereX", "sphereY", "sphereZ"]].to_numpy(
        dtype=float
    )
    maximumNormError = float(
        np.max(np.abs(np.linalg.norm(coordinateValues, axis=1) - 1.0))
    )
    addCheck(
        "centroidRoiAlignment",
        np.array_equal(centroidRoiIds, phenotypeRoiIds),
        "centroid ROI order must exactly match phenotype order",
    )
    addCheck(
        "centroidUnitNorm",
        maximumNormError < 1e-12,
        f"maximumUnitNormError={maximumNormError:.3e}",
    )

    spatialQcFrame = pd.read_csv(requiredPaths["spatialNullQc"])
    addCheck(
        "spatialQcPassed",
        bool(spatialQcFrame["spatialQcPassed"].astype(bool).all()),
        f"phenotypeCount={len(spatialQcFrame)}",
    )
    with np.load(requiredPaths["spatialSurrogates"]) as surrogateArchive:
        surrogateShapes = {
            surrogateKey: list(surrogateArchive[surrogateKey].shape)
            for surrogateKey in surrogateArchive.files
        }
    expectedSurrogateShape = [expectedNullCount, expectedSelectedRois]
    addCheck(
        "spatialSurrogateShapes",
        bool(surrogateShapes)
        and all(shape == expectedSurrogateShape for shape in surrogateShapes.values()),
        f"observed={surrogateShapes}, expectedEach={expectedSurrogateShape}",
    )

    spatialResultFrame = pd.read_csv(requiredPaths["spatialResults"])
    expectedResultCount = len(GENE_SET_NAMES) * 2
    addCheck(
        "spatialResultCount",
        len(spatialResultFrame) == expectedResultCount,
        f"observed={len(spatialResultFrame)}, expected={expectedResultCount}",
    )
    addCheck(
        "spatialPValuesValid",
        bool(spatialResultFrame["pSpatial"].between(0.0, 1.0).all()),
        "all empirical spatial P values must lie in [0, 1]",
    )
    primaryMask = np.ones(len(spatialResultFrame), dtype=bool)
    for columnName, expectedValue in PRIMARY_HYPOTHESIS.items():
        primaryMask &= spatialResultFrame[columnName].astype(str).eq(
            str(expectedValue)
        )
    addCheck(
        "primaryHypothesisPresent",
        int(primaryMask.sum()) == 1,
        f"matchingRows={int(primaryMask.sum())}",
    )

    validationFrame = pd.DataFrame(validationRows)
    VALIDATION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    validationFrame.to_csv(
        VALIDATION_OUTPUT_DIRECTORY / "mainline-validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validationPayload = {
        "allChecksPassed": bool(validationFrame["passed"].all()),
        "checkCount": len(validationFrame),
        "passedCheckCount": int(validationFrame["passed"].sum()),
        "failedChecks": validationFrame.loc[
            ~validationFrame["passed"], "checkName"
        ].tolist(),
        "primaryResult": spatialResultFrame.loc[primaryMask].to_dict("records"),
    }
    writeJson(
        VALIDATION_OUTPUT_DIRECTORY / "mainline-validation.json",
        validationPayload,
    )
    if not validationPayload["allChecksPassed"]:
        raise RuntimeError(
            f"主流程验证失败：{validationPayload['failedChecks']}"
        )
    print(
        f"主流程验证通过：{validationPayload['passedCheckCount']}/"
        f"{validationPayload['checkCount']} 项。"
    )


if __name__ == "__main__":
    main()
