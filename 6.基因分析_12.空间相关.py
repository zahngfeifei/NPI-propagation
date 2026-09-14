from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    ALPHA,
    EMPIRICAL_P_FORMULA,
    GENE_SET_NAMES,
    PHENOTYPE_SPECS,
    PHENOTYPE_PATH,
    SCORE_OUTPUT_DIRECTORY,
    SPATIAL_CORRELATION_OUTPUT_DIRECTORY,
    SPATIAL_NULL_OUTPUT_DIRECTORY,
    TEST_TAIL,
    empiricalTwoSidedP,
    loadConfig,
    partialSpearmanCorrelation,
    residualizeRows,
    residualizeValues,
    spearmanCorrelation,
    summarizeNullDistribution,
    validateSystemResidualDefinition,
    vectorizedRowCorrelations,
)


def main() -> None:
    config = loadConfig()
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    validateSystemResidualDefinition(phenotypeFrame)
    scoreFrame = pd.read_parquet(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-selected-374.parquet"
    )
    spatialNullDirectory = SPATIAL_NULL_OUTPUT_DIRECTORY
    surrogateArchive = np.load(
        spatialNullDirectory / "phenotype-spatial-surrogates.npz"
    )
    spatialQcFrame = pd.read_csv(spatialNullDirectory / "spatial-null-qc.csv").set_index(
        "phenotypeName"
    )
    expectedNullCount = int(config["statistics"]["spatialNullCount"])
    resultRows = []
    nullRows = []
    for phenotypeSpecification in PHENOTYPE_SPECS:
        phenotypeName = phenotypeSpecification["phenotypeName"]
        scoreAdjustment = phenotypeSpecification["scoreAdjustment"]
        surrogateAdjustment = phenotypeSpecification["surrogateAdjustment"]
        phenotypeValues = phenotypeFrame[phenotypeName].to_numpy(dtype=float)
        surrogateKey = phenotypeSpecification["surrogateKey"]
        if surrogateKey not in surrogateArchive.files:
            raise RuntimeError(f"Direct spatial surrogates are missing: {surrogateKey}")
        surrogateValues = residualizeRows(
            surrogateArchive[surrogateKey], phenotypeFrame, surrogateAdjustment
        )
        if len(surrogateValues) != expectedNullCount:
            raise ValueError("Unexpected BrainSMASH surrogate count.")
        phenotypeQc = spatialQcFrame.loc[phenotypeName]
        if not bool(phenotypeQc["spatialQcPassed"]):
            raise RuntimeError(f"Spatial QC failed for {phenotypeName}.")
        for (geneSetName, scoreMethod), groupFrame in scoreFrame.groupby(
            ["geneSetName", "scoreMethod"], sort=True
        ):
            if geneSetName not in GENE_SET_NAMES:
                raise RuntimeError(f"Unexpected gene set in score file: {geneSetName}")
            rawOrderedScores = groupFrame.set_index("ROI_ID_1based").loc[
                phenotypeFrame["ROI_ID_1based"], "score"
            ].to_numpy(dtype=float)
            orderedScores = residualizeValues(
                rawOrderedScores, phenotypeFrame, scoreAdjustment
            )
            observedCorrelation = spearmanCorrelation(orderedScores, phenotypeValues)
            naiveP = float(stats.spearmanr(orderedScores, phenotypeValues).pvalue)
            nullCorrelations = vectorizedRowCorrelations(orderedScores, surrogateValues)
            spatialP = empiricalTwoSidedP(observedCorrelation, nullCorrelations)
            partialSpearmanR = np.nan
            if phenotypeName == "system_residual":
                partialSpearmanR = partialSpearmanCorrelation(
                    rawOrderedScores,
                    phenotypeFrame["raw_beta"].to_numpy(dtype=float),
                    phenotypeFrame,
                    "system",
                )
            resultRows.append(
                {
                    "phenotypeName": phenotypeName,
                    "phenotypeRole": phenotypeSpecification["phenotypeRole"],
                    "geneSetName": geneSetName,
                    "scoreMethod": scoreMethod,
                    "spearmanR": observedCorrelation,
                    "partialSpearmanR": partialSpearmanR,
                    "pNaiveDescriptive": naiveP,
                    "pSpatial": spatialP,
                    "spatialNullCount": len(nullCorrelations),
                    "scoreAdjustment": scoreAdjustment,
                    "surrogateKey": surrogateKey,
                    "surrogateAdjustment": surrogateAdjustment,
                    "testTail": TEST_TAIL,
                    "pFormula": EMPIRICAL_P_FORMULA,
                    "alpha": ALPHA,
                    "observedMoranI": float(phenotypeQc["observedMoranI"]),
                    "nullMoranIMean": float(phenotypeQc["nullMoranIMean"]),
                    "nullMoranISd": float(phenotypeQc["nullMoranISd"]),
                    "nullMoranIQ025": float(phenotypeQc["nullMoranIQ025"]),
                    "nullMoranIQ975": float(phenotypeQc["nullMoranIQ975"]),
                    "variogramFitError": float(phenotypeQc["variogramFitError"]),
                    "variogramQcPassed": bool(phenotypeQc["variogramQcPassed"]),
                    "spatialQcPassed": bool(phenotypeQc["spatialQcPassed"]),
                    "distanceType": str(phenotypeQc["distanceType"]),
                    "hemisphereSeparated": bool(
                        phenotypeQc["hemisphereSeparated"]
                    ),
                    **summarizeNullDistribution(
                        observedCorrelation, nullCorrelations, "spatialNull"
                    ),
                }
            )
            nullRows.extend(
                {
                    "phenotypeName": phenotypeName,
                    "geneSetName": geneSetName,
                    "scoreMethod": scoreMethod,
                    "nullIndex": nullIndex,
                    "nullSpearmanR": float(nullCorrelation),
                }
                for nullIndex, nullCorrelation in enumerate(nullCorrelations)
            )
    outputDirectory = SPATIAL_CORRELATION_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultRows).to_csv(
        outputDirectory / "spatial-correlation-results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(nullRows).to_parquet(
        outputDirectory / "spatial-null-correlations.parquet", index=False
    )


if __name__ == "__main__":
    main()
