from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import gseapy as gp
import numpy as np
import pandas as pd
from scipy import stats

COMMON_SCRIPT_PATH = Path(__file__).resolve().parent / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    AGGREGATE_EXPRESSION_PATH,
    GENE_SET_NAMES,
    GENE_SET_OUTPUT_DIRECTORY,
    PHENOTYPE_SPECS,
    PHENOTYPE_PATH,
    RANKED_GSEA_OUTPUT_DIRECTORY,
    loadConfig,
    normalizeExpressionColumns,
    residualizeRows,
)


def vectorizedSpearman(expressionValues: np.ndarray, phenotypeValues: np.ndarray) -> np.ndarray:
    expressionRanks = stats.rankdata(expressionValues, axis=0, method="average")
    phenotypeRanks = stats.rankdata(phenotypeValues, method="average")
    expressionCentered = expressionRanks - expressionRanks.mean(axis=0, keepdims=True)
    phenotypeCentered = phenotypeRanks - phenotypeRanks.mean()
    return (phenotypeCentered @ expressionCentered) / np.sqrt(
        np.sum(phenotypeCentered**2) * np.sum(expressionCentered**2, axis=0)
    )


def main() -> None:
    config = loadConfig()
    expressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH).set_index(
        "ROI_ID_1based"
    )
    expressionFrame = normalizeExpressionColumns(expressionFrame)
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    selectedExpressionFrame = expressionFrame.loc[
        phenotypeFrame["ROI_ID_1based"].astype(int).tolist()
    ]
    geneSetFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "normalized-gene-sets.csv"
    )
    effectiveGeneSets = {
        geneSetName: sorted(
            groupFrame.loc[groupFrame["presentInAhbaExpression"], "geneSymbol"].tolist()
        )
        for geneSetName, groupFrame in geneSetFrame.groupby("geneSetName", sort=True)
    }
    if set(effectiveGeneSets) != set(GENE_SET_NAMES):
        raise RuntimeError("Unexpected gene sets in ranked-gene analysis.")
    rankingRows = []
    gseaRows = []
    for phenotypeSpecification in PHENOTYPE_SPECS:
        phenotypeName = phenotypeSpecification["phenotypeName"]
        expressionAdjustment = phenotypeSpecification["scoreAdjustment"]
        expressionValues = selectedExpressionFrame.to_numpy(dtype=float)
        if expressionAdjustment != "none":
            expressionValues = residualizeRows(
                expressionValues.T,
                phenotypeFrame,
                expressionAdjustment,
            ).T
        geneCorrelations = vectorizedSpearman(
            expressionValues,
            phenotypeFrame[phenotypeName].to_numpy(dtype=float),
        )
        rankingFrame = pd.DataFrame(
            {"geneSymbol": selectedExpressionFrame.columns, "spearmanR": geneCorrelations}
        ).sort_values("spearmanR", ascending=False)
        rankingFrame.insert(0, "phenotypeName", phenotypeName)
        rankingFrame.insert(1, "expressionAdjustment", expressionAdjustment)
        rankingRows.append(rankingFrame)
        prerankResult = gp.prerank(
            rnk=rankingFrame[["geneSymbol", "spearmanR"]],
            gene_sets=effectiveGeneSets,
            min_size=10,
            max_size=500,
            permutation_num=int(config["statistics"]["rankedGenePermutationCount"]),
            seed=int(config["project"]["seed"]),
            outdir=None,
            verbose=False,
        )
        resultFrame = prerankResult.res2d.copy()
        resultFrame.insert(0, "phenotypeName", phenotypeName)
        resultFrame["expressionAdjustment"] = expressionAdjustment
        resultFrame["calibrationStatus"] = "exploratory_not_spatially_calibrated"
        gseaRows.append(resultFrame)
    outputDirectory = RANKED_GSEA_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    pd.concat(rankingRows, ignore_index=True).to_parquet(
        outputDirectory / "gene-spatial-rankings.parquet", index=False
    )
    pd.concat(gseaRows, ignore_index=True).to_csv(
        outputDirectory / "ranked-gene-gsea-results.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
