from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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

from analysis_common import (
    AGGREGATE_EXPRESSION_PATH,
    GENE_SET_NAMES,
    GENE_SET_OUTPUT_DIRECTORY,
    PHENOTYPE_PATH,
    SCORE_OUTPUT_DIRECTORY,
    normalizeExpressionColumns,
    validateSystemResidualDefinition,
    zscoreColumns,
)


def scoresToLongFrame(scoreFrame: pd.DataFrame) -> pd.DataFrame:
    scoreRows = []
    for geneSetName in scoreFrame.columns:
        scoreRows.extend(
            {
                "ROI_ID_1based": int(roiId),
                "geneSetName": geneSetName,
                "scoreMethod": "mean_z",
                "score": float(scoreValue),
            }
            for roiId, scoreValue in scoreFrame[geneSetName].items()
        )
    return pd.DataFrame(scoreRows)


def main() -> None:
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    validateSystemResidualDefinition(phenotypeFrame)
    selectedRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).tolist()
    expressionFrame = pd.read_parquet(AGGREGATE_EXPRESSION_PATH).set_index(
        "ROI_ID_1based"
    )
    expressionFrame = normalizeExpressionColumns(expressionFrame)
    geneSetFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "normalized-gene-sets.csv"
    )
    effectiveGeneSets = {
        geneSetName: sorted(
            groupFrame.loc[
                groupFrame["presentInAhbaExpression"], "geneSymbol"
            ].tolist()
        )
        for geneSetName, groupFrame in geneSetFrame.groupby(
            "geneSetName", sort=True
        )
    }
    if set(effectiveGeneSets) != set(GENE_SET_NAMES):
        raise RuntimeError("评分输入未严格包含两套预设 SFARI 基因集。")

    standardizedFullExpression = zscoreColumns(expressionFrame)
    meanZFullScores = pd.DataFrame(
        {
            geneSetName: standardizedFullExpression[geneSymbols].mean(axis=1)
            for geneSetName, geneSymbols in effectiveGeneSets.items()
        },
        index=standardizedFullExpression.index,
    )
    fullScoreFrame = scoresToLongFrame(meanZFullScores)
    selectedScoreFrame = scoresToLongFrame(
        meanZFullScores.loc[selectedRoiIds]
    )

    SCORE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fullScoreFrame.to_csv(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-400.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fullScoreFrame.to_parquet(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-400.parquet", index=False
    )
    selectedScoreFrame.to_csv(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-selected-374.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selectedScoreFrame.to_parquet(
        SCORE_OUTPUT_DIRECTORY / "gene-set-scores-selected-374.parquet",
        index=False,
    )

if __name__ == "__main__":
    main()
