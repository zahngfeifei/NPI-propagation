from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import pandas as pd

COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
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
    SFARI_GENE_SCORING_PATH,
    loadConfig,
    sha256File,
    writeJson,
)


def normalizeSymbols(rawSymbols: pd.Series) -> list[str]:
    normalizedSymbols = rawSymbols.dropna().astype(str).str.strip().str.upper()
    return sorted(normalizedSymbols.loc[normalizedSymbols.ne("")].drop_duplicates().tolist())


def main() -> None:
    config = loadConfig()
    sfariPath = SFARI_GENE_SCORING_PATH
    expressionPath = AGGREGATE_EXPRESSION_PATH
    sfariFrame = pd.read_csv(sfariPath)
    requiredColumns = {"gene-symbol", "gene-score", "syndromic"}
    if not requiredColumns.issubset(sfariFrame.columns):
        raise ValueError(f"Missing SFARI columns: {sorted(requiredColumns - set(sfariFrame.columns))}")
    expressionFrame = pd.read_parquet(expressionPath)
    expressionGenes = {
        str(columnName).upper()
        for columnName in expressionFrame.columns
        if str(columnName) != "ROI_ID_1based"
    }
    geneSets = {
        "SFARI_high_confidence_nonsyndromic": normalizeSymbols(
            sfariFrame.loc[
                sfariFrame["gene-score"].eq(1) & sfariFrame["syndromic"].eq(0),
                "gene-symbol",
            ]
        ),
        "SFARI_syndromic": normalizeSymbols(
            sfariFrame.loc[
                sfariFrame["syndromic"].eq(1) & sfariFrame["gene-score"].notna(),
                "gene-symbol",
            ]
        ),
    }
    if set(geneSets) != set(GENE_SET_NAMES):
        raise RuntimeError("Unexpected SFARI gene-set identifiers.")
    if set(geneSets[GENE_SET_NAMES[0]]) & set(geneSets[GENE_SET_NAMES[1]]):
        raise RuntimeError("The two locked SFARI sets must be mutually exclusive.")

    longRows = []
    summaryRows = []
    for geneSetName, geneSymbols in geneSets.items():
        expectedCount = int(config["geneSets"][geneSetName]["expectedRawGeneCount"])
        if len(geneSymbols) != expectedCount:
            raise ValueError(f"{geneSetName} has {len(geneSymbols)} genes, expected {expectedCount}.")
        effectiveGenes = sorted(set(geneSymbols) & expressionGenes)
        missingGenes = sorted(set(geneSymbols) - expressionGenes)
        if len(effectiveGenes) < int(config["project"]["minimumGenesPerSet"]):
            raise ValueError(f"Insufficient AHBA coverage for {geneSetName}.")
        longRows.extend(
            {
                "geneSetName": geneSetName,
                "geneSymbol": geneSymbol,
                "presentInAhbaExpression": geneSymbol in expressionGenes,
            }
            for geneSymbol in geneSymbols
        )
        summaryRows.append(
            {
                "geneSetName": geneSetName,
                "rawGeneCount": len(geneSymbols),
                "effectiveGeneCount": len(effectiveGenes),
                "missingGeneCount": len(missingGenes),
                "coverageFraction": len(effectiveGenes) / len(geneSymbols),
                "missingGenes": ";".join(missingGenes),
                "selectionRule": config["geneSets"][geneSetName]["selectionRule"],
            }
        )
    outputDirectory = GENE_SET_OUTPUT_DIRECTORY
    outputDirectory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(longRows).to_csv(
        outputDirectory / "normalized-gene-sets.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summaryRows).to_csv(
        outputDirectory / "gene-set-summary.csv", index=False, encoding="utf-8-sig"
    )
    writeJson(
        outputDirectory / "gene-set-provenance.json",
        {
            "createdAtUtc": datetime.now(timezone.utc).isoformat(),
            "sfariRelease": "2026 Q2",
            "sourceUrl": "https://gene.sfari.org/database/gene-scoring/",
            "sfariCsvSha256": sha256File(sfariPath),
            "aggregateExpressionSha256": sha256File(expressionPath),
            "mutuallyExclusive": True,
        },
    )


if __name__ == "__main__":
    main()
