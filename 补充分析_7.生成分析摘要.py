from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

COMMON_SCRIPT_PATH = Path(__file__).resolve().parent / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location("analysis_common", COMMON_SCRIPT_PATH)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (
    COHORT_NAME,
    GENE_SET_OUTPUT_DIRECTORY,
    REPORT_OUTPUT_DIRECTORY,
    SUMMARY_OUTPUT_DIRECTORY,
    VALIDATION_OUTPUT_DIRECTORY,
)


GENE_SET_LABELS = {
    "SFARI_high_confidence_nonsyndromic": "SFARI 高置信度非综合征",
    "SFARI_syndromic": "SFARI 综合征",
}
def formatValue(value: float, digits: int = 4) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.{digits}f}"


def main() -> None:
    geneSetSummaryFrame = pd.read_csv(
        GENE_SET_OUTPUT_DIRECTORY / "gene-set-summary.csv"
    )
    resultFrame = pd.read_csv(
        SUMMARY_OUTPUT_DIRECTORY / "all-correlation-results.csv"
    )
    primaryRow = resultFrame.loc[resultFrame["analysisRole"].eq("primary")].iloc[0]
    baselineRow = resultFrame.loc[resultFrame["analysisRole"].eq("baseline")].iloc[0]
    validationRow = resultFrame.loc[
        resultFrame["analysisRole"].eq("gene_set_validation")
    ].iloc[0]
    validationPayload = json.loads(
        (VALIDATION_OUTPUT_DIRECTORY / "validation.json").read_text(encoding="utf-8")
    )
    primaryGeneSetSummary = geneSetSummaryFrame.loc[
        geneSetSummaryFrame["geneSetName"].eq(
            "SFARI_high_confidence_nonsyndromic"
        )
    ].iloc[0]
    coverageLines = []
    for resultRow in geneSetSummaryFrame.itertuples(index=False):
        coverageLines.append(
            f"- {GENE_SET_LABELS[resultRow.geneSetName]}：{resultRow.rawGeneCount} 个官网基因，"
            f"{resultRow.effectiveGeneCount} 个 AHBA 有效基因，"
            f"覆盖率 {resultRow.coverageFraction:.1%}。"
        )

    primarySupported = primaryRow["resultStatus"] == "primary_supported"
    conclusionText = (
        "主要假设得到支持：该负向对应同时通过直接生成的 BrainSMASH 空间零模型和表达性质匹配基因集零模型。"
        if primarySupported
        else "主要假设未得到完整支持：至少一个关键零模型或预设QC条件未通过。"
    )
    validationInterpretation = {
        "validation_supported": "验证分析同时通过两个零模型。",
        "partial_validation": "验证分析仅通过一个零模型，属于部分验证。",
        "not_validated": "验证分析未通过两个零模型。",
    }[validationRow["resultStatus"]]

    primaryTable = pd.DataFrame(
        [
            ("分析组合", "系统残差 × SFARI高置信度非综合征 × Mean-Z"),
            ("nROI", "374"),
            (
                "有效基因",
                f"{int(primaryGeneSetSummary['effectiveGeneCount'])}/"
                f"{int(primaryGeneSetSummary['rawGeneCount'])}",
            ),
            ("Spearman ρ", formatValue(primaryRow["spearmanR"])),
            ("标准partial Spearman敏感性ρ", formatValue(primaryRow["partialSpearmanR"])),
            ("双侧空间P", formatValue(primaryRow["pSpatial"])),
            ("双侧匹配基因集P", formatValue(primaryRow["pGeneSet"])),
            ("conjunction P", formatValue(primaryRow["pConjunction"])),
            (
                "跨两套SFARI空间q（补充）",
                formatValue(primaryRow["qSpatialAcrossTwoSfariSets"]),
            ),
            ("空间零模型次数", str(int(primaryRow["spatialNullCount"]))),
            ("匹配零模型次数", str(int(primaryRow["matchedNullCount"]))),
            (
                "单供体方向一致",
                f"{int(primaryRow['singleDonorSignAgreementCount'])}/"
                f"{int(primaryRow['singleDonorCorrelationCount'])}",
            ),
            (
                "LODO ρ范围",
                f"[{formatValue(primaryRow['lodoMinimumRho'])}, "
                f"{formatValue(primaryRow['lodoMaximumRho'])}]",
            ),
            ("左半球ρ", formatValue(primaryRow["leftHemisphereRho"])),
            ("右半球ρ", formatValue(primaryRow["rightHemisphereRho"])),
            ("空间QC", "PASS" if primaryRow["spatialQcPassed"] else "FAIL"),
            ("匹配QC", "PASS" if primaryRow["matchingQcPassed"] else "FAIL"),
            ("主要判定", str(primaryRow["resultStatus"])),
        ],
        columns=["指标", "结果"],
    )
    baselineTable = pd.DataFrame(
        [
            {
                "Spearman ρ": formatValue(baselineRow["spearmanR"]),
                "空间P": formatValue(baselineRow["pSpatial"]),
                "匹配基因集P": formatValue(baselineRow["pGeneSet"]),
                "用途": "未控制H1–H4系统时的基准描述",
            }
        ]
    )
    validationTable = pd.DataFrame(
        [
            {
                "基因集": GENE_SET_LABELS[validationRow["geneSetName"]],
                "Spearman ρ": formatValue(validationRow["spearmanR"]),
                "空间P": formatValue(validationRow["pSpatial"]),
                "匹配基因集P": formatValue(validationRow["pGeneSet"]),
                "状态": validationRow["resultStatus"],
            }
        ]
    )

    summaryText = f"""# {COHORT_NAME} 两套 SFARI 基因集相关性独立重分析（2 mm）

> 本文件主要汇总基因相关主分析和补充分析。包含样本筛选、ComBat、ROI
> 表型构建及桥接 QC 的全流程报告见
> `{COHORT_NAME}_从表型到基因分析完整结果摘要.md`。

## 冻结的分析层级

- 唯一主要检验：H1–H4系统残差图 × SFARI高置信度非综合征 × Mean-Z。
- 基准分析：原始beta图 × SFARI高置信度非综合征 × Mean-Z。
- 基因集验证：系统残差图 × SFARI综合征 × Mean-Z。
- 排序基因GSEA是未进行空间校准的探索分析。
- AHBA表达使用Schaefer-2018 400分区、7网络、2 mm体积模板构建。

## 基因集覆盖

{chr(10).join(coverageLines)}

## 主要检验

{primaryTable.to_markdown(index=False)}

{conclusionText}

主要检验是分析前唯一指定的假设，因此不与基准或验证结果共同进行FDR。`qSpatialAcrossTwoSfariSets`仅作为补充透明度指标。完整主张要求空间P和匹配基因集P均小于0.05，联合指标定义为两者最大值。

## 基准分析

{baselineTable.to_markdown(index=False)}

基准分析只展示未控制H1–H4系统均值差异时的总体对应，不作为主要显著性结论。

## 基因集验证

{validationTable.to_markdown(index=False)}

{validationInterpretation}

## QC与解释边界

- 每个表型直接生成10,000张左右半球分离的BrainSMASH替代图；`system_residual`不再由raw-beta替代图二次残差化得到。
- 每个基因先在完整400个Schaefer脑区中标准化，再提取374个H1–H4分析脑区并在基因集内求平均。
- 匹配背景排除两套被检验SFARI基因的并集；匹配空间特征使用与BrainSMASH相同球面几何构建的Moran's I。
- 单供体、LODO和半球结果是稳定性指标，不参与显著性判定，也不影响代码验收。
- GSEA标记为`exploratory_not_spatially_calibrated`，不作为空间显著性证据。
- 自动代码与数据验收：{'PASS' if validationPayload['overallPassed'] else 'FAIL'}。
- 该分析检验正常成人AHBA表达脑图与组水平影像表型的空间对应，不支持个体层面、疾病表达变化方向或因果结论。
"""
    REPORT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (REPORT_OUTPUT_DIRECTORY / "analysis-summary.md").write_text(
        summaryText, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
