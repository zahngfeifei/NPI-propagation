from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SUPPLEMENT_ROOT = Path(__file__).resolve().parent
COHORT_NAME = SUPPLEMENT_ROOT.name.removesuffix("补充分析")
if COHORT_NAME not in {"ABIDE1", "ABIDE2"}:
    raise RuntimeError(f"无法从补充分析目录识别队列：{SUPPLEMENT_ROOT}")

MAIN_PROJECT_ROOT = (
    SUPPLEMENT_ROOT.parent / f"{COHORT_NAME}_6.基因分析_2mm"
)
MAIN_COMMON_SCRIPT_PATH = MAIN_PROJECT_ROOT / "6.基因分析_14.公共配置与函数.py"
mainModuleSpec = importlib.util.spec_from_file_location(
    f"{COHORT_NAME.lower()}_main_analysis_common",
    MAIN_COMMON_SCRIPT_PATH,
)
if mainModuleSpec is None or mainModuleSpec.loader is None:
    raise RuntimeError(f"无法加载主线公共配置：{MAIN_COMMON_SCRIPT_PATH}")
mainAnalysisCommon = importlib.util.module_from_spec(mainModuleSpec)
sys.modules[mainModuleSpec.name] = mainAnalysisCommon
mainModuleSpec.loader.exec_module(mainAnalysisCommon)

for exportedName in dir(mainAnalysisCommon):
    if not exportedName.startswith("_"):
        globals()[exportedName] = getattr(mainAnalysisCommon, exportedName)

# 主线数据只读；所有补充结果都重定向到当前补充分析目录。
PROJECT_ROOT = SUPPLEMENT_ROOT
MATCHED_NULL_OUTPUT_DIRECTORY = (
    SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果1_匹配基因集零模型"
)
SENSITIVITY_OUTPUT_DIRECTORY = (
    SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果2_供体与半球敏感性"
)
SUMMARY_OUTPUT_DIRECTORY = SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果3_汇总统计"
RANKED_GSEA_OUTPUT_DIRECTORY = SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果4_排名基因GSEA"
VALIDATION_OUTPUT_DIRECTORY = SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果5_结果验证"
REPORT_OUTPUT_DIRECTORY = SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充结果6_分析摘要"
RUN_LOG_OUTPUT_DIRECTORY = SUPPLEMENT_ROOT / f"{COHORT_NAME}_补充分析运行日志"

