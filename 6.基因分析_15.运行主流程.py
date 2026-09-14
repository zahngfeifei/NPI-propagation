from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
KNOWN_ENVIRONMENT_PYTHON = (
    WORKSPACE_ROOT
    / "6.基于2"
    / "analysis_asd_geneset_spatial"
    / ".venv"
    / "Scripts"
    / "python.exe"
)
PYTHON_EXECUTABLE = (
    KNOWN_ENVIRONMENT_PYTHON
    if KNOWN_ENVIRONMENT_PYTHON.exists()
    else Path(sys.executable).resolve()
)
RUN_LOG_OUTPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程运行日志"
INPUT_DIRECTORY = PROJECT_ROOT / "ABIDE2_主流程必要输入"

MAINLINE_SCRIPT_NAMES = (
    "6.基因分析_1.ROI传播指标构建.py",
    "6.基因分析_2.ROI传播指标combat.py",
    "6.基因分析_3.耦合指标贡献图.py",
    "6.基因分析_4.构建ROI组间效应图.py",
    "6.基因分析_5.精确桥接验证.py",
    "6.基因分析_6.构建AHBA表型.py",
    "6.基因分析_7.AHBA表达矩阵_2mm.py",
    "6.基因分析_8.SFARI基因集.py",
    "6.基因分析_9.基因集表达评分.py",
    "6.基因分析_10.构建Schaefer400球面质心.py",
    "6.基因分析_11.空间零模型.py",
    "6.基因分析_12.空间相关.py",
    "6.基因分析_13.主流程结果验证.py",
)


def sha256File(inputPath: Path, blockSize: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with inputPath.open("rb") as inputFile:
        while block := inputFile.read(blockSize):
            digest.update(block)
    return digest.hexdigest()


def buildFileRecord(
    inputName: str,
    localPath: Path,
    role: str,
    recordedSource: str,
    officialSource: str | None = None,
) -> dict[str, Any]:
    if not localPath.exists():
        raise FileNotFoundError(f"外部输入不存在：{localPath}")
    record: dict[str, Any] = {
        "inputName": inputName,
        "inputType": "file",
        "localPath": str(localPath),
        "role": role,
        "recordedSource": recordedSource,
        "bytes": localPath.stat().st_size,
        "sha256": sha256File(localPath),
    }
    if officialSource:
        record["officialSource"] = officialSource
    return record


def buildDirectoryRecord(
    inputName: str,
    localPath: Path,
    role: str,
    recordedSource: str,
    pattern: str = "*",
    officialSource: str | None = None,
    hashFileContents: bool = True,
    displayedLocalPath: str | None = None,
) -> dict[str, Any]:
    if not localPath.exists():
        raise FileNotFoundError(f"外部输入目录不存在：{localPath}")
    matchedPaths = sorted(
        (inputPath for inputPath in localPath.rglob(pattern) if inputPath.is_file()),
        key=lambda inputPath: inputPath.as_posix(),
    )
    manifestDigest = hashlib.sha256()
    totalBytes = 0
    for matchedPath in matchedPaths:
        relativePath = matchedPath.relative_to(localPath).as_posix()
        fileBytes = matchedPath.stat().st_size
        totalBytes += fileBytes
        manifestDigest.update(relativePath.encode("utf-8"))
        manifestDigest.update(str(fileBytes).encode("ascii"))
        if hashFileContents:
            manifestDigest.update(sha256File(matchedPath).encode("ascii"))
    record: dict[str, Any] = {
        "inputName": inputName,
        "inputType": "directory",
        "localPath": displayedLocalPath or str(localPath),
        "role": role,
        "recordedSource": recordedSource,
        "filePattern": pattern,
        "fileCount": len(matchedPaths),
        "totalBytes": totalBytes,
        "directoryManifestSha256": manifestDigest.hexdigest(),
        "manifestIncludesFileContentHashes": hashFileContents,
    }
    if officialSource:
        record["officialSource"] = officialSource
    return record


def writeInputProvenance() -> Path:
    provenanceRecords = [
        buildDirectoryRecord(
            "roiPropagationArrays",
            PROJECT_ROOT / "ABIDE2_新结果1" / "curves_per_subject",
            "第1步计算每名被试的400个ROI早期传播斜率",
            "旧工程的ABIDE2步进传播分析输出，已复制到当前工作区",
            pattern="*_V_all.npy",
        ),
        buildFileRecord(
            "systemPropagationMetrics",
            PROJECT_ROOT / "ABIDE2_新结果1" / "EC_SEC_metrics_all_subjects.csv",
            "第1、2步的H1-H4系统斜率重建质控",
            "旧工程的ABIDE2步进传播分析输出，已复制到当前工作区",
        ),
        buildFileRecord(
            "strictFourSystemLongTable",
            PROJECT_ROOT / "ABIDE2_结果1" / "result1_input_long_table_strict.csv",
            "第3、5步重建原始四系统行为耦合指标",
            "旧工程的ABIDE2四系统梯度整合输出，已复制到当前工作区",
        ),
        buildFileRecord(
            "subjectCovariates",
            PROJECT_ROOT / "subject_info_for_stats.csv",
            "ComBat及ROI组间模型的组别、站点、年龄、性别和FIQ协变量",
            "旧工程的ABIDE2统计协变量冻结表，已复制到当前工作区",
        ),
        buildFileRecord(
            "roiDefinitions",
            INPUT_DIRECTORY / "roi-definitions.csv",
            "374个H1-H4 ROI的索引与系统归属",
            str(
                WORKSPACE_ROOT
                / "基因"
                / "ABIDE1_6.基因分析_2mm"
                / "ABIDE1_主流程必要输入"
                / "roi-definitions.csv"
            ),
        ),
        buildFileRecord(
            "schaeferAtlas2mm",
            INPUT_DIRECTORY
            / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
            "AHBA样本到Schaefer-400分区的2 mm空间匹配",
            "ABIDE1主流程已核验的同版Schaefer-400 2 mm图谱冻结副本",
            "https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal",
        ),
        buildFileRecord(
            "schaeferAtlasLabels",
            INPUT_DIRECTORY / "Schaefer2018_400Parcels_7Networks_order.txt",
            "Schaefer-400标签顺序",
            "ABIDE1主流程已核验的同版Schaefer-400标签冻结副本",
            "https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal",
        ),
        buildFileRecord(
            "sfariGeneScoring2026Q2",
            INPUT_DIRECTORY / "sfari-gene-scoring-2026-q2-original.csv",
            "预设SFARI高可信非综合征与综合征基因集",
            "SFARI Gene 2026 Q2冻结导出",
            "https://gene.sfari.org/database/gene-scoring/",
        ),
        buildDirectoryRecord(
            "allenHumanBrainAtlasMicroarray",
            Path.home() / "abagen-data" / "microarray",
            "六名成人供体的AHBA微阵列表达与样本注释",
            "abagen本地数据缓存",
            pattern="*",
            officialSource="https://human.brain-map.org/static/download",
            hashFileContents=False,
            displayedLocalPath=r"%USERPROFILE%\abagen-data\microarray",
        ),
    ]
    provenancePayload = {
        "generatedAt": datetime.now().isoformat(),
        "projectRoot": str(PROJECT_ROOT),
        "pythonExecutable": redactLogText(str(PYTHON_EXECUTABLE)),
        "mainlineOnly": True,
        "excludedAnalyses": [
            "matched random gene-set null",
            "donor and hemisphere sensitivity",
            "ranked-gene GSEA",
            "extended summary/report",
        ],
        "inputs": provenanceRecords,
    }
    outputPath = INPUT_DIRECTORY / "input-provenance.json"
    outputPath.write_text(
        json.dumps(provenancePayload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputPath


def redactLogText(rawLogText: str) -> str:
    """将本机用户目录替换为可复现且不暴露用户名的占位符。"""
    userProfilePath = str(Path.home())
    return rawLogText.replace(userProfilePath, "%USERPROFILE%")


def runScript(scriptPath: Path, logFile) -> None:
    stageMessage = f"[ABIDE2主流程] 运行 {scriptPath.name}"
    print(stageMessage, flush=True)
    logFile.write(stageMessage + "\n")
    processEnvironment = os.environ.copy()
    processEnvironment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [str(PYTHON_EXECUTABLE), str(scriptPath)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=processEnvironment,
    )
    if process.stdout is None:
        raise RuntimeError(f"无法读取脚本输出：{scriptPath}")
    for outputLine in process.stdout:
        consoleEncoding = sys.stdout.encoding or "utf-8"
        consoleSafeLine = outputLine.encode(
            consoleEncoding, errors="backslashreplace"
        ).decode(consoleEncoding)
        print(consoleSafeLine, end="", flush=True)
        logFile.write(redactLogText(outputLine))
        logFile.flush()
    returnCode = process.wait()
    if returnCode != 0:
        raise subprocess.CalledProcessError(
            returnCode,
            [str(PYTHON_EXECUTABLE), str(scriptPath)],
        )


def parseArguments() -> argparse.Namespace:
    argumentParser = argparse.ArgumentParser(description="运行 ABIDE2 2 mm 主流程。")
    argumentParser.add_argument(
        "--start-step",
        type=int,
        default=1,
        help="从指定步骤开始，默认从第1步开始。",
    )
    argumentParser.add_argument(
        "--end-step",
        type=int,
        default=len(MAINLINE_SCRIPT_NAMES),
        help="运行到指定步骤结束，默认运行到第13步。",
    )
    return argumentParser.parse_args()


def main() -> None:
    arguments = parseArguments()
    startStep = int(arguments.start_step)
    endStep = int(arguments.end_step)
    if startStep < 1 or endStep > len(MAINLINE_SCRIPT_NAMES) or startStep > endStep:
        raise ValueError(
            f"步骤范围无效：start={startStep}, end={endStep}, "
            f"允许范围为 1..{len(MAINLINE_SCRIPT_NAMES)}。"
        )
    selectedScriptNames = MAINLINE_SCRIPT_NAMES[startStep - 1 : endStep]
    scriptPaths = tuple(PROJECT_ROOT / scriptName for scriptName in selectedScriptNames)
    missingScripts = [scriptPath for scriptPath in scriptPaths if not scriptPath.exists()]
    if missingScripts:
        raise FileNotFoundError(f"缺少主流程脚本：{missingScripts}")
    RUN_LOG_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    provenancePath = writeInputProvenance()
    logName = (
        "ABIDE2-mainline-2mm-run.log"
        if startStep == 1 and endStep == len(MAINLINE_SCRIPT_NAMES)
        else f"ABIDE2-mainline-2mm-run-step{startStep}-to-{endStep}.log"
    )
    logPath = RUN_LOG_OUTPUT_DIRECTORY / logName
    with logPath.open("w", encoding="utf-8") as logFile:
        logFile.write(f"开始时间：{datetime.now().isoformat()}\n")
        logFile.write(f"Python：{redactLogText(str(PYTHON_EXECUTABLE))}\n")
        logFile.write(f"输入来源清单：{provenancePath}\n")
        logFile.write(f"运行步骤：{startStep}..{endStep}\n")
        for scriptPath in scriptPaths:
            runScript(scriptPath, logFile)
        logFile.write(f"完成时间：{datetime.now().isoformat()}\n")
    print(f"[ABIDE2主流程] 全部完成，日志：{logPath}", flush=True)


if __name__ == "__main__":
    main()
