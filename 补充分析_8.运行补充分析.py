from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


SUPPLEMENT_ROOT = Path(__file__).resolve().parent
COMMON_SCRIPT_PATH = SUPPLEMENT_ROOT / "补充分析_1.公共配置与函数.py"
commonModuleSpec = importlib.util.spec_from_file_location(
    "supplement_analysis_common_runner", COMMON_SCRIPT_PATH
)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载补充分析配置：{COMMON_SCRIPT_PATH}")
supplementCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = supplementCommon
commonModuleSpec.loader.exec_module(supplementCommon)

COHORT_NAME = supplementCommon.COHORT_NAME
MAIN_PROJECT_ROOT = supplementCommon.MAIN_PROJECT_ROOT
RUN_LOG_OUTPUT_DIRECTORY = supplementCommon.RUN_LOG_OUTPUT_DIRECTORY
PYTHON_EXECUTABLE = Path(sys.executable).resolve()

SUPPLEMENT_SCRIPT_NAMES = (
    "补充分析_2.匹配基因集零模型.py",
    "补充分析_3.供体与半球敏感性.py",
    "补充分析_4.汇总统计.py",
    "补充分析_5.排名基因GSEA.py",
    "补充分析_6.结果验证.py",
    "补充分析_7.生成分析摘要.py",
)


def sha256File(inputPath: Path, blockSize: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with inputPath.open("rb") as inputFile:
        while inputBlock := inputFile.read(blockSize):
            digest.update(inputBlock)
    return digest.hexdigest()


def validateMainline() -> Path:
    validationPath = (
        MAIN_PROJECT_ROOT
        / f"{COHORT_NAME}_主流程结果13_主流程验证"
        / "mainline-validation.json"
    )
    if not validationPath.exists():
        raise FileNotFoundError(f"缺少主线验证文件：{validationPath}")
    validationPayload = json.loads(validationPath.read_text(encoding="utf-8"))
    mainlinePassed = bool(
        validationPayload.get(
            "allChecksPassed",
            validationPayload.get("overallPassed", False),
        )
    )
    if not mainlinePassed:
        raise RuntimeError(f"{COHORT_NAME} 主线验证未通过，不能运行补充分析。")
    return validationPath


def writeInputProvenance(mainlineValidationPath: Path) -> Path:
    provenancePath = SUPPLEMENT_ROOT / "supplement-input-provenance.json"
    provenancePayload = {
        "generatedAt": datetime.now().isoformat(),
        "cohort": COHORT_NAME,
        "mainProjectRoot": str(MAIN_PROJECT_ROOT),
        "mainlineValidationPath": str(mainlineValidationPath),
        "mainlineValidationSha256": sha256File(mainlineValidationPath),
        "mainlineReadOnly": True,
        "supplementOutputRoot": str(SUPPLEMENT_ROOT),
        "pythonExecutable": redactLogText(str(PYTHON_EXECUTABLE)),
        "analyses": [
            "expression-property-matched gene-set null",
            "single-donor and leave-one-donor-out sensitivity",
            "hemisphere direction sensitivity",
            "ranked-gene exploratory GSEA",
            "integrated summary and validation",
        ],
    }
    provenancePath.write_text(
        json.dumps(provenancePayload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return provenancePath


def redactLogText(rawLogText: str) -> str:
    return rawLogText.replace(str(Path.home()), "%USERPROFILE%")


def runScript(scriptPath: Path, logFile) -> None:
    stageMessage = f"[{COHORT_NAME}补充分析] 运行 {scriptPath.name}"
    print(stageMessage, flush=True)
    logFile.write(stageMessage + "\n")
    processEnvironment = os.environ.copy()
    processEnvironment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        [str(PYTHON_EXECUTABLE), str(scriptPath)],
        cwd=SUPPLEMENT_ROOT,
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
        print(outputLine, end="", flush=True)
        logFile.write(redactLogText(outputLine))
        logFile.flush()
    returnCode = process.wait()
    if returnCode != 0:
        raise subprocess.CalledProcessError(
            returnCode, [str(PYTHON_EXECUTABLE), str(scriptPath)]
        )


def parseArguments() -> argparse.Namespace:
    argumentParser = argparse.ArgumentParser(
        description=f"运行 {COHORT_NAME} 独立补充分析。"
    )
    argumentParser.add_argument("--start-step", type=int, default=1)
    argumentParser.add_argument(
        "--end-step", type=int, default=len(SUPPLEMENT_SCRIPT_NAMES)
    )
    return argumentParser.parse_args()


def main() -> None:
    arguments = parseArguments()
    startStep = int(arguments.start_step)
    endStep = int(arguments.end_step)
    if startStep < 1 or endStep > len(SUPPLEMENT_SCRIPT_NAMES) or startStep > endStep:
        raise ValueError(
            f"步骤范围无效：start={startStep}, end={endStep}, "
            f"允许范围为 1..{len(SUPPLEMENT_SCRIPT_NAMES)}。"
        )
    selectedNames = SUPPLEMENT_SCRIPT_NAMES[startStep - 1 : endStep]
    scriptPaths = [SUPPLEMENT_ROOT / scriptName for scriptName in selectedNames]
    missingScripts = [scriptPath for scriptPath in scriptPaths if not scriptPath.exists()]
    if missingScripts:
        raise FileNotFoundError(f"缺少补充分析脚本：{missingScripts}")

    mainlineValidationPath = validateMainline()
    provenancePath = writeInputProvenance(mainlineValidationPath)
    RUN_LOG_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logName = (
        f"{COHORT_NAME}-supplement-run.log"
        if startStep == 1 and endStep == len(SUPPLEMENT_SCRIPT_NAMES)
        else f"{COHORT_NAME}-supplement-run-step{startStep}-to-{endStep}.log"
    )
    logPath = RUN_LOG_OUTPUT_DIRECTORY / logName
    with logPath.open("w", encoding="utf-8") as logFile:
        logFile.write(f"开始时间：{datetime.now().isoformat()}\n")
        logFile.write(f"主线验证：{mainlineValidationPath}\n")
        logFile.write(f"输入来源：{provenancePath}\n")
        for scriptPath in scriptPaths:
            runScript(scriptPath, logFile)
        logFile.write(f"完成时间：{datetime.now().isoformat()}\n")
    print(f"[{COHORT_NAME}补充分析] 全部完成，日志：{logPath}", flush=True)


if __name__ == "__main__":
    main()
