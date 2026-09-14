from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


THIRD_OUTPUT_FOLDER = Path(__file__).resolve().parent
THIRD_WORKSPACE_JSON_PATH = THIRD_OUTPUT_FOLDER / "third-citation-audit-workspace.json"
AUDIT_CSV_PATH = THIRD_OUTPUT_FOLDER / "citation_audit.csv"
AUDIT_REPORT_PATH = THIRD_OUTPUT_FOLDER / "citation_audit_report.md"
SECOND_GENERATOR_PATH = (
    THIRD_OUTPUT_FOLDER.parents[1]
    / "第二次审核"
    / "审核结果"
    / "generate-second-citation-audit.py"
)

moduleSpec = importlib.util.spec_from_file_location(
    "secondAuditGenerator", SECOND_GENERATOR_PATH
)
if moduleSpec is None or moduleSpec.loader is None:
    raise RuntimeError("Unable to load the shared audit generator")

auditGenerator = importlib.util.module_from_spec(moduleSpec)
moduleSpec.loader.exec_module(auditGenerator)

auditGenerator.WORKSPACE_JSON_PATH = THIRD_WORKSPACE_JSON_PATH
auditGenerator.AUDIT_CSV_PATH = AUDIT_CSV_PATH
auditGenerator.AUDIT_REPORT_PATH = AUDIT_REPORT_PATH
auditGenerator.EXCLUDED_PLACEHOLDERS = {"REF3"}

auditGenerator.RATINGS = {
    "C001": "A",
    "C002": "B",
    "C004": "B",
    "C005": "B",
    "C006": "B",
    "C007": "B",
    "C008": "B",
    "C009": "B",
    "C010": "B",
    "C011": "B",
    "C012": "B",
    "C013": "A",
    "C014": "B",
    "C015": "C",
    "C016": "B",
    "C017": "B",
    "C018": "A",
    "C019": "A",
    "C020": "B",
    "C021": "B",
    "C022": "A",
    "C023": "A",
    "C024": "B",
    "C025": "B",
    "C026": "B",
    "C027": "B",
    "C028": "C",
    "C029": "C",
    "C030": "B",
    "C031": "B",
    "C032": "B",
}

auditGenerator.EVIDENCE_PHRASES.update(
    {
        "REF4": "directed flow between regions",
        "REF8": "communication strategies",
        "REF10": "recurrent excitation/inhibition",
        "REF13": "core-periphery",
        "REF14": "Response dissociation",
        "REF19": "sensory-information, goal-directed tasks",
        "REF25": "intrinsic coordinate system",
        "REF27": "specific directionality",
    }
)
auditGenerator.OCCURRENCE_EVIDENCE_PHRASES.update(
    {
        "C004": "directed flow between regions",
        "C008": "relative position of a cortical location",
        "C012": "specific directionality",
        "C014": "communication strategies",
        "C016": "recurrent excitation/inhibition",
        "C020": "core-periphery interactions",
        "C021": "interactions between cortical areas",
        "C024": "sensory-information, goal-directed tasks",
    }
)

auditGenerator.GROUP_DETAILS[range(2, 5)] = (
    "REF2 支持健康 FC 主梯度；替换后的 REF4 直接使用定向信息流研究全脑功能层级，但仍不检验本文两个 ASD 队列的 NPI excitatory EC-G1。",
    "本文队列结果、健康 FC 比较与一般 directed-flow 层级证据仍被合并在同一论断链中。",
    "把两个队列的 EC-G1 作为本研究结果；REF2 只支撑健康梯度比较，REF4 只作为定向层级背景，并注明其方法不是 NPI excitatory EC。"
)
auditGenerator.GROUP_DETAILS[range(8, 10)] = (
    "REF25 明确把皮层梯度描述为内在连续坐标及相对位置，REF26 支持高维功能拓扑的低维表示；但两者均未直接测量本文 excitatory output specialization。",
    "梯度相对位置的解释得到改善，但“减少专业化”仍是对本文 EC-G1 距离结果的推断。",
    "保留方法学解释，将结论降低为“consistent with reduced differentiation of distributed output profiles”。"
)
auditGenerator.GROUP_DETAILS[range(10, 13)] = (
    "REF7 支持刺激后传播和网络功能分化；REF18 支持 ASD 连接轮廓收缩；替换后的 REF27 直接报告 ASD 中皮层下至初级感觉区的定向影响异常。",
    "这些研究仍未直接测量本文的 EC-G1 facilitatory architecture 或感觉活动进入 attentional/control/transmodal 系统的传播特异性。",
    "使用“may reflect”或“is consistent with”；REF27 仅用于说明 ASD 感觉通路存在定向影响异常。"
)
auditGenerator.GROUP_DETAILS[range(13, 16)] = (
    "REF29 直接区分模型 EC 与包含网络效应的 dynamic flow；替换后的 REF8 系统讨论多路径和间接网络通信；REF9 全文仍然错误。",
    "传播句包含 recurrent、indirect、negative 和 initial-state 多项成分，REF8 只覆盖其中的网络通信/间接路径。",
    "保留 REF29 与 REF8 的部分支持；以正确 NeuroImage 全文替换 REF9 后重新审核，并为 negative influence 与 initial state 补充模型内直接证据。"
)
auditGenerator.GROUP_DETAILS[range(16, 18)] = (
    "替换后的 REF10 直接在 ASD 模型中联系宏观连接异常、复发性兴奋/抑制和皮层下输入；REF30 支持局部复发及神经变异性影响传播。",
    "两篇仍未直接检验本文控制/默认网络的减弱放大、感觉—注意输入汇聚或 EC-G1—传播解耦。",
    "保留为可能机制并使用“could”; 不要写成已证实的 control/DMN 特异机制。"
)
auditGenerator.GROUP_DETAILS[range(18, 22)] = (
    "REF11/REF12 直接支持 ASD 功能层级异常；替换后的 REF13 讨论 ASD core-periphery 动态，REF14 直接研究 ASD 层级皮层回路反应解耦。",
    "REF13 是综述，REF14 的具体视觉回路反应解耦与本文全脑 excitatory output/propagation 指标仍不同。",
    "hierarchical accounts 可保留；对本文 output specialization 和 propagation efficacy 使用“may involve”，并区分各研究的层级尺度。"
)
auditGenerator.GROUP_DETAILS[range(24, 26)] = (
    "替换后的 REF19 直接分析感觉决策中 DMN、显著性和中央执行网络的定向交互；REF24 支持 DMN 整合外部输入、先验与内部表征。",
    "REF19 来自非 ASD 感知决策任务，REF24 为一般 DMN 观点；均不证明 ASD 中路由较不特异。",
    "把“routes them less specifically”明确归于本研究；引用仅支撑相关网络具备感觉—控制/情境整合功能。"
)
auditGenerator.GROUP_DETAILS[range(28, 30)] = (
    "REF17 本地全文缺失，无法核验；REF28 仅在非 ASD TMS 研究中证明中外侧前额叶目标表征对情境性控制的因果作用。",
    "没有本地全文证据或直接研究证明 ASD 中 control/DMN 早期募集减少会限制网络状态转换。",
    "补齐 REF17.pdf 后重新审核；当前句使用“may limit”，并将 REF28 仅用于控制功能背景。"
)


def patchMissingFullTextRow(auditRows: list[dict[str, str]]) -> None:
    for auditRow in auditRows:
        if auditRow["citationPlaceholder"] != "REF17":
            continue
        auditRow["referenceFullText"] = (
            "未提供 REF17.pdf，无法从本地参考文献全文提取或核验原文。"
        )
        auditRow["evidenceLocation"] = "无法定位（本地全文缺失）"
        auditRow["rating"] = "C"
        auditRow["ratingReason"] = (
            "本地全文缺失；按仅限 A/B/C 的规则归入最低档 C，但该 C 不表示存在概念支持。"
            "不得依据题名、摘要映射或预设评级推断文献支持该处论述。"
        )
        auditRow["problem"] = (
            "参考文献文件夹缺少 REF17.pdf，无法核对 DOI、正文证据、页码及正式版本内容。"
        )
        auditRow["recommendation"] = (
            "补充 DOI 10.1002/hbm.26750 的正式全文后，对 C028 重新进行全文审核。"
        )


def writeThirdAuditReport(
    auditRows: list[dict[str, str]], workspace: dict[str, object]
) -> None:
    ratingCounts = Counter(row["rating"] for row in auditRows)
    totalCount = len(auditRows)
    reportLines = [
        "# 第三次论文正文逐处引用审核报告",
        "",
        "## 审核范围与方法",
        "",
        f"- 论文正文：`{workspace['sourceFiles']['manuscriptPath']}`",
        f"- 引用映射：`{workspace['sourceFiles']['citationCsvPath']}`",
        f"- 本地全文：`{workspace['sourceFiles']['literatureFolder']}`",
        "- 审核日期：2026-07-29。",
        f"- 正文共识别 32 个逐处引用；排除超过 70 MB 的 REF3 后，本报告审核 {totalCount} 处。",
        "- 第三次映射表中的预设评级仅用于比较；所有判断均重新依据本地全文完成。REF17 因全文缺失，不使用题名或映射摘要代替证据。",
        "",
        "## 评级统计",
        "",
        "| 评级 | 数量 | 比例 |",
        "|---|---:|---:|",
    ]
    for rating in ["A", "B", "C"]:
        count = ratingCounts[rating]
        reportLines.append(
            f"| {rating} | {count} | {count / totalCount * 100:.1f}% |"
        )

    reportLines.extend(
        [
            "",
            "## DOI、全文与发表状态核对",
            "",
            "- 30 个映射 DOI 均为唯一值，未发现重复。",
            "- REF3 超过 70 MB，本轮排除；REF17.pdf 缺失；其余 28 份本地 PDF 中，27 份与映射题名和 DOI 一致。",
            "- REF9 全文仍然错误：映射为 NeuroImage 论文 `10.1016/j.neuroimage.2021.118546`，PDF 实际为 `NOMES DE LUGAR: CONFIM`（DOI `10.2307/26459823`）。",
            "- REF13 的首页题名、作者和期刊与映射一致，但 PDF 文本层未提取出 DOI；REF24 的 DOI 末字符在文本层断行，首页题名与作者可确认文件正确。",
            "- 纳入且文件正确的文献均可确认正式期刊发表状态；未见撤稿声明或会议摘要。REF13、REF16、REF20、REF21、REF22、REF24、REF25 属综述/观点类来源。",
            "",
            "## 与第三次映射表预设评级的比较",
            "",
            "- 排除 REF3 后，28 处评级与预设一致，3 处下调，没有上调。",
            "- C015（REF9）由 B 下调为 C：全文文件错误。",
            "- C028（REF17）由 B 下调为 C：本地全文缺失，无法核验。",
            "- C029（REF28）由 B 下调为 C：仅提供非 ASD 的前额叶控制机制，不能直接支持 ASD 早期募集与网络状态转换。",
            "",
            "## 无法核验或必须替换",
            "",
            "- **REF9 / C015：必须替换错误全文。**",
            "- **REF17 / C028：必须补充正式全文后重审。**",
            "- **REF3 / C003：PDF 超过 70 MB，本轮不纳入评级与比例。**",
            "",
            "## 替换文献后的主要改善",
            "",
            "- REF4：从一般多模态梯度替换为直接使用定向信息流研究全脑功能层级的研究，C004 从上轮 C 提升到 B。",
            "- REF8：新的脑网络通信综述直接覆盖多路径和间接传播，C014 从上轮 C 提升到 B。",
            "- REF10：新的 ASD connectome/microcircuit 模型研究直接涉及复发性兴奋/抑制与外部输入，C016 从上轮 C 提升到 B。",
            "- REF19：新的感觉决策原始研究直接分析 DMN、显著性和执行网络定向交互，C024 从上轮 C 提升到 B。",
            "- REF25、REF27：分别改善梯度相对位置解释及 ASD 感觉通路定向影响证据。",
            "",
            "## 仍需谨慎的复合论断",
            "",
            "- C008–C009：梯度方法学得到支持，但“减少专业化”仍是本文解释。",
            "- C014–C017：单篇文献仍不能覆盖 recurrent、indirect、negative、initial state 及 control/DMN 特异机制的全部内容。",
            "- C020–C021：ASD 层级动态证据与本文全脑 excitatory output/propagation 指标并不相同。",
            "- C024–C025：一般感觉—控制/DMN 网络功能不能直接证明 ASD 路由较不特异。",
            "",
            "## 每一处引用的详细审核结果",
            "",
        ]
    )

    for rowNumber, row in enumerate(auditRows, start=1):
        reportLines.extend(
            [
                f"### {rowNumber}. {row['citationPlaceholder']} — {row['rating']}",
                "",
                f"- **DOI：** {row['doi']}",
                f"- **题名：** {row['title']}",
                f"- **正文原句：** {row['manuscriptFullText']}",
                f"- **参考文献原文：** {row['referenceFullText']}",
                f"- **证据位置：** {row['evidenceLocation']}",
                f"- **评级理由：** {row['ratingReason']}",
                f"- **问题：** {row['problem']}",
                f"- **修改建议：** {row['recommendation']}",
                "",
            ]
        )

    AUDIT_REPORT_PATH.write_text("\n".join(reportLines), encoding="utf-8")


def main() -> None:
    workspace = json.loads(THIRD_WORKSPACE_JSON_PATH.read_text(encoding="utf-8"))
    auditRows = auditGenerator.buildAuditRows(workspace)
    patchMissingFullTextRow(auditRows)
    auditGenerator.writeAuditCsv(auditRows)
    writeThirdAuditReport(auditRows, workspace)
    print(f"rows={len(auditRows)}")
    print(AUDIT_CSV_PATH)
    print(AUDIT_REPORT_PATH)


if __name__ == "__main__":
    main()
