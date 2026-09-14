from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


OUTPUT_FOLDER = Path(__file__).resolve().parent
WORKSPACE_JSON_PATH = OUTPUT_FOLDER / "second-citation-audit-workspace.json"
AUDIT_CSV_PATH = OUTPUT_FOLDER / "citation_audit.csv"
AUDIT_REPORT_PATH = OUTPUT_FOLDER / "citation_audit_report.md"
EXCLUDED_PLACEHOLDERS = {"REF3"}

RATINGS = {
    "C001": "A",
    "C002": "B",
    "C004": "C",
    "C005": "B",
    "C006": "B",
    "C007": "B",
    "C008": "C",
    "C009": "B",
    "C010": "B",
    "C011": "B",
    "C012": "C",
    "C013": "A",
    "C014": "C",
    "C015": "C",
    "C016": "C",
    "C017": "B",
    "C018": "A",
    "C019": "A",
    "C020": "B",
    "C021": "B",
    "C022": "A",
    "C023": "A",
    "C024": "C",
    "C025": "B",
    "C026": "B",
    "C027": "B",
    "C028": "C",
    "C029": "C",
    "C030": "B",
    "C031": "B",
    "C032": "B",
}

EVIDENCE_PHRASES = {
    "REF1": "By systematically perturbing all regions",
    "REF2": "principal gradient of connectivity",
    "REF4": "Bridging local and global cortical organization",
    "REF5": "Starting from an input matrix",
    "REF6": "undirected component",
    "REF7": "stimulation-evoked responses",
    "REF8": "parallel communication",
    "REF10": "propagation of activity along",
    "REF11": "gradient is overall similar",
    "REF12": "atypical cortical hierarchy is a hallmark",
    "REF13": "reduced segregation between unimodal and transmodal",
    "REF14": "hierarchically structured predictions",
    "REF15": "atypical integration of sensory-to",
    "REF16": "bottom-up",
    "REF17": "sensory and social impairments",
    "REF18": "impaired segregation and integration",
    "REF19": "gatekeeper to executive control",
    "REF20": "goal-consistent behaviors",
    "REF21": "integrative role for the DMN",
    "REF22": "neurovascular coupling",
    "REF23": "Future longitudinal studies",
    "REF24": "integrates incoming extrinsic information",
    "REF25": "sensory-fugal gradient",
    "REF26": "low dimensional representation",
    "REF27": "spatial specificity",
    "REF28": "context-sensitive cognitive control",
    "REF29": "dynamic flow",
    "REF30": "recurrent circuitry",
}

OCCURRENCE_EVIDENCE_PHRASES = {
    "C006": "Positive entries in the EBC",
    "C013": "discrepancy between the flow and MOU-EC",
    "C014": "parallel communication",
    "C016": "propagation of activity along",
    "C017": "recurrent circuitry",
    "C024": "gating executive control",
    "C026": "goal-consistent behaviors",
    "C027": "integrative role for the DMN",
    "C030": "resting-state functional magnetic resonance",
    "C031": "neurovascular coupling",
}

GROUP_DETAILS = {
    range(1, 2): (
        "NPI 原始论文直接说明替代脑扰动如何获得一对多 EC，并同时给出方向、强度及兴奋/抑制符号。",
        "该处把模型中的“兴奋性响应”解释为区域输出属性；仍应避免把它写成细胞水平兴奋。",
        "可保留 REF1，并在方法或限制中继续明确这是模型响应符号，不是突触兴奋。"
    ),
    range(2, 5): (
        "文献支持健康皮层的感觉—跨模态梯度或多模态组织，但不检验本文两个队列的 EC-G1，也未直接分析 directed excitatory influence。",
        "本文结果、健康 FC 比较和跨模态背景被写在同一论断中，容易把背景证据误当作对新结果的验证。",
        "把两个队列的 EC-G1 作为本研究结果；REF2 仅放在“resembled the healthy-cortex principal gradient”之后。REF4 应改为多模态背景而非 directed EC 证据。"
    ),
    range(5, 8): (
        "文献分别支持梯度由分布式轮廓相似性构成、NPI EC 的定向带符号属性或 FC 的无向统计，但没有单篇覆盖整个 EC-G1/FC-G1 对比。",
        "一般方法定义被外推到本文新构建的梯度。",
        "拆分 FC 与 EC 定义；明确 EC-G1 的具体构建来自本研究方法，而不是引用论文已经定义的指标。"
    ),
    range(8, 10): (
        "文献讨论宏观梯度、结构—功能耦合或高维拓扑轮廓，但未直接证明 EC-G1 距离代表“专业化”而非单连接强度。",
        "由梯度位置差异推断输出专业化降低，仍存在解释性跨越。",
        "降低为“consistent with reduced differentiation of distributed output profiles”，避免把 specialization 写成已直接测量。"
    ),
    range(10, 13): (
        "文献支持网络间传播、功能分化、ASD 连接轮廓收缩或感觉编码异常的一部分，但不直接检验本文的 facilitatory EC architecture 及其感觉至跨模态转换。",
        "静态 FC、一般刺激模型和感觉综述被用于支撑新的定向 EC 机制。",
        "将该机制明确标为解释性假设，使用“may reflect”或“is consistent with”，并寻找直接的 ASD 有效连接/扰动研究。"
    ),
    range(13, 16): (
        "REF29 直接区分模型 EC 与包含网络效应的 dynamic flow；REF8 仅支持多路径/并行通信背景；REF9 全文文件错误。",
        "复合句列出 recurrent、indirect、negative 和 initial-state 多项成分，现有单篇证据不能全部覆盖。",
        "保留 REF29 支撑 EC 与传播的区别；拆分传播组成，并以正确 NeuroImage 全文替换 REF9 后重新审核。"
    ),
    range(16, 18): (
        "文献支持层级动态或局部复发对信号传播的作用，但不检验 ASD 控制/默认网络中的减弱放大、输入汇聚或正负 EC 相互作用。",
        "多个具体机制均属本文结果后的推测，且外部研究对象与尺度不一致。",
        "改为明确的可能机制列表并使用“could”; REF30 仅支撑复发影响传播的背景，不应支撑 ASD 网络特异结论。"
    ),
    range(18, 22): (
        "文献直接证明 ASD 存在异常功能层级或层级预测加工差异；但后两篇并未测量本文的定向输出分化和传播效率。",
        "不同层级指标被并列为同一构念，包括 FC 梯度、发育轨迹、振荡模式与预测加工。",
        "概括性 hierarchical accounts 可保留；讨论本文机制时需区分各论文的层级操作化，并把 output/propagation 结论标为本研究发现。"
    ),
    range(22, 24): (
        "两篇文献直接讨论 ASD 中低阶感觉与高阶/跨模态加工之间的异常联系。",
        "REF16 是综述，适合框架性论述但不能替代原始研究证明本文的 EC/传播机制。",
        "可保留该背景句；若进一步声称具体定向路由机制，应补充原始有效连接或刺激研究。"
    ),
    range(24, 26): (
        "文献分别支持前岛在认知控制中的门控作用及 DMN 对外部输入与内部先验的整合，但不支持 ASD 中“路由较不特异”的本文结论。",
        "一般网络功能被用于解释 ASD 定向输出异常，研究对象和指标不一致。",
        "把“routes them less specifically”明确归于本研究；引用只放在 control/DMN 功能说明之后。"
    ),
    range(26, 28): (
        "REF20 支持目标一致的灵活控制，REF21 支持 DMN 的内部叙事、模型和信息整合；每篇只覆盖复合句的一侧。",
        "一个复合句把控制与 DMN 两套功能合并，单篇文献均非整句完全支持。",
        "拆成两句并分别引用：REF20 对应 control，REF21 对应 DMN。"
    ),
    range(28, 30): (
        "文献仅支持感觉异常与社会功能、或中外侧前额叶的情境性目标控制；没有测量 ASD 中控制/DMN 的早期募集或网络状态转换。",
        "相关功能背景被用于支撑本文具体动态结果的后果。",
        "改为“may limit”并明确这是机制解释；不要把 REF17/REF28 作为 early recruitment 的直接证据。"
    ),
    range(30, 32): (
        "REF1 说明 NPI 基于静息态 fMRI 的替代模型，REF22 说明神经活动与血流动力学信号之间存在复杂、可变的耦合；两者共同支持谨慎解释。",
        "NPI 论文未直接证明所有传播量都不能对应神经传递，REF22 又主要基于神经血管耦合综述及啮齿动物证据。",
        "保留限制性表述，并写成“model-derived regional response dynamics in BOLD-related signals”，避免过度外推到细胞机制。"
    ),
    range(32, 33): (
        "横断面研究直接指出 ASD 感觉诱发反应具有年龄相关变化，并明确提出需要纵向研究。",
        "该研究未测量 excitatory output EC，因此只能支持研究设计需要，不能预判输出分化的具体发育机制。",
        "保留未来研究建议，使用“to test whether”，并明确纵向研究需直接测量 EC-G1 和传播。"
    ),
}


def normalizeWhitespace(rawText: str) -> str:
    return re.sub(r"\s+", " ", rawText).strip()


def findGroupDetails(occurrenceNumber: int) -> tuple[str, str, str]:
    for occurrenceRange, details in GROUP_DETAILS.items():
        if occurrenceNumber in occurrenceRange:
            return details
    raise KeyError(f"No details configured for occurrence {occurrenceNumber}")


def chooseEvidence(
    occurrence: dict[str, object], referenceRecord: dict[str, object]
) -> tuple[str, str]:
    occurrenceId = occurrence["occurrenceId"]
    citationPlaceholder = occurrence["citationPlaceholder"]

    if citationPlaceholder == "REF9":
        firstPageText = referenceRecord["pages"][0]["text"]
        return normalizeWhitespace(firstPageText), "PDF p.1（文件首页；证明全文与映射不符）"

    preferredPhrase = OCCURRENCE_EVIDENCE_PHRASES.get(
        occurrenceId, EVIDENCE_PHRASES.get(citationPlaceholder, "")
    ).lower()
    candidatePassages = [
        {"pageNumber": pageRecord["pageNumber"], "passage": passage}
        for pageRecord in referenceRecord["pages"]
        for passage in pageRecord["passages"]
    ]

    for candidatePassage in candidatePassages:
        if preferredPhrase and preferredPhrase in candidatePassage["passage"].lower():
            return (
                normalizeWhitespace(candidatePassage["passage"]),
                f"PDF p.{candidatePassage['pageNumber']}（最相关连续段落）",
            )

    if occurrence["candidateEvidence"]:
        topEvidence = occurrence["candidateEvidence"][0]
        return (
            normalizeWhitespace(topEvidence["passage"]),
            f"PDF p.{topEvidence['pageNumber']}（最相关连续段落）",
        )

    return "本地 PDF 未提取到可核验的相关原文。", "无法定位"


def buildAuditRows(workspace: dict[str, object]) -> list[dict[str, str]]:
    auditRows: list[dict[str, str]] = []

    for occurrence in workspace["citationOccurrences"]:
        citationPlaceholder = occurrence["citationPlaceholder"]
        if citationPlaceholder in EXCLUDED_PLACEHOLDERS:
            continue

        occurrenceId = occurrence["occurrenceId"]
        occurrenceNumber = int(occurrenceId[1:])
        referenceRecord = workspace["references"][citationPlaceholder]
        referenceFullText, evidenceLocation = chooseEvidence(
            occurrence, referenceRecord
        )
        generalReason, generalProblem, generalRecommendation = findGroupDetails(
            occurrenceNumber
        )
        rating = RATINGS[occurrenceId]
        ratingPrefix = {
            "A": "全文直接覆盖该处主要论断，研究对象、变量关系和结论方向基本一致。",
            "B": "全文只覆盖该句的一部分，或存在研究对象、指标、尺度或因果强度差异。",
            "C": "全文仅提供一般概念、背景机制或邻近主题，没有直接检验该处具体论断。",
        }[rating]

        if citationPlaceholder == "REF9":
            ratingPrefix = (
                "本地 REF9.pdf 与映射 DOI、题名和研究领域均不符；按仅限 A/B/C 的规则归入最低档 C，"
                "但该 C 不表示存在概念支持。"
            )
            generalProblem = (
                "全文文件错误：实际为 Massimo Cacciari 的“NOMES DE LUGAR: CONFIM”，"
                "DOI 10.2307/26459823，而非 NeuroImage 论文。"
            )
            generalRecommendation = (
                "以 DOI 10.1016/j.neuroimage.2021.118546 的正式全文替换 REF9.pdf，随后重新审核 C015。"
            )

        auditRows.append(
            {
                "citationPlaceholder": citationPlaceholder,
                "doi": occurrence["metadata"]["doi"],
                "title": occurrence["metadata"]["title"],
                "manuscriptFullText": occurrence["manuscriptFullText"],
                "referenceFullText": referenceFullText,
                "evidenceLocation": evidenceLocation,
                "rating": rating,
                "ratingReason": f"{ratingPrefix}{generalReason}",
                "problem": generalProblem,
                "recommendation": generalRecommendation,
            }
        )

    return auditRows


def writeAuditCsv(auditRows: list[dict[str, str]]) -> None:
    fieldNames = [
        "citationPlaceholder",
        "doi",
        "title",
        "manuscriptFullText",
        "referenceFullText",
        "evidenceLocation",
        "rating",
        "ratingReason",
        "problem",
        "recommendation",
    ]
    with AUDIT_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(auditRows)


def writeAuditReport(
    auditRows: list[dict[str, str]], workspace: dict[str, object]
) -> None:
    ratingCounts = Counter(row["rating"] for row in auditRows)
    totalCount = len(auditRows)
    reportLines = [
        "# 第二次论文正文逐处引用审核报告",
        "",
        "## 审核范围与方法",
        "",
        f"- 论文正文：`{workspace['sourceFiles']['manuscriptPath']}`",
        f"- 引用映射：`{workspace['sourceFiles']['citationCsvPath']}`",
        f"- 本地全文：`{workspace['sourceFiles']['literatureFolder']}`",
        "- 审核日期：2026-07-29。",
        f"- 正文共识别 32 个逐处引用；延续用户要求排除超过 70 MB 的 REF3 后，本报告审核 {totalCount} 处，涉及 29 篇本地全文。",
        "- 映射表中的预设评级未直接采用；所有 A/B/C 均依据本地 PDF 全文重新判断。",
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
            "- 排除 REF3 后，29 份纳入 PDF 中有 28 份的题名、作者、年份、期刊及正式发表信息与映射一致。",
            "- REF9 全文错误：映射为 NeuroImage 论文 `10.1016/j.neuroimage.2021.118546`，本地 PDF 实际为 2005 年葡萄牙语哲学文章 `10.2307/26459823`。",
            "- REF24 的 DOI 在文本抽取时末字符被断行，但首页题名、作者和期刊信息与映射一致。",
            "- REF17 为作者接收稿版式，但首页 PII/DOI 对应正式期刊论文；REF29 为期刊排版前稿/作者稿，题名与 DOI 对应正式发表记录。",
            "- REF3 为正式 Nature 论文，但 PDF 约 73.4 MB，本轮按用户要求不作支持性评级。",
            "- 本地全文首页未见撤稿、撤稿声明、会议摘要或仅以预印本作为最终版本的情况。",
            "",
            "## 与映射表预设评级的比较",
            "",
            "- 评级一致 15 处；本轮下调 11 处，上调 5 处。REF3 因文件大小未比较。",
            "- 下调为 C：C004、C008、C012、C014、C015、C016、C024、C028、C029。主要原因是文献只提供一般背景，或本地全文错误。",
            "- 由 A 下调为 B：C026、C027。REF20 与 REF21 分别只支持复合句的 control 或 DMN 一侧，单篇不能完全支持整句。",
            "- 由 B 上调为 A：C013、C018、C019、C022、C023。对应全文直接覆盖 EC/动态流区别、ASD 层级异常或感觉—高阶整合的概括性论述。",
            "",
            "## 无法核验或必须替换",
            "",
            "- **REF9 / C015：必须替换全文。** 当前 PDF 与 DOI、题名和学科均不匹配。",
            "- **REF3 / C003：本轮排除。** PDF 超过 70 MB，不纳入评级与比例。",
            "",
            "## 需要优先修改的引用",
            "",
            "- C004：REF4 是多模态梯度研究，没有测量 directed excitatory influence，不能支撑该具体结论。",
            "- C008–C009：宏观梯度论文没有直接证明“gradient position 不是单连接强度”及“专业化降低”的完整解释。",
            "- C011–C012：静态连接收缩和感觉综述不能直接证明 ASD 的 facilitatory EC architecture 或感觉至跨模态转换特异性。",
            "- C014–C017：传播复合机制证据分散；REF9 全文错误，REF8/REF10/REF30 只能支持部分机制。",
            "- C020–C021：ASD 层级或预测加工研究没有测量本文的 directed output specialization 与 propagation efficacy。",
            "- C024–C025：控制/DMN 的一般功能不能直接证明 ASD 中路由较不特异。",
            "- C028–C029：文献没有测量本文所说的控制与 DMN 早期募集减少及其网络状态后果。",
            "",
            "## 综述与观点文献",
            "",
            "- REF16、REF17、REF19、REF20、REF21、REF22、REF24、REF27 属于综述、观点或综合性文章。它们适合定义和解释背景，不应替代直接研究来证明本文的 ASD 动态传播结果。",
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
    workspace = json.loads(WORKSPACE_JSON_PATH.read_text(encoding="utf-8"))
    auditRows = buildAuditRows(workspace)
    writeAuditCsv(auditRows)
    writeAuditReport(auditRows, workspace)
    print(f"rows={len(auditRows)}")
    print(AUDIT_CSV_PATH)
    print(AUDIT_REPORT_PATH)


if __name__ == "__main__":
    main()
