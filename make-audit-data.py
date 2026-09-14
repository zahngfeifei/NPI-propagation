from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from pathlib import Path


TASK_ROOT = Path(r"F:\NPI-4-code\.codex-temp\introduction-citation-audit-second")
MAPPING_PATH = Path(
    r"F:\NPI-4-code\引言\第二次审核\Introduction_reference_mapping_A80_B20_no_Elsevier.csv"
)
OLD_AUDIT_PATH = Path(
    r"F:\NPI-4-code\.codex-temp\introduction-citation-audit\audit-data.json"
)
OUTPUT_PATH = TASK_ROOT / "audit-data.json"


def loadMapping() -> dict[str, dict[str, str]]:
    with MAPPING_PATH.open(encoding="utf-8-sig", newline="") as mappingStream:
        return {
            row["citationPlaceholder"]: row
            for row in csv.DictReader(mappingStream)
        }


def loadOldReviewsByDoi() -> dict[str, dict[str, object]]:
    oldAudit = json.loads(OLD_AUDIT_PATH.read_text(encoding="utf-8"))
    oldReviewsByDoi: dict[str, dict[str, object]] = {}
    for record in oldAudit["records"]:
        doi = str(record["doi"]).lower()
        if doi != "未提供" and doi not in oldReviewsByDoi:
            oldReviewsByDoi[doi] = record
    return oldReviewsByDoi


def reuseOldReview(
    newRef: str,
    mappingRow: dict[str, str],
    oldReviewsByDoi: dict[str, dict[str, object]],
) -> dict[str, str]:
    doi = mappingRow["doiOrIdentifier"]
    oldRecord = deepcopy(oldReviewsByDoi[doi.lower()])
    oldRef = str(oldRecord["ref"])
    selfReferencePattern = re.compile(rf"\b{re.escape(oldRef)}\b")
    reusedReview: dict[str, str] = {}
    for fieldName in ("rating", "location", "quote", "reason", "issue", "suggestion"):
        fieldValue = str(oldRecord[fieldName])
        reusedReview[fieldName] = selfReferencePattern.sub(newRef, fieldValue)
    reusedReview["location"] = reusedReview["location"].replace(
        f"{oldRef}.pdf", f"{newRef}.pdf"
    )
    return reusedReview


def buildCustomReviews() -> dict[str, dict[str, str]]:
    return {
        "REF2": {
            "rating": "B",
            "location": "REF2.pdf, PDF p.1, Introduction",
            "quote": "“functional gradients that extend from unimodal regions through the cortical association areas”",
            "reason": (
                "论文明确采用从单模态区到联合皮层的宏观功能梯度框架，但主要研究问题是儿童至青春期的发育变化。"
            ),
            "issue": (
                "对总体轴线有直接文字支持，对成人经典主梯度的来源和注意/控制系统精确排序则是间接支持。"
            ),
            "suggestion": (
                "可保留为发育性佐证；如需压缩引文数量，优先保留直接界定成人主梯度的REF1和REF3。"
            ),
        },
        "REF3": {
            "rating": "A",
            "location": "REF3.pdf, PDF pp.1, 4, Abstract/Results",
            "quote": (
                "“the SA axis is anchored at one end by primary unimodal cortex and at "
                "the other end by high-order association cortex coinciding with the "
                "default mode network”"
            ),
            "reason": (
                "本地全文直接界定成人经典感觉-联合主轴的两个端点，并报告默认、控制和注意网络在主要功能轴上的相对组织，直接覆盖正文的空间层级论断。"
            ),
            "issue": (
                "论文为全生命周期梯度研究，控制/注意网络还通过调制-表征轴单独刻画；正文将多轴信息压缩为一句概括，但未造成实质性方向错误。"
            ),
            "suggestion": "无需修改；与REF1共同作为该句的主要直接证据。",
        },
        "REF4": {
            "rating": "B",
            "location": "REF4.pdf, PDF p.1, Abstract",
            "quote": "“described using two common functional gradients”",
            "reason": (
                "研究直接确认等皮层、海马与小脑之间存在可对应的功能连接梯度，支持功能梯度作为稳定低维组织轴。"
            ),
            "issue": (
                "重点是跨脑结构的梯度对应关系，并不直接检验经典皮层主梯度的两个端点或注意/控制网络的中间位置。"
            ),
            "suggestion": (
                "可保留为补充性证据；当前句的主证据应由REF1和REF3承担，若需压缩引文可优先删除REF4。"
            ),
        },
        "REF7": {
            "rating": "A",
            "location": "REF7.pdf, PDF p.1, Abstract",
            "quote": "“increased idiosyncrasy in default mode, somatomotor and attention networks”",
            "reason": (
                "研究直接报告默认、躯体运动与注意网络的ASD特异功能组织异常，并关联症状严重度，适合支持“网络特异性异常”。"
            ),
            "issue": (
                "对感觉-跨模态过渡的支持不如REF5和REF6直接，但能完整支撑同句中的网络特异异常分句。"
            ),
            "suggestion": "无需修改。",
        },
        "REF10": {
            "rating": "B",
            "location": "REF10.pdf, PDF p.1, Abstract/Introduction",
            "quote": "“EC ... causal influence ... FC ... activity synchrony between locations”",
            "reason": (
                "全文明确区分FC的同步/统计依赖与EC的方向性影响，能够支撑正文的概念定义。"
            ),
            "issue": (
                "该文的主要研究问题是中年人血管暴露标志物，并非专门建立FC/EC定义的方法学论文；作为定义性来源的权威性和针对性有限。"
            ),
            "suggestion": (
                "可保留现有引用，但建议再补充一篇直接讨论有效连接定义或模型框架的非Elsevier方法学来源；若不补充，正文可将“estimates”保留为较审慎措辞。"
            ),
        },
        "REF13": {
            "rating": "A",
            "location": "REF13.pdf, PDF p.1, Abstract/Introduction",
            "quote": "“adaptive allocation of cognitive resources ... changes in task demands”",
            "reason": (
                "论文直接论证多需求控制系统在多种任务中的灵活、领域一般性参与及其随当前任务需求调配认知资源的作用。"
            ),
            "issue": (
                "对注意/控制系统按需求协调加工的支持直接；不单独支撑同句后半的DMN内外信息整合。"
            ),
            "suggestion": "无需修改；保持REF13紧随该句前半分句最清晰。",
        },
        "REF14": {
            "rating": "A",
            "location": "REF14.pdf, PDF p.1, Abstract/Significance",
            "quote": (
                "“FPCNA is connected to the default network ... whereas FPCNB is "
                "connected to the dorsal attention network”"
            ),
            "reason": (
                "论文直接显示额顶控制网络存在分别偏向默认网络和背侧注意网络的子系统，并将其联系到内省加工与外部知觉注意；与REF13合用可直接支持按目标协调内外信息加工。"
            ),
            "issue": (
                "“internally represented goals and contextual information”是对文中目标导向控制、内省与知觉调节结果的综合表述，而非单一实验变量。"
            ),
            "suggestion": "无需更换；保持REF13支持需求驱动控制、REF14支持control-DMN/DAN分化的分工。",
        },
        "REF16": {
            "rating": "A",
            "location": "REF16.pdf, PDF p.1, Abstract",
            "quote": (
                "“FC between right insular and medial prefrontal cortices predicted "
                "more severe social responsiveness impairments in ASD”"
            ),
            "reason": (
                "研究直接报告ASD中显著性网络与默认网络过度连接，并显示岛叶-内侧前额叶连接预测更严重的社会反应性损害，直接覆盖网络异常与社会功能关联。"
            ),
            "issue": "样本来自ABIDE且为横断面关联，不能解释因果方向，但不影响当前关联性表述。",
            "suggestion": "无需修改。",
        },
        "REF17": {
            "rating": "A",
            "location": "REF17.pdf, PDF p.1, Abstract",
            "quote": (
                "“convergent hypoconnectivity in sensorimotor and attention regions "
                "and convergent hyperconnectivity between frontoparietal and default "
                "mode networks”"
            ),
            "reason": (
                "大样本规范模型研究直接显示ASD连接异常具有尺度依赖的异质性，同时在感觉运动/注意及额顶-默认网络层面出现收敛异常，且偏离模式预测社会与认知能力。"
            ),
            "issue": "研究为32站点横断面规范模型；连接偏离与社会、认知指标之间属于预测关联，不能据此推断因果方向。",
            "suggestion": "无需修改；该文是当前句中覆盖临床异质性与多网络异常最完整的来源。",
        },
        "REF18": {
            "rating": "B",
            "location": "REF18.pdf, PDF pp.1, 13, Abstract/Conclusion",
            "quote": "“most changes occur in SN, CEN, DMN, FPN ... but the severity of changes depends ... on age”",
            "reason": (
                "全文直接报告默认、显著性-执行和额顶网络的年龄相关低连接/高连接异常，可覆盖正文列举的网络异常范围。"
            ),
            "issue": (
                "研究主要比较不同年龄组的连接差异；对个体层面的临床异质性及社会沟通差异仅有间接讨论，不能单独承担整句后半主张。"
            ),
            "suggestion": (
                "将REF18理解为支持网络异常和年龄效应；临床异质性与社会沟通关联主要由REF15-REF17承担。"
            ),
        },
        "REF19": {
            "rating": "A",
            "location": "REF19.pdf, PDF pp.2, 6, Introduction/Methods",
            "quote": (
                "“all hypotheses were initially tested using the ABIDE1 dataset and "
                "subsequently validated in the independent ABIDE2 dataset”"
            ),
            "reason": (
                "全文明确把ABIDE1作为发现数据、把独立收集的ABIDE2作为验证数据，并说明ABIDE1来自17个国际中心，直接支持两个独立多站点ASD队列的表述。"
            ),
            "issue": (
                "该文证明其自身使用的ABIDE1/ABIDE2队列独立；本文最终样本仍需在方法中报告纳入、排除和参与者去重规则。"
            ),
            "suggestion": (
                "无需更换；在方法中明确ABIDE I与ABIDE II样本无重叠，并分别给出站点数和最终样本量。"
            ),
        },
    }


def buildRecords(
    mappingByRef: dict[str, dict[str, str]],
    reviewByRef: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    principalGradientSentence = (
        "A principal functional connectivity gradient provides a prominent representation "
        "of this organization, extending from visual and somatomotor regions at one extreme "
        "to transmodal regions of the default mode network at the other, with attention and "
        "control systems occupying intermediate positions (REF1-REF4)."
    )
    autismGradientSentence = (
        "Connectome-gradient and related studies have reported atypical differentiation "
        "between sensory and transmodal systems, altered sensory-to-higher-order integration, "
        "and network-specific connectivity abnormalities in ASD (REF5-REF9)."
    )
    fcEcSentence = (
        "Whereas FC quantifies statistical covariation, EC estimates directional interactions "
        "within a model of brain dynamics (REF10)."
    )
    npiSentence = (
        "Neural perturbational inference (NPI) provides a data-driven framework for estimating "
        "participant-specific whole-brain EC by learning regional dynamics and quantifying "
        "predicted responses to virtual perturbation of individual source regions (REF11)."
    )
    stepwiseSentence = (
        "Sensory signals are unlikely to engage higher-order association systems in a single "
        "transition; stepwise connectivity studies instead suggest a progression from primary "
        "sensory systems through intermediate attention and control territories toward "
        "higher-order association cortex (REF12)."
    )
    controlDmnSentence = (
        "Within this architecture, attention and control systems support the selection and "
        "coordination of distributed processing according to current demands, whereas "
        "interactions involving control and default mode systems contribute to integrating "
        "external information with internally represented goals and contextual information "
        "(REF13, REF14)."
    )
    autismNetworksSentence = (
        "This distinction may be particularly relevant to ASD, in which altered sensory, "
        "salience, control, and default mode connectivity has been associated with clinical "
        "heterogeneity and social-communicative differences (REF15-REF18), and previous studies "
        "have suggested atypical transitions between sensory and transmodal systems (REF5, REF6)."
    )
    cohortSentence = (
        "To test this possibility, we combined NPI-derived positive output EC, cortical gradient "
        "mapping, and sensory-seeded propagation analyses in two independent multisite ASD "
        "cohorts from ABIDE I and ABIDE II (REF19, REF20)."
    )

    occurrenceSpecification = [
        ("REF1", principalGradientSentence),
        ("REF2", principalGradientSentence),
        ("REF3", principalGradientSentence),
        ("REF4", principalGradientSentence),
        ("REF5", autismGradientSentence),
        ("REF6", autismGradientSentence),
        ("REF7", autismGradientSentence),
        ("REF8", autismGradientSentence),
        ("REF9", autismGradientSentence),
        ("REF10", fcEcSentence),
        ("REF11", npiSentence),
        ("REF12", stepwiseSentence),
        ("REF13", controlDmnSentence),
        ("REF14", controlDmnSentence),
        ("REF15", autismNetworksSentence),
        ("REF16", autismNetworksSentence),
        ("REF17", autismNetworksSentence),
        ("REF18", autismNetworksSentence),
        ("REF5", autismNetworksSentence),
        ("REF6", autismNetworksSentence),
        ("REF19", cohortSentence),
        ("REF20", cohortSentence),
    ]

    records: list[dict[str, object]] = []
    for position, (referenceLabel, sentence) in enumerate(
        occurrenceSpecification, start=1
    ):
        mappingRow = mappingByRef[referenceLabel]
        review = reviewByRef[referenceLabel]
        records.append(
            {
                "position": position,
                "ref": referenceLabel,
                "rating": review["rating"],
                "doi": mappingRow["doiOrIdentifier"],
                "title": mappingRow["title"],
                "sentence": sentence,
                "location": review["location"],
                "quote": review["quote"],
                "reason": review["reason"],
                "issue": review["issue"],
                "suggestion": review["suggestion"],
            }
        )
    return records


def main() -> None:
    mappingByRef = loadMapping()
    oldReviewsByDoi = loadOldReviewsByDoi()
    customReviews = buildCustomReviews()

    reviewByRef: dict[str, dict[str, str]] = {}
    for referenceLabel, mappingRow in mappingByRef.items():
        if referenceLabel in customReviews:
            reviewByRef[referenceLabel] = customReviews[referenceLabel]
        else:
            reviewByRef[referenceLabel] = reuseOldReview(
                referenceLabel, mappingRow, oldReviewsByDoi
            )

    records = buildRecords(mappingByRef, reviewByRef)
    auditData = {
        "sourceDocx": (
            r"F:\NPI-4-code\引言\第二次审核\Introduction_REF_A80_B20_no_Elsevier.docx"
        ),
        "pdfDirectory": r"F:\NPI-4-code\引言\第二次审核\参考文献",
        "providedPdfCount": 21,
        "citedReferenceCount": 20,
        "unusedReferences": ["REF22"],
        "records": records,
    }
    OUTPUT_PATH.write_text(
        json.dumps(auditData, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
