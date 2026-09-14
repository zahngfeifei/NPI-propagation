import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", re.IGNORECASE)


def normalizeDoi(doiValue):
    return re.sub(r"\s+", "", doiValue or "").lower().rstrip(".,;)")


def normalizeTitle(titleValue):
    normalizedValue = (titleValue or "").lower().replace("‐", "-").replace("–", "-").replace("—", "-")
    normalizedValue = re.sub(r"[^a-z0-9]+", " ", normalizedValue)
    return " ".join(normalizedValue.split())


def titleCoverage(expectedTitle, observedText):
    expectedNormalized = normalizeTitle(expectedTitle)
    observedNormalized = normalizeTitle(observedText)
    expectedTokens = set(expectedNormalized.split())
    observedTokens = set(observedNormalized.split())
    tokenCoverage = len(expectedTokens & observedTokens) / len(expectedTokens) if expectedTokens else 0.0
    sequenceCoverage = SequenceMatcher(None, expectedNormalized, observedNormalized[:max(len(expectedNormalized) * 3, 500)]).ratio()
    return max(tokenCoverage, sequenceCoverage)


def cleanSourceFileName(sourceFileValue):
    cleanedValue = re.sub(r"（.*?）", "", sourceFileValue or "").strip()
    return cleanedValue


def readMapping(mappingPath):
    with Path(mappingPath).open("r", encoding="utf-8-sig", newline="") as mappingStream:
        return list(csv.DictReader(mappingStream))


def readExtraction(extractionPath):
    return json.loads(Path(extractionPath).read_text(encoding="utf-8"))["pdfRecords"]


def findPrimaryDoi(firstPagesText, metadataValues):
    combinedValue = "\n".join([firstPagesText, *metadataValues])
    doiMatches = [normalizeDoi(matchValue) for matchValue in DOI_PATTERN.findall(combinedValue)]
    uniqueMatches = []
    for doiMatch in doiMatches:
        if doiMatch and doiMatch not in uniqueMatches:
            uniqueMatches.append(doiMatch)
    return uniqueMatches[:8]


def verifyRecords(mappingRecords, pdfRecords):
    pdfRecordByName = {record["fileName"].lower(): record for record in pdfRecords}
    verificationRecords = []
    referencedSourceFiles = set()
    for mappingRecord in mappingRecords:
        sourceFileName = cleanSourceFileName(mappingRecord["sourceFile"])
        referencedSourceFiles.add(sourceFileName.lower())
        pdfRecord = pdfRecordByName.get(sourceFileName.lower())
        if pdfRecord is None:
            verificationRecords.append({
                "newRef": mappingRecord["newRef"],
                "sourceFile": sourceFileName,
                "expectedDoi": mappingRecord["doi"],
                "expectedTitle": mappingRecord["title"],
                "pageCount": 0,
                "textCharacterCount": 0,
                "doiFound": False,
                "titleCoverage": 0.0,
                "metadataTitle": "",
                "observedDoiCandidates": "",
                "status": "missing_file",
                "notes": "映射所需 PDF 不存在。",
            })
            continue

        textPath = Path(pdfRecord["textPath"])
        fullText = textPath.read_text(encoding="utf-8", errors="replace")
        whitespaceCollapsedText = re.sub(r"\s+", "", fullText).lower()
        expectedDoi = normalizeDoi(mappingRecord["doi"])
        doiFound = expectedDoi in whitespaceCollapsedText
        metadataTitle = pdfRecord.get("metadata", {}).get("/Title", "") or ""
        observedTitleText = "\n".join([
            metadataTitle,
            pdfRecord.get("firstPageText", ""),
            pdfRecord.get("secondPageText", ""),
        ])
        coverageValue = titleCoverage(mappingRecord["title"], observedTitleText)
        observedDoiCandidates = findPrimaryDoi(
            "\n".join([pdfRecord.get("firstPageText", ""), pdfRecord.get("secondPageText", "")]),
            [str(value) for value in pdfRecord.get("metadata", {}).values()],
        )

        if coverageValue >= 0.82 and (doiFound or expectedDoi in observedDoiCandidates):
            status = "verified"
            notes = "题名与 DOI 均与映射一致。"
        elif coverageValue >= 0.92:
            status = "verified_title_only"
            notes = "题名高度一致，但 PDF 文本层未检出 DOI；需以首页视觉信息复核。"
        elif doiFound and coverageValue >= 0.60:
            status = "verified_doi_partial_title"
            notes = "DOI 一致，题名文本覆盖不完整；需以首页视觉信息复核。"
        elif not doiFound and coverageValue < 0.50:
            status = "mismatch"
            notes = "题名与 DOI 均不匹配，属于错误全文。"
        else:
            status = "manual_review"
            notes = "自动核验信号不完整，需要人工复核首页。"

        verificationRecords.append({
            "newRef": mappingRecord["newRef"],
            "sourceFile": sourceFileName,
            "expectedDoi": mappingRecord["doi"],
            "expectedTitle": mappingRecord["title"],
            "pageCount": pdfRecord["pageCount"],
            "textCharacterCount": pdfRecord["textCharacterCount"],
            "doiFound": doiFound,
            "titleCoverage": round(coverageValue, 4),
            "metadataTitle": metadataTitle,
            "observedDoiCandidates": ";".join(observedDoiCandidates),
            "status": status,
            "notes": notes,
        })

    extraRecords = []
    for pdfRecord in pdfRecords:
        if pdfRecord["fileName"].lower() in referencedSourceFiles:
            continue
        fullText = Path(pdfRecord["textPath"]).read_text(encoding="utf-8", errors="replace")
        observedDoiCandidates = findPrimaryDoi(
            "\n".join([pdfRecord.get("firstPageText", ""), pdfRecord.get("secondPageText", "")]),
            [str(value) for value in pdfRecord.get("metadata", {}).values()],
        )
        extraRecords.append({
            "fileName": pdfRecord["fileName"],
            "pageCount": pdfRecord["pageCount"],
            "textCharacterCount": pdfRecord["textCharacterCount"],
            "metadataTitle": pdfRecord.get("metadata", {}).get("/Title", "") or "",
            "observedDoiCandidates": ";".join(observedDoiCandidates),
            "firstPagePreview": pdfRecord.get("firstPageText", "")[:500],
            "status": "not_in_final_mapping",
        })
    return verificationRecords, extraRecords


def writeCsv(outputPath, records):
    if not records:
        Path(outputPath).write_text("", encoding="utf-8-sig")
        return
    with Path(outputPath).open("w", encoding="utf-8-sig", newline="") as outputStream:
        writer = csv.DictWriter(outputStream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main():
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("--mapping", required=True)
    argumentParser.add_argument("--extraction", required=True)
    argumentParser.add_argument("--output-dir", required=True)
    parsedArguments = argumentParser.parse_args()

    outputDirectory = Path(parsedArguments.output_dir)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    verificationRecords, extraRecords = verifyRecords(
        readMapping(parsedArguments.mapping),
        readExtraction(parsedArguments.extraction),
    )
    writeCsv(outputDirectory / "reference-verification.csv", verificationRecords)
    writeCsv(outputDirectory / "extra-files.csv", extraRecords)
    summaryRecord = {
        "targetCount": len(verificationRecords),
        "statusCounts": {
            statusValue: sum(1 for record in verificationRecords if record["status"] == statusValue)
            for statusValue in sorted({record["status"] for record in verificationRecords})
        },
        "extraFileCount": len(extraRecords),
    }
    (outputDirectory / "summary.json").write_text(
        json.dumps(summaryRecord, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
