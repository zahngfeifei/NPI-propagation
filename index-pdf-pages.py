from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber


def normalizeText(rawText: str) -> str:
    return re.sub(r"\s+", " ", rawText).strip()


def indexPdf(pdfPath: Path, outputPath: Path) -> None:
    pageRecords: list[dict[str, object]] = []
    with pdfplumber.open(pdfPath) as pdfDocument:
        for pageIndex, page in enumerate(pdfDocument.pages, start=1):
            pageText = normalizeText(page.extract_text() or "")
            pageRecords.append({"pageNumber": pageIndex, "text": pageText})
    outputPath.write_text(json.dumps(pageRecords, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pageCount": len(pageRecords)}, ensure_ascii=False))


def main() -> None:
    argumentParser = argparse.ArgumentParser(description="Extract searchable text from every PDF page.")
    argumentParser.add_argument("pdfPath", type=Path)
    argumentParser.add_argument("outputPath", type=Path)
    parsedArguments = argumentParser.parse_args()
    indexPdf(parsedArguments.pdfPath, parsedArguments.outputPath)


if __name__ == "__main__":
    main()
