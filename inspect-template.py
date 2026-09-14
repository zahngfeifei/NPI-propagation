from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn


def emuToInches(value: Any) -> float | None:
    if value is None:
        return None
    return round(value.inches, 4)


def lengthToPoints(value: Any) -> float | None:
    if value is None:
        return None
    return round(value.pt, 3)


def getStyleRecord(style: Any) -> dict[str, Any]:
    paragraphFormat = style.paragraph_format
    font = style.font
    fontNames: dict[str, str] = {}
    runProperties = style.element.rPr
    if runProperties is not None and runProperties.rFonts is not None:
        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
            value = runProperties.rFonts.get(qn(f"w:{key}"))
            if value:
                fontNames[key] = value

    return {
        "name": style.name,
        "styleId": style.style_id,
        "baseStyle": style.base_style.name if style.base_style is not None else None,
        "font": {
            "name": font.name,
            "fontNames": fontNames,
            "sizePt": lengthToPoints(font.size),
            "bold": font.bold,
            "italic": font.italic,
            "color": str(font.color.rgb) if font.color.rgb is not None else None,
        },
        "paragraph": {
            "alignment": str(paragraphFormat.alignment) if paragraphFormat.alignment is not None else None,
            "spaceBeforePt": lengthToPoints(paragraphFormat.space_before),
            "spaceAfterPt": lengthToPoints(paragraphFormat.space_after),
            "lineSpacing": str(paragraphFormat.line_spacing) if paragraphFormat.line_spacing is not None else None,
            "lineSpacingRule": str(paragraphFormat.line_spacing_rule)
            if paragraphFormat.line_spacing_rule is not None
            else None,
            "leftIndentIn": emuToInches(paragraphFormat.left_indent),
            "rightIndentIn": emuToInches(paragraphFormat.right_indent),
            "firstLineIndentIn": emuToInches(paragraphFormat.first_line_indent),
            "keepWithNext": paragraphFormat.keep_with_next,
            "keepTogether": paragraphFormat.keep_together,
            "pageBreakBefore": paragraphFormat.page_break_before,
        },
    }


def getTableRecord(table: Any, tableIndex: int) -> dict[str, Any]:
    gridColumns = table._tbl.tblGrid.gridCol_lst
    return {
        "index": tableIndex,
        "style": table.style.name if table.style is not None else None,
        "alignment": str(table.alignment) if table.alignment is not None else None,
        "autofit": table.autofit,
        "rowCount": len(table.rows),
        "columnCount": len(table.columns),
        "gridWidthsTwips": [int(column.w) for column in gridColumns],
        "cellTexts": [[cell.text for cell in row.cells] for row in table.rows],
    }


def main() -> None:
    inputPath = Path(sys.argv[1])
    outputPath = Path(sys.argv[2])
    document = Document(inputPath)
    keyStyleNames = {
        "Normal",
        "Title",
        "Subtitle",
        "Heading 1",
        "Heading 2",
        "Field Label",
        "Evidence Quote",
        "List Bullet",
        "TOC Item",
    }

    packageParts = []
    with zipfile.ZipFile(inputPath) as archive:
        for archiveInfo in sorted(archive.infolist(), key=lambda info: info.filename):
            partBytes = archive.read(archiveInfo.filename)
            packageParts.append(
                {
                    "path": archiveInfo.filename,
                    "size": len(partBytes),
                    "sha256": hashlib.sha256(partBytes).hexdigest(),
                    "classification": "editable"
                    if archiveInfo.filename == "word/document.xml"
                    else "preserve-only",
                }
            )

    evidence = {
        "reference": {
            "path": str(inputPath),
            "size": inputPath.stat().st_size,
            "sha256": hashlib.sha256(inputPath.read_bytes()).hexdigest(),
        },
        "styles": [
            getStyleRecord(style)
            for style in document.styles
            if style.name in keyStyleNames
        ],
        "tables": [
            getTableRecord(table, tableIndex)
            for tableIndex, table in enumerate(document.tables)
        ],
        "headers": [
            [paragraph.text for paragraph in section.header.paragraphs]
            for section in document.sections
        ],
        "footers": [
            [paragraph.text for paragraph in section.footer.paragraphs]
            for section in document.sections
        ],
        "packageParts": packageParts,
    }
    outputPath.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
