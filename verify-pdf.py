import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageChops, ImageDraw

REVIEW_DIR = Path(__file__).parent
MANUSCRIPT_DIR = Path('F:/NPI-4-code/NPI-Progation-document/V31')
sourceText = (MANUSCRIPT_DIR / 'main.tex').read_text(encoding='utf-8')
orderedKeys = [key.strip() for key in re.search(r'\\nocite\{(.*?)\}', sourceText, re.S)[1].replace('%', '').split(',')]
bibliographyText = (REVIEW_DIR / 'main.bbl').read_text(encoding='utf-8')
compiledKeys = re.findall(r'\\entry\{([^}]+)\}', bibliographyText)
assert compiledKeys == orderedKeys
citationGroups = re.findall(r'\\textbf\{\(([\d,\s-]+)\)\}', sourceText)
citationNumbers = []
for citationGroup in citationGroups:
    for citationPart in citationGroup.split(','):
        numberBounds = [int(value) for value in citationPart.strip().split('-')]
        citationNumbers.extend(range(numberBounds[0], numberBounds[-1] + 1))
assert list(dict.fromkeys(citationNumbers)) == list(range(1, len(orderedKeys) + 1))
compileLog = (REVIEW_DIR / 'main.log').read_text(encoding='utf-8', errors='replace')
compileIssues = [line for line in compileLog.splitlines() if re.search(r'Overfull|Missing character|LaTeX Warning|Package .+ Warning|^!', line)]
assert not compileIssues, compileIssues
beforePdf = fitz.open(MANUSCRIPT_DIR / 'main-beta-corrected.pdf')
afterPdf = fitz.open(REVIEW_DIR / 'main.pdf')
assert len(beforePdf) == len(afterPdf) == 17
changedPages = []
pageImages = []
for pageIndex in range(len(afterPdf)):
    oldPixels = beforePdf[pageIndex].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    newPixels = afterPdf[pageIndex].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    oldImage = Image.frombytes('RGB', (oldPixels.width, oldPixels.height), oldPixels.samples)
    newImage = Image.frombytes('RGB', (newPixels.width, newPixels.height), newPixels.samples)
    pageImages.append(newImage)
    if ImageChops.difference(oldImage, newImage).getbbox():
        changedPages.append(pageIndex + 1)
        afterPdf[pageIndex].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(REVIEW_DIR / f'page-{pageIndex + 1}.png')
for batchIndex, firstPage in enumerate(range(0, len(afterPdf), 6)):
    overviewImage = Image.new('RGB', (1050, 1040), 'white')
    overviewDraw = ImageDraw.Draw(overviewImage)
    for pageIndex in range(firstPage, min(firstPage + 6, len(afterPdf))):
        thumbnailImage = pageImages[pageIndex].resize((350, 494))
        cellIndex = pageIndex - firstPage
        positionX = (cellIndex % 3) * 350
        positionY = (cellIndex // 3) * 520
        overviewImage.paste(thumbnailImage, (positionX, positionY + 20))
        overviewDraw.text((positionX + 5, positionY + 4), f'Page {pageIndex + 1}', fill='black')
    overviewImage.save(REVIEW_DIR / f'overview-{batchIndex + 1}.png')
verification = {'pages': len(afterPdf), 'changedPages': changedPages, 'references': len(orderedKeys), 'citationGroups': len(citationGroups), 'compileIssues': compileIssues}
(REVIEW_DIR / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification))
