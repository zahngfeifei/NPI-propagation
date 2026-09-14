import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def naturalPageNumber(imagePath):
    numericCharacters = "".join(character for character in imagePath.stem if character.isdigit())
    return int(numericCharacters or 0)


def makeContactSheets(inputDirectory, outputDirectory, columns, rows, thumbnailWidth):
    imagePaths = sorted(Path(inputDirectory).glob("*.png"), key=naturalPageNumber)
    outputPath = Path(outputDirectory)
    outputPath.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    labelHeight = 24
    pageGap = 16
    pageCountPerSheet = columns * rows
    for sheetIndex in range(math.ceil(len(imagePaths) / pageCountPerSheet)):
        sheetImagePaths = imagePaths[sheetIndex * pageCountPerSheet:(sheetIndex + 1) * pageCountPerSheet]
        openedImages = [Image.open(imagePath).convert("RGB") for imagePath in sheetImagePaths]
        thumbnailHeight = max(round(image.height * thumbnailWidth / image.width) for image in openedImages)
        sheetWidth = columns * thumbnailWidth + (columns + 1) * pageGap
        sheetHeight = rows * (thumbnailHeight + labelHeight) + (rows + 1) * pageGap
        contactSheet = Image.new("RGB", (sheetWidth, sheetHeight), "#D9DDE3")
        sheetDraw = ImageDraw.Draw(contactSheet)
        for imageIndex, (imagePath, pageImage) in enumerate(zip(sheetImagePaths, openedImages)):
            columnIndex = imageIndex % columns
            rowIndex = imageIndex // columns
            scaledHeight = round(pageImage.height * thumbnailWidth / pageImage.width)
            thumbnailImage = pageImage.resize((thumbnailWidth, scaledHeight), Image.Resampling.LANCZOS)
            xCoordinate = pageGap + columnIndex * (thumbnailWidth + pageGap)
            yCoordinate = pageGap + rowIndex * (thumbnailHeight + labelHeight + pageGap)
            contactSheet.paste(thumbnailImage, (xCoordinate, yCoordinate))
            sheetDraw.text(
                (xCoordinate, yCoordinate + thumbnailHeight + 4),
                f"Page {naturalPageNumber(imagePath)}",
                fill="#111111",
                font=font,
            )
        sheetFilePath = outputPath / f"contact-sheet-{sheetIndex + 1:02d}.png"
        contactSheet.save(sheetFilePath)


def main():
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("inputDirectory")
    argumentParser.add_argument("outputDirectory")
    argumentParser.add_argument("--columns", type=int, default=2)
    argumentParser.add_argument("--rows", type=int, default=4)
    argumentParser.add_argument("--thumbnail-width", type=int, default=600)
    parsedArguments = argumentParser.parse_args()
    makeContactSheets(
        parsedArguments.inputDirectory,
        parsedArguments.outputDirectory,
        parsedArguments.columns,
        parsedArguments.rows,
        parsedArguments.thumbnail_width,
    )


if __name__ == "__main__":
    main()
