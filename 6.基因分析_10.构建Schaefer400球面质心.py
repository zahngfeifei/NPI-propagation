from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from brainspace.datasets import load_conte69, load_parcellation


COMMON_SCRIPT_PATH = Path(__file__).with_name("6.基因分析_14.公共配置与函数.py")
commonModuleSpec = importlib.util.spec_from_file_location(
    "analysis_common", COMMON_SCRIPT_PATH
)
if commonModuleSpec is None or commonModuleSpec.loader is None:
    raise RuntimeError(f"无法加载公共配置：{COMMON_SCRIPT_PATH}")
geneAnalysisCommon = importlib.util.module_from_spec(commonModuleSpec)
sys.modules[commonModuleSpec.name] = geneAnalysisCommon
commonModuleSpec.loader.exec_module(geneAnalysisCommon)

from analysis_common import (  # noqa: E402
    ANALYSIS_CONFIG,
    CENTROID_OUTPUT_DIRECTORY,
    PHENOTYPE_PATH,
    ROI_CENTROID_PATH,
    sha256File,
    writeJson,
)


def buildSchaeferSphereCentroids(parcelCount: int) -> np.ndarray:
    """计算 Conte69 球面上每个 Schaefer 分区的单位向量质心。"""
    sphereSurfaces = load_conte69(
        as_sphere=True,
        with_normals=False,
        join=False,
    )
    parcelLabels = load_parcellation(
        "schaefer",
        scale=parcelCount,
        join=False,
    )
    centroidRows: list[tuple[int, np.ndarray]] = []
    for sphereSurface, hemisphereLabels in zip(sphereSurfaces, parcelLabels):
        spherePoints = np.asarray(sphereSurface.Points, dtype=float)
        positiveParcelLabels = sorted(
            int(parcelLabel)
            for parcelLabel in np.unique(hemisphereLabels)
            if parcelLabel > 0
        )
        for parcelLabel in positiveParcelLabels:
            parcelVertexMask = np.asarray(hemisphereLabels) == parcelLabel
            rawCentroid = spherePoints[parcelVertexMask].mean(axis=0)
            centroidNorm = float(np.linalg.norm(rawCentroid))
            if not np.isfinite(centroidNorm) or centroidNorm <= 0:
                raise ValueError(f"ROI {parcelLabel} 的球面质心无法归一化。")
            unitCentroid = rawCentroid / centroidNorm
            centroidRows.append((parcelLabel, unitCentroid))

    centroidRows.sort(key=lambda row: row[0])
    observedParcelIds = [row[0] for row in centroidRows]
    expectedParcelIds = list(range(1, parcelCount + 1))
    if observedParcelIds != expectedParcelIds:
        raise ValueError("BrainSpace Schaefer 标签与预期的 1-based ROI 顺序不一致。")
    return np.vstack([row[1] for row in centroidRows])


def main() -> None:
    parcelCount = int(ANALYSIS_CONFIG["project"]["expectedFullRois"])
    expectedSelectedRoiCount = int(
        ANALYSIS_CONFIG["project"]["expectedSelectedRois"]
    )
    phenotypeFrame = pd.read_parquet(PHENOTYPE_PATH)
    selectedRoiIds = phenotypeFrame["ROI_ID_1based"].astype(int).to_numpy()
    hemisphereValues = phenotypeFrame["hemisphere"].astype(str).to_numpy()

    if len(selectedRoiIds) != expectedSelectedRoiCount:
        raise ValueError(
            f"表型 ROI 数量为 {len(selectedRoiIds)}，预期为 {expectedSelectedRoiCount}。"
        )
    if len(np.unique(selectedRoiIds)) != len(selectedRoiIds):
        raise ValueError("表型 ROI_ID_1based 存在重复值。")
    if selectedRoiIds.min() < 1 or selectedRoiIds.max() > parcelCount:
        raise ValueError("表型 ROI_ID_1based 超出 Schaefer-400 范围。")

    fullSphereCentroids = buildSchaeferSphereCentroids(parcelCount)
    selectedCoordinates = fullSphereCentroids[selectedRoiIds - 1]
    coordinateNorms = np.linalg.norm(selectedCoordinates, axis=1)
    if not np.allclose(coordinateNorms, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("生成的球面质心不是单位向量。")

    coordinateFrame = pd.DataFrame(
        {
            "ROI_ID_1based": selectedRoiIds,
            "hemisphere": hemisphereValues,
            "sphereX": selectedCoordinates[:, 0],
            "sphereY": selectedCoordinates[:, 1],
            "sphereZ": selectedCoordinates[:, 2],
            "x": selectedCoordinates[:, 0],
            "y": selectedCoordinates[:, 1],
            "z": selectedCoordinates[:, 2],
        }
    )
    CENTROID_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    coordinateFrame.to_csv(ROI_CENTROID_PATH, index=False, encoding="utf-8-sig")

    writeJson(
        CENTROID_OUTPUT_DIRECTORY / "centroid-provenance.json",
        {
            "method": (
                "BrainSpace Conte69 spherical surface vertices were averaged within "
                "each Schaefer-400 parcel and normalized to unit length; the 374 "
                "phenotype ROI IDs were then selected in phenotype order."
            ),
            "coordinateMeaning": "unit-sphere coordinates, not MNI millimetres",
            "sourceDataset": "BrainSpace Conte69 sphere and bundled Schaefer parcellation",
            "sourceDocumentation": "https://brainspace.readthedocs.io/",
            "brainspaceVersion": importlib.metadata.version("brainspace"),
            "phenotypePath": str(PHENOTYPE_PATH),
            "phenotypeSha256": sha256File(PHENOTYPE_PATH),
            "outputPath": str(ROI_CENTROID_PATH),
            "outputSha256": sha256File(ROI_CENTROID_PATH),
            "parcelCount": parcelCount,
            "selectedRoiCount": len(coordinateFrame),
            "leftHemisphereCount": int(
                coordinateFrame["hemisphere"].eq("LH").sum()
            ),
            "rightHemisphereCount": int(
                coordinateFrame["hemisphere"].eq("RH").sum()
            ),
            "maximumUnitNormError": float(np.max(np.abs(coordinateNorms - 1.0))),
        },
    )
    print(f"球面质心已生成：{ROI_CENTROID_PATH}")


if __name__ == "__main__":
    main()
