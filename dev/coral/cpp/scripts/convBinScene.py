import sys
import numpy as np
from PIL import Image
from typing import Dict, Tuple

def getSceneGridSizes() -> Dict[int, Tuple[int, int]]:
    return {
        3052: (20, 21), 18008: (24, 24), 29032: (21, 21), 29041: (21, 21),
        29044: (20, 21), 32030: (21, 21), 32035: (20, 21), 32037: (20, 21),
        34029: (21, 21), 34033: (21, 21), 34037: (21, 21), 35029: (21, 21),
        35035: (20, 21), 39035: (20, 21), 50024: (21, 21), 63012: (23, 23),
        63013: (23, 23), 64012: (23, 23), 64015: (22, 22), 66014: (22, 23)
    }

def sceneBinPath(sceneId: int) -> str:
    # Hardcoded output path
    outputRoot = "/mnt/sdcard/output/stitched"
    return f"{outputRoot}/stitched_scene_{sceneId}.bin"

def scenePngPath(sceneId: int) -> str:
    outputRoot = "/mnt/sdcard/output/stitched"
    return f"{outputRoot}/stitched_scene_{sceneId}.png"

def bin_to_png(sceneId: int):
    grid = getSceneGridSizes()
    if sceneId not in grid:
        raise ValueError(f"Scene ID {sceneId} not found.")
    cols, rows = grid[sceneId]
    patchHeight, patchWidth = 192, 192
    width = cols * patchWidth
    height = rows * patchHeight

    bin_path = sceneBinPath(sceneId)
    png_path = scenePngPath(sceneId)

    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size != width * height:
        raise ValueError(f"Expected {width * height} values, found {data.size} in {bin_path}")

    img = data.reshape((height, width))
    img_min = img.min()
    img_max = img.max()
    img_norm = 255 * (img - img_min) / (img_max - img_min) if img_max != img_min else img * 0
    img_uint8 = img_norm.astype(np.uint8)
    Image.fromarray(img_uint8).save(png_path)
    print(f"Saved PNG: {png_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: convBinScene.py <sceneId>")
        sys.exit(1)
    sceneId = int(sys.argv[1])
    bin_to_png(sceneId)