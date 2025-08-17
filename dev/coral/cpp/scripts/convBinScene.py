import sys
import os
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
    patchHeight, patchWidth = 384, 384
    width = cols * patchWidth
    height = rows * patchHeight

    bin_path = sceneBinPath(sceneId)
    png_path = scenePngPath(sceneId)

    # Expect float32 stitched image of shape (height, width)
    expected_vals = width * height
    file_size = os.path.getsize(bin_path)
    if file_size != expected_vals * 4:
        raise ValueError(f"Expected {expected_vals*4} bytes, found {file_size} in {bin_path}")

    # Map input without loading into RAM
    data = np.memmap(bin_path, dtype=np.float32, mode="r", shape=(height, width))

    # Create a temporary uint8 memmap for the normalized image
    tmp_u8_path = png_path + ".tmp.u8"
    out = np.memmap(tmp_u8_path, dtype=np.uint8, mode="w+", shape=(height, width))

    # Since model outputs come from sigmoid, assume values in [0,1]
    # Process in row chunks to keep memory low
    chunk_rows = 512
    scale = np.float32(255.0)
    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        chunk = data[y0:y1]  # float32 view
        # Clip to [0,1], scale to [0,255], cast to uint8
        buf = np.clip(chunk, 0.0, 1.0).astype(np.float32, copy=False)
        np.multiply(buf, scale, out=buf)
        out[y0:y1] = buf.astype(np.uint8, copy=False)

    out.flush()
    del out
    del data

    # Save PNG from the uint8 memmap
    u8 = np.memmap(tmp_u8_path, dtype=np.uint8, mode="r", shape=(height, width))
    img = Image.fromarray(np.asarray(u8), mode="L")
    img.save(png_path, optimize=True)
    del u8
    os.remove(tmp_u8_path)

    print(f"Saved PNG: {png_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: convBinScene.py <sceneId>")
        sys.exit(1)
    sceneId = int(sys.argv[1])
    bin_to_png(sceneId)