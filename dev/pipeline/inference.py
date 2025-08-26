from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tqdm import tqdm
import math
import time
import os
from model import softJaccardLoss, diceCoefficient, diceLoss
from tensorflow_model_optimization.quantization.keras import quantize_scope
import tensorflow.image as tfi
from PIL import Image
from pathlib import Path

# --- Config ---
batchSize     = 16
imgSize       = (192,192)            # None for (384,384)
singleSceneID = None                 # None - full test set, all scenes. 0 for random scene, or specify ID
Upsample      = imgSize is not None  # only if not full resolution
BaseFolder    = Path(r"c:\Users\andre\Documents\BA\dev\pipeline\results\runs")
runFolder     = BaseFolder / "run_20250817_154643_3ch_192"
modelPath     = runFolder / "endModel.h5"
saveFolder    = runFolder / "evaluation" / "inference"

# --- Load Data (testDS only used when singleSceneID is not None) ---
(_, _, _, _, testDS, singleSceneID_loaded) = buildDS(
    includeTestDS=True,
    batchSize=batchSize,
    imgSize=imgSize,
    singleSceneID=singleSceneID
)

# --- Scene metadata ---
sceneGridSizes = getSceneGridSizes()

# --- Load Model ---
def bceDiceLoss(yTrue, yPred):
    return tf.keras.losses.binary_crossentropy(yTrue, yPred) + 0.5 * diceLoss(yTrue, yPred)

with quantize_scope():
    model = tf.keras.models.load_model(
        modelPath,
        custom_objects={'softJaccardLoss': softJaccardLoss,
                        'diceCoefficient': diceCoefficient,
                        'bceDiceLoss'    : bceDiceLoss}
    )

os.makedirs(saveFolder, exist_ok=True)

def runScene(sceneId: int):
    """Infer, (optionally) upsample, stitch, and save for a single scene."""
    cols, rows = sceneGridSizes[sceneId]
    total = math.ceil((cols * rows) / batchSize)

    # Build a per-scene dataset so we only keep one scene in memory
    (_, _, _, _, sceneDS, _) = buildDS(
        includeTestDS=True,
        batchSize=batchSize,
        imgSize=imgSize,
        singleSceneID=sceneId
    )

    preds = []
    start = time.time()
    for xBatch, _ in tqdm(sceneDS, total=total, desc=f"Scene {sceneId}"):
        yPred = model(xBatch, training=False)   # [B,H,W,1]
        preds.extend([p.numpy() for p in tf.unstack(yPred)])

    # Optional per-patch upsampling (still only one scene in RAM)
    if Upsample:
        preds = [tfi.resize(p, [384, 384], method="bilinear").numpy() for p in preds]

    # Stitch scene and save
    predArray = np.array([np.squeeze(p) for p in preds])   # [N,H,W]
    stitched  = stitchPatches(predArray, sceneId)          # {sceneId: canvas}

    canvas = stitched[sceneId]
    np.save(os.path.join(saveFolder, f"scene_{sceneId}.npy"), canvas)
    imgArray = (canvas * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(imgArray).save(os.path.join(saveFolder, f"scene_{sceneId}.png"))

    print(f"⏱️ Scene {sceneId} done in {time.time() - start:.2f}s → saved")

# --- Dispatch ---
if singleSceneID_loaded is not None:
    # Single scene mode
    print(f"🧩 Inference for Scene {singleSceneID_loaded}")
    runScene(singleSceneID_loaded)
else:
    # Full dataset: iterate scene-by-scene to keep memory low
    print("🧩 Inference for full test set (scene-by-scene)")
    for sid in sorted(sceneGridSizes.keys()):
        runScene(sid)