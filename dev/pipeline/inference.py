from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import time
import os
from model import softJaccardLoss, diceCoefficient
from tensorflow_model_optimization.quantization.keras import quantize_scope
import tensorflow.image as tfi
from PIL import Image
from pathlib import Path

# --- Config ---
batchSize     = 1
imgSize       = (192,192)
singleSceneID = 3052
Upsample      = imgSize is not None  # <-- 🔧 your new flag
BaseFolder    = Path(r"c:\Users\andre\Documents\BA\dev\pipeline\results\runs")
runFolder     = BaseFolder / "run_20250806_171557"
modelPath     = runFolder  / "endModel.h5"
saveFolder    = runFolder  / "evaluation\inference"

# --- Load Data ---
(trainDS, valDS, trainSteps, valSteps, testDS, singleSceneID) = buildDS(
    includeTestDS=True,
    batchSize=batchSize,
    imgSize=imgSize,
    singleSceneID=singleSceneID  # 0 for random
)

# --- Prepare Inference, Define Size ---
sceneGridSizes = getSceneGridSizes()    
if singleSceneID is not None:
    cols, rows = sceneGridSizes[singleSceneID]
    total = math.ceil((cols * rows) / batchSize)
    print(f"🧩 Inference for Scene {singleSceneID} ({cols}×{rows} patches)")
else:
    total = math.ceil(9201 / batchSize)
    print("🧩 Inference for full test set")

# --- Load Model ---
with quantize_scope():
    model = tf.keras.models.load_model(
        modelPath,
        custom_objects={'softJaccardLoss': softJaccardLoss,
                        'diceCoefficient': diceCoefficient}
    )

# --- Run Inference ---
predictions = []
start = time.time()
for xBatch, _ in tqdm(testDS, total=total):
    yPred = model(xBatch)  # shape: [B, H, W, 1]
    predictions.extend([p.numpy() for p in tf.unstack(yPred)])
print(f"⏱️ Inference completed in {time.time() - start:.2f} seconds")

# --- Upsample if enabled ---
if Upsample:
    predictions = [
        tfi.resize(p, [384, 384], method="bilinear").numpy()
        for p in predictions
    ]

# --- Prepare Prediction Array ---
predictionsArray = np.array([np.squeeze(p) for p in predictions])  # shape: [N, H, W]

# --- Stitch Output ---
stitchedScenes = stitchPatches(predictionsArray, singleSceneID)

# --- Save stitched scenes ---
os.makedirs(saveFolder, exist_ok=True)
for sceneId, sceneArray in stitchedScenes.items():
    npyPath = os.path.join(saveFolder, f"scene_{sceneId}.npy")
    np.save(npyPath, sceneArray)
    # Normalize prediction to [0, 255] for PNG saving
    imgArray = (sceneArray * 255).clip(0, 255).astype(np.uint8)
    pngPath = os.path.join(saveFolder, f"scene_{sceneId}.png")
    Image.fromarray(imgArray).save(pngPath)
    print(f"✅ Scene {sceneId} stitched and saved")

# --- Optional: Show sample
sceneId = singleSceneID
plt.figure(figsize=(12, 12))
plt.imshow(stitchedScenes[sceneId], cmap="gray")
plt.title(f"Scene {sceneId}")
plt.axis('off')
plt.show()