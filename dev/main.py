from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import time

# --- Config ---
batchSize   = 32
imgSize     = (192,192)
modelPath   = r"C:\Users\andre\Documents\BA\Dev\!Archiv\!Models\Simple_savedmodel"

# --- Load Data ---
(trainDS, valDS, trainSteps, valSteps, testDS, singleSceneID) = buildDS(
    includeTestDS=True,
    batchSize=batchSize,
    imgSize=imgSize,
    singleSceneID=None  # 0 for random
)

# --- Prepare Inference ---
sceneGridSizes = getSceneGridSizes()
if singleSceneID is not None:
    cols, rows = sceneGridSizes[singleSceneID]
    total = math.ceil((cols * rows) / batchSize)
    print(f"🧩 Inference for Scene {singleSceneID} ({cols}×{rows} patches)")
else:
    total = math.ceil(9201 / batchSize)
    print("🧩 Inference for full test set")

# --- Run Inference ---
model = load_model(modelPath)
predictions = []

start = time.time()
for xBatch, _ in tqdm(testDS, total=total):
    yPred = model(xBatch)  # shape: [B, H, W, 1]
    predictions.extend([p.numpy() for p in tf.unstack(yPred)])
print(f"⏱️ Inference completed in {time.time() - start:.2f} seconds")

# --- Prepare Prediction Array ---
predictionsArray = np.array([np.squeeze(p) for p in predictions])
np.save("predictions_array.npy", predictionsArray)

# --- Stitch Output ---
stitchedScenes = stitchPatches(predictionsArray, singleSceneID)

sceneId = singleSceneID  # or manually set it, e.g., 3052

plt.figure(figsize=(12, 12))
plt.imshow(stitchedScenes[sceneId], cmap="gray")
plt.title(f"Scene {sceneId}")
plt.axis('off')
plt.show()