from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import time
from tensorflow_model_optimization.quantization.keras import quantize_scope
from model import softJaccardLoss



# --- Config ---
batchSize   = 1
imgSize     = (384,384)
singleSceneID=29044

# --- Load Data ---
(trainDS, valDS, trainSteps, valSteps, testDS, singleSceneID) = buildDS(
    includeTestDS=True,
    batchSize=batchSize,
    imgSize=None,
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

# --- Run Inference ---

with quantize_scope({'softJaccardLoss': softJaccardLoss}):
    model = tf.keras.models.load_model('dev/uNetq.h5')

predictions = []

start = time.time()
for xBatch, _ in tqdm(testDS, total=total):
    yPred = model(xBatch)  # shape: [B, H, W, 1]
    predictions.extend([p.numpy() for p in tf.unstack(yPred)])
print(f"⏱️ Inference completed in {time.time() - start:.2f} seconds")

# --- Prepare Prediction Array ---
predictionsArray = np.array([np.squeeze(p) for p in predictions])
#np.save("predictions_array.npy", predictionsArray)

# --- Stitch Output ---
stitchedScenes = stitchPatches(predictionsArray, singleSceneID)

sceneId = singleSceneID  # or manually set it, e.g., 3052

plt.figure(figsize=(12, 12))
plt.imshow(stitchedScenes[sceneId], cmap="gray")
plt.title(f"Scene {sceneId}")
plt.axis('off')
plt.show()
