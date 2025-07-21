from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import time
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.quantization.keras import quantize_scope
from model import softJaccardLoss, diceCoefficient



# --- Config ---
batchSize     = 1
imgSize       = (192,192)
singleSceneID = 0

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

# --- Run Inference ---

tfModelPath = r"C:\Users\andre\Documents\BA\dev\results\run_20250719_170647\endModel.h5"

with quantize_scope():
    model = tf.keras.models.load_model(
        tfModelPath,
        custom_objects={'softJaccardLoss': softJaccardLoss,
                        'diceCoefficient': diceCoefficient}
    )

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
