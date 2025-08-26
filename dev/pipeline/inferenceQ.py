from pathlib import Path
import os, math, time
import numpy as np
import tensorflow as tf
import tensorflow.image as tfi
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from load import buildDS, stitchPatches, getSceneGridSizes  # dataset + stitching

# ------------ Config ------------
batchSize     = 16                    #EXPERIMENTAL SETUP
imgSize       = (192, 192)
singleSceneID = 3052                  # 0 for random, or pick a specific ID
upsample      = imgSize is not None
upsampleSize  = (384, 384)

baseFolder    = Path(r"c:\Users\andre\Documents\BA\dev\pipeline\results\runs")
runFolder     = baseFolder / "run_20250719_170647"
tfliteModelPath = runFolder / "quant.tflite" 
saveFolder    = runFolder / "evaluationQ" / "inference"

# ------------ Data ------------
(trainDS, valDS, trainSteps, valSteps, testDS, singleSceneID) = buildDS(
    includeTestDS=True,
    batchSize=batchSize,   
    imgSize=imgSize,
    singleSceneID=singleSceneID
)

sceneGrid = getSceneGridSizes()
if singleSceneID is not None:
    cols, rows = sceneGrid[singleSceneID]
    totalBatches = math.ceil((cols * rows) / batchSize)
    print(f"🧩 Inference for Scene {singleSceneID} ({cols}×{rows} patches)")
else:
    totalBatches = math.ceil(9201 / batchSize)
    print("🧩 Inference for full test set")

# ------------ TFLite Interpreter ------------
interpreter = tf.lite.Interpreter(model_path=str(tfliteModelPath), num_threads=os.cpu_count() or 4)
interpreter.allocate_tensors()
inDet  = interpreter.get_input_details()[0]
outDet = interpreter.get_output_details()[0]

# EXPERIMENTAL
expectedB = int(inDet['shape'][0])  # likely 16
assert expectedB == batchSize, f"Model expects batch={expectedB}, but batchSize={batchSize}."

# ------------ Run Inference ------------
predictions = []
start = time.time()

for xBatch, _ in tqdm(testDS, total=totalBatches):
    # xBatch: [B,H,W,C], float32 [0,1]
    x = xBatch.numpy() if tf.is_tensor(xBatch) else xBatch
    b = x.shape[0]

    if b == expectedB:
        feed = x
        take = b
    else:
        # last partial batch - pad by repeating last sample to size 16
        feed = np.concatenate([x, np.repeat(x[-1:,...], expectedB - b, axis=0)], axis=0)
        take = b

    interpreter.set_tensor(inDet['index'], feed.astype(inDet['dtype']))
    interpreter.invoke()

    y = interpreter.get_tensor(outDet['index']).astype(np.float32)  # [16,H,W,1] float (FP32 model)
    y = y[:take]  # slice back to real batch size

    # accumulate
    for p in y:
        predictions.append(p)  # [H,W,1]

print(f"⏱️ Inference completed in {time.time() - start:.2f} seconds for {len(predictions)} patches")

# ------------ Stitch per-scene (upsample once per scene) ------------
predictionsArray = np.array([np.squeeze(p) for p in predictions], dtype=np.float32)
stitchedScenes = stitchPatches(predictionsArray, singleSceneID)

saveFolder.mkdir(parents=True, exist_ok=True)
factor = (upsampleSize[0] // imgSize[0]) if upsample else 1

for sceneId, sceneArray in stitchedScenes.items():
    if upsample and factor != 1:
        h, w = sceneArray.shape
        sceneArray = tfi.resize(sceneArray[..., None], [h * factor, w * factor],
                                method="bilinear").numpy().squeeze()

    npyPath = saveFolder / f"scene_{sceneId}.npy"
    pngPath = saveFolder / f"scene_{sceneId}.png"
    np.save(npyPath, sceneArray)
    Image.fromarray((np.clip(sceneArray, 0.0, 1.0) * 255).astype(np.uint8)).save(pngPath)
    print(f"✅ Scene {sceneId} stitched and saved → {pngPath.name}")

# ------------ Optional preview ------------
if singleSceneID is not None:
    showImg = stitchedScenes[singleSceneID]
    if upsample and factor != 1:
        showImg = tfi.resize(showImg[..., None], [showImg.shape[0]*factor, showImg.shape[1]*factor],
                             method="bilinear").numpy().squeeze()
    plt.figure(figsize=(10, 10))
    plt.imshow(showImg, cmap="gray")
    plt.title(f"Scene {singleSceneID}")
    plt.axis('off')
    plt.show()