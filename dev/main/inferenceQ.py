from load import buildDS, stitchPatches, getSceneGridSizes
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import math
import time



# --- Config ---
batchSize     = 1
imgSize       = (192,192)
singleSceneID = 29044

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

# --- Load TFLite Model ---

tfliteModelPath = "dev/model192epochs1/quant.tflite"
interpreter = tf.lite.Interpreter(model_path=tfliteModelPath)
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()


predictions = []

start = time.time()


for xBatch, _ in tqdm(testDS, total=total):
    inputArray = xBatch.numpy() if tf.is_tensor(xBatch) else xBatch

    scale, zeroPoint = inputDetails[0]['quantization']
    quantizedInput = np.round(inputArray / scale + zeroPoint).astype(np.int8)

    print('putThatIn')
    interpreter.set_tensor(inputDetails[0]['index'], quantizedInput)
    print('putThatIN')
    interpreter.invoke()
    print('putThatOut')
    outputData = interpreter.get_tensor(outputDetails[0]['index'])
    print('putThatOUT')
    outScale, outZeroPoint = outputDetails[0]['quantization']
    outputDequant = (outputData.astype(np.float32) - outZeroPoint) * outScale
    # Save as probabilities, squeeze if needed
    for p in outputDequant:
        predictions.append(np.squeeze(p))


print(f"⏱️ Inference completed in {time.time() - start:.2f} seconds")

# --- Prepare Prediction Array ---
predictionsArray = np.array(predictions)

# --- Stitch Output ---
stitchedScenes = stitchPatches(predictionsArray, singleSceneID)

sceneId = singleSceneID

plt.figure(figsize=(12, 12))
plt.imshow(stitchedScenes[sceneId], cmap="gray")
plt.title(f"Scene {sceneId}")
plt.axis('off')
plt.show()
