from load import *  #buildDS, stitchPatches, getSceneGridSizes
from model import *
from convert import asBatchOne, ConvertToTflite, convertToEdge
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import shutil, datetime
import json
import os
import numpy as np
import tensorflow as tf
from evaluate import evaluatePRC
from pathlib import Path
import sys

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # flush so you see it immediately
    def flush(self):
        for f in self.files:
            f.flush()


batchSize         = 1
imgSize           = (192,192)
numFilters        = 32
numEpochs         = 1
modelArchitecture = cloudNetQ
valRatio          = 0.2
trainValDSSize    = 100
numCalBatches     = 1

(trainDS, valDS, trainSteps, 
 valSteps, testDS, singleSceneID) = buildDS(includeTestDS=False, batchSize=batchSize, 
                                            imgSize=imgSize, valRatio=valRatio, trainValDSSize=trainValDSSize)

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
runFolder = f"dev/pipeline/results/runs/run_{now}"
os.makedirs(runFolder, exist_ok=True)

log_file = open(Path(runFolder) / "history.txt", 'w')
tee = Tee(sys.stdout, log_file)
sys.stdout = tee
sys.stderr = tee  # Optional: log errors too

model   = modelArchitecture(batchShape=(batchSize, *imgSize, 3))
model.summary()

shutil.copy('dev/pipeline/model.py', f'{runFolder}/my_model.py')
plot_model(model, to_file=f'{runFolder}/model.pdf', show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint(f'{runFolder}/modelCheckpoint.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)

run_config = {
    "batch_size": batchSize,
    "epochs": numEpochs,
    "optimizer": "adam",
    "loss": "binarycrossentropy",
    "metrics": ["MeanIoU", "diceCoefficient"],
    "learning_rate": model.optimizer.lr.numpy() if hasattr(model.optimizer, 'lr') else None,
    "train_steps": trainSteps,
    "val_steps": valSteps,
    "random_seed": 42,
    "model_architecture": modelArchitecture.__name__
    # Add anything else relevant!
}

# Save as a JSON file
with open(f'{runFolder}/experiment_config.json', "w") as f:
    json.dump(run_config, f, indent=4, default=str)

with open(f'{runFolder}/model_architecture.json', "w") as f:
    f.write(model.to_json())

historyuNet = model.fit(trainDS, validation_data=valDS, epochs=numEpochs, callbacks=[checkpoint, earlyStop, lrReduce], 
                        steps_per_epoch=trainSteps, validation_steps=valSteps, validation_freq=5, verbose=2)

model.save(f'{runFolder}/endModel.h5')
print('-' * 40)
print ('Training end, model saved successfully!')
print('-' * 40)

with open(f'{runFolder}/training_history.json', "w") as f:
    json.dump(historyuNet.history, f, indent=4, default=str)

# --- PRC on validation (pooled pixels) ---
yScoresList, yTrueList = [], []
for xBatch, gtBatch in valDS.take(valSteps):     # valDS is repeated → limit by valSteps
    yPred = model(xBatch, training=False)        # [B,H,W,1], sigmoid outputs
    yScoresList.append(tf.squeeze(yPred, -1).numpy())  # [B,H,W]
    yTrueList.append(gtBatch.numpy())                    # [B,H,W]

yScores = np.concatenate(yScoresList, axis=0)    # [N,H,W]
yTrue   = np.concatenate(yTrueList,   axis=0)    # [N,H,W]

plotPath = Path(runFolder) / "val_prc.png"
bestThr, bestF1 = evaluatePRC(yTrue, yScores, showPlot=False, savePlotPath=plotPath, title="Validation PRC (pooled pixels)")

with open(f'{runFolder}/val_threshold.json', "w") as f:
    json.dump({"best_threshold": float(bestThr), "best_f1": float(bestF1)}, f, indent=2)

print(f"Validation PRC -> Thr* = {bestThr:.6f}, F1 = {bestF1:.4f} (saved to val_threshold.json)")

model = asBatchOne(model, modelArchitecture, imgSize, numFilters)
model = ConvertToTflite(model, runFolder, imgSize, numCalBatches)
convertToEdge(runFolder)
