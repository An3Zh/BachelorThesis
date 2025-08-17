from load import *  # buildDS, stitchPatches, getSceneGridSizes
from model import *  # uses cloudNetQ, diceLoss, diceCoefficient
from convert import asBatchOne, ConvertToTflite, convertToEdge
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import BinaryIoU, AUC, Precision, Recall
import shutil, datetime
import json
import os
import numpy as np
import tensorflow as tf
from evaluate import evaluatePRC
from pathlib import Path
import sys
import random

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# -----------------------------
# Config (tune as needed)
# -----------------------------
batchSize         = 16      # increase if memory allows; if dropping to 1–2, consider freezing BN
imgSize           = (192, 192)
numFilters        = 32
numEpochs         = 200
modelArchitecture = cloudNetQ
valRatio          = 0.2
trainValDSSize    = 5155    # raise above debug values if you have data
numCalBatches     = 64      # more batches => better PTQ calibration

# -----------------------------
# Data
# -----------------------------
(trainDS, valDS, trainSteps,
 valSteps, testDS, singleSceneID) = buildDS(
    includeTestDS=False,
    batchSize=batchSize,
    imgSize=imgSize,
    valRatio=valRatio,
    trainValDSSize=trainValDSSize
)

# Optional: lightweight augmentation (applied on batches)
def augmentBatch(xBatch, yBatch):
    def _aug(x, y):
        # ---------------- Flips ----------------
        flipH = tf.random.uniform([]) < 0.5
        flipV = tf.random.uniform([]) < 0.5
        x2 = tf.cond(flipH, lambda: tf.image.flip_left_right(x), lambda: x)
        y2 = tf.cond(flipH, lambda: tf.image.flip_left_right(y[..., tf.newaxis]), lambda: y[..., tf.newaxis])
        y2 = tf.squeeze(y2, -1)
        x3 = tf.cond(flipV, lambda: tf.image.flip_up_down(x2), lambda: x2)
        y3 = tf.cond(flipV, lambda: tf.image.flip_up_down(y2[..., tf.newaxis]), lambda: y2[..., tf.newaxis])
        y3 = tf.squeeze(y3, -1)

        # ---------------- Rotations ----------------
        doRotate = tf.random.uniform([]) < 0.5
        ifRotate = lambda: tf.image.rot90(x3, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
        ifRotateMask = lambda: tf.image.rot90(y3[..., tf.newaxis], k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))
        x4 = tf.cond(doRotate, ifRotate, lambda: x3)
        y4 = tf.cond(doRotate, ifRotateMask, lambda: y3[..., tf.newaxis])
        y4 = tf.squeeze(y4, -1)

        # ---------------- Random crop + resize ----------------
        doCrop = tf.random.uniform([]) < 0.5
        cropFrac = 0.9  # keep 90% of area
        cropSize = [tf.cast(tf.shape(x4)[0] * cropFrac, tf.int32),
                    tf.cast(tf.shape(x4)[1] * cropFrac, tf.int32)]
        ifCrop = lambda: (tf.image.resize(tf.image.random_crop(x4, size=[cropSize[0], cropSize[1], 4]), (192, 192)),
                          tf.image.resize(tf.image.random_crop(y4[..., tf.newaxis], size=[cropSize[0], cropSize[1], 1]), (192, 192)))
        ifNoCrop = lambda: (x4, y4[..., tf.newaxis])
        x5, y5 = tf.cond(doCrop, ifCrop, ifNoCrop)
        y5 = tf.squeeze(y5, -1)

        # ---------------- Brightness/contrast ----------------
        x5 = tf.image.random_brightness(x5, max_delta=0.05)
        x5 = tf.image.random_contrast(x5, lower=0.95, upper=1.05)
        x5 = tf.clip_by_value(x5, 0.0, 1.0)

        return x5, y5

    xBatch, yBatch = tf.map_fn(lambda z: _aug(z[0], z[1]),
                               (xBatch, yBatch),
                               fn_output_signature=(tf.float32, tf.float32))
    return xBatch, yBatch

# apply augmentation only to training stream
trainDS = trainDS.map(augmentBatch, num_parallel_calls=tf.data.AUTOTUNE)

# -----------------------------
# Logging / Run folder
# -----------------------------
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
runFolder = f"dev/pipeline/results/runs/run_{now}"
os.makedirs(runFolder, exist_ok=True)

log_file = open(Path(runFolder) / "history.txt", 'w')
tee = Tee(sys.stdout, log_file)
sys.stdout = tee
sys.stderr = tee

# -----------------------------
# Model
# -----------------------------
model = modelArchitecture(batchShape=(batchSize, *imgSize, 4), filters=numFilters)
model.summary()

shutil.copy('dev/pipeline/model.py', f'{runFolder}/my_model.py')
plot_model(model, to_file=f'{runFolder}/model.pdf', show_shapes=True, show_layer_names=True)

# -----------------------------
# Optimizer, loss, metrics
# -----------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

def bceDiceLoss(yTrue, yPred):
    return tf.keras.losses.binary_crossentropy(yTrue, yPred) + 0.5 * diceLoss(yTrue, yPred)

model.compile(
    optimizer=optimizer,
    loss=bceDiceLoss,
    metrics=[
        'accuracy',
        BinaryIoU(threshold=0.5),        # interpretable per-epoch IoU
        diceCoefficient,                  # your dice metric
        AUC(curve='PR', name='AUC_PR'),   # aligns with PRC thresholding step
        Precision(name='precision'),
        Recall(name='recall')
    ]
)

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = ModelCheckpoint(
    f'{runFolder}/modelCheckpoint.h5',
    monitor='val_diceCoefficient', mode='max',
    save_best_only=True, verbose=1
)
earlyStop = EarlyStopping(
    monitor='val_diceCoefficient', mode='max',
    patience=10, restore_best_weights=True
)
lrReduce = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4,
    min_lr=1e-6, verbose=1
)

# -----------------------------
# Save experiment config
# -----------------------------
run_config = {
    "batch_size": batchSize,
    "epochs": numEpochs,
    "optimizer": "adam",
    "loss": "bce + 0.5*dice",
    "metrics": ["BinaryIoU@0.5", "diceCoefficient", "AUC_PR", "Precision", "Recall"],
    "learning_rate": float(model.optimizer.lr.numpy()) if hasattr(model.optimizer, 'lr') else None,
    "train_steps": int(trainSteps),
    "val_steps": int(valSteps),
    "random_seed": SEED,
    "model_architecture": modelArchitecture.__name__,
    "img_size": imgSize,
}
with open(f'{runFolder}/experiment_config.json', "w") as f:
    json.dump(run_config, f, indent=4, default=str)

with open(f'{runFolder}/model_architecture.json', "w") as f:
    f.write(model.to_json())

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    trainDS,
    validation_data=valDS,
    epochs=numEpochs,
    callbacks=[checkpoint, earlyStop, lrReduce],
    steps_per_epoch=trainSteps,
    validation_steps=valSteps,
    validation_freq=1,   # validate every epoch so callbacks work optimally
    verbose=2
)

model.save(f'{runFolder}/endModel.h5')
print('-' * 40)
print('Training end, model saved successfully!')
print('-' * 40)

with open(f'{runFolder}/training_history.json', "w") as f:
    json.dump(history.history, f, indent=4, default=str)

# -----------------------------
# PRC on validation (pooled pixels) — keep your existing evaluation
# -----------------------------
yScoresList, yTrueList = [], []
for xBatch, gtBatch in valDS.take(valSteps):
    yPred = model(xBatch, training=False)           # [B,H,W,1]
    yScoresList.append(tf.squeeze(yPred, -1).numpy())  # [B,H,W]
    yTrueList.append(gtBatch.numpy())                   # [B,H,W]

yScores = np.concatenate(yScoresList, axis=0)
yTrue   = np.concatenate(yTrueList,   axis=0)

plotPath = Path(runFolder) / "val_prc.png"
bestThr, bestF1 = evaluatePRC(yTrue, yScores, showPlot=False, savePlotPath=plotPath, title="Validation PRC (pooled pixels)")

with open(f'{runFolder}/val_threshold.json', "w") as f:
    json.dump({"best_threshold": float(bestThr), "best_f1": float(bestF1)}, f, indent=2)

print(f"Validation PRC -> Thr* = {bestThr:.6f}, F1 = {bestF1:.4f} (saved to val_threshold.json)")

# -----------------------------
# Export for edge
# -----------------------------
model = asBatchOne(model, modelArchitecture, imgSize, numFilters)
model = ConvertToTflite(model, runFolder, imgSize, numCalBatches)
convertToEdge(runFolder)