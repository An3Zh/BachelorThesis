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
    seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)

    # ensure mask has channel dim
    if yBatch.shape.rank == 3:
        yBatch = yBatch[..., tf.newaxis]

    # flips (same seeds for x/y)
    xBatch = tf.image.stateless_random_flip_left_right(xBatch, seed=seed)
    yBatch = tf.image.stateless_random_flip_left_right(yBatch, seed=seed)

    xBatch = tf.image.stateless_random_flip_up_down(xBatch, seed=seed + 1)
    yBatch = tf.image.stateless_random_flip_up_down(yBatch, seed=seed + 1)

    # 0/90/180/270 rotation (mask-safe)
    k = tf.random.stateless_uniform([], minval=0, maxval=4, dtype=tf.int32, seed=seed + 2)
    xBatch = tf.image.rot90(xBatch, k)
    yBatch = tf.image.rot90(yBatch, k)

    # random crop (one box for the whole batch → tiny & fast)
    def applyCrop(x, y):
        cropFrac = tf.constant(0.9, tf.float32)  # keep 90% area
        y1 = tf.random.stateless_uniform([], seed=seed + 3, minval=0.0, maxval=1.0 - cropFrac)
        x1 = tf.random.stateless_uniform([], seed=seed + 4, minval=0.0, maxval=1.0 - cropFrac)
        y2, x2 = y1 + cropFrac, x1 + cropFrac

        bsz = tf.shape(x)[0]
        boxes = tf.tile(tf.reshape(tf.stack([y1, x1, y2, x2]), [1, 4]), [bsz, 1])
        boxIdx = tf.range(bsz)

        # keep original HxW after crop
        targetSize = tf.shape(x)[1:3]
        x = tf.image.crop_and_resize(x, boxes, boxIdx, targetSize, method='bilinear')
        y = tf.image.crop_and_resize(y, boxes, boxIdx, targetSize, method='nearest')
        return x, y

    doCrop = tf.random.stateless_uniform([], seed=seed + 5) > 0.5
    xBatch, yBatch = tf.cond(doCrop, lambda: applyCrop(xBatch, yBatch), lambda: (xBatch, yBatch))

    # photometric (image only)
    xBatch = tf.image.stateless_random_brightness(xBatch, max_delta=0.05, seed=seed + 6)
    xBatch = tf.image.stateless_random_contrast(xBatch, lower=0.95, upper=1.05, seed=seed + 7)
    xBatch = tf.clip_by_value(xBatch, 0.0, 1.0)

    return xBatch, tf.squeeze(yBatch, -1)

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
    filepath=f'{runFolder}/ckpt',
    monitor='val_diceCoefficient', mode='max',
    save_best_only=True, save_weights_only=True, verbose=1
)
earlyStop = EarlyStopping(
    monitor='val_diceCoefficient', mode='max',
    patience=12, restore_best_weights=True, verbose=1
)
lrReduce = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=6,
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
model.save(f'{runFolder}/endModel')
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