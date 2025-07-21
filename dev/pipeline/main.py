from load import *  #buildDS, stitchPatches, getSceneGridSizes
from model import uNetQ
from convert import asBatchOne, ConvertToTflite, convertToEdge
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import shutil, datetime
import json
import os


batchSize         = 1
imgSize           = (192,192)
numEpochs         = 1
modelArchitecture = uNetQ
valRatio          = 0.1 
trainValDSSize    = 10
numCalBatches     = 1

(trainDS, valDS, trainSteps, 
 valSteps, testDS, singleSceneID) = buildDS(includeTestDS=False, batchSize=batchSize, 
                                            imgSize=imgSize, valRatio=valRatio, trainValDSSize=trainValDSSize)

model   = modelArchitecture(batchShape=(batchSize, *imgSize, 4))
model.summary()

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
runFolder = f"dev/main/results/runs/run_{now}"
os.makedirs(runFolder, exist_ok=True)
shutil.copy('dev/main/model.py', f'{runFolder}/my_model.py')
plot_model(model, to_file=f'{runFolder}/model.pdf', show_shapes=True, show_layer_names=True)

checkpoint = ModelCheckpoint(f'{runFolder}/modelCheckpoint.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)

run_config = {
    "batch_size": batchSize,
    "epochs": numEpochs,
    "optimizer": "adam",
    "loss": "softJaccardLoss",
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
                        steps_per_epoch=trainSteps, validation_steps=valSteps)

model.save(f'{runFolder}/endModel.h5')
print('-' * 40)
print ('Training end, model saved successfully!')
print('-' * 40)

model = asBatchOne(model, modelArchitecture, imgSize)
model = ConvertToTflite(model, runFolder, imgSize, numCalBatches)
convertToEdge(runFolder)
