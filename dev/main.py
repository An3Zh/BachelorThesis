from load import *  #buildDS, stitchPatches, getSceneGridSizes
from model import uNetQ
from convert import asBatchOne, ConvertToTflite, convertToEdge
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


batchSize   = 4
imgSize     = (192,192)

(trainDS, valDS, trainSteps, 
 valSteps, testDS, singleSceneID) = buildDS(includeTestDS=False, batchSize=batchSize, 
                                            imgSize=imgSize, valRatio=0.1)

model   = uNetQ(batchShape=(batchSize, *imgSize, 4))

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lrReduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6, verbose=1)

with open(f"dev/model_architecture.json", "w") as f:
    f.write(model.to_json())

historyuNet = model.fit(trainDS, validation_data=valDS, epochs=10, callbacks=[checkpoint, earlyStop, lrReduce], 
                        steps_per_epoch=trainSteps, validation_steps=valSteps)

model.save('endModel.h5')

model = asBatchOne(model, uNetQ, imgSize)
model = ConvertToTflite(model, imgSize)
convertToEdge()
