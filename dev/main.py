from load import *  #buildDS, stitchPatches, getSceneGridSizes
from model import uNetq, uNet
from convert import ConvertToTflite, convertToEdge


batchSize   = 1
imgSize     = (192,192)

(trainDS, valDS, trainSteps, 
 valSteps, testDS, singleSceneID) = buildDS(includeTestDS=False, batchSize=batchSize, 
                                            imgSize=imgSize, trainValDSSize=100)

model   = uNetq(batchShape=(batchSize, 192, 192, 4))
historyuNet = model.fit(trainDS, validation_data=None, epochs=1, 
                        steps_per_epoch=trainSteps, validation_steps=None)

#model.save("dev/uNet.h5")

model = ConvertToTflite(model)
convertToEdge(model)
