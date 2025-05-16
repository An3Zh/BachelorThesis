from load import buildDS
from model import simple
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

baseDir          = Path(r"C:\Users\andre\Documents\BA\Dev\Data\38-Cloud_training")
csvFilenames = Path(r"C:\Users\andre\Documents\BA\Dev\Data\training_patches_38-cloud_nonempty.csv")
imgSize = (192,192)
batchSize = 8

(trainDS, 
 valDS, 
 trainSteps, 
 valSteps) = buildDS(csvFilenames, baseDir, batchSize=batchSize, imgSize=imgSize, seed=42)

modelSimple   = simple(inputShape=(192,192,4))
historySimple = modelSimple.fit(trainDS, 
                                validation_data=valDS, 
                                epochs=3, 
                                steps_per_epoch=trainSteps, 
                                validation_steps=valSteps)
modelSimple.save("Simple_savedmodel", save_format="tf")