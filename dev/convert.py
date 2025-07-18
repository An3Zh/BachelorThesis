import tensorflow as tf
import subprocess
from pathlib import Path
#from model import softJaccardLoss
from load import buildDS



def representativeDatasetGen(imgSize):
    batchSize = 1       

    trainDS, _, _, _, _, _ = buildDS(includeTestDS=False, batchSize=batchSize, imgSize=imgSize)
    i = 0
    for xBatch, _ in trainDS.take(1):  # You can increase this if needed
        print(f"Calibration batch {i}")
        i+=1
        yield [xBatch]

def asBatchOne(model, modelArchitecture, imgSize):
    modelBatchOne = modelArchitecture(batchShape=(1, *imgSize, 4))
    modelBatchOne.set_weights(model.get_weights())
    return modelBatchOne

def ConvertToTflite(model, imgSize):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representativeDatasetGen(imgSize)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tfliteModel = converter.convert()
    tflite_model_quant_file = Path("dev/quant.tflite")
    tflite_model_quant_file.write_bytes(tfliteModel)
    print("saved converted model")
    return tfliteModel

def convertToEdge():
    command = [
        "wsl", "edgetpu_compiler",
        "dev/quant.tflite", "-o",
        "dev/"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
