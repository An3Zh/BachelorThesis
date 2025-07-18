import tensorflow as tf
import subprocess
from pathlib import Path
from load import buildDS
import os
import platform



def representativeDatasetGen(imgSize):
    batchSize = 1       

    trainDS, _, _, _, _, _ = buildDS(includeTestDS=False, batchSize=batchSize, imgSize=imgSize)
    i = 0
    for xBatch, _ in trainDS.take(20):  # You can increase this if needed
        print(f"Calibration batch {i}")
        i+=1
        yield [xBatch]

def asBatchOne(model, modelArchitecture, imgSize):
    modelBatchOne = modelArchitecture(batchShape=(1, *imgSize, 4))
    modelBatchOne.set_weights(model.get_weights())
    return modelBatchOne

def ConvertToTflite(model, runFolder, imgSize):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representativeDatasetGen(imgSize)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tfliteModel = converter.convert()
    tflite_model_quant_file = Path(runFolder) / "quant.tflite"
    tflite_model_quant_file.write_bytes(tfliteModel)
    print('-' * 40)
    print("Saved converted model successfully!")
    print('-' * 40)
    return tfliteModel

def to_wsl_path(path):
    abs_path = os.path.abspath(path)
    drive, rest = abs_path[0], abs_path[2:]
    return f"/mnt/{drive.lower()}{rest.replace(os.sep, '/')}"

def convertToEdge(runFolder):
    tflite_path = os.path.join(runFolder, "quant.tflite")
    output_dir = runFolder
    system_type = platform.system().lower()
    
    if system_type == "windows":
        # Use WSL and convert paths
        tflite_path = to_wsl_path(tflite_path)
        output_dir = to_wsl_path(output_dir)
        command = [
            "wsl", "edgetpu_compiler",
            tflite_path, "-o", output_dir
        ]
    elif system_type == "linux":
        # Native Linux—use standard paths
        command = [
            "edgetpu_compiler",
            tflite_path, "-o", output_dir
        ]
    else:
        raise RuntimeError(f"Unsupported platform: {system_type}")
    
    print('Running EdgeTPU Compiler now:')
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
