import tensorflow as tf
import math
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import random
import json
from matplotlib import pyplot as plt
import os

def getFolders(baseDir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    trainFolders = {
        "red":   baseDir / "38-Cloud_training" / "train_red",
        "green": baseDir / "38-Cloud_training" / "train_green",
        "blue":  baseDir / "38-Cloud_training" / "train_blue",
        "nir":   baseDir / "38-Cloud_training" / "train_nir",
        "gt":    baseDir / "38-Cloud_training" / "train_gt",
    }

    testFolders = {
        "red":   baseDir / "38-Cloud_test" / "test_red",
        "green": baseDir / "38-Cloud_test" / "test_green",
        "blue":  baseDir / "38-Cloud_test" / "test_blue",
        "nir":   baseDir / "38-Cloud_test" / "test_nir",
    }

    return trainFolders, testFolders

def getSceneGridSizes() -> Dict[int, tuple]:
    sceneGridSizes = {
        3052: (20, 21), 18008: (24, 24), 29032: (21, 21), 29041: (21, 21),
        29044: (20, 21), 32030: (21, 21), 32035: (20, 21), 32037: (20, 21),
        34029: (21, 21), 34033: (21, 21), 34037: (21, 21), 35029: (21, 21),
        35035: (20, 21), 39035: (20, 21), 50024: (21, 21), 63012: (23, 23),
        63013: (23, 23), 64012: (23, 23), 64015: (22, 22), 66014: (22, 23)
    }
    return sceneGridSizes

def loadTIF(path: tf.Tensor) -> tf.Tensor:
    def _loadTIFNumpy(pathBytes):
        path = pathBytes.decode("utf-8")
        #print("Loading TIF:", path)
        return np.array(Image.open(path), dtype=np.uint16)
    
    tensor = tf.numpy_function(_loadTIFNumpy, [path], tf.uint16)
    tensor.set_shape([None, None])  
    return tensor

def getPaths(
    filename: tf.Tensor,
    folders: Dict[str, Path],
    isTestDS: Optional[bool] = False,
) -> Union[
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
]:
    
    red   = tf.strings.join([str(folders["red"]),    "/", "red_",   filename, ".TIF"])
    green = tf.strings.join([str(folders["green"]),  "/", "green_", filename, ".TIF"])
    blue  = tf.strings.join([str(folders["blue"]),   "/", "blue_",  filename, ".TIF"])
    nir   = tf.strings.join([str(folders["nir"]),    "/", "nir_",   filename, ".TIF"])
    
    if isTestDS:
        return red, green, blue, nir, filename
    else:
        gt    = tf.strings.join([str(folders["gt"]), "/", "gt_",    filename, ".TIF"])
        return red, green, blue, nir, gt

def loadDS(
    red: tf.Tensor,
    green: tf.Tensor,
    blue: tf.Tensor,
    nir: tf.Tensor,
    gt: Optional[tf.Tensor] = None,
    filename: Optional[tf.Tensor] = None,
    imgSize: Optional[Tuple[int, int]] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:

    r  = loadTIF(red)
    g  = loadTIF(green)
    b  = loadTIF(blue)
    n  = loadTIF(nir)

    x = tf.stack([r, g, b, n], axis=-1)                     # [H,W,4], uint16
    x = tf.cast(x, tf.float32) / 65535.0                    # [0,1] float32
    if imgSize is not None:
        x = tf.image.resize(x, imgSize, method='bilinear')                     # [192,192,4]

    if gt is not None:
        gt = loadTIF(gt)
        gt = tf.cast(gt > 0, tf.float32)                          # [0,1] float32
        if imgSize is not None:
            gt = tf.image.resize(gt[..., tf.newaxis], imgSize, method='nearest')    # [192,192,1]
            gt = tf.squeeze(gt, axis=-1)                          # [192,192]
        return x, gt
    else:
        return x, filename

def buildDS(
    includeTestDS: Optional[bool] = False,
    valRatio: float = 0.2,
    batchSize: int = 8,
    imgSize: Optional[Tuple[int, int]] = None,
    trainValDSSize: int = 5155,
    testDSSize: int = 9201,
    shuffleBuffTrain: int = 512,
    seed: int = 42,
    singleSceneID: Optional[int] = None
) -> Union[
    Tuple[tf.data.Dataset, tf.data.Dataset, int, int],
    Tuple[tf.data.Dataset, tf.data.Dataset, int, int, tf.data.Dataset, int]
]:
    

    # Load config from config.json (if present)
    configPath = Path(__file__).parent / "config.json"
    if configPath.exists():
        with open(configPath, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("couldn't find config.json")

    baseDir      = Path(config["baseDir"])
    csvTrainVal  = Path(config["csvTrainVal"])
    csvTest      = Path(config["csvTest"])

    trainFolders, testFolders = getFolders(baseDir)

    trainValPathsFn = lambda filename: getPaths(filename, trainFolders)
    trainValLoadFn  = lambda r, g, b, n, gt: loadDS(r, g, b, n, gt, imgSize=imgSize)

    # Retrieve both train and val datasets from .csv
    trainValDS = tf.data.TextLineDataset(str(csvTrainVal)).skip(1)
    trainValDS = trainValDS.shuffle(trainValDSSize, seed, reshuffle_each_iteration=False)

    # Calculate sizes and steps for automated tf dataset handling
    valSize    = int(trainValDSSize * valRatio)
    trainSize  = trainValDSSize - valSize
    trainSteps = math.ceil(trainSize / batchSize)
    valSteps   = math.ceil(valSize / batchSize)

    trainDS    = trainValDS.take(trainSize) # Take train DS
    valDS      = trainValDS.skip(trainSize) # The rest is for validation
    
    # Expand train dataset and fill it with images
    trainDS = trainDS.map(trainValPathsFn, num_parallel_calls=tf.data.AUTOTUNE)
    trainDS = trainDS.map(trainValLoadFn,  num_parallel_calls=tf.data.AUTOTUNE)
    trainDS = trainDS.shuffle(shuffleBuffTrain, seed, reshuffle_each_iteration=False)
    trainDS = trainDS.repeat()
    trainDS = trainDS.batch(batchSize).prefetch(tf.data.AUTOTUNE)

    # Expand val dataset and fill it with images
    valDS = valDS.map(trainValPathsFn, num_parallel_calls=tf.data.AUTOTUNE)
    valDS = valDS.map(trainValLoadFn,  num_parallel_calls=tf.data.AUTOTUNE)
    valDS = valDS.repeat()
    valDS = valDS.batch(batchSize).prefetch(tf.data.AUTOTUNE)

    # Similarly create test dataset
    if includeTestDS is True:
        if singleSceneID is not None:
            if singleSceneID == 0:
                sceneIDs = list(getSceneGridSizes().keys())
                singleSceneID = random.choice(sceneIDs)
                print(f"🔀 Randomly selected scene ID: {singleSceneID}")

            sceneCsvName = f"scene_{singleSceneID:05d}.csv"
            csvTest = csvTest.parent / "split_scenes" / sceneCsvName
            print(f"📂 Using test CSV: {csvTest}")


        testPathsFn = lambda filename: getPaths(filename, testFolders, isTestDS=True)
        testLoadFn  = lambda r, g, b, n, filename: loadDS(r, g, b, n, imgSize=imgSize, filename=filename)
        testDS = tf.data.TextLineDataset(str(csvTest)).skip(1)
        #testDS = testDS.shuffle(testDSSize, seed, reshuffle_each_iteration=False) DO NOT SHUFFLE, unless you are saving new .csv, used later for stitching order
        testDS = testDS.map(testPathsFn, num_parallel_calls=tf.data.AUTOTUNE)
        testDS = testDS.map(testLoadFn, num_parallel_calls=tf.data.AUTOTUNE)
        testDS = testDS.batch(batchSize).prefetch(tf.data.AUTOTUNE)
    else:
        testDS = None
        singleSceneID = None

    return trainDS, valDS, trainSteps, valSteps, testDS, singleSceneID
    
def stitchPatches(yPred: np.ndarray, singleSceneID: int = None):

    outputDir = Path(r"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes")
    outputDir.mkdir(parents=True, exist_ok=True)
    sceneGridSizes = getSceneGridSizes()
    scenesToProcess = (
        {singleSceneID: sceneGridSizes[singleSceneID]} if singleSceneID is not None
        else sceneGridSizes)
    stitchedScenes = {}
    i = 0

    for sceneId, (cols, rows) in scenesToProcess.items():

        patchH, patchW = yPred[i].shape
        canvas = np.zeros((rows * patchH, cols * patchW), dtype=yPred[i].dtype)

        expected = cols * rows
        if i + expected > len(yPred):
            raise ValueError(f"Scene {sceneId}: expected {expected} patches, but only {len(yPred) - i} available.")

        for row in range(rows):
            for col in range(cols):
                patch = yPred[i]
                y0, y1 = row * patchH, (row + 1) * patchH
                x0, x1 = col * patchW, (col + 1) * patchW
                canvas[y0:y1, x0:x1] = patch
                i += 1

        stitchedScenes[sceneId] = canvas

        imgPath = outputDir / f"scene_{sceneId}.png"
        npyPath = outputDir / f"scene_{sceneId}.npy"

        np.save(npyPath, canvas)

        if canvas.dtype != np.uint8:
            img = (canvas * 255).clip(0, 255).astype(np.uint8)
        else:
            img = canvas

        Image.fromarray(img).save(imgPath)

        print(f"✅ Scene {sceneId} stitched and saved → {imgPath.name}")
        
    return stitchedScenes

if __name__ == "__main__":

    batchSize = 2
    imgSize = (192,192)

    (trainDS, 
     valDS, 
     trainSteps, 
     valSteps,
     testDS, singleSceneID) = buildDS(includeTestDS=True, batchSize=batchSize, imgSize=imgSize, singleSceneID=3052)


    os.makedirs("train_vis", exist_ok=True)
    os.makedirs("val_vis", exist_ok=True)

    # --- Training batch ---
    for xb, mb in trainDS.take(1):
        print("\n--- Training Batch ---")
        print(f"x batch: {xb.shape}, {xb.dtype}, min={tf.reduce_min(xb).numpy()}, max={tf.reduce_max(xb).numpy()}")
        print(f"m batch: {mb.shape}, {mb.dtype}, unique={tf.unique(tf.reshape(mb, [-1])).y.numpy()}")

        for i in range(xb.shape[0]):
            fig, axs = plt.subplots(1, 5, figsize=(15, 4))
            axs[0].imshow(xb[i, :, :, 0], cmap='Reds')
            axs[0].set_title("Red")
            axs[1].imshow(xb[i, :, :, 1], cmap='Greens')
            axs[1].set_title("Green")
            axs[2].imshow(xb[i, :, :, 2], cmap='Blues')
            axs[2].set_title("Blue")
            axs[3].imshow(xb[i, :, :, 3], cmap='gray')
            axs[3].set_title("NIR")
            axs[4].imshow(mb[i], cmap='gray', vmin=0, vmax=1)
            axs[4].set_title("Mask")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"train_vis/train_batch_0_sample_{i}.png")
            plt.close(fig)

    # --- Validation batch ---
    for xb, mb in valDS.take(1):
        print("\n--- Validation Batch ---")
        print(f"x batch: {xb.shape}, {xb.dtype}, min={tf.reduce_min(xb).numpy()}, max={tf.reduce_max(xb).numpy()}")
        print(f"m batch: {mb.shape}, {mb.dtype}, unique={tf.unique(tf.reshape(mb, [-1])).y.numpy()}")
        for i in range(xb.shape[0]):
            fig, axs = plt.subplots(1, 5, figsize=(15, 4))
            axs[0].imshow(xb[i, :, :, 0], cmap='Reds')
            axs[0].set_title("Red")
            axs[1].imshow(xb[i, :, :, 1], cmap='Greens')
            axs[1].set_title("Green")
            axs[2].imshow(xb[i, :, :, 2], cmap='Blues')
            axs[2].set_title("Blue")
            axs[3].imshow(xb[i, :, :, 3], cmap='gray')
            axs[3].set_title("NIR")
            axs[4].imshow(mb[i], cmap='gray', vmin=0, vmax=1)
            axs[4].set_title("Mask")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"val_vis/val_batch_0_sample_{i}.png")
            plt.close(fig)