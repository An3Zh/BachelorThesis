import tensorflow as tf
import os
import math
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

def getFolders(baseDir: Path) -> Dict[str, Path]:
    return {
        "red":   baseDir / "train_red",
        "green": baseDir / "train_green",
        "blue":  baseDir / "train_blue",
        "nir":   baseDir / "train_nir",
        "gt":    baseDir / "train_gt",
    }

def loadTIF(path: tf.Tensor) -> tf.Tensor:
    def _loadTIFNumpy(pathBytes):
        path = pathBytes.decode("utf-8")
        return np.array(Image.open(path), dtype=np.uint16)
    
    tensor = tf.numpy_function(_loadTIFNumpy, [path], tf.uint16)
    tensor.set_shape([None, None])  
    return tensor

def getPaths(
    filename: tf.Tensor,
    folders: Dict[str, Path]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    red   = tf.strings.join([str(folders["red"]),   "/", "red_",   filename, ".TIF"])
    green = tf.strings.join([str(folders["green"]), "/", "green_", filename, ".TIF"])
    blue  = tf.strings.join([str(folders["blue"]),  "/", "blue_",  filename, ".TIF"])
    nir   = tf.strings.join([str(folders["nir"]),   "/", "nir_",   filename, ".TIF"])
    gt    = tf.strings.join([str(folders["gt"]),    "/", "gt_",    filename, ".TIF"])
    return red, green, blue, nir, gt

def loadDS(
    red: tf.Tensor,
    green: tf.Tensor,
    blue: tf.Tensor,
    nir: tf.Tensor,
    gt: tf.Tensor,
    imgSize: Optional[Tuple[int, int]] = None) -> Tuple[tf.Tensor, tf.Tensor]:

    r  = loadTIF(red)
    g  = loadTIF(green)
    b  = loadTIF(blue)
    n  = loadTIF(nir)
    gt = loadTIF(gt)

    x = tf.stack([r, g, b, n], axis=-1)                     # [H,W,4], uint16
    x = tf.cast(x, tf.float32) / 65535.0                    # [0,1] float32
    if imgSize is not None:
        x = tf.image.resize(x, imgSize)                     # [192,192,4]

    gt = tf.cast(gt > 0, tf.float32)                          # [0,1] float32
    if imgSize is not None:
        gt = tf.image.resize(gt[..., tf.newaxis], imgSize, method='nearest')    # [192,192,1]
        gt = tf.squeeze(gt, axis=-1)                          # [192,192]
    
    return x, gt

def buildDS(
    csvWithFilenames: Path,
    baseDir: Path,
    valRatio: float = 0.2,
    batchSize: int = 8,
    imgSize: Optional[Tuple[int, int]] = None,
    fullDSSize: int = 5155,
    shuffleBuffTrain: int = 512,
    seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    
    folders = getFolders(baseDir)
    def _pathsFn(filename):      return getPaths(filename, folders)
    def _loadFn(r, g, b, n, gt): return loadDS(r, g, b, n, gt, imgSize)

    ds = tf.data.TextLineDataset(str(csvWithFilenames)).skip(1)
    ds = ds.shuffle(fullDSSize, seed, reshuffle_each_iteration=False)

    valSize   = int(fullDSSize * valRatio)
    trainSize = fullDSSize - valSize
    trainDS = ds.take(trainSize)
    valDS   = ds.skip(trainSize)
    trainSteps = math.ceil(trainSize / batchSize)
    valSteps   = math.ceil(valSize / batchSize)

    trainDS = trainDS.map(_pathsFn, num_parallel_calls=tf.data.AUTOTUNE)
    trainDS = trainDS.map(_loadFn,  num_parallel_calls=tf.data.AUTOTUNE)
    trainDS = trainDS.shuffle(shuffleBuffTrain, seed, reshuffle_each_iteration=False)
    trainDS = trainDS.repeat()
    trainDS = trainDS.batch(batchSize).prefetch(tf.data.AUTOTUNE)

    valDS = valDS.map(_pathsFn, num_parallel_calls=tf.data.AUTOTUNE)
    valDS = valDS.map(_loadFn,  num_parallel_calls=tf.data.AUTOTUNE)
    valDS = valDS.repeat()
    valDS = valDS.batch(batchSize).prefetch(tf.data.AUTOTUNE)

    return trainDS, valDS, trainSteps, valSteps


if __name__ == "__main__":

    baseDir      = Path(r"C:\Users\andre\Documents\BA\Dev\Data\38-Cloud_training")
    csvFilenames = Path(r"C:\Users\andre\Documents\BA\Dev\Data\training_patches_38-cloud_nonempty.csv")
    batchSize = 1
    imgSize = (192,192)

    (trainDS, 
     valDS, 
     trainSteps, 
     valSteps) = buildDS(csvFilenames, baseDir, batchSize=batchSize, imgSize=None, seed=42)

    for xb, mb in trainDS.take(1):      # for xb, mb, pathb in trainDS.take(1):
        print("\n--- Training Batch ---")
        print(f"x batch: {xb.shape}, {xb.dtype}, min={tf.reduce_min(xb).numpy()}, max={tf.reduce_max(xb).numpy()}")
        print(f"m batch: {mb.shape}, {mb.dtype}, unique={tf.unique(tf.reshape(mb, [-1])).y.numpy()}")

        for i in range(xb.shape[0]):
            #path_str = pathb[i].numpy().decode("utf-8")
            #print(f"Shown mask path: {path_str}")
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
            plt.show()

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
            plt.show()

