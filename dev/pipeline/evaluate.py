import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from PIL import Image
import numpy as np
import os
from pathlib import Path

def computeMetrics(gtMask, predMask):
    assert gtMask.shape == predMask.shape, "Masks must have same shape"
    gt = gtMask.astype(bool)
    pred = predMask.astype(bool)

    # Confusion
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()           
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    # IoU
    iou = tp / (tp + fp + fn + 1e-6)
    # Dice
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    # Precision
    precision = tp / (tp + fp + 1e-6)
    # Recall
    recall = tp / (tp + fn + 1e-6)
    # Accuracy (if you want)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    print(f"IoU (Jaccard):   {iou:.4f}")
    print(f"Dice (F1):       {dice:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")

    return dict(iou=iou, dice=dice, precision=precision, recall=recall, accuracy=accuracy)

def evaluatePRC(yTrueArray: np.ndarray, yPredArray: np.ndarray, showPlot: bool = True):
    """
    Evaluates precision, recall, F1 score across thresholds using PRC.
    Expects:
      - yTrueArray: binary ground truth masks, shape [N, H, W]
      - yPredArray: predicted probability masks, shape [N, H, W]

    Returns:
      - bestThreshold: float
      - bestF1: float
    """
    yTrueFlat = yTrueArray.flatten()
    yPredFlat = yPredArray.flatten()

    precisions, recalls, thresholds = precision_recall_curve(yTrueFlat, yPredFlat)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    bestIdx = np.argmax(f1s)
    bestThreshold = thresholds[bestIdx]
    bestF1 = f1s[bestIdx]

    print(f"✅ Best threshold: {bestThreshold:.3f}")
    print(f"📈 Precision: {precisions[bestIdx]:.3f}, Recall: {recalls[bestIdx]:.3f}, F1 Score: {bestF1:.3f}")

    if showPlot:
        plt.figure(figsize=(8,6))
        plt.plot(recalls, precisions, label="PR Curve")
        plt.scatter(recalls[bestIdx], precisions[bestIdx], c="red", label=f"Best F1 = {bestF1:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return bestThreshold, bestF1

def unpadToMatch(predMask: np.ndarray, gtMask: np.ndarray) -> np.ndarray:
    """
    Crops predMask from the center using floor-based cropping to match gtMask.
    Handles odd differences like the MATLAB 'unzeropad' function.
    """
    predH, predW = predMask.shape
    gtH, gtW = gtMask.shape

    diffH = predH - gtH
    diffW = predW - gtW

    if diffH < 0 or diffW < 0:
        raise ValueError(f"GT mask is larger than prediction. Cannot crop safely: pred={predMask.shape}, gt={gtMask.shape}")

    top = diffH // 2
    left = diffW // 2

    return predMask[top:top + gtH, left:left + gtW]

def overlayPredictionVsGT(gt: np.ndarray, pred: np.ndarray, alpha=0.5):
    """
    Displays an overlay of GT (red) vs prediction (green). White = match, other colors = mismatch.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch! GT: {gt.shape}, Pred: {pred.shape}")

    overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.float32)

    # GT = red channel, Pred = green channel
    overlay[..., 0] = gt     # Red
    overlay[..., 1] = pred   # Green

    plt.figure(figsize=(12, 12))
    plt.title("Overlay — GT (Red) vs Prediction (Green)")
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    # 🔧 CONFIGURE THESE
    sceneId = "3052"
    predPath = fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\scene_{sceneId}.npy"
    gtPath = fr"C:\Users\andre\Documents\BA\dev\pipeline\Data\38-Cloud_test\Entire_scene_gts\edited_corrected_gts_LC08_L1TP_003052_20160120_20170405_01_T1.TIF"
    savePath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\unpadded_scene_{sceneId}")
    overlaySavePath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\overlay_scene_{sceneId}.png")

#   📥 Load prediction and GT
#   pred = np.load(predPath)
#   gt = np.array(Image.open(gtPath), dtype=np.uint8)
#
#   print("Loaded:")
#   print(f"  Prediction shape: {pred.shape}")
#   print(f"  GT shape:         {gt.shape}")
#
#   # ✂️ Unpad
#   cropped = unpadToMatch(pred, gt)
#   print(f"Unpadded prediction shape: {cropped.shape}")
#
#   # 💾 Save cropped result
#   np.save(savePath.with_suffix(".npy"), cropped)
#   img = (cropped * 255).clip(0, 255).astype(np.uint8) if cropped.dtype != np.uint8 else cropped
#   Image.fromarray(img).save(savePath.with_suffix(".png"))
#   print(f"✅ Unpadded prediction saved to: {savePath}")


#   predPath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\unpadded_scene_{sceneId}.npy")
#   saveNpyPath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\binmask_scene_{sceneId}.npy")
#   savePngPath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\binmask_scene_{sceneId}.png")
#
#   pred = np.load(predPath)
    gt = np.array(Image.open(gtPath), dtype=np.uint8)
#   bestThreshold, bestF1 = evaluatePRC(gt, pred, showPlot=True)
#
#   binarizedMask = (pred >= bestThreshold).astype(np.uint8)
#   np.save(saveNpyPath, binarizedMask)
#   Image.fromarray(binarizedMask * 255).save(savePngPath)

    binmaskPath = Path(fr"C:\Users\andre\Documents\BA\dev\pipeline\results\scenes\unpadded\binmask_scene_{sceneId}.npy")
    binmask = np.load(binmaskPath)

    
    metrics = computeMetrics(gt, binmask)


#🔍 Overlay GT vs Prediction, SAVE DIRECTLY (NO plt)
#   overlay = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
#   overlay[..., 0] = gt * 255        # Red
#   overlay[..., 1] = cropped * 255   # Green
#   # overlay[..., 2] stays 0 (blue)

#   Image.fromarray(overlay).save(overlaySavePath)
#   print(f"✅ Overlay image saved to: {overlaySavePath}")
