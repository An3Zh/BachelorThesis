import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from PIL import Image
import numpy as np


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
