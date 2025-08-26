import re
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, confusion_matrix
from matplotlib import pyplot as plt

from load import getSceneGridSizes

def getSceneId(path: Path):
    m = re.search(r"scene_(\d+)\.npy$", path.name)
    return int(m.group(1)) if m else None

def findGt(gtBase: Path, sceneId: int):
    return next(iter(gtBase.glob(f"**/*_{sceneId:06d}_*.TIF")), None)

def loadGtMask(gtPath: Path):
    gt = np.array(Image.open(gtPath), dtype=np.uint8)
    return (gt > 0).astype(np.uint8)


def ensureDirs(evalDir: Path):
    infDir = evalDir / "inference"
    unpDir = evalDir / "unpadded"
    binDir = evalDir / "binarized"
    unpDir.mkdir(parents=True, exist_ok=True)
    binDir.mkdir(parents=True, exist_ok=True)
    return infDir, unpDir, binDir

# ---- helpers ----

def unpadToMatch(pred, gt):
    ph, pw = pred.shape
    gh, gw = gt.shape
    y0 = (ph - gh) // 2
    x0 = (pw - gw) // 2
    return pred[y0:y0 + gh, x0:x0 + gw]

def evaluatePRC(gtBatch, predBatch, showPlot=False, savePlotPath: Path = None, title: str = None):
    y_true = gtBatch.flatten()
    y_scores = predBatch.flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr, best_f1 = thresholds[best_idx], f1_scores[best_idx]

    if showPlot or savePlotPath is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label="PR curve")
        ax.scatter(recall[best_idx], precision[best_idx], c="r", s=40, zorder=3, label=f"Best F1={best_f1:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision–Recall Curve")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="lower left")
        if savePlotPath is not None:
            fig.savefig(savePlotPath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return best_thr, best_f1



def computeMetrics(gt, predBin):
    tn, fp, fn, tp = confusion_matrix(gt.flatten(), predBin.flatten(), labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "accuracy": accuracy}


def savePngGray(path: Path, arr: np.ndarray) -> None:
    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def savePngBinary(path: Path, mask: np.ndarray) -> None:
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


# --- main evaluation ---

def evaluateAll(runDir: Path, gtBase: Path, fixedThreshold: float = None):
    evalDir = runDir / "evaluation"
    infDir, unpDir, binDir = ensureDirs(evalDir)
    grid = getSceneGridSizes()

    files = sorted(infDir.glob("scene_*.npy"))
    if not files:
        print(f"No inference files in {infDir}")
        return
    
    if fixedThreshold is not None:
        csvPath = evalDir / "metricsValThr.csv"
    else:
        csvPath = evalDir / "metricsTestThr.csv"

    with open(csvPath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sceneId", "bestThreshold", "bestF1", "iou", "dice", "precision", "recall", "accuracy"],
        )
        writer.writeheader()

        for p in files:
            sceneId = getSceneId(p)
            if sceneId is None or sceneId not in grid:
                print(f"skip {p.name}")
                continue

            gtPath = findGt(gtBase, sceneId)
            if gtPath is None:
                print(f"no GT for {sceneId}")
                continue

            pred = np.load(p)
            if pred.ndim != 2:
                pred = np.squeeze(pred)
            gt = loadGtMask(gtPath)

            # Unpad
            predUnp = unpadToMatch(pred, gt)
            #unpNpy = unpDir / f"unpadded_scene_{sceneId}.npy"          #optional .npy and .png unpadded scenes saving
            #np.save(unpNpy, predUnp)
            #savePngGray(unpNpy.with_suffix(".png"), predUnp)

            # Threshold selection
            if fixedThreshold is not None:
                thr = fixedThreshold
                f1 = None
            else:
                thr, f1 = evaluatePRC(gt[np.newaxis, ...], predUnp[np.newaxis, ...], showPlot=False)

            # Binarize
            binMask = (predUnp >= thr).astype(np.uint8)
            binNpy = binDir / f"binmask_scene_{sceneId}.npy"
            #np.save(binNpy, binMask)                                   #optional binarized mask .npy saving
            savePngBinary(binNpy.with_suffix(".png"), binMask)

            # Metrics
            m = computeMetrics(gt, binMask)
            writer.writerow({
                "sceneId": sceneId,
                "bestThreshold": f"{thr:.6f}",
                "bestF1": "" if f1 is None else f"{f1:.6f}",
                **{k: f"{v:.6f}" for k, v in m.items()},
            })

            f1_str = f", F1={f1:.4f}" if f1 is not None else ""
            print(
                f"scene {sceneId} → τ={thr:.4f}{f1_str}, "
                f"IoU={m['iou']:.4f}, Dice={m['dice']:.4f}, "
                f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, Accuracy={m['accuracy']:.4f}"
            )

    print(f"done → {csvPath}")


if __name__ == "__main__":
    BaseFolder = Path(r"c:\Users\andre\Documents\BA\dev\pipeline\results\runs")
    runDir = BaseFolder / "run_20250817_130734_3ch_384"
    gtBase = Path(r"C:\Users\andre\Documents\BA\dev\pipeline\Data\38-Cloud_test\Entire_scene_gts")
    evaluateAll(runDir, gtBase)#, fixedThreshold=0.33622974157333374)  # specify Threshold if you have it