import re
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve, confusion_matrix

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

def evaluatePRC(gtBatch, predBatch, showPlot=True):
    y_true = gtBatch.flatten()
    y_scores = predBatch.flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]



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

def evaluateAll(runDir: Path, gtBase: Path):
    evalDir = runDir / "evaluation"
    infDir, unpDir, binDir = ensureDirs(evalDir)
    grid = getSceneGridSizes()

    files = sorted(infDir.glob("scene_*.npy"))
    if not files:
        print(f"No inference files in {infDir}")
        return

    csvPath = evalDir / "metrics.csv"
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

            # Unpad and save NPY + PNG
            predUnp = unpadToMatch(pred, gt)
            unpNpy = unpDir / f"unpadded_scene_{sceneId}.npy"
            np.save(unpNpy, predUnp)
            savePngGray(unpNpy.with_suffix(".png"), predUnp)

            # PRC → binarize and save NPY + PNG
            thr, f1 = evaluatePRC(gt[np.newaxis, ...], predUnp[np.newaxis, ...], showPlot=False)
            binMask = (predUnp >= thr).astype(np.uint8)
            binNpy = binDir / f"binmask_scene_{sceneId}.npy"
            np.save(binNpy, binMask)
            savePngBinary(binNpy.with_suffix(".png"), binMask)

            # Metrics
            m = computeMetrics(gt, binMask)
            writer.writerow({
                "sceneId": sceneId,
                "bestThreshold": f"{thr:.6f}",
                "bestF1": f"{f1:.6f}",
                **{k: f"{v:.6f}" for k, v in m.items()},
            })

            print(
                f"scene {sceneId} → τ={thr:.4f}, F1={f1:.4f}, "
                f"IoU={m['iou']:.4f}, Dice={m['dice']:.4f}, "
                f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, Accuracy={m['accuracy']:.4f}"
            )

    print(f"done → {csvPath}")


if __name__ == "__main__":
    BaseFolder = Path(r"c:\Users\andre\Documents\BA\dev\pipeline\results\runs")
    runDir = BaseFolder / "run_20250719_170647"
    # raw strings cannot end with a single backslash → drop it
    gtBase = Path(r"C:\Users\andre\Documents\BA\dev\pipeline\Data\38-Cloud_test\Entire_scene_gts")
    evaluateAll(runDir, gtBase)