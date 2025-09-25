import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a:[Na,4], b:[Nb,4] in xyxy format
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)

def load_sets(model, csv_path: Path, tiles_root: Path, nms_thresh: float):
    df = pd.read_csv(csv_path)
    imgs = sorted(df["image_path"].unique())
    GT, PRED = [], []
    for rel in imgs:
        g = df[df.image_path == rel][["xmin","ymin","xmax","ymax"]].values.astype("float32")
        boxes_gt = torch.tensor(g) if len(g) else torch.zeros((0, 4), dtype=torch.float32)
        img = np.array(Image.open(tiles_root / rel).convert("RGB"))
        # apply fixed NMS fijo and score=0.0
        model.model.score_thresh = 0.0
        model.model.nms_thresh = nms_thresh
        p = model.predict_image(img)
        if p is None or p.empty:
            boxes_pr = torch.zeros((0, 4), dtype=torch.float32)
            scores   = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes_pr = torch.tensor(p[["xmin","ymin","xmax","ymax"]].values.astype("float32"))
            scores   = torch.tensor(p["score"].values.astype("float32"))
        GT.append(boxes_gt)
        PRED.append((boxes_pr, scores))
    return GT, PRED

def prf1_at_threshold(GT, PRED, thr: float, iou_thr: float):
    TP = FP = FN = 0
    for gt, (pr, sc) in zip(GT, PRED):
        keep = (sc >= thr)
        if keep.any():
            pr = pr[keep][torch.argsort(sc[keep], descending=True)]
        else:
            pr = pr[:0]
        if pr.numel() == 0:
            FN += len(gt); continue
        if gt.numel() == 0:
            FP += len(pr); continue
        ious = iou_matrix(pr, gt)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        for i in range(len(pr)):
            j = torch.argmax(ious[i]).item()
            if ious[i, j] >= iou_thr and not matched[j]:
                TP += 1; matched[j] = True
            else:
                FP += 1
        FN += int((~matched).sum().item())
    P = TP/(TP+FP) if TP+FP>0 else 0.0
    R = TP/(TP+FN) if TP+FN>0 else 0.0
    F1 = (2*P*R/(P+R)) if P+R>0 else 0.0
    return P, R, F1