from typing import Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

def plot_predictions(image_array: np.ndarray,
                     pred_df,
                     title: Optional[str] = None,
                     save_path: Optional[Path] = None,
                     show_scores: bool = True,
                     figsize=(8, 8)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_array)

    if pred_df is not None and len(pred_df):
        for _, r in pred_df.iterrows():
            x0, y0, x1, y1 = map(int, [r["xmin"], r["ymin"], r["xmax"], r["ymax"]])
            w, h = x1 - x0, y1 - y0
            ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=2, color="gold"))
            if show_scores and "score" in r:
                sc = float(r["score"])
                ax.text(x0, max(y0 - 3, 0), f"Tree {sc:.2f}",
                        fontsize=8, color="black",
                        bbox=dict(facecolor="gold", alpha=0.5, pad=1))
    else:
        ax.set_title("No predictions")

    if title:
        ax.set_title(title)

    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()