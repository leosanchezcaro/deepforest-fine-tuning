from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
import tempfile, os

def fit_folder(ref_path: Path, dir_path: Path, padval: int = 0) -> None:
    """Center crop/pad ALL .tif in dir_path to the size of ref_path, in-place, preserving CRS/transform."""
    # target size
    with rasterio.open(ref_path) as ref:
        tw, th = ref.width, ref.height

    for tif in sorted(dir_path.glob("*.tif")):
        with rasterio.open(tif) as src:
            w, h = src.width, src.height
            bands = min(src.count, 3)  # RGB
            data = src.read(list(range(1, bands+1)))      # (b, h, w)
            prof = src.profile.copy()
            prof.update(width=tw, height=th, count=bands, driver="GTiff", crs=src.crs)

            # base transform
            a: Affine = src.transform

            # -------- centered CROP (if larger) --------
            # crop window
            left  = max((w - tw)//2, 0)
            top   = max((h - th)//2, 0)
            right = min(left + tw, w)
            bot   = min(top + th, h)
            data  = data[:, top:bot, left:right]
            # shift transform by the crop
            a = a * Affine.translation(left, top)

            # -------- centered PAD (if smaller) --------
            cur_h, cur_w = data.shape[1], data.shape[2]
            if cur_w < tw or cur_h < th:
                out = np.full((bands, th, tw), padval, dtype=data.dtype)
                dx = (tw - cur_w)//2
                dy = (th - cur_h)//2
                out[:, dy:dy+cur_h, dx:dx+cur_w] = data
                data = out
                # padding on left/top shifts the pixel origin negative by (dx, dy)
                a = a * Affine.translation(-dx, -dy)

            prof.update(transform=a)

        # write to temp and replace atomically (file is closed here)
        with tempfile.NamedTemporaryFile(delete=False, dir=tif.parent, suffix=".tif") as ntf:
            tmp = Path(ntf.name)
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(data)
        os.replace(tmp, tif)