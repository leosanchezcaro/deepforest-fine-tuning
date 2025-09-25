"""
Convert all .tif tiles in a folder from RGBA to RGB in-place (drop alpha if present).
Keeps 3-band RGB as is. Skips images with <3 bands.

"""

import time
import os
import tempfile
from pathlib import Path
import rasterio

def convert_folder_inplace(tiles_dir: Path) -> tuple[int, int, int]:
    tiles = sorted(tiles_dir.glob("*.tif"))
    n_rgb, n_rgba, n_other = 0, 0, 0

    for src_path in tiles:
        # read and close before writing
        with rasterio.open(src_path) as src:
            band_count = src.count
            if band_count == 3:
                n_rgb += 1
                continue
            if band_count < 4:
                print(f"Skipped (not RGB/RGBA): {src_path.name}")
                n_other += 1
                continue

            profile = src.profile.copy()
            profile.update(count=3)
            data = src.read([1, 2, 3])
            tags = {}
            try:
                tags = src.tags()
            except Exception:
                pass

        # write to tmp
        with tempfile.NamedTemporaryFile(delete=False, dir=src_path.parent, suffix=".tif") as ntf:
            tmp_path = Path(ntf.name)

        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(data)
            if tags:
                try:
                    dst.update_tags(**tags)
                except Exception:
                    pass

        for _ in range(5):
            try:
                os.replace(tmp_path, src_path)
                break
            except PermissionError:
                time.sleep(0.2)
        else:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise

        n_rgba += 1

    return n_rgb, n_rgba, n_other

def main():
    ap = argparse.ArgumentParser(description="Convert RGBA→RGB in-place for all .tif in a folder.")
    ap.add_argument("--dir", dest="tiles_dir", type=Path, required=True, help="Folder with .tif tiles")
    args = ap.parse_args()

    n_rgb, n_rgba, n_other = convert_folder_inplace(args.tiles_dir)
    print(f"Done. Kept RGB: {n_rgb} | Converted RGBA→RGB: {n_rgba} | Other: {n_other}")

if __name__ == "__main__":
    main()