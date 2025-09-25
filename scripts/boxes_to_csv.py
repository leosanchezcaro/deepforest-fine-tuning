"""
Build DeepForest CSV from a shapefile of annotated boxes and split into train/val/test.

- Reads:  data/labels/BOXES.shp
- Tiles:  data/tiles/*.tif
- Writes: data/labels/deepforest_labels.csv
          data/labels/df_labels_train.csv
          data/labels/df_labels_valid.csv
          data/labels/df_labels_test.csv
- Splits: data/labels/splits.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box

# defaults relative to repo root (scripts/.. is repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
TILES_DIR = REPO_ROOT / "data" / "tiles"
LABELS_DIR = REPO_ROOT / "data" / "labels"
VECT_PATH = LABELS_DIR / "BOXES.shp"
OUT_CSV = LABELS_DIR / "deepforest_labels.csv"
SPLITS_JSON = LABELS_DIR / "splits.json"

def build_deepforest_csv(tiles_dir: Path = TILES_DIR,
                         vect_path: Path = VECT_PATH,
                         out_csv: Path = OUT_CSV) -> None:
    gdf = gpd.read_file(vect_path)
    rows = []

    # check CRS in the first tile vs shapefile (NO reprojection)
    first_tile = next(tiles_dir.glob("*.tif"), None)
    if first_tile:
        with rasterio.open(first_tile) as s0:
            if s0.crs and gdf.crs and s0.crs.to_string() != gdf.crs.to_string():
                print(f"[WARNING] CRS' don't match (tile: {s0.crs}, vect: {gdf.crs}). "
                      f"process continues WITHOUT reprojection")

    for tif in sorted(tiles_dir.glob("*.tif")):
        with rasterio.open(tif) as src:
            tile_bounds = src.bounds
            tile_poly = box(*tile_bounds)

            ann_tile = gdf[gdf.intersects(tile_poly)].copy()
            if ann_tile.empty:
                print(f"No intersection: {tif.name}")
                continue

            for geom in ann_tile.geometry:
                if geom is None or geom.is_empty:
                    continue

                inter = geom.intersection(tile_poly)
                if inter.is_empty:
                    continue

                minx, miny, maxx, maxy = inter.bounds

                # rasterio.index -> (row, col)
                rmin, cmin = src.index(minx, maxy)  # upper left
                rmax, cmax = src.index(maxx, miny)  # lower right

                xmin = int(np.clip(min(cmin, cmax), 0, src.width  - 1))
                xmax = int(np.clip(max(cmin, cmax), 0, src.width  - 1))
                ymin = int(np.clip(min(rmin, rmax), 0, src.height - 1))
                ymax = int(np.clip(max(rmin, rmax), 0, src.height - 1))

                # discard boxes that are too small
                if (xmax - xmin) < 3 or (ymax - ymin) < 3:
                    continue

                rows.append({
                    "image_path": tif.name,  # relative to DeepForest root_dir
                    "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                    "label": "Tree"
                })

    df = pd.DataFrame(rows, columns=["image_path","xmin","ymin","xmax","ymax","label"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"OK → {out_csv.resolve()} | boxes: {len(df)}")

def split_labels(labels_dir: Path = LABELS_DIR,
                 base_csv: Path = OUT_CSV,
                 splits_json: Path = SPLITS_JSON) -> None:
    with open(splits_json, "r", encoding="utf-8") as f:
        S = json.load(f)

    df = pd.read_csv(base_csv)

    # normalize image_path to just the filename (in case it has subfolders)
    df["fname"] = df["image_path"].apply(lambda x: Path(x).name)

    train_names = set(S.get("train", []))
    valid_names = set(S.get("valid", []))
    test_names  = set(S.get("test", []))

    train_df = df[df["fname"].isin(train_names)].drop(columns="fname")
    valid_df = df[df["fname"].isin(valid_names)].drop(columns="fname")
    test_df  = df[df["fname"].isin(test_names)].drop(columns="fname")

    train_df.to_csv(labels_dir / "df_labels_train.csv", index=False)
    valid_df.to_csv(labels_dir / "df_labels_valid.csv", index=False)
    test_df.to_csv(labels_dir / "df_labels_test.csv", index=False)

    print("Rows:",
          "train:", len(train_df),
          "| valid:", len(valid_df),
          "| test:", len(test_df))

def main():
    build_deepforest_csv(TILES_DIR, VECT_PATH, OUT_CSV)
    split_labels(LABELS_DIR, OUT_CSV, SPLITS_JSON)

if __name__ == "__main__":
    main()