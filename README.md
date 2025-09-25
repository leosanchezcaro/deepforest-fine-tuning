# DeepForest Tree Fine-Tuning (template repo)

This repo provides a minimal, reproducible workflow to fine-tune [DeepForest](https://github.com/weecology/DeepForest) for tree detection on your own tiles and box annotations.  

I annotated around 1200 boxes across multiple locations in my study area. After 10–20 training epochs, both metrics and visual inspection showed clear improvements, meaning that with relatively little extra work you can significantly boost performance for a specific region. The key is to digitize boxes consistently and include examples that represent the full variety of trees present in your area. 

Installation on Windows can be trickier, but it should work if you follow the environment steps below. If you use another OS (or need GPU support), refer to the official DeepForest documentation and resources.


## Environment (DeepForest installed separately)

Create a separate venv and install DeepForest from source:

```.bat
# clone DeepForest
git clone https://github.com/weecology/DeepForest.git
cd DeepForest

# venv (Windows example)
python -m venv %USERPROFILE%\df_src
%USERPROFILE%\df_src\Scripts\activate
python -m pip install -U pip setuptools wheel

# Torch CPU (change to CUDA wheels if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# avoid native build of stringzilla (optional pin)
pip install "stringzilla==3.12.6"

# install DeepForest (editable)
pip install -e .

# jupyter kernel (optional)
python -m pip install ipykernel
python -m ipykernel install --user --name df_src --display-name "Python (df_src)"
```

## Repo layout

```
.
├─ notebooks/
│  ├─ 00_data_preparation         # tiles RGB, size fit, boxes to CSV, train/val/test splits
│  ├─ 01_train.ipynb              # fine-tune and save .pt
│  └─ 03_evaluate.ipynb           # compute metrics, compare with base model
├─ scripts/
│  ├─ rgba_to_rgb.py              # drop alpha band
│  ├─ clip_tiles.py               # center crop/pad tiles to a reference
│  ├─ boxes_to_csv.py             # BOXES.shp → deepforest_labels.csv + split CSVs
│  ├─ eval_metrics.py             # IoU, max-F1, Precision, Recall
│  └─ vis_utils.py                # quick prediction plotting
├─ data/
│  ├─ tiles/                      # local GeoTIFF tiles
│  └─ labels/
│     ├─ BOXES.shp                # annotated boxes (same CRS as tiles)
│     ├─ splits.json              # {"train":[...], "valid":[...], "test":[...]}
│     ├─ deepforest_labels.csv    # generated
│     ├─ df_labels_train.csv      # generated
│     ├─ df_labels_valid.csv      # generated
│     └─ df_labels_test.csv       # generated
├─ models/
│  └─ deepforest_ft.pt            # generated
├─ requirements.txt
└─ README.md
```

## Data expectations

- **Tiles**: GeoTIFF, RGB or RGBA. Sizes may differ (handled in prep).  
- **Boxes**: `data/labels/BOXES.shp` (boxes per crown), same CRS as tiles.  

## Run the notebooks

1) **Data prep**: `notebooks/00_data_preparation.ipynb`
   - (optional) Convert RGBA to RGB in place.
   - (optional) Fit all tiles to a reference size (center crop/pad).
   - Build `deepforest_labels.csv` from `BOXES.shp`.
   - Generate `df_labels_{train,valid,test}.csv` using `splits.json`.

2) **Train**: `notebooks/01_train.ipynb`
   - Load pretrained DeepForest.
   - Point to `df_labels_train.csv` / `df_labels_valid.csv`.
   - Set epochs/lr/batch, train, and save `models/deepforest_ft.pt`.

3) **Evaluate**: `notebooks/03_evaluate.ipynb`
   - Evaluate base and fine-tuned with DeepForest’s internal metrics.
   - Sweep score threshold on VALID (fixed NMS) to pick max-F1. Then, the "optimal" score threshold is used for box_precision and box_recall.
   - Report P/R/F1 on TEST using that threshold.

## Author

Leonardo Sánchez Caro - Geospatial Data Science | Water Resources  
[LinkedIn](https://www.linkedin.com/in/leonardo-sanchezcaro/)