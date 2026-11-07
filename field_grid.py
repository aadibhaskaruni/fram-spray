# Step 4 - Cut every drone image into an 8x8 grid of cells and measure each cell.
# Run from the project root. Shift+Enter on each "# %%" cell.
# Input : data/field_images/<class>/*.jpg   (made in Step 2)
# Output: data/tiles/tile_features.csv  (one row per cell)   data/tiles/image_index.csv  (one row per image, with train/val/test split)

# %% 1. Settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

IMG_DIR = Path("data/field_images")
OUT_DIR = Path("data/tiles")
N = 8                                  # grid size: N x N cells (same as the simulator field)
SPLIT = (0.70, 0.15, 0.15)             # train / validation / test as BLOCKS of consecutive images (less leakage)
SEED = 0
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
classes = sorted(p.name for p in IMG_DIR.iterdir() if p.is_dir())
print("Classes found:", {c: len(list((IMG_DIR / c).glob('*.jpg'))) for c in classes})

# %% 2. The measuring function: one image -> N x N cells x a few colour features
def cell_features(img: Image.Image, n: int = N) -> pd.DataFrame:
    img = img.convert("RGB")
    W, H = img.size
    w, h = W // n, H // n                              # cell size in pixels
    img = img.crop((0, 0, w * n, h * n))               # drop the leftover border pixels
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    total = R + G + B + 1e-6
    exg = (2 * G - R - B) / total                      # "excess green": high for healthy green vegetation
    green = ((hue >= 70) & (hue <= 170) & (sat > 0.20)).astype(np.float32)           # green pixels
    yellow = ((hue >= 20) & (hue < 70) & (sat > 0.25) & (val > 0.25)).astype(np.float32)  # yellow / brown pixels
    dark = (val < 0.20).astype(np.float32)                                             # shadow / soil

    def per_cell(a):                                   # average a (H, W) array inside each cell -> (n, n)
        return a.reshape(n, h, n, w).mean(axis=(1, 3))

    feats = {
        "mean_r": per_cell(R), "mean_g": per_cell(G), "mean_b": per_cell(B),
        "exg_mean": per_cell(exg), "exg_std": np.sqrt(np.maximum(per_cell(exg ** 2) - per_cell(exg) ** 2, 0)),
        "green_frac": per_cell(green), "yellow_frac": per_cell(yellow), "dark_frac": per_cell(dark),
    }
    rows = [dict(row=r, col=c, **{k: float(v[r, c]) for k, v in feats.items()}) for r in range(n) for c in range(n)]
    return pd.DataFrame(rows)

test_img = Image.open(sorted((IMG_DIR / classes[0]).glob("*.jpg"))[0])
print("Example image size:", test_img.size)
print(cell_features(test_img).head(3).round(3))

# %% 3. Process ALL images -> assign the split per image -> save two CSV files
# Drone photos are taken one after another and neighbouring frames overlap, so a RANDOM split would put
# near-identical pictures in both train and test. Instead we split in blocks of consecutive images.
index_rows, tile_tables = [], []
for c in classes:
    files = sorted((IMG_DIR / c).glob("*.jpg"))
    order = range(len(files))              # keep the original order
    n_tr, n_va = int(SPLIT[0] * len(files)), int(SPLIT[1] * len(files))
    for rank, i in enumerate(order):
        split = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
        image_id = f"{c}/{files[i].name}"
        index_rows.append(dict(image_id=image_id, cls=c, split=split))
        t = cell_features(Image.open(files[i]))
        t.insert(0, "image_id", image_id); t.insert(1, "cls", c); t.insert(2, "split", split)
        tile_tables.append(t)
    print(f"done {c}: {len(files)} images")

tiles = pd.concat(tile_tables, ignore_index=True)
index = pd.DataFrame(index_rows)
tiles.to_csv(OUT_DIR / "tile_features.csv", index=False)
index.to_csv(OUT_DIR / "image_index.csv", index=False)
print(f"\nSaved {len(tiles)} cells from {len(index)} images")
print(index.groupby(["cls", "split"]).size().unstack())

# %% 4. Do the classes actually look different? (This decides whether Step 5 can work.)
cols = ["green_frac", "yellow_frac", "dark_frac", "exg_mean", "exg_std"]
summary = tiles.groupby("cls")[cols].agg(["mean", "std"]).round(3)
print(summary)
print("\nRule of thumb: if the class means are close compared with the std, colour alone cannot tell them apart.")

# %% 5. Picture: image | green map | yellow/brown map for one image per class
fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3.2 * len(classes)))
axes = np.atleast_2d(axes)
for r, c in enumerate(classes):
    f = sorted((IMG_DIR / c).glob("*.jpg"))[0]
    t = tiles[tiles.image_id == f"{c}/{f.name}"]
    axes[r, 0].imshow(Image.open(f)); axes[r, 0].set_title(c, fontsize=10, loc="left")
    axes[r, 1].imshow(t.green_frac.values.reshape(N, N), cmap="Greens", vmin=0, vmax=1); axes[r, 1].set_title("green fraction per cell", fontsize=9)
    axes[r, 2].imshow(t.yellow_frac.values.reshape(N, N), cmap="YlOrBr", vmin=0, vmax=1); axes[r, 2].set_title("yellow/brown fraction per cell", fontsize=9)
    for a in axes[r]: a.axis("off")
fig.savefig("outputs/step04_tile_maps.png", dpi=130, bbox_inches="tight")
plt.show()
print("Step 4 finished. Next: Step 5 - train a small classifier on the cells.")
# %%
