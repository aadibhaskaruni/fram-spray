# Step 2 - Prepare the FIELD (drone) image dataset: MH-SoyaHealthVision, UAV part.
# Run from the project root (the folder containing data/ and outputs/). Shift+Enter on each "# %%" cell.
#
# BEFORE running: download the dataset and extract it into  data/raw/
#   Page: https://data.mendeley.com/datasets/hkbgh5s3b7   (check the file size on the page first)
#   It contains BOTH leaf images and UAV (drone) images. We only use the UAV part.
#   Put it at e.g.  data/raw/MH-SoyaHealthVision   (any folder name under data/raw works)

# %% 1. Settings
import shutil
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path("data/raw")            # the script searches inside this folder
OUT_DIR = Path("data/field_images")   # clean, resized UAV images used by the project
MAX_PER_CLASS = 150                   # images kept per class (picked evenly spaced, NOT the first 150)
MAX_SIDE = 1024                       # resize big drone photos (saves disk space)
EXTS = {".jpg", ".jpeg", ".png"}
assert RAW_DIR.exists() and any(RAW_DIR.iterdir()), f"Nothing in {RAW_DIR.resolve()} - download and extract the dataset there first."

# %% 2. Scan everything: class = folder name, and flag folders that belong to the UAV part
by_class = defaultdict(list)
for p in RAW_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in EXTS:
        by_class[p.parent].append(p)

print(f"{'images':>6}  {'UAV?':5}  folder")
for folder, files in sorted(by_class.items(), key=lambda kv: str(kv[0])):
    is_uav = "uav" in str(folder).lower() or "aerial" in str(folder).lower() or "drone" in str(folder).lower()
    print(f"{len(files):6d}  {'yes' if is_uav else '-':5}  {folder.relative_to(RAW_DIR)}")

# %% 3. Choose the folders to use  (automatic: every folder whose path contains 'uav')
USE = [f for f in by_class if "uav" in str(f).lower() or "aerial" in str(f).lower() or "drone" in str(f).lower()]
# If the automatic choice is wrong, set USE by hand, e.g.:
# USE = [RAW_DIR / "UAV" / "Healthy_Soyabean", RAW_DIR / "UAV" / "Soyabean_Rust"]
assert USE, "No UAV folders detected. Look at the table above and set USE by hand."
print("Using these classes:")
for f in USE:
    print(f"  {f.name}  ({len(by_class[f])} images)")

# %% 4. Copy + resize into data/field_images/<class>/
import re
import numpy as np
from PIL import Image

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
counts = {}
for f in USE:
    dest = OUT_DIR / slug(f.name)
    dest.mkdir(parents=True, exist_ok=True)
    allf = sorted(by_class[f])
    pick = np.linspace(0, len(allf) - 1, min(MAX_PER_CLASS, len(allf))).round().astype(int)   # evenly spaced over the whole class
    for i, src in enumerate(allf[j] for j in pick):
        im = Image.open(src).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(dest / f"{i:04d}.jpg", quality=90)
    counts[slug(f.name)] = len(list(dest.glob("*.jpg")))
print("Saved:", counts)
size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*.jpg")) / 1e6
print(f"Disk used by the clean subset: {size_mb:.0f} MB")

# %% 5. Show one image per class with the 8x8 simulation grid drawn on top
import matplotlib.pyplot as plt
N = 8   # the simulator field is an 8 x 8 grid of cells
classes = sorted(p.name for p in OUT_DIR.iterdir() if p.is_dir())
fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4))
axes = [axes] if len(classes) == 1 else list(axes)
for ax, c in zip(axes, classes):
    im = Image.open(sorted((OUT_DIR / c).glob("*.jpg"))[0])
    w, h = im.size
    ax.imshow(im)
    for k in range(1, N):
        ax.axvline(k * w / N, color="white", lw=0.7, alpha=0.8)
        ax.axhline(k * h / N, color="white", lw=0.7, alpha=0.8)
    ax.set_title(c, fontsize=10); ax.axis("off")
Path("outputs").mkdir(exist_ok=True)
fig.suptitle("Each drone image = one field; each grid square = one cell the agent decides on", fontsize=11)
fig.savefig("outputs/step02_field_grid.png", dpi=130, bbox_inches="tight")
plt.show()
print("Step 2 finished. Next: Step 3 - weather data.")
# %%
