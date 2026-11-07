# Step 5 - Build the "sensor": a small classifier that turns a cell's colour features into an infestation score.
# Run from the project root. Shift+Enter on each "# %%" cell.
# Needs:   pip install scikit-learn
# Input :  data/tiles/tile_features.csv   (from Step 4)
# Output:  data/model/cell_classifier.joblib, data/tiles/cell_scores.csv, data/tiles/score_bank.npz

# %% 1. Settings and data
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HEALTHY, RUST, PEST = "healthy_soyabean", "rust", "pest_attack"   # class folder names from Step 2
FEATS = ["mean_r", "mean_g", "mean_b", "exg_mean", "exg_std", "green_frac", "yellow_frac", "dark_frac"]
STRUCTURE = ["dark_frac", "exg_std"]        # describe soil / canopy density, not disease
COLOUR = [f for f in FEATS if f not in STRUCTURE]

tiles = pd.read_csv("data/tiles/tile_features.csv")
Path("data/model").mkdir(parents=True, exist_ok=True); Path("outputs").mkdir(exist_ok=True)
print(tiles.groupby(["cls", "split"]).size().unstack())

# %% 2. Helper: train a "negative class vs positive class" model and evaluate it on the TEST images
def run(feats, pos, neg=HEALTHY):
    part = {s: tiles[(tiles.split == s) & tiles.cls.isin([neg, pos])] for s in ("train", "val", "test")}
    y = {s: part[s].cls.eq(pos).astype(int) for s in part}
    candidates = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0),
    }
    best = None
    for name, m in candidates.items():                     # pick the model by VALIDATION auc
        m.fit(part["train"][feats], y["train"])
        auc = roc_auc_score(y["val"], m.predict_proba(part["val"][feats])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, name, m)
    _, name, model = best
    te = part["test"].assign(score=model.predict_proba(part["test"][feats])[:, 1], y=y["test"])
    cell_acc = float(((te.score > 0.5) == te.y).mean())
    img = te.groupby("image_id").agg(score=("score", "mean"), y=("y", "first"))
    img_acc = float(((img.score > 0.5) == img.y).mean())
    out = dict(model=name, cell_acc=cell_acc, cell_auc=float(roc_auc_score(te.y, te.score)),
               image_acc=img_acc, image_auc=float(roc_auc_score(img.y, img.score)), n_test_images=len(img))
    return out, model

# %% 3. Main sensor: healthy vs rust (rust is a visible foliar disease -> a real signal in the images)
res, sensor = run(FEATS, RUST)
print(f"Model chosen on validation data: {res['model']}")
print(f"TEST cells : accuracy {res['cell_acc']:.3f} | AUC {res['cell_auc']:.3f}")
print(f"TEST images: accuracy {res['image_acc']:.3f} | AUC {res['image_auc']:.3f}   ({res['n_test_images']} images - small, so treat with care)")

# %% 4. Honesty checks: is the classifier seeing DISEASE, or just soil / canopy density?
rows = []
for label, feats, pos in [
    ("rust vs healthy, all features", FEATS, RUST),
    ("rust vs healthy, colour only", COLOUR, RUST),
    ("rust vs healthy, yellow_frac only", ["yellow_frac"], RUST),
    ("pest vs healthy, all features", FEATS, PEST),
    ("pest vs healthy, colour only", COLOUR, PEST),
    ("pest vs healthy, soil/canopy only", STRUCTURE, PEST),
]:
    r, _ = run(feats, pos)
    rows.append(dict(test=label, cell_AUC=round(r["cell_auc"], 3), image_AUC=round(r["image_auc"], 3)))
print(pd.DataFrame(rows).to_string(index=False))
print("""
How to read this:
 - If 'rust vs healthy, yellow_frac only' is high, the rust signal is real disease colour.
 - If 'pest vs healthy, soil/canopy only' is about as high as 'all features', the pest images differ from
   healthy mainly by canopy density (growth stage / camera height), NOT by pest damage -> do not claim pest detection.""")

# %% 5. Save the sensor and build the SCORE BANK used by the simulator (cells the model never trained on)
tiles["score"] = sensor.predict_proba(tiles[FEATS])[:, 1]
tiles[["image_id", "cls", "split", "row", "col", "score"]].to_csv("data/tiles/cell_scores.csv", index=False)
joblib.dump(dict(model=sensor, feats=FEATS), "data/model/cell_classifier.joblib")

held = tiles[tiles.split.isin(["val", "test"])]
bank = {"healthy": held[held.cls == HEALTHY].score.values, "rust": held[held.cls == RUST].score.values,
        "pest": held[held.cls == PEST].score.values}
np.savez("data/tiles/score_bank.npz", **bank)
print(f"Score bank sizes: { {k: len(v) for k, v in bank.items()} }")
print(f"Sensor false alarms : {100 * (bank['healthy'] > 0.5).mean():.1f}% of healthy cells score > 0.5")
print(f"Sensor misses       : {100 * (bank['rust'] < 0.5).mean():.1f}% of rust cells score < 0.5")
print("These real error rates are what the agent has to live with in the simulator.")

# %% 6. Picture: score distributions + score maps for one test image per class
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(2, 1, 1)
for k, c in (("healthy", "tab:green"), ("rust", "tab:orange"), ("pest", "tab:red")):
    ax.hist(bank[k], bins=30, range=(0, 1), alpha=0.5, color=c, density=True, label=f"{k} cells")
ax.set_xlabel("sensor score (probability of rust)"); ax.set_ylabel("density"); ax.legend()
ax.set_title("What the agent 'sees' for each true cell type (held-out cells)")
for j, c in enumerate([HEALTHY, PEST, RUST]):
    a = fig.add_subplot(2, 3, 4 + j)
    pick = tiles[(tiles.cls == c) & (tiles.split == "test")].image_id.iloc[0]
    a.imshow(tiles[tiles.image_id == pick].score.values.reshape(8, 8), cmap="Reds", vmin=0, vmax=1)
    a.set_title(f"{c}: score per cell", fontsize=9); a.axis("off")
fig.tight_layout(); fig.savefig("outputs/step05_sensor.png", dpi=130)
plt.show()
print("Step 5 finished. Next: Step 6 - build the simulator.")
# %%
