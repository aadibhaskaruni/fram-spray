# Step 1 - Fresh setup check for the farm spraying RL project.
# In VS Code: click "Run Cell" above each "# %%" line (or press Shift+Enter).

# %% 1. Python version and free disk space
import sys, shutil
print("Python:", sys.version.split()[0])
print(f"Free disk space: {shutil.disk_usage('.').free / 1e9:.1f} GB")

# %% 2. Check the libraries
import importlib
for name in ["numpy", "pandas", "matplotlib", "PIL", "gymnasium"]:
    try:
        mod = importlib.import_module(name)
        print(f"OK       {name:11s} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"MISSING  {name:11s} -> run: pip install numpy pandas matplotlib pillow gymnasium")

# %% 3. Create the project folders
from pathlib import Path
for folder in ["data/images", "data/weather", "src", "outputs"]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print("ready:", folder)

# %% 4. Test the internet + the location (geocoding) API
import json, urllib.parse, urllib.request
place = "Thiruvananthapuram"          # change to any place name
url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": place, "count": 1})
try:
    with urllib.request.urlopen(url, timeout=20) as resp:
        hit = json.load(resp)["results"][0]
    print(f"API works: {hit['name']}, {hit.get('country', '')} -> lat {hit['latitude']}, lon {hit['longitude']}")
except Exception as e:
    print("Could not reach the API:", e)
    print("Check your internet connection / firewall, then run this cell again.")

# %% 5. Count dataset images (empty until Step 2)
exts = {".jpg", ".jpeg", ".png"}
n_images = sum(1 for p in Path("data/images").rglob("*") if p.suffix.lower() in exts)
print(f"Images found in data/images: {n_images}")
print("Step 1 finished. Next: Step 2 - download the dataset.")

# %%
