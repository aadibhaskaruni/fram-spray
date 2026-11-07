# Step 3 - Choose a location and download REAL hourly weather (Open-Meteo, free, no API key).
# Run from the project root. Shift+Enter on each "# %%" cell.
# Requires the file src/weather_api.py (create the folder src/ and save that file inside it).

# %% 1. Settings - change PLACE to test any location
import sys
sys.path.insert(0, ".")                 # so "from src.weather_api import ..." works
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.weather_api import load_or_fetch

PLACE = "Pune"                          # soybean dataset is from Maharashtra; try "Thiruvananthapuram" too
START, END = "2026-06-01", "2026-09-30" # a monsoon (kharif) season; archive data lags a few days behind today
WIND_LIMIT_KMH = 20.0                   # ASSUMPTION: no spraying above this wind speed. Set it from the pesticide label / local guidance.

# %% 2. Find the location and download the weather (cached after the first run)
df, info = load_or_fetch(PLACE, START, END)
print(f"Location: {info['name']}, {info['admin']}, {info['country']}  (lat {info['latitude']:.3f}, lon {info['longitude']:.3f})")
print(f"Hours downloaded: {len(df)}  ({len(df)/24:.0f} days)")
print(df.head())

# %% 3. Summary - what does this weather mean for spraying?
rainy = df["rain_mm"] > 0.2
windy = df["wind_kmh"] > WIND_LIMIT_KMH
print(f"Mean temperature      : {df['temp_c'].mean():.1f} C   (min {df['temp_c'].min():.1f}, max {df['temp_c'].max():.1f})")
print(f"Total rainfall        : {df['rain_mm'].sum():.0f} mm   | rainy hours: {100*rainy.mean():.0f}%")
print(f"Mean / max wind       : {df['wind_kmh'].mean():.1f} / {df['wind_kmh'].max():.1f} km/h")
print(f"Hours too windy to spray (> {WIND_LIMIT_KMH:.0f} km/h): {100*windy.mean():.0f}%")
print(f"Hours that are dry AND calm (sprayable): {100*(~rainy & ~windy).mean():.0f}%")

# %% 4. Plot the weather (daily values) and save the picture
daily = df.set_index("time").resample("D").agg({"temp_c": "mean", "rain_mm": "sum", "wind_kmh": "mean"})
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(daily.index, daily["temp_c"], color="tab:red");  ax[0].set_ylabel("Temp (C), daily mean")
ax[1].bar(daily.index, daily["rain_mm"], color="tab:blue"); ax[1].set_ylabel("Rain (mm/day)")
ax[2].plot(daily.index, daily["wind_kmh"], color="tab:green")
ax[2].axhline(WIND_LIMIT_KMH, ls="--", color="k", lw=1, label=f"assumed spray limit ({WIND_LIMIT_KMH:.0f} km/h)")
ax[2].set_ylabel("Wind (km/h), daily mean"); ax[2].legend(fontsize=8)
fig.suptitle(f"Real weather: {info['name']}, {START} to {END} (Open-Meteo)")
Path("outputs").mkdir(exist_ok=True)
fig.savefig("outputs/step03_weather.png", dpi=130, bbox_inches="tight")
plt.show()

# %% 5. Compare two locations - this shows the location selector really changes the data
other = "Thiruvananthapuram" if PLACE != "Thiruvananthapuram" else "Pune"
df2, info2 = load_or_fetch(other, START, END)
print(f"\n{'':22s} {info['name']:>18s} {info2['name']:>18s}")
print(f"{'Mean temp (C)':22s} {df['temp_c'].mean():18.1f} {df2['temp_c'].mean():18.1f}")
print(f"{'Total rain (mm)':22s} {df['rain_mm'].sum():18.0f} {df2['rain_mm'].sum():18.0f}")
print(f"{'Mean wind (km/h)':22s} {df['wind_kmh'].mean():18.1f} {df2['wind_kmh'].mean():18.1f}")
print("Step 3 finished. Next: Step 4 - turn the drone images into a field grid.")
# %%
