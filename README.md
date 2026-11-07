# Adaptive Pesticide Spraying with Deep Reinforcement Learning

A research prototype that compares reinforcement-learning policies and hand-written rules for adaptive pesticide dosing in a partially observable, simulated soybean field.

> **Scope:** This repository is a simulator-based project. It does not control a real drone or sprayer and does not report a real field trial. “Yield” in the result tables means simulated mean cell health—not harvested crop mass. Pesticide amounts are simulator dose units, not verified litres.

## Project objective

The project asks whether an agent can choose a useful spray dose at each field cell while receiving only noisy local observations, using a limited tank, and responding to changing weather. It compares PPO and DQN with no-spray, fixed-dose, threshold, context-aware, and oracle reference policies.

The field is an **8 × 8 grid (64 cells)**. The sprayer follows a fixed serpentine route for **three sweeps**, giving **192 decisions per simulated season**. The agent chooses one of four actions: none, low, medium, or high dose.

## MDP/POMDP formulation

The environment is partially observable, so the implementation is best described as a **POMDP with hard-coded action gates**:

1. **Hidden state:** true infestation, crop health, and pesticide residue for all cells, along with weather, tank level, route position, and season time.
2. **Observation:** noisy local sensor-score samples and nearby/current-cell and weather/resource features. The observation contains 13 values; a four-step observation history is stacked into 52 inputs. The complete true infestation and residue maps are hidden from the agent.
3. **Action:** PPO, DQN, or a baseline chooses no/low/medium/high, corresponding to dose values `0`, `1/3`, `2/3`, or `1` simulator units.
4. **Action gates:** a requested spray is executed as no spray if configured wind, rain, or tank-capacity conditions block it. These are simulator assumptions, not pesticide-label recommendations.
5. **Transition:** the simulator applies the dose and updates infestation, crop health, residue, tank, and time. In the visited cell, the spray response includes `p ← p × exp(−3u)`; the subsequent disease and crop dynamics are modeled equations.
6. **Reward:** the base reward is `r = 1.0·ΔH − 0.10·C − 0.15·U − 0.15·E`, balancing change in simulated cell health against dose use, unnecessary spray, and a wind-based drift proxy.

PPO training adds a shaping term with weight `0.3`; reported evaluation uses the base reward. The project has hard-coded feasibility gates, but does not implement a separate constrained-optimization objective, so the narrower POMDP description is used here.

## Implementation overview

| Stage | Main files | What the stage does |
|---|---|---|
| Environment check | `setup.py` | Prints Python/library and disk checks, creates expected folders, and checks access to Open-Meteo geocoding. It is a check script, not an installer. |
| Image preparation | `field_dataset.py` | Finds the UAV-image classes in `data/raw/`, selects/resizes images, and writes `data/field_images/`. |
| Weather | `weather.py`, `src/weather_api.py` | Loads or fetches hourly temperature, rain, and wind data from Open-Meteo and saves CSVs under `data/weather/`. No API key is needed. |
| Cell features | `field_grid.py` | Splits images into 8×8 cells, computes eight colour/vegetation features, and writes `data/tiles/tile_features.csv` and `image_index.csv`. |
| Sensor-score model | `sensor.py` | Compares logistic regression and histogram gradient boosting for healthy-vs-rust cell scores; writes the model, score CSV, held-out score bank, and sensor plot. |
| Simulator | `src/spray_env.py`, `simulator.py` | Implements the Gymnasium environment, action/observation spaces, field transitions, hard gates, reward, and example sanity checks. |
| Baselines | `src/policies.py`, `src/evaluation.py`, `baselines.py`, `tank_check.py` | Defines and evaluates simple policies, tunes threshold rules on training scenarios, and compares tank settings. |
| PPO / DQN | `train_ppo.py`, `robust.py`, `shaping.py`, `dqn.py` | Trains or evaluates PPO/DQN models, repeats PPO seeds, and compares shaping settings. |
| Final evaluation | `tradeoff.py` | Compares policies on a separate set of fresh simulated seasons and recomputes reward under alternative weights without retraining. |
| Results and demo | `outputs/`, `graphs/`, `plot_evaluation_graphs.py`, `make_demo.py` | Stores CSVs, plots, and HTML demonstrations. Check plot provenance before using a graph as measured evidence. |

## Data included and not included

The repository contains derived image-cell feature/score files, a held-out sensor-score bank, model checkpoints, weather CSVs for Pune and Thiruvananthapuram, experiment tables, figures, and demo HTML. The original raw UAV-image folders are **not** included. To rebuild the image pipeline from the beginning, obtain the referenced MH-SoyaHealthVision dataset and put its UAV images under `data/raw/` as described in `field_dataset.py`.

The weather helper calls Open-Meteo geocoding and, if a cache is unavailable, its archive API. The existing weather CSVs are included, but the current helper geocodes the place before checking its cache, so internet access is still needed when the scripts call `load_or_fetch`.

Image-derived scores and hourly weather are data inputs. Infestation growth/spread, crop-health damage, spray efficacy, residue behavior, and final “yield” are simulator assumptions or outputs; they are not field-measured biology.

## Dependencies

Use **Python 3.10 or 3.11** as a practical starting point. The project has no pinned lock file; install mutually compatible package versions. `setup.py` checks a subset of the dependencies but does not install them.

- **Core/data and plots:** NumPy, pandas, Matplotlib, Pillow.
- **Environment:** Gymnasium.
- **Image classifier:** scikit-learn and joblib.
- **Deep RL:** PyTorch and Stable-Baselines3.
- **Weather:** Python standard library plus internet access to Open-Meteo; no API key.

### Windows PowerShell setup

From the cloned repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU PyTorch avoids installing an unnecessary CUDA build for a CPU-only run.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pandas matplotlib Pillow gymnasium scikit-learn joblib stable-baselines3

# Readiness checks only; this does not install packages.
python setup.py
```

If PowerShell blocks virtual-environment activation, open a new PowerShell window or use the Python executable directly at `.venv\Scripts\python.exe`.

### Linux/macOS setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy pandas matplotlib Pillow gymnasium scikit-learn joblib stable-baselines3
python setup.py
```

## Running the project

Run scripts from the **repository root** so relative paths such as `data/tiles/score_bank.npz` resolve. Files are ordinary Python scripts with `# %%` cell markers: you can run them top-to-bottom with `python <script>.py`, or execute cells sequentially in VS Code/Jupyter. Training can take a long time; inspect each script before running it.

### A. Use the supplied derived data and checkpoints

The repository already includes `data/tiles/score_bank.npz` and weather CSVs, so the raw-image preparation steps can be skipped for using the supplied simulation inputs.

```bash
python setup.py
python simulator.py
python baselines.py
python tank_check.py
python train_ppo.py
python robust.py
python shaping.py
python tradeoff.py
```

What to expect:

- `simulator.py` checks the Gymnasium environment, prints example steps, evaluates simple policies, and saves `outputs/step06_simulator.png`.
- `baselines.py` writes `outputs/step07_baselines.csv` and a comparison plot.
- `tank_check.py` writes tank-capacity comparisons under `outputs/`.
- `train_ppo.py` first runs a 20,000-step smoke test, then starts the full 600,000-step PPO run. It saves a model in `models/` and learning/evaluation outputs in `outputs/`.
- `robust.py` compares multiple PPO training seeds and a stronger context rule. Existing checkpoints can be reused; missing seeds may be trained.
- `shaping.py` compares reward-shaping settings and may train missing checkpoints.
- `tradeoff.py` performs a separate evaluation on 100 fresh simulated seasons; it does not retrain policies.

The supplied checkpoints let you avoid training in many workflows, but the scripts do not all provide a single “evaluate only” command-line mode. Check the code and use its model-loading branches before starting a long run.

### B. Rebuild from raw imagery

Only do this if the source dataset is available. Put the downloaded UAV portion of MH-SoyaHealthVision under `data/raw/`, then run the steps in order:

```bash
python field_dataset.py  # data/raw/ -> data/field_images/
python weather.py        # Open-Meteo -> data/weather/
python field_grid.py     # images -> data/tiles/tile_features.csv
python sensor.py         # features -> classifier and score_bank.npz
python simulator.py     # sanity-check the resulting simulation inputs
```

Then continue with the baselines and training/evaluation steps in section A. Keep source image licenses and dataset terms with the data; they may not permit public redistribution.

### DQN and graph-generation cautions

- **DQN:** the checked-in `dqn.py` has a visible mismatch at its `SmartThreshold(tau=0.15)` call—the `SmartThreshold` implementation uses a different constructor signature. Fix and verify this before rerunning DQN evaluation. The supplied `outputs/step12_table.csv` should be treated as a reported artifact until its provenance is reconciled with the visible script.
- **Report-style graphs:** `plot_evaluation_graphs.py` hard-codes some values and includes a synthetic cumulative-regret panel. Do not cite every panel as a measured result; prefer CSV-backed step-specific outputs.
- **Demo generator:** `make_demo.py` contains a machine-specific absolute Windows path to a generator not included in this repository, so it is not portable as supplied.

## Important results and reproducibility notes

- Keep the 30-season robustness analysis (`outputs/step09_summary.csv`) separate from the 100-fresh-season trade-off analysis (`outputs/step11_table.csv`); they use different evaluation seeds/protocols.
- The classifier code prints accuracy/AUC, but do not report those metrics unless the actual run output is captured.
- The 20% sensor-flip scenario is a synthetic stress test, not a measured camera failure rate. The “harsh weather” scenario uses training-week weather rows, not an independent future monsoon.
- Reward weights, weather gates, dose response, pest/crop dynamics, tank capacity, and dose units are modeling/design choices. They are not agronomic, regulatory, or pesticide-label guidance.
- “Gain vs no spray” is a difference in simulated reward points under the same evaluation setup—not a monetary or percentage gain.

## Main outputs

- `outputs/step07_baselines.csv` — rule-baseline results.
- `outputs/step08_results_seed0.csv` and `outputs/step08_learning_curve.png` — PPO training/evaluation artifacts.
- `outputs/step09_summary.csv` — multi-seed robustness summary.
- `outputs/step10_shaping.csv` — reward-shaping comparison.
- `outputs/step11_table.csv` and `outputs/step11_tradeoff.png` — fresh-season health/chemical trade-off.
- `outputs/step12_table.csv` — supplied DQN result table (see caveat above).
- `docs/research_report.pdf` — submitted research report, included as provided.
- `docs/rl_spraying_project_beginner_guide.pdf` — extended beginner walkthrough.
- `docs/adaptive_farm_spraying_mdp_diagram.drawio` — editable diagrams.net MDP/POMDP diagram.

## License and responsible use

No project license is asserted by this README. Before reusing or redistributing the code, data, models, report, or third-party assets, verify their applicable permissions and licenses. The simulator is for research and educational comparisons; it must not be used as real-world spraying or safety guidance.
