# Adaptive Pesticide Spraying with Deep Reinforcement Learning

This repository contains the adaptive-spraying project source code, derived data, trained model files, experiment outputs, figures, and a beginner-friendly walkthrough.

## Project summary

The project studies whether a reinforcement-learning policy can choose among no, low, medium, and high pesticide doses in an **8×8 simulated soybean field**. A season uses a fixed serpentine route with three sweeps (192 decisions). The agent receives noisy local observations rather than the full true infestation and residue maps. The base reward balances simulated cell-health change against pesticide use, unnecessary spraying, and a wind-based drift proxy.

This is a **simulation research prototype**, not a real sprayer controller or field trial. Hourly weather inputs and image-derived sensor-score samples are included, but crop/pest growth, crop damage, pesticide response, residue, and reported “yield” are simulator outputs/assumptions. “Yield” in the results means simulated mean cell health, not harvested crop mass. Pesticide values are simulator dose units, with no verified litres conversion.

## Repository contents

- `src/` — simulator, policies, evaluation helpers, Stable-Baselines utilities, and weather helper.
- `*.py` — data preparation, sensor, weather, baseline, training, evaluation, and plotting scripts.
- `data/` — derived image-cell features and sensor scores, classifier, and weather CSVs.
- `models/` — supplied PPO/DQN model checkpoints.
- `outputs/` — experiment tables, learning curves, figures, and HTML demos.
- `graphs/` — report-style graph assets. Check the data provenance before treating every graph as a measured output; the supplied plotting script hard-codes some values and includes a synthetic regret panel.
- `docs/research_report.pdf` — submitted research report, included as provided.
- `docs/rl_spraying_project_beginner_guide.pdf` and `.docx` — detailed walkthrough in PDF and editable Word formats.
- `docs/adaptive_farm_spraying_mdp_diagram.drawio` — editable diagrams.net project MDP/POMDP interaction loop.

## Getting started

1. Use Python 3 and install the project dependencies described/checked by `setup.py`. The ML/training workflow uses packages including NumPy, pandas, Matplotlib, Pillow, Gymnasium, scikit-learn, joblib, PyTorch, and Stable-Baselines3.
2. Run commands from the repository root so relative paths resolve.
3. Review `setup.py` before running it. Several source files are notebook-style Python scripts with `# %%` cells; training and data-preparation scripts may take time or perform network requests when run.
4. The project uses cached weather CSVs under `data/weather/`. If data must be rebuilt, `weather.py` can query Open-Meteo when the cache is missing.
5. Original raw UAV-image folders are not present in the submitted archive. Rebuilding the image preprocessing from scratch requires obtaining the referenced source dataset and placing it in the path expected by `field_dataset.py`.

## Important reproducibility notes

- Keep the 30-season robustness results (`outputs/step09_summary.csv`) separate from the 100-fresh-season trade-off results (`outputs/step11_table.csv`). They use different evaluation sets and protocols.
- The checked-in DQN table and visible evaluation code need reconciliation before describing the table as reproduced directly by the current script.
- Do not quote classifier accuracy/AUC unless the sensor script is rerun and its actual output is recorded.
- Treat the configured weather gates and all crop/pest dynamics as simulation assumptions, not professional spraying guidance or field-validated biology.
- Do not silently overwrite original results when changing parameters, seeds, data splits, reward weights, or dynamics. Save and label a new experiment.

## License and data

No license is asserted by this README. Before reusing or redistributing the included data, model files, and third-party assets, verify the applicable permissions and licenses.
