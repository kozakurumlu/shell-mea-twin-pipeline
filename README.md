# Shell MEA Resonant Digital Twin Pipeline

State-conditional digital twinning of human neural organoid recordings from a
folded shell MEA ([DANDI 001336](https://dandiarchive.org/dandiset/001336)):

1. **PP-GLM directed connectivity** — point-process GLM (logistic link) per
   organoid state: signed coupling matrix, driver/receiver hubs, directed 3D
   shell graph.
2. **Resonant digital twins** — a Resonant Reservoir Network (RRN) per state,
   fitted with the NSGA-III multi-objective optimisation methodology of
   Sethi, Faraz & Wong-Lin, *"Multi-Objective Optimisation with Oscillatory
   Dynamics in Spontaneous and Decision Spiking Neural Networks"*
   ([arXiv:2605.25224](https://arxiv.org/abs/2605.25224)):
   - each candidate is simulated for 15 independent noise realisations and
     each objective is the RMSE over those trials (paper Eqn. 3), normalised
     by its target, `F = RMSE / (target + eps)`;
   - three objectives stay separate for NSGA-III: population firing rate,
     dominant envelope oscillation frequency, envelope spectral containment;
   - 25 generations x generation size 50, Das-Dennis reference directions,
     SBX (eta=30, p=1.0) crossover, polynomial mutation (eta=20)
     (Deb & Jain 2014 settings);
   - the twin is the Pareto-front member with the lowest composite RMSE
     `sqrt(sum_j F_j^2)`; the per-generation convergence trace and Pareto
     front are saved per state (`ga_pareto_convergence.png`).
   Global RRN parameters and the recurrent weights (on a fixed sparse
   skeleton) are evolved jointly.
3. **CEBRA validation** — latent embeddings of organoid vs twin activity,
   alignment via ridge R^2 and Procrustes distance.
4. **Longitudinal twinning** — per-session targets, twin fidelity, parameters
   and connectivity tracked over time.

## Figures

`figures/` contains a complete run over 9 recordings (subject BO2: seven
stimulation states, 5-60 uA; subject SO1: two sessions, spontaneous):
per-state PP-GLM matrices/hubs/3D graphs, twin reports, NSGA-III
Pareto-front + convergence reports, CEBRA latents, and longitudinal
summaries.

## Running

```bash
python -m venv .venv
.venv/bin/pip install numpy scipy matplotlib "plotly<6" scikit-learn pynwb pymoo kaleido==0.2.1
# optional, for the CEBRA stage:
.venv/bin/pip install torch cebra

# raw data is NOT in this repo (see .gitignore); fetch NWB files from
# DANDI 001336 into data/DANDI_001336/, then simply:
.venv/bin/python shell_mea_twin_pipeline.py
```

When `data/DANDI_001336/` exists next to the script, it is the default data
root and all plots go to `figures/` - no environment variables needed.
`SHELL_MEA_DATA_ROOT`, `SHELL_MEA_FIGURES_DIR` and `SHELL_MEA_OUTPUT_ROOT`
still override the defaults, and without a local data folder the script keeps
its original Colab behaviour (timestamped run folders on Drive).

Per twinned state, `figures/<subject>/twinning/<state>/` contains
`twin_report.png` (objectives dashboard), `ga_pareto_convergence.png`
(NSGA-III Pareto front + convergence), `twin_activity_traces.png`
(paper Fig. 4-style organoid-vs-twin firing-rate traces and PSDs) and
`twin_params.json` (full twin + optimiser provenance).
