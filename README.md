# Shell MEA Resonant Digital Twin Pipeline

State-conditional digital twinning of human neural organoid recordings from a
folded shell MEA ([DANDI 001336](https://dandiarchive.org/dandiset/001336)):

1. **PP-GLM directed connectivity** — point-process GLM (logistic link) per
   organoid state: signed coupling matrix, driver/receiver hubs, directed 3D
   shell graph.
2. **Resonant digital twins** — a Resonant Reservoir Network (RRN,
   [Kramer, arXiv:2506.17083](https://arxiv.org/abs/2506.17083)) per state:
   15 damped AR(2) resonators whose centre frequencies span 0.015-0.44 Hz as
   two interleaved **golden-ratio ladders** (adjacent ratio sqrt(phi), the
   paper's golden organisation at reservoir capacity; damping evolvable to
   near-critical so nodes ring selectively with the data). Fitted with the
   NSGA-III multi-objective optimisation methodology of
   Sethi, Faraz & Wong-Lin, *"Multi-Objective Optimisation with Oscillatory
   Dynamics in Spontaneous and Decision Spiking Neural Networks"*
   ([arXiv:2605.25224](https://arxiv.org/abs/2605.25224)):
   - the twin's population rate comes from a **doubly-stochastic burst
     readout**: the reservoir's standardised mean resonator amplitude acts as
     a latent excitability modulating the probability (kappa) and size (beta)
     of discrete burst events with log-normal amplitudes and a 1-2 s decay
     kernel, riding on a quiescent baseline. Sparse events reproduce the
     aperiodic impulsive bursting of the stimulated states (near-flat rate
     spectrum); dense, strongly-modulated events recover smooth rhythmic
     envelopes - one mechanism spans both organoid phenotypes;
   - each candidate is simulated for 15 independent noise realisations and
     each objective is the RMSE over those trials (paper Eqn. 3), normalised
     by its target, `F = RMSE / (target + eps)`;
   - FIVE objectives stay separate for NSGA-III: population firing rate;
     dominant-oscillation mismatch; envelope spectral containment; a
     teacher-forced synchronisation loss (the same reservoir driven by the
     organoid envelope must track it through a ridge readout on held-out
     data - generalized synchronisation); and a Wasserstein rate-distribution
     loss that matches fluctuation/burst amplitude;
   - a **dominant oscillation** is only accepted when the smoothed band
     spectrum has an interior peak whose prominence over a median-filtered
     local background beats the 95th percentile of the same statistic on
     bin-shuffled surrogates of the envelope (an exact permutation test:
     tilted/monotone 1/f burst states are reported explicitly as "no dominant
     oscillation", and the same calibrated criterion scores spurious twin
     peaks);
   - 25 generations x generation size 50, Das-Dennis reference directions,
     SBX (eta=30, p=1.0) crossover, polynomial mutation (eta=20)
     (Deb & Jain 2014 settings);
   - the twin is selected from the pooled non-dominated front: within 25% of
     the minimum composite RMSE `sqrt(mean_j F_j^2)`, members with spurious
     oscillations are excluded and the lowest composite among the remainder
     wins; the per-generation convergence trace and Pareto front are saved
     per state (`ga_pareto_convergence.png`).
   Global RRN parameters (13) and the recurrent weights (45, on a fixed
   sparse skeleton) are evolved jointly.
3. **CEBRA validation** — latent embeddings of the organoid's channel
   envelopes vs the DRIVEN twin's resonator amplitudes (the reservoir
   synchronised to the recording - an autonomous realisation cannot align
   trajectory-wise by construction), alignment via ridge R^2 and Procrustes
   distance.
4. **Longitudinal twinning** — per-session targets, twin fidelity, parameters
   and connectivity tracked over time.

## Figures

`figures/` contains a complete run over 16 recordings (subject BO14: seven
stimulation states, 5-60 uA, 16 channels; subject BO2: seven ~90 s
stimulation states, 3 channels; subject SO1: two spontaneous sessions):
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
(paper Fig. 4-style, three panels: (A) organoid vs ONE autonomous twin
realisation with marginal rate histograms, (B) normalised PSDs with the
dominant-peak calls, (C) the driven twin synchronised to the organoid with
the held-out sync r) and `twin_params.json` (full twin + optimiser
provenance).
