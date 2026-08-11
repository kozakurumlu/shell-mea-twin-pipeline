#!/usr/bin/env python3
"""
State-Conditional Resonant Digital Twin Pipeline (Shell MEA / DANDI 001336)

Narrative
---------
1. PP-GLM connectivity: point-process GLM (Truccolo-style, logistic link) on
   detected spikes gives a signed, directed coupling matrix per organoid state
   (replaces CCM/GMN; same outputs: matrix heatmap, driver/receiver hubs,
   3D shell plot - now with direction arrows and excitatory/inhibitory sign).
2. Resonant digital twins: KongFatt-style twinning (Sethi, Faraz & Wong-Lin,
   "Multi-Objective Optimisation with Oscillatory Dynamics in Spontaneous and
   Decision Spiking Neural Networks", arXiv:2605.25224) but optimising a
   Resonant Reservoir Network instead of an Izhikevich RSNN. In addition to
   the global RRN parameters, the recurrent connection weights are evolved on
   a FIXED sparse skeleton (KongFatt-style connectivity evolution, ported to
   the RRN). Per state, the optimisation follows the paper's NSGA-III
   methodology exactly:
     - each candidate parameter set is simulated for TWIN_N_TRIALS independent
       noise realisations and each objective is the RMSE over those trials
       (paper Eqn. 3), normalised by its target, F = RMSE/(target + eps);
     - the three objectives are kept SEPARATE for NSGA-III - (a) population
       firing rate, (b) dominant oscillation frequency of the population-rate
       envelope, (c) envelope spectral containment;
     - NSGA-III (Das-Dennis reference directions, SBX eta=30/p=1.0,
       polynomial mutation eta=20, i.e. Deb & Jain 2014 settings) runs for
       TWIN_N_GEN generations with generation size TWIN_POP_SIZE (paper: 25
       generations, generation size 50);
     - the twin is the Pareto-front member with the lowest overall composite
       RMSE sqrt(sum_j F_j^2) (paper Sec. III), and the per-generation
       convergence + Pareto front are saved as figures (paper Figs. 3D/5C).
   The twin is AUTONOMOUS (noise + parametric rhythmic drive), i.e. it
   generates the state rather than filtering a replay of the recording.
3. CEBRA validation: latent embeddings of organoid activity vs twin activity
   per state; alignment (ridge R^2, Procrustes distance).
4. Longitudinal twinning: per-session twins tracked across sessions (SO1).

Twin timescale note: KongFatt's organoid twin targeted 0.09 Hz firing rate and
0.195 Hz dominant oscillation - far below the old pipeline's 5 Hz floor. The
twin therefore operates on the population firing-rate envelope (1 s bins) with
resonators spanning 0.02-0.5 Hz.
"""

import os
import re
import json
import warnings
import gc
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotly.graph_objects as go

from pynwb import NWBHDF5IO
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, detrend
from scipy.sparse import random as sparse_random, csr_matrix
from scipy.special import expit
from scipy.sparse.linalg import eigs
from scipy.spatial import procrustes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
plt.rcParams["agg.path.chunksize"] = 10000

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
RUN_PPGLM = True         # Point-process GLM directed connectivity
RUN_TWINNING = True      # KongFatt-style RRN digital twins
RUN_CEBRA = True         # CEBRA latent comparison organoid vs twin
RUN_LONGITUDINAL = True  # Per-session tracking (subjects with >1 session)

# =============================================================================
# SPIKE DETECTION CONFIGURATION
# =============================================================================
SPIKE_FS = 10000.0       # Hz - raw stream is decimated to this for spike detection
SPIKE_HIGHPASS_HZ = 300.0
SPIKE_THR_K = 4.5        # threshold = -K * noise (median absolute deviation based)
SPIKE_REFRACTORY_MS = 1.5
MIN_SPIKES_PER_CHANNEL = 50   # channels below this are excluded as GLM sources

# =============================================================================
# PP-GLM CONFIGURATION
# =============================================================================
GLM_BIN_MS = 1.0         # spike trains binned at 1 kHz
GLM_MAX_LAG_MS = 20      # coupling/history filter length
GLM_MAX_SAMPLES = 60000  # rows subsampled for the logistic fit
GLM_C = 1.0              # L2 regularisation (inverse)
GLM_EDGE_TOP_N = 50      # max edges kept (same role as old GMN_TOP_N_EDGES)
GLM_NULL_PERCENTILE = 95 # edge threshold = this percentile of |null couplings|

# =============================================================================
# TWIN CONFIGURATION (KongFatt-style objectives, RRN model)
# =============================================================================
ENV_BIN_S = 1.0          # population rate envelope bin size (s) -> envelope Fs = 1 Hz
MIN_ENV_BINS = 300       # need >= 5 min of data for sub-Hz twinning
MAX_ENV_BINS = 14400     # cap at 4 h
TWIN_FMIN = 0.02         # resonator bank floor (Hz) - covers KongFatt's 0.195 Hz
TWIN_FMAX = 0.5          # resonator bank ceiling (Hz) = Fs/2 at envelope Fs = 1 Hz
TWIN_K = 12              # FIXED resonator count -> fixed weight-matrix dimension.
                         # Required so the recurrent-weight genome stays aligned
                         # across the whole NSGA-III search (we evolve weight
                         # VALUES on a fixed skeleton, a la KongFatt connectivity).
TWIN_FSTEP = (TWIN_FMAX - TWIN_FMIN) / (TWIN_K - 1)  # frange is linspace, not arange
TWIN_SKELETON_SPARSITY = 0.35  # fixed recurrent-connectivity density (the skeleton)
TWIN_PEAK_BAND = (0.03, 0.45)  # band for dominant-frequency target
TWIN_SIGMA = 0.005       # RRN paper default (noise-driven spontaneous dynamics)
# --- NSGA-III settings following Sethi, Faraz & Wong-Lin (arXiv:2605.25224) ---
TWIN_N_GEN = 25          # generations (paper: 25, chosen on convergence; the
                         # per-generation convergence trace is saved to check)
TWIN_POP_SIZE = 50       # generation size (paper: 50)
TWIN_REF_PARTITIONS = 8  # Das-Dennis partitions for 3 objectives -> C(10,2)=45
                         # reference directions <= pop size
TWIN_N_TRIALS = 15       # independent noise realisations per parameter set;
                         # each objective is the RMSE over these trials (paper
                         # Eqn. 3; their Case-1 setting). Trials are batched in
                         # one vectorised RRN pass, so extra trials are cheap.
TWIN_RMSE_EPS = 1e-6     # eps in the paper's normalisation F = RMSE/(target+eps)
TWIN_F_CAP = 100.0       # clamp for valid-sim normalised RMSEs (keeps NSGA-III
                         # objective normalisation sane)
TWIN_FAIL_F = 1e3        # dominated penalty objective vector for failed sims
TWIN_SEED = 42

# =============================================================================
# CEBRA CONFIGURATION
# =============================================================================
CEBRA_MAX_ITER = 1000
CEBRA_OUT_DIM = 3
CEBRA_MAX_SAMPLES = 20000

RANDOM_SEED = 42

# =============================================================================
# Path configuration
# =============================================================================
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/DANDI_001336_human_neural_organoids_shell_MEA_neuromodulation")
DEFAULT_OUTPUT_ROOT = Path("/content/drive/MyDrive/RRN_Paper/colab_outputs")


def _resolve_path(value, default):
    if value:
        try:
            return Path(value).expanduser().resolve()
        except Exception:
            pass
    return default.resolve()


DATA_ROOT = _resolve_path(os.getenv("SHELL_MEA_DATA_ROOT"), DEFAULT_DATA_ROOT)
OUTPUT_ROOT = _resolve_path(os.getenv("SHELL_MEA_OUTPUT_ROOT"), DEFAULT_OUTPUT_ROOT)

_figures_dir = os.getenv("SHELL_MEA_FIGURES_DIR")
if _figures_dir:
    # Fixed output folder (e.g. ./figures for local runs); reruns overwrite.
    # OUTPUT_ROOT is deliberately untouched here (it may be a Colab path).
    main_output_path = Path(_figures_dir).expanduser().resolve()
    main_output_path.mkdir(parents=True, exist_ok=True)
else:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_path = OUTPUT_ROOT / f"Shell_MEA_Twin_Analysis_{timestamp}"
    # ponytail: never overwrite a prior run; bump a suffix if the timestamp collides
    counter = 0
    while main_output_path.exists():
        counter += 1
        main_output_path = OUTPUT_ROOT / f"Shell_MEA_Twin_Analysis_{timestamp}_{counter:02d}"
    main_output_path.mkdir(parents=True)
main_output_dir = str(main_output_path)
print(f"Run output folder: {main_output_dir}")


def _enough_disk_space(path, need_bytes):
    try:
        return shutil.disk_usage(path).free >= need_bytes
    except Exception:
        return True


def _safe_savefig(fig, path, dpi=300, bbox_inches="tight"):
    if not _enough_disk_space(os.path.dirname(path) or ".", 25_000_000):
        print("   Low disk space - skipping save")
        return
    try:
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    except Exception as e:
        print(f"   Error saving figure: {e}")


# =============================================================================
# 3D Electrode Positions (folded shell MEA)
# Same geometry as shell.py; embedded here so the pipeline is self-contained
# (no separate shell.py needed when uploaded to Colab).
# =============================================================================
SHELL_R = 2.0
SHELL_MAX_COVERAGE_ANGLE = np.pi / 0.95
SHELL_D_TIP, SHELL_D_SIDE, SHELL_D_INNER, SHELL_W_OFFSET = 0.35, 0.50, 0.80, 0.2
SHELL_ARM_COLORS = {'East': 'red', 'West': 'green', 'South': 'blue', 'North': 'orange'}
SHELL_FLAT_COORDS = {
    0: (SHELL_D_TIP, 0), 1: (SHELL_D_SIDE, -SHELL_W_OFFSET),
    2: (SHELL_D_INNER, 0), 3: (SHELL_D_SIDE, SHELL_W_OFFSET),
    4: (-SHELL_D_TIP, 0), 5: (-SHELL_D_SIDE, SHELL_W_OFFSET),
    6: (-SHELL_D_INNER, 0), 7: (-SHELL_D_SIDE, -SHELL_W_OFFSET),
    8: (0, -SHELL_D_TIP), 9: (-SHELL_W_OFFSET, -SHELL_D_SIDE),
    10: (0, -SHELL_D_INNER), 11: (SHELL_W_OFFSET, -SHELL_D_SIDE),
    12: (0, SHELL_D_TIP), 13: (SHELL_W_OFFSET, SHELL_D_SIDE),
    14: (0, SHELL_D_INNER), 15: (-SHELL_W_OFFSET, SHELL_D_SIDE),
}
SHELL_ARM_OF = ['East'] * 4 + ['West'] * 4 + ['South'] * 4 + ['North'] * 4


def get_electrode_positions_3d():
    """{idx: {'pos': np.array([x,y,z]), 'arm', 'color', 'flat_coords': (u,v)}}.

    3D layout of the 16-channel folded-shell probe: each electrode's flat (u,v)
    is mapped onto the sphere by arc length, arms coloured for grouping.
    """
    positions = {}
    for i in range(16):
        u, v = SHELL_FLAT_COORDS[i]
        arm = SHELL_ARM_OF[i]
        r_flat = np.sqrt(u**2 + v**2)
        phi = np.arctan2(v, u)
        theta = r_flat * SHELL_MAX_COVERAGE_ANGLE
        pos = np.array([SHELL_R * np.sin(theta) * np.cos(phi),
                        SHELL_R * np.sin(theta) * np.sin(phi),
                        SHELL_R * np.cos(theta)])
        positions[i] = {'pos': pos, 'arm': arm,
                        'color': SHELL_ARM_COLORS[arm], 'flat_coords': (u, v)}
    return positions


# =============================================================================
# Resonant Reservoir Network (envelope-scale twin)
# =============================================================================
def _rescale_spectral_radius(W, radius, K):
    """Scale sparse W to the requested spectral radius (dense eig for small K).

    ARPACK (scipy eigs) is overkill and occasionally flaky for the tiny
    K x K matrices used here, and this runs inside the GA hot loop.
    """
    try:
        if K <= 64:
            max_eig = float(np.max(np.abs(np.linalg.eigvals(W.toarray()))))
        else:
            vals = eigs(W.astype(np.float64), k=1, which='LM',
                        return_eigenvectors=False)
            max_eig = float(np.abs(vals[0]))
        if max_eig > 0:
            return W * (float(radius) / max_eig)
    except Exception:
        pass
    return W


class ReservoirNetwork:
    def __init__(self, Fs=1.0, fmin=0.02, fstep=0.02, sigma=0.005, sparsity=0.35,
                 spectral_radius=0.43, base_geometric_ratio=0.93, random_state=42,
                 skeleton=None):
        self.Fs = float(Fs)
        self.fmin = float(fmin)
        self.fmax = float(TWIN_FMAX)
        self.fstep = float(fstep)
        self.sigma = float(sigma)
        self.sparsity = float(sparsity)
        self.spectral_radius = float(spectral_radius)
        self.base_geometric_ratio = float(base_geometric_ratio)
        self.random_state = random_state
        self.skeleton = skeleton  # fixed (rows, cols) nonzeros, or None
        self.history_weights, self.frange = self.generate_history_weights()
        self.K = int(np.size(self.history_weights['w_t_minus_1']))
        if self.K == 0:
            self.K = 1
            self.history_weights = {'w_t_minus_1': np.array([0.5]),
                                    'w_t_minus_2': np.array([-0.25])}
            self.frange = np.array([float(fmin)])
        a1 = self.history_weights["w_t_minus_1"]
        a2 = self.history_weights["w_t_minus_2"]
        dt = 1.0 / self.Fs
        a2_safe = np.where(a2 != 0, a2, -1e-10)
        self.omega = np.sqrt(np.abs((a1 + a2 - 1) / (a2_safe * dt**2)))
        self.beta = -(a1 + 2 * a2) / (2 * a2_safe * dt)
        self.rng = np.random.default_rng(self.random_state)
        self.reset_states()
        self.W_in = np.ones((self.K, 1))
        # Fixed skeleton -> recurrent weights are evolved externally via
        # build_W_res; otherwise fall back to a random recurrent matrix
        # (self-contained twin with no weight evolution).
        self.W_res = (csr_matrix((self.K, self.K))
                      if skeleton is not None else self.generate_reservoir_weights())

    def generate_reservoir_weights(self):
        if self.K < 2:
            return csr_matrix(np.zeros((self.K, self.K)))
        W_res = sparse_random(self.K, self.K, density=self.sparsity,
                              data_rvs=lambda n: self.rng.uniform(-1, 0, size=n),
                              format='csr', random_state=self.random_state)
        W_res.setdiag(0)
        if self.K > 2:
            W_res = _rescale_spectral_radius(W_res, self.spectral_radius, self.K)
        return W_res

    def generate_skeleton(self):
        """Fixed sparse connectivity pattern (which (i,j) pairs are nonzero).

        Drawn once per state with a fixed seed so the recurrent-weight genome
        stays aligned across the whole NSGA-III search. Returns (rows, cols).
        """
        W = sparse_random(self.K, self.K, density=TWIN_SKELETON_SPARSITY,
                          data_rvs=lambda n: self.rng.uniform(-1, 0, size=n),
                          format='csr', random_state=self.random_state)
        W.setdiag(0)
        W.eliminate_zeros()
        coo = W.tocoo()
        return (coo.row.tolist(), coo.col.tolist())

    def build_W_res(self, values, spectral_radius):
        """Build W_res from the fixed skeleton and evolved nonzero `values`."""
        rows, cols = self.skeleton
        W = csr_matrix((np.asarray(values, dtype=np.float64), (rows, cols)),
                       shape=(self.K, self.K))
        if self.K > 2:
            W = _rescale_spectral_radius(W, spectral_radius, self.K)
        self.W_res = W

    def generate_history_weights(self):
        """Fixed resonator bank covering the sub-Hz envelope band.

        K oscillators evenly span [TWIN_FMIN, TWIN_FMAX] via linspace so K is
        CONSTANT across the whole search (the recurrent-weight genome must stay
        aligned). Unstable oscillators are *clamped* to a stable default rather
        than dropped, so K never changes with base_geometric_ratio.
        """
        frange = np.linspace(TWIN_FMIN, TWIN_FMAX, TWIN_K)
        r = self.base_geometric_ratio
        w_t_minus_1 = 2 * r * np.cos(2 * np.pi * frange / self.Fs)
        w_t_minus_2 = (-r**2) * np.ones_like(w_t_minus_1)
        discriminant = w_t_minus_1**2 + 4 * w_t_minus_2 + 0j
        sqrt_discriminant = np.sqrt(discriminant)
        z1 = (w_t_minus_1 + sqrt_discriminant) / 2
        z2 = (w_t_minus_1 - sqrt_discriminant) / 2
        valid = ((np.abs(z1) < 1) & (np.abs(z2) < 1)
                 & (w_t_minus_1 > 0) & (frange < self.Fs / 2.0))
        # Clamp unstable oscillators to a stable damped default; keep all K.
        w_t_minus_1 = np.where(valid, w_t_minus_1, 0.5)
        w_t_minus_2 = np.where(valid, w_t_minus_2, -0.25)
        return ({'w_t_minus_1': w_t_minus_1, 'w_t_minus_2': w_t_minus_2}, frange)

    def reset_states(self, n_trials=1):
        """State arrays are (n_trials, K): independent noise realisations of
        the SAME network run as one vectorised batch (paper Eqn. 3 trials)."""
        B = int(n_trials)
        self.x_t = np.zeros((B, self.K))
        self.x_t_minus_1 = np.zeros((B, self.K))
        self.x_t_minus_2 = np.zeros((B, self.K))
        self.A_t = np.zeros((B, self.K))
        self.phi_t = np.zeros((B, self.K))

    def update(self, u_t):
        B = self.x_t.shape[0]
        noise = self.rng.normal(0, self.sigma, size=(B, self.K))
        drive = self.W_in.dot(np.atleast_1d(u_t)).reshape(1, self.K)
        recur = self.W_res.dot(expit(np.cos(self.phi_t)).T).T
        pre_activation = (drive
                          + self.history_weights['w_t_minus_1'] * self.x_t_minus_1
                          + self.history_weights['w_t_minus_2'] * self.x_t_minus_2
                          + recur + noise)
        self.x_t = pre_activation
        vt = (self.x_t - self.x_t_minus_1) * self.Fs
        omega_safe = np.where(self.omega != 0, self.omega, 1e-10)
        x_t_safe = np.where(self.x_t != 0, self.x_t, 1e-10)
        self.A_t = np.sqrt(self.x_t**2 + ((vt + self.beta * self.x_t) / omega_safe) ** 2)
        self.phi_t = np.arctan2((vt + self.beta * self.x_t), (omega_safe * x_t_safe))
        self.x_t_minus_2[:] = self.x_t_minus_1
        self.x_t_minus_1[:] = self.x_t

    def collect_states(self, input_sequence, n_trials=1):
        T = input_sequence.shape[0]
        B = int(n_trials)
        self.reset_states(B)
        states = np.zeros((B, T, self.K), dtype=np.float32)
        amplitudes = np.zeros((B, T, self.K), dtype=np.float32)
        for t in range(T):
            self.update(input_sequence[t])
            states[:, t] = self.x_t
            amplitudes[:, t] = self.A_t
        return states, amplitudes


# =============================================================================
# Data discovery and loading
# =============================================================================
def discover_recordings(data_root):
    """Find NWB files and label each with (subject, session, stim state)."""
    records = []
    for p in sorted(Path(data_root).rglob("*.nwb")):
        sub = re.search(r"sub-([A-Za-z0-9]+)", str(p))
        ses = re.search(r"ses-([A-Za-z0-9]+)", str(p))
        stim = re.search(r"(\d+)uA", p.name)
        if sub:
            subject = sub.group(1)
        elif "BO14" in p.name:
            subject = "BO14"
        elif "SO1" in p.name:
            subject = "SO1"
        else:
            subject = p.parent.name
        session = ses.group(1) if ses else "unknown"
        state = f"{stim.group(1)}uA" if stim else "spontaneous"
        records.append({'path': p, 'subject': subject, 'session': session,
                        'state': state})
    return records


def load_recording(path):
    """Load 2D acquisition decimated to ~SPIKE_FS. Returns dict or None."""
    try:
        with NWBHDF5IO(str(path), "r") as io:
            nwbfile = io.read()
            session_start = getattr(nwbfile, "session_start_time", None)
            for _, d in nwbfile.acquisition.items():
                if hasattr(d, "data") and len(d.data.shape) == 2:
                    rate = float(getattr(d, "rate", 30000.0))
                    step = int(max(1, round(rate / SPIKE_FS)))
                    raw = d.data[::step].astype(np.float32)
                    np.nan_to_num(raw, copy=False)
                    return {'raw': raw, 'fs': rate / step,
                            'session_start': session_start}
    except Exception as e:
        print(f"      Error loading {path.name}: {e}")
    return None


# =============================================================================
# Spike detection -> binned spike trains -> rate envelopes
# =============================================================================
def detect_spikes(raw, fs):
    """Threshold-crossing spike detection on the high-passed ~10 kHz stream.

    Returns spk (T_ms x C int8 at 1 kHz) and per-channel spike counts.
    """
    n_samples, n_channels = raw.shape
    nyq = fs / 2.0
    if nyq <= SPIKE_HIGHPASS_HZ:
        sos = butter(4, 0.05, output='sos')  # stream too slow; mild detrend only
    else:
        sos = butter(4, SPIKE_HIGHPASS_HZ / nyq, btype='highpass', output='sos')
    filt = sosfiltfilt(sos, raw, axis=0).astype(np.float32)

    decim = int(round(fs / 1000.0))  # ms bins
    n_bins = n_samples // decim
    spk = np.zeros((n_bins, n_channels), dtype=np.int8)
    counts = np.zeros(n_channels, dtype=int)
    refractory = int(SPIKE_REFRACTORY_MS * fs / 1000.0)
    for ch in range(n_channels):
        x = filt[:, ch]
        noise = np.median(np.abs(x)) / 0.6745
        if noise <= 0:
            continue
        thr = -SPIKE_THR_K * noise
        peaks, _ = find_peaks(-x, height=-thr, distance=max(1, refractory))
        counts[ch] = len(peaks)
        if len(peaks):
            spk[np.minimum(peaks // decim, n_bins - 1), ch] = 1
    return spk, counts


def compute_envelopes(spk):
    """Population and per-channel firing-rate envelopes at ENV_BIN_S bins."""
    bin_ms = int(ENV_BIN_S * 1000)
    n_bins = spk.shape[0] // bin_ms
    n_bins = min(n_bins, MAX_ENV_BINS)
    trimmed = spk[:n_bins * bin_ms]
    per_channel = trimmed.reshape(n_bins, bin_ms, spk.shape[1]).sum(axis=1) / ENV_BIN_S
    population = per_channel.sum(axis=1)
    return population.astype(np.float32), per_channel.astype(np.float32)


# =============================================================================
# PP-GLM (point-process GLM, logistic link) directed connectivity
# =============================================================================
def _lagged_design(spk, max_lag):
    """Lagged spike features: column block j holds channel j at lags 1..max_lag."""
    T, C = spk.shape
    X = np.zeros((T, C * max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        X[lag:, (lag - 1)::max_lag] = spk[:-lag].astype(np.float32)
    return X


def _fit_ppglm_matrix(spk, rng):
    """Fit one logistic PP-GLM per target channel. Returns signed C x C matrix."""
    T, C = spk.shape
    max_lag = GLM_MAX_LAG_MS
    X = _lagged_design(spk, max_lag)
    rows = np.arange(T)
    if T > GLM_MAX_SAMPLES:
        rows = np.sort(rng.choice(T, size=GLM_MAX_SAMPLES, replace=False))
    W = np.zeros((C, C), dtype=np.float64)
    for target in range(C):
        y = spk[rows, target].astype(np.int32)
        if y.sum() < MIN_SPIKES_PER_CHANNEL or y.sum() >= len(rows) - MIN_SPIKES_PER_CHANNEL:
            continue
        try:
            clf = LogisticRegression(penalty='l2', C=GLM_C, solver='lbfgs',
                                     max_iter=300)
            clf.fit(X[rows], y)
            coef = clf.coef_[0].reshape(C, max_lag)  # block j = source channel j
            for src in range(C):
                if src != target:
                    W[src, target] = coef[src].sum()  # signed coupling strength
        except Exception:
            continue
    return W


def run_ppglm(spk, counts):
    """PP-GLM coupling matrix + circular-shift null threshold + edge selection."""
    rng = np.random.default_rng(RANDOM_SEED)
    active = counts >= MIN_SPIKES_PER_CHANNEL
    spk_use = spk[:, active]
    if spk_use.shape[1] < 2:
        return None
    W = _fit_ppglm_matrix(spk_use, rng)

    # Null: independently circular-shift each channel (preserves rates, breaks timing)
    spk_null = np.zeros_like(spk_use)
    for ch in range(spk_use.shape[1]):
        spk_null[:, ch] = np.roll(spk_use[:, ch], rng.integers(1, spk_use.shape[0]))
    W_null = _fit_ppglm_matrix(spk_null, rng)
    offdiag = np.abs(W_null)[~np.eye(W_null.shape[0], dtype=bool)]
    thr = np.percentile(offdiag, GLM_NULL_PERCENTILE) if offdiag.size else 0.0

    C = W.shape[0]
    edges = [(int(s), int(t), float(W[s, t]))
             for s in range(C) for t in range(C)
             if s != t and abs(W[s, t]) > thr]
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    edges = edges[:GLM_EDGE_TOP_N]
    node_strengths = {
        i: {'in': float(np.sum(np.abs(W[:, i]))),
            'out': float(np.sum(np.abs(W[i, :]))),
            'total': float(np.sum(np.abs(W[:, i])) + np.sum(np.abs(W[i, :])))}
        for i in range(C)}
    return {'W': W, 'threshold': float(thr), 'edges': edges,
            'node_strengths': node_strengths, 'active_channels': np.where(active)[0]}


# =============================================================================
# PP-GLM visualisation (replaces CCM matrix / GMN 3D / node importance)
# =============================================================================
def plot_ppglm_matrix(W, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1.0
    im = ax.imshow(W, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Target Channel', fontsize=12)
    ax.set_ylabel('Source Channel', fontsize=12)
    ax.set_title(f'{title}\n(Row i, Col j) = signed PP-GLM coupling i -> j', fontsize=13)
    n = W.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    plt.colorbar(im, ax=ax, label='Coupling strength (+exc / -inh)')
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


def plot_node_importance(node_strengths, title, output_path):
    n_channels = len(node_strengths)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    arm_colors_list = (['red'] * 4 + ['green'] * 4 + ['blue'] * 4 + ['orange'] * 4)[:n_channels]

    in_vals = [node_strengths[i]['in'] for i in range(n_channels)]
    axes[0].bar(range(n_channels), in_vals, color=arm_colors_list)
    axes[0].set_xlabel('Channel'); axes[0].set_ylabel('In-Strength')
    axes[0].set_title('Receivers (Sum |Incoming Coupling|)')
    axes[0].set_xticks(range(n_channels))

    out_vals = [node_strengths[i]['out'] for i in range(n_channels)]
    axes[1].bar(range(n_channels), out_vals, color=arm_colors_list)
    axes[1].set_xlabel('Channel'); axes[1].set_ylabel('Out-Strength')
    axes[1].set_title('Drivers (Sum |Outgoing Coupling|)')
    axes[1].set_xticks(range(n_channels))

    net_vals = [out_vals[i] - in_vals[i] for i in range(n_channels)]
    bar_colors = ['forestgreen' if v > 0 else 'crimson' for v in net_vals]
    axes[2].bar(range(n_channels), net_vals, color=bar_colors)
    axes[2].set_xlabel('Channel'); axes[2].set_ylabel('Net Influence')
    axes[2].set_title('Net Influence (Out - In)\nGreen=Driver, Red=Receiver')
    axes[2].set_xticks(range(n_channels))
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


def plot_ppglm_3d(positions, active_channels, edges, node_strengths, title, output_path):
    """3D directed connectivity on the shell geometry (Plotly HTML + PNG).

    `active_channels` maps each spike-train column (0..n-1, the indices used in
    `edges`/`node_strengths`) back to its original electrode id, so the graph is
    placed correctly even when some shell electrodes were dropped for low spike
    count. Drawn whenever every active electrode has a known shell position.
    """
    n_channels = len(active_channels)
    R = 2.0
    arm_colors = {'East': 'red', 'West': 'green', 'South': 'blue', 'North': 'orange'}
    # column k (as used by edges/node_strengths) -> original electrode id -> position
    elec = [int(active_channels[k]) for k in range(n_channels)]
    pos = [positions[elec[k]]['pos'] for k in range(n_channels)]
    xs = [pos[k][0] for k in range(n_channels)]
    ys = [pos[k][1] for k in range(n_channels)]
    zs = [pos[k][2] for k in range(n_channels)]
    arms = [positions[elec[k]]['arm'] for k in range(n_channels)]

    fig = go.Figure()

    u_grid = np.linspace(0, 2 * np.pi, 50)
    v_grid = np.linspace(0, np.pi, 50)
    fig.add_trace(go.Surface(
        x=R * np.outer(np.cos(u_grid), np.sin(v_grid)),
        y=R * np.outer(np.sin(u_grid), np.sin(v_grid)),
        z=R * np.outer(np.ones(np.size(u_grid)), np.cos(v_grid)),
        colorscale=[[0, 'lightgray'], [1, 'lightgray']], showscale=False,
        opacity=0.15, hoverinfo='skip', name='Shell Surface'))

    if edges:
        max_w = max(abs(e[2]) for e in edges)
        cone_x, cone_y, cone_z, cone_u, cone_v, cone_w, cone_c = [], [], [], [], [], [], []
        for src, tgt, w in edges:
            nw = abs(w) / max_w if max_w > 0 else 0.5
            color = ('rgba(214,39,40,%.2f)' % (0.35 + 0.6 * nw)) if w > 0 \
                else ('rgba(31,119,180,%.2f)' % (0.35 + 0.6 * nw))
            p0, p1 = positions[src]['pos'], positions[tgt]['pos']
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode='lines', line=dict(color=color, width=2 + 6 * nw),
                showlegend=False, hoverinfo='text',
                hovertext=f"PP-GLM: {src} -> {tgt}<br>Coupling: {w:+.3f} "
                          f"({'excitatory' if w > 0 else 'inhibitory'})"))
            # Arrowhead at 85% along the edge, pointing at the target
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                tip = p0 + 0.85 * (p1 - p0)
                cone_x.append(tip[0]); cone_y.append(tip[1]); cone_z.append(tip[2])
                cone_u.append(direction[0]); cone_v.append(direction[1]); cone_w.append(direction[2])
                cone_c.append('red' if w > 0 else 'blue')
        if cone_x:
            fig.add_trace(go.Cone(
                x=cone_x, y=cone_y, z=cone_z, u=cone_u, v=cone_v, w=cone_w,
                sizemode='absolute', sizeref=0.25, anchor='tail',
                colorscale=[[0, 'blue'], [1, 'red']], showscale=False,
                hoverinfo='skip'))

    for arm in ['East', 'West', 'South', 'North']:
        idx = [i for i in range(n_channels) if arms[i] == arm]
        max_s = max(node_strengths[i]['total'] for i in range(n_channels)) or 1
        sizes = [8 + 12 * node_strengths[i]['total'] / max_s for i in idx]
        fig.add_trace(go.Scatter3d(
            x=[xs[i] for i in idx], y=[ys[i] for i in idx], z=[zs[i] for i in idx],
            mode='markers+text',
            marker=dict(size=sizes, color=arm_colors[arm],
                        line=dict(width=2, color='black')),
            text=[str(i) for i in idx], textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            name=f'{arm} Arm',
            hovertext=[f"<b>Electrode {i}</b><br>Arm: {arms[i]}<br>"
                       f"In: {node_strengths[i]['in']:.3f}<br>"
                       f"Out: {node_strengths[i]['out']:.3f}" for i in idx],
            hoverinfo='text'))

    fig.update_layout(
        title=dict(text=f'<b>{title}</b><br><sub>Arrows: directed PP-GLM coupling '
                        f'(red=excitatory, blue=inhibitory)</sub>',
                   x=0.5, font=dict(size=14)),
        scene=dict(xaxis=dict(title='X', range=[-2.5, 2.5]),
                   yaxis=dict(title='Y', range=[-2.5, 2.5]),
                   zaxis=dict(title='Z', range=[-2.5, 2.5]),
                   aspectmode='cube',
                   camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))),
        width=1000, height=900, hovermode='closest',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'))

    html_path = output_path.replace('.png', '.html')
    try:
        fig.write_html(html_path)
        print(f"         Saved interactive HTML: {html_path}")
    except Exception as e:
        print(f"         Warning: could not save HTML: {e}")

    # Static stills from several camera angles (front = default output_path)
    base, ext = os.path.splitext(output_path)
    views = [
        ("front", dict(x=1.8, y=1.8, z=1.2)),
        ("back", dict(x=-1.8, y=-1.8, z=1.2)),
        ("left", dict(x=-1.8, y=1.8, z=1.2)),
        ("right", dict(x=1.8, y=-1.8, z=1.2)),
        ("top", dict(x=0.01, y=0.01, z=2.6)),
    ]
    saved_any = False
    for view_name, eye in views:
        vpath = output_path if view_name == "front" else f"{base}_{view_name}{ext}"
        try:
            fig.update_layout(scene=dict(camera=dict(eye=eye)))
            fig.write_image(vpath, width=1000, height=900, scale=2)
            saved_any = True
        except Exception as e:
            print(f"         Warning: Plotly PNG failed ({view_name} view): {e}")
    if not saved_any:
        print(f"         Plotly PNG failed for all views; using matplotlib fallback")
        _plot_ppglm_3d_fallback(positions, active_channels, edges, node_strengths, title, output_path)


def _plot_ppglm_3d_fallback(positions, active_channels, edges, node_strengths, title, output_path):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    n_channels = len(active_channels)
    R = 2.0
    elec = [int(active_channels[k]) for k in range(n_channels)]
    colors = [positions[elec[i]]['color'] for i in range(n_channels)]
    max_s = max(node_strengths[i]['total'] for i in range(n_channels)) or 1
    sizes = [100 + 400 * node_strengths[i]['total'] / max_s for i in range(n_channels)]

    if edges:
        max_w = max(abs(e[2]) for e in edges)
        for src, tgt, w in edges:
            nw = abs(w) / max_w if max_w > 0 else 0.5
            p0, p1 = positions[elec[src]]['pos'], positions[elec[tgt]]['pos']
            color = 'red' if w > 0 else 'blue'
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color=color, linewidth=1 + 3 * nw, alpha=0.3 + 0.5 * nw)
            tip = p0 + 0.85 * (p1 - p0)
            d = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-10)
            ax.quiver(tip[0], tip[1], tip[2], d[0], d[1], d[2],
                      length=0.2, color=color, alpha=0.9)

    u_grid = np.linspace(0, 2 * np.pi, 30)
    v_grid = np.linspace(0, np.pi, 30)
    ax.plot_surface(R * np.outer(np.cos(u_grid), np.sin(v_grid)),
                    R * np.outer(np.sin(u_grid), np.sin(v_grid)),
                    R * np.outer(np.ones(np.size(u_grid)), np.cos(v_grid)),
                    alpha=0.05, color='gray')
    for i in range(n_channels):
        p = positions[elec[i]]['pos']
        ax.scatter(p[0], p[1], p[2], c=colors[i], s=sizes[i],
                   edgecolors='black', linewidths=1.5, alpha=0.9, zorder=10)
        ax.text(p[0], p[1], p[2] + 0.25, str(elec[i]), fontsize=9, ha='center',
                fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([-2.5, 2.5]); ax.set_ylim([-2.5, 2.5]); ax.set_zlim([-2.5, 2.5])
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


# =============================================================================
# KongFatt-style twinning: targets and objectives
# =============================================================================
def compute_twin_targets(pop_env):
    """KongFatt observables: mean population rate + dominant envelope frequency."""
    fs_env = 1.0 / ENV_BIN_S
    target_rate = float(np.mean(pop_env))
    x = detrend(pop_env.astype(np.float64))
    nperseg = int(min(256, max(32, len(x) // 4)))
    freqs, psd = welch(x, fs=fs_env, nperseg=nperseg)
    band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
    if not np.any(band) or np.max(psd[band]) <= 0:
        return None
    band_freqs = freqs[band]
    band_psd = psd[band]
    target_freq = float(band_freqs[np.argmax(band_psd)])
    psd_norm = band_psd / np.sum(band_psd)
    return {'target_rate': target_rate, 'target_freq': target_freq,
            'psd_freqs': band_freqs, 'psd_norm': psd_norm, 'nperseg': nperseg}


def simulate_twin(params, T, skeleton=None, W_values=None, n_trials=1):
    """Autonomous RRN twin: sinusoidal rhythmic drive + noise, no data input.

    With `skeleton`+`W_values` the recurrent matrix uses the evolved nonzero
    weights on a fixed skeleton (KongFatt-style connectivity evolution); without
    them the network falls back to a random recurrent matrix (self-contained).

    `n_trials` independent noise realisations are simulated in one vectorised
    batch (paper Eqn. 3 trials). The RRN rng is re-seeded per call, so every
    candidate is evaluated on the SAME noise streams (common random numbers):
    objectives are deterministic in the parameters, which stabilises the GA.

    Returns (activity, amplitudes, frange): activity is (n_trials, T) per-trial
    population activity; amplitudes is trial 0's (T, K) resonator amplitudes.
    """
    rrn = ReservoirNetwork(
        Fs=1.0 / ENV_BIN_S, sigma=TWIN_SIGMA, sparsity=TWIN_SKELETON_SPARSITY,
        spectral_radius=params['spectral_radius'],
        base_geometric_ratio=params['base_geometric_ratio'],
        random_state=TWIN_SEED, skeleton=skeleton)
    if skeleton is not None and W_values is not None:
        rrn.build_W_res(W_values, params['spectral_radius'])
    t = np.arange(T) * ENV_BIN_S
    u = (params['drive_amp'] * np.sin(2 * np.pi * params['drive_freq'] * t)
         ).reshape(-1, 1)
    _, amplitudes = rrn.collect_states(u, n_trials=n_trials)  # (B, T, K)
    activity = amplitudes.mean(axis=2)  # (B, T) population activity per trial
    return activity, amplitudes[0], rrn.frange


def _twin_metrics(params, pop_env, targets, skeleton=None, W_values=None,
                  n_trials=None):
    """Return metric dict or None if the twin diverged/died.

    Paper-faithful objectives (Sethi, Faraz & Wong-Lin, arXiv:2605.25224):
    each candidate is simulated for `n_trials` independent noise realisations
    and each objective is the RMSE over trials (Eqn. 3),

        RMSE_x = sqrt( (1/N) * sum_i (x_i - x_target)^2 ),

    normalised by its target, F_x = RMSE_x / (x_target + eps). The three
    normalised RMSEs (rate, dominant frequency, spectral containment with
    target containment 1) stay SEPARATE as the NSGA-III objectives; `overall`
    is the paper's composite sqrt(sum_j F_j^2) used afterwards to pick one
    twin off the Pareto front.
    """
    if n_trials is None:
        n_trials = TWIN_N_TRIALS
    T = len(pop_env)
    try:
        activity, _, _ = simulate_twin(params, T, skeleton, W_values,
                                       n_trials=n_trials)  # (B, T)
        if not np.all(np.isfinite(activity)) or np.any(activity.std(axis=1) < 1e-12):
            return None  # diverged or dead twin in any trial
        rates = params['output_gain'] * activity.mean(axis=1)  # (B,)
        x = detrend(activity.astype(np.float64), axis=1)
        freqs, psd = welch(x, fs=1.0 / ENV_BIN_S, nperseg=targets['nperseg'],
                           axis=1)  # psd is (B, n_freqs)
        band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
        band_psd = psd[:, band]
        band_tot = band_psd.sum(axis=1)
        if not np.any(band) or not np.all(band_tot > 0):
            return None
        band_freqs = freqs[band]
        twin_freqs = band_freqs[np.argmax(band_psd, axis=1)]  # (B,)
        twin_psd_norm = band_psd / band_tot[:, None]
        org_psd = targets['psd_norm']
        m = min(org_psd.size, twin_psd_norm.shape[1])
        containments = np.minimum(org_psd[None, :m],
                                  twin_psd_norm[:, :m]).sum(axis=1)  # (B,)

        # Eqn. 3 RMSE over trials, then the paper's target normalisation.
        rmse_rate = np.sqrt(np.mean((rates - targets['target_rate']) ** 2))
        rmse_freq = np.sqrt(np.mean((twin_freqs - targets['target_freq']) ** 2))
        rmse_spec = np.sqrt(np.mean((1.0 - containments) ** 2))
        F = np.array([
            rmse_rate / (targets['target_rate'] + TWIN_RMSE_EPS),
            rmse_freq / (targets['target_freq'] + TWIN_RMSE_EPS),
            rmse_spec / 1.0,  # containment target is 1, already normalised
        ])
        if not np.all(np.isfinite(F)):
            return None
        F = np.minimum(F, TWIN_F_CAP)
        overall = float(np.sqrt(np.sum(F ** 2)))  # paper composite RMSE
        return {'pred_rate': float(rates.mean()),
                'twin_freq': float(np.median(twin_freqs)),
                'containment': float(containments.mean()),
                'rmse_rate': float(F[0]), 'rmse_freq': float(F[1]),
                'rmse_spec': float(F[2]), 'overall': overall}
    except Exception:
        return None


# Global twin parameters searched by NSGA-III (the recurrent-weight VALUES are
# added as extra free variables on a fixed skeleton; see run_twinning). The two
# log-uniform variables are searched in log10 space then exponentiated back.
TWIN_PARAM_SPECS = [
    ('drive_freq', 0.03, 0.45, None),
    ('drive_amp', 1e-3, 5.0, 'log'),
    ('base_geometric_ratio', 0.70, 0.99, None),
    ('spectral_radius', 0.05, 1.2, None),
    ('output_gain', 1e-3, 1e3, 'log'),
]


def _specs_to_bounds():
    xl, xu = [], []
    for _, lo, hi, tr in TWIN_PARAM_SPECS:
        if tr == 'log':
            lo, hi = np.log10(lo), np.log10(hi)
        xl.append(float(lo)); xu.append(float(hi))
    return np.array(xl), np.array(xu)


def _vec_to_params(X):
    params = {}
    for (name, _lo, _hi, tr), x in zip(TWIN_PARAM_SPECS, X):
        if tr == 'log':
            x = 10 ** float(x)
        params[name] = float(x)
    return params


def run_twinning(pop_env):
    """NSGA-III multi-objective search of the RRN twin for one state.

    Follows the optimisation methodology of Sethi, Faraz & Wong-Lin
    (arXiv:2605.25224): each candidate is evaluated as the target-normalised
    RMSE over TWIN_N_TRIALS noise realisations per objective (Eqn. 3), the
    three objectives stay separate for NSGA-III (Das-Dennis reference
    directions; Deb & Jain 2014 SBX/polynomial-mutation operators; paper-scale
    25 generations x generation size 50), and the returned twin is the
    Pareto-front member with the lowest composite RMSE sqrt(sum_j F_j^2).
    Per-generation convergence is logged for the GA report figure.
    Returns the same dict shape as before (plus GA diagnostics), or None.
    """
    if len(pop_env) < MIN_ENV_BINS:
        print(f"      Too short for sub-Hz twinning ({len(pop_env)} bins < "
              f"{MIN_ENV_BINS}); skipping")
        return None
    targets = compute_twin_targets(pop_env)
    if targets is None:
        print("      Could not compute twin targets; skipping")
        return None
    print(f"      Targets: rate={targets['target_rate']:.4f} Hz, "
          f"dom. freq={targets['target_freq']:.4f} Hz")

    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.callback import Callback
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.optimize import minimize
    except ImportError:
        print("      pymoo not installed - skipping twinning (pip install pymoo)")
        return None

    # Fixed recurrent-weight skeleton for this state (constant across the
    # whole search). The nonzero weight VALUES are free NSGA-III variables,
    # a la KongFatt's connectivity evolution but in the RRN framework.
    skel_net = ReservoirNetwork(Fs=1.0 / ENV_BIN_S, random_state=TWIN_SEED,
                                sparsity=TWIN_SKELETON_SPARSITY, skeleton=([], []))
    skeleton = skel_net.generate_skeleton()
    n_w = len(skeleton[0])
    n_g = len(TWIN_PARAM_SPECS)

    xl_spec, xu_spec = _specs_to_bounds()
    xl = np.concatenate([xl_spec, np.full(n_w, -1.0)])
    xu = np.concatenate([xu_spec, np.full(n_w, 1.0)])
    ref_dirs = get_reference_directions("das-dennis", 3,
                                        n_partitions=TWIN_REF_PARTITIONS)

    # Running best-ever composite individual, recorded at EVALUATION time:
    # NSGA-III niching can drop it from survivor populations, and pymoo's
    # `res.opt` holds only the niche representatives of the first front.
    best_seen = {'comp': float('inf'), 'X': None, 'F': None}

    class _TwinProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            params = _vec_to_params(X[:n_g])
            W_values = X[n_g:]
            m = _twin_metrics(params, pop_env, targets, skeleton, W_values)
            if m is not None:
                Fv = np.array([m['rmse_rate'], m['rmse_freq'], m['rmse_spec']])
                comp = float(np.sqrt(np.sum(Fv ** 2)))
                if comp < best_seen['comp']:
                    best_seen.update(comp=comp, X=np.array(X, dtype=float), F=Fv)
                out["F"] = Fv
            else:
                # penalise failed sims with a dominated vector
                out["F"] = np.full(3, TWIN_FAIL_F)

    class _ConvergenceLog(Callback):
        """Per-generation composite-RMSE trace (paper: generations chosen on
        convergence), so every run documents its own convergence."""

        def __init__(self):
            super().__init__()
            self.data["best"] = []
            self.data["median_valid"] = []
            self.data["n_valid"] = []

        def notify(self, algorithm):
            F = np.asarray(algorithm.pop.get("F"), dtype=float)
            comp = np.sqrt((F ** 2).sum(axis=1))
            valid = np.all(F < TWIN_FAIL_F, axis=1)
            self.data["best"].append(float(comp.min()))
            self.data["median_valid"].append(
                float(np.median(comp[valid])) if np.any(valid) else float('nan'))
            self.data["n_valid"].append(int(valid.sum()))

    # Deb & Jain (2014) NSGA-III operator settings (also pymoo defaults),
    # made explicit for reproducibility: SBX eta=30 prob=1.0, polynomial
    # mutation eta=20 prob=1/n_var. Paper scale: 25 gens x pop 50.
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=TWIN_POP_SIZE,
                      crossover=SBX(eta=30, prob=1.0), mutation=PM(eta=20),
                      eliminate_duplicates=True)
    res = minimize(_TwinProblem(), algorithm, ('n_gen', TWIN_N_GEN),
                   seed=RANDOM_SEED, callback=_ConvergenceLog(), verbose=False)
    conv = getattr(res.algorithm, "callback", None)
    conv_data = dict(conv.data) if conv is not None and hasattr(conv, "data") else {}
    # JSON-safe convergence trace (the raw X/F arrays stay out of the report)
    conv_json = {k: conv_data.get(k, []) for k in ("best", "median_valid", "n_valid")}

    # Select the Pareto-front member with the lowest composite RMSE
    # sqrt(sum_j F_j^2) (paper Sec. III). Pool res.opt (only the niche
    # representatives in pymoo's NSGA-III), the full final population and the
    # running best-ever individual, then take the pool's non-dominated front:
    # the min-composite member is always non-dominated, so it survives this.
    pools_F, pools_X = [], []
    for pool in (getattr(res, "opt", None), getattr(res, "pop", None)):
        if pool is not None and len(pool) > 0:
            pools_F.append(np.atleast_2d(pool.get("F")))
            pools_X.append(np.atleast_2d(pool.get("X")))
    if best_seen['X'] is not None:
        pools_F.append(np.atleast_2d(best_seen['F']))
        pools_X.append(np.atleast_2d(best_seen['X']))
    if pools_F:
        F = np.vstack(pools_F)
        X = np.vstack(pools_X)
    else:
        F = np.atleast_2d(res.F)
        X = np.atleast_2d(res.X)
    valid = np.all(F < TWIN_FAIL_F, axis=1)
    if not np.any(valid):
        print("      Optimisation failed to find a valid twin")
        return None
    F, X = F[valid], X[valid]
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F, X = F[nd], X[nd]
    agg = np.sqrt(np.sum(F ** 2, axis=1))
    best_i = int(np.argmin(agg))
    best_X = X[best_i]
    best_params = _vec_to_params(best_X[:n_g])
    best_W = best_X[n_g:]
    # Common random numbers make this re-evaluation identical to the GA's.
    m = _twin_metrics(best_params, pop_env, targets, skeleton, best_W)
    if m is None:
        print("      Optimisation failed to find a valid twin")
        return None
    details = {k: m[k] for k in ('pred_rate', 'twin_freq', 'containment',
                                 'rmse_rate', 'rmse_freq', 'rmse_spec', 'overall')}
    optimiser = {
        'algorithm': 'NSGA-III (Sethi, Faraz & Wong-Lin, arXiv:2605.25224)',
        'pop_size': TWIN_POP_SIZE, 'n_gen': TWIN_N_GEN,
        'n_trials_per_eval': TWIN_N_TRIALS,
        'ref_dirs': f'das-dennis p={TWIN_REF_PARTITIONS} ({ref_dirs.shape[0]} dirs)',
        'crossover': 'SBX(eta=30, prob=1.0)', 'mutation': 'PM(eta=20, prob=1/n_var)',
        'n_var': int(len(xl)), 'n_recurrent_weights': int(n_w),
        'objectives': 'F = RMSE_over_trials/(target+eps): rate, dom_freq, 1-containment',
        'composite': 'sqrt(sum_j F_j^2)', 'seed': RANDOM_SEED,
    }
    print(f"      Best composite RMSE sqrt(sum F^2): {m['overall']:.4f} | "
          f"rate {m['pred_rate']:.4f} vs {targets['target_rate']:.4f} Hz | "
          f"freq {m['twin_freq']:.4f} vs {targets['target_freq']:.4f} Hz | "
          f"containment {m['containment']:.3f} | front size {len(F)} | "
          f"recurrent weights {n_w} | {TWIN_POP_SIZE}x{TWIN_N_GEN} gens, "
          f"{TWIN_N_TRIALS} trials/eval")
    return {'params': best_params, 'targets': targets, 'details': details,
            'overall': m['overall'], 'skeleton': skeleton, 'W_values': best_W,
            'pareto_F': F.tolist(), 'pareto_agg': agg.tolist(),
            'best_index': best_i, 'convergence': conv_json,
            'optimiser': optimiser}


def plot_twin_report(twin, pop_env, title, output_path, skeleton=None, W_values=None):
    """2x2 twin report: envelope overlay, PSD match, resonator recruitment, summary."""
    params, targets, details = twin['params'], twin['targets'], twin['details']
    activity, amplitudes, frange = simulate_twin(
        params, len(pop_env), skeleton, W_values, n_trials=TWIN_N_TRIALS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    t = np.arange(len(pop_env)) * ENV_BIN_S
    org_n = (pop_env - pop_env.mean()) / (pop_env.std() + 1e-10)
    act0 = activity[0]
    act_n = (act0 - act0.mean()) / (act0.std() + 1e-10)
    ax.plot(t, org_n, 'b-', lw=0.8, alpha=0.8, label='Organoid rate envelope')
    ax.plot(t, act_n, 'r-', lw=0.8, alpha=0.8,
            label=f'RRN twin activity (trial 1/{TWIN_N_TRIALS})')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Normalised activity')
    ax.set_title('Population dynamics (z-scored)'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(targets['psd_freqs'], targets['psd_norm'], 'b-', lw=1.5,
                label='Organoid')
    x = detrend(activity.astype(np.float64), axis=1)
    freqs, psd = welch(x, fs=1.0 / ENV_BIN_S, nperseg=targets['nperseg'], axis=1)
    band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
    band_psd = psd[:, band]
    band_norm = band_psd / (band_psd.sum(axis=1, keepdims=True) + 1e-30)
    for row in band_norm:  # per-trial spectra (paper Eqn. 3 trials)
        ax.semilogy(freqs[band], row, 'r-', lw=0.5, alpha=0.25)
    ax.semilogy(freqs[band], band_norm.mean(axis=0), 'r-', lw=1.5,
                label=f'Twin (mean of {TWIN_N_TRIALS} trials)')
    ax.axvline(targets['target_freq'], color='blue', ls='--', lw=1,
               label=f"target {targets['target_freq']:.3f} Hz")
    ax.axvline(details['twin_freq'], color='red', ls='--', lw=1,
               label=f"twin {details['twin_freq']:.3f} Hz")
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Normalised PSD')
    ax.set_title('Dominant oscillation (KongFatt objective)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    recruitment = amplitudes.mean(axis=0)
    ax.stem(frange, recruitment, linefmt='g-', markerfmt='go', basefmt=' ')
    ax.set_xscale('log')
    ax.set_xlabel('Resonator frequency (Hz)'); ax.set_ylabel('Mean amplitude')
    ax.set_title('Resonator recruitment (interpretability: which rhythms make the state)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"KongFatt objectives, RMSE over {TWIN_N_TRIALS} trials / target (Eqn. 3)\n"
        f"  Population rate:  {targets['target_rate']:.4f} -> {details['pred_rate']:.4f} Hz "
        f"(F {details['rmse_rate']:.3f})\n"
        f"  Dominant freq:    {targets['target_freq']:.4f} -> {details['twin_freq']:.4f} Hz "
        f"(F {details['rmse_freq']:.3f})\n"
        f"  PSD containment:  {details['containment']:.3f} "
        f"(F {details['rmse_spec']:.3f})\n"
        f"  COMPOSITE sqrt(sum F^2): {details['overall']:.4f}\n"
        f"  NSGA-III: pop {TWIN_POP_SIZE} x {TWIN_N_GEN} gens "
        f"(arXiv:2605.25224 settings)\n\n"
        f"Twin parameters\n"
        f"  drive_freq={params['drive_freq']:.4f} Hz, drive_amp={params['drive_amp']:.3f}\n"
        f"  bgr={params['base_geometric_ratio']:.3f}, sr={params['spectral_radius']:.3f}\n"
        f"  output_gain={params['output_gain']:.3f}, sigma={TWIN_SIGMA} (fixed)\n"
        f"  recurrent weights: {len(skeleton[0]) if skeleton else 'random skeleton'}"
        f" nonzero on fixed skeleton (evolved)")
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


def plot_ga_report(twin, title, output_path):
    """NSGA-III optimisation report a la the paper's Figs. 3D/5C: pairwise
    Pareto-front projections coloured by composite RMSE (darker = lower, best
    member arrowed) plus the per-generation convergence trace that justifies
    the generation count."""
    F = np.asarray(twin.get('pareto_F', []), dtype=float)
    agg = np.asarray(twin.get('pareto_agg', []), dtype=float)
    conv = twin.get('convergence', {}) or {}
    if F.ndim != 2 or F.size == 0 or agg.size != F.shape[0]:
        return
    best_i = int(twin.get('best_index', int(np.argmin(agg))))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    names = ['F firing rate', 'F dominant freq', 'F spectral containment']
    pairs = [(0, 1), (0, 2), (1, 2)]
    order = np.argsort(-agg)  # draw worse (lighter) first so best stays on top
    for ax, (i, j) in zip(axes.flat[:3], pairs):
        sc = ax.scatter(F[order, i], F[order, j], c=agg[order], cmap='Purples_r',
                        s=48, edgecolors='black', linewidths=0.4)
        ax.scatter([F[best_i, i]], [F[best_i, j]], marker='*', s=380,
                   facecolor='purple', edgecolor='black', linewidths=0.8,
                   zorder=5)
        ax.annotate('lowest composite RMSE', xy=(F[best_i, i], F[best_i, j]),
                    xytext=(0.55, 0.9), textcoords='axes fraction',
                    color='purple', fontweight='bold', fontsize=9,
                    arrowprops=dict(color='purple', arrowstyle='->', lw=1.8))
        ax.set_xlabel(f'{names[i]} (normalised RMSE)')
        ax.set_ylabel(f'{names[j]} (normalised RMSE)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='composite sqrt(sum F^2), darker = lower')

    ax = axes.flat[3]
    best = conv.get('best', [])
    med = conv.get('median_valid', [])
    if best:
        ax.plot(np.arange(1, len(best) + 1), best, 'o-', color='purple',
                label='best composite RMSE')
    if med:
        ax.plot(np.arange(1, len(med) + 1), med, 's--', color='gray',
                label='median composite RMSE (valid)')
    if best or med:
        ax.set_yscale('log')
        ax.legend(fontsize=9)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Composite RMSE sqrt(sum F^2)')
    ax.set_title(f'Convergence (pop {TWIN_POP_SIZE}, {TWIN_N_TRIALS} trials/eval)')
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'{title}\nPareto front, {F.shape[0]} non-dominated solutions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


# =============================================================================
# CEBRA: organoid vs twin latent comparison
# =============================================================================
def _cebra_fit_transform(features):
    import cebra
    model = cebra.CEBRA(model_architecture='offset10-model', batch_size=512,
                        learning_rate=3e-4, output_dimension=CEBRA_OUT_DIM,
                        max_iterations=CEBRA_MAX_ITER, distance='cosine',
                        conditional='time_delta', device='cuda_if_available',
                        verbose=False)
    model.fit(features)
    return model.transform(features)


def _subsample(x, max_n, idx=None):
    if len(x) <= max_n:
        return x
    return x[idx]


def run_cebra_comparison(org_features, twin_features, label, out_dir):
    """CEBRA latents for organoid vs twin + alignment metrics."""
    try:
        rng = np.random.default_rng(RANDOM_SEED)
        m_full = min(len(org_features), len(twin_features))
        org_features, twin_features = org_features[:m_full], twin_features[:m_full]
        # Shared indices: organoid and twin rows are the same timepoints
        idx = (np.sort(rng.choice(m_full, size=CEBRA_MAX_SAMPLES, replace=False))
               if m_full > CEBRA_MAX_SAMPLES else np.arange(m_full))
        org = StandardScaler().fit_transform(_subsample(org_features, CEBRA_MAX_SAMPLES, idx))
        twin = StandardScaler().fit_transform(_subsample(twin_features, CEBRA_MAX_SAMPLES, idx))
        emb_org = _cebra_fit_transform(org)
        emb_twin = _cebra_fit_transform(twin)

        m = min(len(emb_org), len(emb_twin))
        eo, et = emb_org[:m], emb_twin[:m]
        split = int(0.7 * m)
        r2_ot = Ridge().fit(eo[:split], et[:split]).score(eo[split:], et[split:])
        r2_to = Ridge().fit(et[:split], eo[:split]).score(et[split:], eo[split:])
        _, _, disparity = procrustes(eo, et)

        fig = plt.figure(figsize=(16, 7))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(eo[:, 0], eo[:, 1], eo[:, 2], c=np.arange(m), cmap='viridis', s=3)
        ax.set_title(f'Organoid latent - {label}')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(et[:, 0], et[:, 1], et[:, 2], c=np.arange(m), cmap='viridis', s=3)
        ax.set_title(f'RRN twin latent - {label}')
        plt.suptitle(f'CEBRA latents | ridge R2 org->twin={r2_ot:.3f}, '
                     f'twin->org={r2_to:.3f}, procrustes={disparity:.3f}')
        plt.tight_layout()
        _safe_savefig(fig, os.path.join(out_dir, f'cebra_{label}.png'), dpi=150)
        plt.close(fig)
        return {'r2_org_to_twin': float(r2_ot), 'r2_twin_to_org': float(r2_to),
                'procrustes': float(disparity)}
    except Exception as e:
        print(f"      CEBRA comparison failed for {label}: {e}")
        return None


def run_cebra_grouped(feature_list, labels, title, output_path):
    """One CEBRA space across states/sessions, coloured by label (state separation)."""
    try:
        X = np.vstack(feature_list)
        y = np.concatenate([np.full(len(f), lab) for f, lab in zip(feature_list, labels)])
        X = StandardScaler().fit_transform(X)
        emb = _cebra_fit_transform(X)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        uniq = sorted(set(y))
        for lab in uniq:
            idx = y == lab
            ax.scatter(emb[idx, 0], emb[idx, 1], emb[idx, 2], s=3, alpha=0.6, label=lab)
        ax.set_title(title)
        ax.legend(fontsize=8, markerscale=3)
        plt.tight_layout()
        _safe_savefig(fig, output_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"      Grouped CEBRA failed ({title}): {e}")


# =============================================================================
# Per-recording analysis
# =============================================================================
def process_recording(rec, out_base):
    """Full per-state pipeline: spikes -> PP-GLM -> twin -> CEBRA comparison."""
    subject, session, state = rec['subject'], rec['session'], rec['state']
    label = state if session == 'unknown' else f'{session}_{state}'
    print(f"\n   [{subject}] {label} ({rec['path'].name})")

    rec_out = os.path.join(out_base, subject)
    os.makedirs(rec_out, exist_ok=True)
    result = {'subject': subject, 'session': session, 'state': state,
              'session_start': None, 'ppglm': None, 'twin': None, 'cebra': None}

    loaded = load_recording(rec['path'])
    if loaded is None:
        return result
    raw, fs = loaded['raw'], loaded['fs']
    if loaded['session_start'] is not None:
        result['session_start'] = str(loaded['session_start'])
    print(f"      Stream: {fs:.0f} Hz, shape {raw.shape}")

    spk, counts = detect_spikes(raw, fs)
    del raw
    gc.collect()
    duration_s = spk.shape[0] / 1000.0
    print(f"      Spikes: {int(counts.sum())} total over {duration_s:.0f}s "
          f"across {(counts >= MIN_SPIKES_PER_CHANNEL).sum()} active channels")
    pop_env, ch_env = compute_envelopes(spk)

    if RUN_PPGLM:
        pp = run_ppglm(spk, counts)
        if pp is not None:
            pp_dir = os.path.join(rec_out, 'ppglm', label)
            os.makedirs(pp_dir, exist_ok=True)
            n_ch = pp['W'].shape[0]
            plot_ppglm_matrix(pp['W'], f'{subject} - {label} - PP-GLM Coupling',
                              os.path.join(pp_dir, 'ppglm_matrix.png'))
            plot_node_importance(pp['node_strengths'],
                                 f'{subject} - {label} - Node Importance',
                                 os.path.join(pp_dir, 'node_importance.png'))
            # Directed PP-GLM graph on the 3D organoid shell (shell.py layout).
            # Applied to any folded-shell recording: every active electrode must
            # map onto a known shell position (true for the 16-ch shell MEA even
            # when some electrodes drop out for low spike count).
            positions = get_electrode_positions_3d()
            active_channels = pp['active_channels']
            if len(active_channels) and all(int(c) < len(positions) for c in active_channels):
                plot_ppglm_3d(positions, active_channels, pp['edges'], pp['node_strengths'],
                              f'{subject} - {label} - Directed PP-GLM Network',
                              os.path.join(pp_dir, 'ppglm_3d.png'))
            np.save(os.path.join(pp_dir, 'ppglm_matrix.npy'), pp['W'])
            summary = {
                'state': label, 'n_channels': int(n_ch),
                'null_threshold': pp['threshold'], 'n_edges': len(pp['edges']),
                'top_edges': [(s, t, w) for s, t, w in pp['edges'][:10]],
                'top_drivers': sorted(range(n_ch),
                                      key=lambda x: pp['node_strengths'][x]['out'],
                                      reverse=True)[:3],
                'top_receivers': sorted(range(n_ch),
                                        key=lambda x: pp['node_strengths'][x]['in'],
                                        reverse=True)[:3],
                'active_channels': pp['active_channels'].tolist(),
            }
            with open(os.path.join(pp_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            result['ppglm'] = {'n_edges': len(pp['edges']),
                               'mean_abs_coupling': float(np.mean(np.abs(pp['W']))),
                               'threshold': pp['threshold']}
            print(f"      PP-GLM: {len(pp['edges'])} edges above null threshold "
                  f"{pp['threshold']:.4f}")

    twin_result, twin_amps = None, None
    if RUN_TWINNING:
        twin_result = run_twinning(pop_env)
        if twin_result is not None:
            tw_dir = os.path.join(rec_out, 'twinning', label)
            os.makedirs(tw_dir, exist_ok=True)
            skel = twin_result.get('skeleton')
            Wv = twin_result.get('W_values')
            plot_twin_report(twin_result, pop_env,
                             f'{subject} - {label} - RRN Digital Twin',
                             os.path.join(tw_dir, 'twin_report.png'),
                             skeleton=skel, W_values=Wv)
            plot_ga_report(twin_result,
                           f'{subject} - {label} - NSGA-III optimisation',
                           os.path.join(tw_dir, 'ga_pareto_convergence.png'))
            _, twin_amps, _ = simulate_twin(twin_result['params'], len(pop_env), skel, Wv)
            params_json = {
                'state': label, 'overall_rmse': twin_result['overall'],
                'targets': {'rate_hz': twin_result['targets']['target_rate'],
                            'dom_freq_hz': twin_result['targets']['target_freq']},
                'achieved': twin_result['details'],
                'rrn_params': {k: float(v) for k, v in twin_result['params'].items()},
                'recurrent_skeleton': ({'rows': list(skel[0]), 'cols': list(skel[1])}
                                       if skel is not None else None),
                'recurrent_weights': (np.asarray(Wv).tolist()
                                      if Wv is not None else None),
                'sigma': TWIN_SIGMA,
                'optimiser': twin_result.get('optimiser'),
                'convergence': twin_result.get('convergence'),
                'pareto_front_F': twin_result.get('pareto_F'),
            }
            with open(os.path.join(tw_dir, 'twin_params.json'), 'w') as f:
                json.dump(params_json, f, indent=2)
            result['twin'] = {'overall': twin_result['overall'],
                              'target_rate': twin_result['targets']['target_rate'],
                              'target_freq': twin_result['targets']['target_freq'],
                              'pred_rate': twin_result['details']['pred_rate'],
                              'twin_freq': twin_result['details']['twin_freq'],
                              'params': {k: float(v)
                                         for k, v in twin_result['params'].items()}}

    if RUN_CEBRA and twin_amps is not None:
        ce_dir = os.path.join(rec_out, 'cebra')
        os.makedirs(ce_dir, exist_ok=True)
        result['cebra'] = run_cebra_comparison(ch_env, twin_amps, label, ce_dir)

    if RUN_CEBRA:
        # small (n_bins x C float32) - kept for the grouped cross-state CEBRA plot
        result['_ch_env'] = ch_env
    else:
        del ch_env
    del spk, pop_env
    if twin_amps is not None:
        del twin_amps
    gc.collect()
    return result


# =============================================================================
# Longitudinal analysis
# =============================================================================
def run_longitudinal(subject, results, out_base):
    """Track per-session targets, twin fidelity, params and connectivity over time."""
    sessions = {}
    for r in results:
        if r['subject'] != subject:
            continue
        key = r['session_start'] or r['session']
        sessions.setdefault(key, []).append(r)
    if len(sessions) < 2:
        print(f"   Longitudinal: only {len(sessions)} session(s) for {subject}; skipping")
        return
    ordered = sorted(sessions.items(), key=lambda kv: str(kv[0]))

    rate_t, freq_t, rmse_t, edges_t, coupling_t, labels = [], [], [], [], [], []
    param_track = {}
    for key, rs in ordered:
        twins = [r['twin'] for r in rs if r['twin'] is not None]
        pps = [r['ppglm'] for r in rs if r['ppglm'] is not None]
        labels.append(str(key)[:19])
        rate_t.append(np.mean([t['target_rate'] for t in twins]) if twins else np.nan)
        freq_t.append(np.mean([t['target_freq'] for t in twins]) if twins else np.nan)
        rmse_t.append(np.mean([t['overall'] for t in twins]) if twins else np.nan)
        edges_t.append(np.mean([p['n_edges'] for p in pps]) if pps else np.nan)
        coupling_t.append(np.mean([p['mean_abs_coupling'] for p in pps]) if pps else np.nan)
        for t in twins:
            for k, v in t['params'].items():
                param_track.setdefault(k, []).append(v)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        (rate_t, 'Population firing rate (Hz)', 'tab:blue'),
        (freq_t, 'Dominant oscillation freq (Hz)', 'tab:orange'),
        (rmse_t, 'Twin overall RMSE (lower = better twin)', 'tab:green'),
        (edges_t, 'PP-GLM edges above null', 'tab:red'),
        (coupling_t, 'Mean |PP-GLM coupling|', 'tab:purple'),
    ]
    for ax, (vals, name, color) in zip(axes.flat, panels):
        ax.plot(x, vals, 'o-', color=color)
        ax.set_ylabel(name); ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
    ax = axes.flat[5]
    for k, vals in param_track.items():
        ax.plot(np.arange(len(vals)), vals, 'o-', label=k)
    ax.set_title('Twin parameter trajectories')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'{subject} - Longitudinal state tracking', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    lon_dir = os.path.join(out_base, subject, 'longitudinal')
    os.makedirs(lon_dir, exist_ok=True)
    _safe_savefig(fig, os.path.join(lon_dir, 'longitudinal.png'))
    plt.close(fig)
    with open(os.path.join(lon_dir, 'longitudinal_summary.json'), 'w') as f:
        json.dump({'sessions': labels, 'rate': rate_t, 'dom_freq': freq_t,
                   'twin_rmse': rmse_t, 'ppglm_edges': edges_t,
                   'mean_abs_coupling': coupling_t}, f, indent=2)
    print(f"   Longitudinal report saved for {subject} ({len(labels)} sessions)")


# =============================================================================
# Main
# =============================================================================
def main():
    print("Shell MEA State-Conditional Resonant Digital Twin Pipeline")
    print(f"   PP-GLM: {RUN_PPGLM} | Twinning: {RUN_TWINNING} | "
          f"CEBRA: {RUN_CEBRA} | Longitudinal: {RUN_LONGITUDINAL}")
    print(f"   Data root: {DATA_ROOT}")

    records = discover_recordings(DATA_ROOT)
    if not records:
        print(f"   No NWB files found in {DATA_ROOT}")
        return
    subjects = sorted({r['subject'] for r in records})
    print(f"   Found {len(records)} recordings | subjects: {subjects}")

    results = []
    for rec in records:
        try:
            results.append(process_recording(rec, main_output_dir))
        except Exception as e:
            print(f"      Error processing {rec['path'].name}: {e}")
            import traceback
            traceback.print_exc()

    # Grouped CEBRA: state separation (BO14 by stim, SO1 by session)
    if RUN_CEBRA:
        for subject in subjects:
            subj_results = [r for r in results if r['subject'] == subject]
            feature_list, labels = [], []
            for r in subj_results:
                # envelopes are recomputed cheaply from nothing saved; skip if absent
                feats = r.get('_ch_env') if isinstance(r, dict) else None
                if feats is not None:
                    feature_list.append(feats)
                    lab = r['state'] if r['session'] == 'unknown' else r['session']
                    labels.append(lab)
            if len(feature_list) >= 2:
                run_cebra_grouped(
                    feature_list, labels,
                    f'{subject} - CEBRA organoid states',
                    os.path.join(main_output_dir, subject,
                                 f'cebra_states_{subject}.png'))

    if RUN_LONGITUDINAL:
        for subject in subjects:
            run_longitudinal(subject, results, main_output_dir)

    with open(os.path.join(main_output_dir, 'run_summary.json'), 'w') as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith('_')}
                   for r in results], f, indent=2, default=str)
    print(f"\nComplete. Results in {main_output_dir}")


if __name__ == "__main__":
    main()
