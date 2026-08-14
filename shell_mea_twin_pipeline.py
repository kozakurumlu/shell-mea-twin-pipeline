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
     - FIVE objectives are kept SEPARATE for NSGA-III - (a) population firing
       rate, (b) dominant-oscillation mismatch (peak-frequency error when the
       organoid has a real interior spectral peak; spurious-peak fraction when
       it does not - bursty 1/f states are handled explicitly), (c) envelope
       spectral containment, (d) teacher-forced synchronisation loss (the same
       reservoir driven by the organoid envelope must track it via a ridge
       readout on held-out data - generalized synchronisation), (e) a
       Wasserstein rate-distribution loss that matches fluctuation/burst
       amplitude so flat twins are penalised;
     - NSGA-III (Das-Dennis reference directions, SBX eta=30/p=1.0,
       polynomial mutation eta=20, i.e. Deb & Jain 2014 settings) runs for
       TWIN_N_GEN generations with generation size TWIN_POP_SIZE (paper: 25
       generations, generation size 50);
     - the twin is the Pareto-front member with the lowest overall composite
       RMSE, the RMS of the normalised RMSEs sqrt(mean_j F_j^2) (paper
       Sec. III), and the per-generation convergence + Pareto front + paper
       Fig. 4-style activity traces are saved as figures.
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
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, detrend, lfilter
from scipy.ndimage import median_filter
from scipy.stats import wasserstein_distance
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
MIN_ENV_BINS = 64        # minimum envelope bins for twinning. Long states
                         # (BO14/SO1, 5-8 min) give well-resolved sub-Hz
                         # spectra; short ~90 s states (BO2) are still twinned
                         # on rate/bursts/sync, with coarser spectral targets.
MAX_ENV_BINS = 14400     # cap at 4 h
TWIN_FMIN = 0.015        # resonator bank floor (Hz): slowest envelope rhythm
                         # resolvable over the 5-8 min recordings (~7 cycles)
TWIN_FMAX = 0.5          # resonator bank ceiling (Hz) = Fs/2 at envelope Fs = 1 Hz
# Resonator centre frequencies follow the GOLDEN-RATIO organisation of the
# RRN of Kramer, "Brain-inspired, interpretable, resonant recurrent neural
# networks" (arXiv:2506.17083): geometric spacing with an irrational ratio
# mirrors the observed organisation of in vivo brain rhythms and
# outperformed factor-of-two, Euler and (worst) LINEAR spacing in that
# paper's Table 1. A single phi ladder puts only 8 nodes in the sub-Hz
# band, which measurably weakened the teacher-forced synchronisation
# readout (fewer basis amplitudes), so the bank uses TWO INTERLEAVED golden
# ladders - adjacent ratio sqrt(phi) ~ 1.272, every second node exactly phi
# apart - anchored at TWIN_FMIN and climbing to Nyquist: K = 15 nodes,
# 0.015 ... 0.437 Hz, preserving the irrational geometric structure at the
# reservoir capacity the sync objective needs.
TWIN_GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
TWIN_GOLDEN_STEP = np.sqrt(TWIN_GOLDEN)
TWIN_FRANGE = TWIN_FMIN * TWIN_GOLDEN_STEP ** np.arange(
    int(np.floor(np.log(TWIN_FMAX / TWIN_FMIN) / np.log(TWIN_GOLDEN_STEP))) + 1)
TWIN_FRANGE = TWIN_FRANGE[TWIN_FRANGE < TWIN_FMAX]
TWIN_K = int(TWIN_FRANGE.size)  # FIXED resonator count -> fixed weight-matrix
                         # dimension, so the recurrent-weight genome stays
                         # aligned across the whole NSGA-III search (weight
                         # VALUES are evolved on a fixed skeleton).
TWIN_SKELETON_SPARSITY = 0.35  # fixed recurrent-connectivity density (the skeleton)
TWIN_PEAK_BAND = (0.03, 0.45)  # band for dominant-frequency target
TWIN_PEAK_MIN_RATIO = 1.5      # a dominant peak must top this x band-median
                               # power; also must be an INTERIOR local max, so
                               # a monotone 1/f decay (argmax on the band edge)
                               # is explicitly "no dominant oscillation"
TWIN_SIGMA = 0.005       # default noise scale (also an evolvable gene now:
                         # fluctuation/burst amplitude needs a knob)
TWIN_BURNIN_S = 90       # reservoir washout (s) discarded from every sim:
                         # the zero-state equilibration transient must not
                         # reach the objectives (the exp readout would turn
                         # it into a fake burst) or the figures. 90 s covers
                         # the slow-pole parameter corner (bgr ~0.99) where a
                         # 30 s washout left a visible residual decay
TWIN_SYNC_TRAIN_FRAC = 0.7  # ridge-readout train fraction for the sync loss
# --- NSGA-III settings following Sethi, Faraz & Wong-Lin (arXiv:2605.25224) ---
TWIN_N_GEN = 25          # generations (paper: 25, chosen on convergence; the
                         # per-generation convergence trace is saved to check)
TWIN_POP_SIZE = 50       # generation size (paper: 50)
TWIN_REF_PARTITIONS = 3  # Das-Dennis partitions for the FIVE objectives
                         # (rate, dom-osc, spectrum, sync, distribution)
                         # -> C(3+4,4)=35 reference directions <= pop size
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
# Local checkout: data/DANDI_001336 and figures/ next to this script are the
# defaults, so `python shell_mea_twin_pipeline.py` needs no environment
# variables. Colab keeps its Drive defaults. SHELL_MEA_DATA_ROOT /
# SHELL_MEA_FIGURES_DIR / SHELL_MEA_OUTPUT_ROOT still override everything.
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_DATA_ROOT = _SCRIPT_DIR / "data" / "DANDI_001336"
DEFAULT_DATA_ROOT = (_LOCAL_DATA_ROOT if _LOCAL_DATA_ROOT.is_dir() else
                     Path("/content/drive/MyDrive/DANDI_001336_human_neural_organoids_shell_MEA_neuromodulation"))
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
if not _figures_dir and _LOCAL_DATA_ROOT.is_dir():
    _figures_dir = str(_SCRIPT_DIR / "figures")  # local default: ./figures
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
        """Fixed golden-ratio resonator bank covering the sub-Hz envelope band.

        K oscillators at TWIN_FRANGE (golden-ratio spacing, Kramer
        arXiv:2506.17083 Eq. 4): w1 = 2 r cos(2 pi f / Fs), w2 = -r^2, giving
        AR(2) poles r e^{+-i 2 pi f / Fs} - stable for ANY f < Fs/2 whenever
        r < 1, so no sign constraint on w1 is needed. (The previous
        implementation clamped every node with w1 <= 0, i.e. every centre
        frequency above Fs/4 = 0.25 Hz, silently parking the upper half of a
        linear 0.02-0.5 Hz bank on a single duplicate resonator.) K is
        CONSTANT across the whole search (the recurrent-weight genome must
        stay aligned); a pathological damping value is clamped, not dropped.
        """
        frange = TWIN_FRANGE.astype(np.float64).copy()
        r = min(max(float(self.base_geometric_ratio), 0.0), 0.99999)
        w_t_minus_1 = 2 * r * np.cos(2 * np.pi * frange / self.Fs)
        w_t_minus_2 = (-r**2) * np.ones_like(w_t_minus_1)
        valid = frange < self.Fs / 2.0
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
    """Load the raw ephys stream decimated to ~SPIKE_FS. Returns dict or None.

    DANDI 001336 files carry two 2D acquisition series: 'ES' (the recorded
    electrode stream) and 'ES_STIM' (the stimulation command waveform). Select
    the recorded stream explicitly - never the stim channel - instead of
    relying on dict iteration order.
    """
    try:
        with NWBHDF5IO(str(path), "r") as io:
            nwbfile = io.read()
            session_start = getattr(nwbfile, "session_start_time", None)
            twod = {name: d for name, d in nwbfile.acquisition.items()
                    if hasattr(d, "data") and len(d.data.shape) == 2}
            if not twod:
                print(f"      No 2D acquisition series in {path.name}")
                return None
            if 'ES' in twod:
                name = 'ES'
            else:
                non_stim = [n for n in sorted(twod) if 'STIM' not in n.upper()]
                name = non_stim[0] if non_stim else sorted(twod)[0]
            d = twod[name]
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
def _smooth_band(band_psd):
    """Edge-corrected 3-bin moving average: single-bin estimator spikes cannot
    fake a peak. (A plain mode='same' average would bias the edge bins down
    and shift a monotone spectrum's argmax to bin 1 - must divide by the
    actual kernel coverage instead.)"""
    kernel = np.ones(3)
    return (np.convolve(band_psd, kernel, mode='same')
            / np.convolve(np.ones_like(band_psd), kernel, mode='same'))


def _band_peak_stat(band_psd):
    """(interior peak index, prominence) of a band spectrum.

    Prominence is measured against a LOCAL background - a wide median filter
    of the smoothed spectrum - not against the global median. This is the
    lightweight form of spectral parameterization: a smooth 1/f-type TILT
    (which burst width/autocorrelation produces) is absorbed into the
    background, while a genuine RHYTHM is a narrow local excess above it.
    Without this, any red-tilted burst state shows its smoothed argmax at an
    interior low-frequency bin and masquerades as an oscillation.
    """
    sm = _smooth_band(np.asarray(band_psd, dtype=float))
    n = sm.size
    k = max(5, (n // 3) | 1)  # odd, ~1/3 of the band: wider than any rhythm bump
    bg = median_filter(sm, size=k, mode='nearest')
    r = sm / np.maximum(bg, 1e-300)
    i = int(np.argmax(r[1:-1])) + 1  # interior bins only
    return i, float(r[i])


def _peak_ratio_threshold(x, nperseg, n_surr=200, q=95.0):
    """False-alarm calibration for the dominant-peak criterion.

    A peakless spectrum estimated over few Welch segments fluctuates enough
    that its interior maximum clears a fixed 1.5 x median bar most of the
    time (~80% measured on impulse trains at T=480), so a fixed ratio
    silently converts "no oscillation" states into fake peaks - and, on the
    twin side, punishes honest flat-spectrum twins through the spurious-peak
    objective, steering the GA AWAY from the correct solution. Calibrate
    against a PERMUTATION null instead: shuffling the envelope bins destroys
    all timing structure while preserving the exact amplitude distribution
    (critical for heavy-tailed burst envelopes, whose spectral estimates
    fluctuate far more than Gaussian noise), so the threshold is the q-th
    percentile of the interior-max / median ratio of the same
    Welch + 3-bin-smoothing estimator over shuffled surrogates: an exact
    level-(100-q)% test of "oscillation beyond the amplitude distribution".
    Deterministic (fixed seed); TWIN_PEAK_MIN_RATIO stays as a minimum
    effect size.
    """
    rng = np.random.default_rng(TWIN_SEED + 2)
    x = np.asarray(x, dtype=np.float64)
    ratios = []
    for _ in range(n_surr):
        xs = rng.permutation(x)
        f, p = welch(xs, fs=1.0 / ENV_BIN_S, nperseg=nperseg)
        b = (f >= TWIN_PEAK_BAND[0]) & (f <= TWIN_PEAK_BAND[1])
        pb = p[b]
        if pb.size < 3:
            continue
        if np.max(pb) > 0:
            ratios.append(_band_peak_stat(pb)[1])
    if not ratios:
        return TWIN_PEAK_MIN_RATIO
    return max(TWIN_PEAK_MIN_RATIO, float(np.percentile(ratios, q)))


def _dominant_freq(band_freqs, band_psd, min_ratio=None):
    """Frequency of the dominant spectral peak, or None when there is none.

    A real dominant oscillation must be (a) an INTERIOR local maximum of the
    band (the edges cannot carry a resolvable rhythm), and (b) SIGNIFICANTLY
    prominent above the local spectral background (_band_peak_stat): at
    least min_ratio x the median-filtered background, where min_ratio should
    come from _peak_ratio_threshold (permutation null of the same signal and
    estimator; defaults to the fixed TWIN_PEAK_MIN_RATIO floor). Both a
    monotone 1/f decay and a red burst-width tilt are absorbed into the
    background and report None - "no dominant oscillation".
    """
    band_psd = np.asarray(band_psd, dtype=float)
    if band_psd.size < 3 or not np.all(np.isfinite(band_psd)) \
            or np.max(band_psd) <= 0:
        return None
    if min_ratio is None:
        min_ratio = TWIN_PEAK_MIN_RATIO
    i, stat = _band_peak_stat(band_psd)
    if stat < float(min_ratio):
        return None
    return float(band_freqs[i])


def _fmt_hz(v):
    return f"{v:.4f} Hz" if v is not None else "none"


def compute_twin_targets(pop_env):
    """KongFatt observables: mean population rate + dominant envelope frequency.

    `target_freq` is None when the state has no dominant oscillation (bursty
    1/f-type spectrum); twinning still runs, with the oscillation objective
    flipped to penalising spurious twin peaks.
    """
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
    # significance threshold for "dominant peak": permutation null of THIS
    # state's envelope (same length, estimator and amplitude distribution),
    # shared by the organoid target and the twin-trial spurious-peak test
    peak_min_ratio = _peak_ratio_threshold(x, nperseg)
    target_freq = _dominant_freq(band_freqs, band_psd, peak_min_ratio)
    psd_norm = band_psd / np.sum(band_psd)
    return {'target_rate': target_rate, 'target_freq': target_freq,
            'has_oscillation': target_freq is not None,
            'peak_min_ratio': peak_min_ratio,
            'psd_freqs': band_freqs, 'psd_norm': psd_norm, 'nperseg': nperseg}


def simulate_twin(params, T, skeleton=None, W_values=None, n_trials=1,
                  input_env=None):
    """RRN twin simulation: autonomous by default, coupled when teacher-forced.

    Autonomous mode (input_env=None): sinusoidal rhythmic drive + noise, no
    data input - the twin GENERATES the state. Coupled mode (input_env=the
    organoid envelope): the drive is replaced by the z-scored organoid
    envelope scaled by params['sync_gain'] - master-slave coupling for the
    generalized-synchronisation objective.

    With `skeleton`+`W_values` the recurrent matrix uses the evolved nonzero
    weights on a fixed skeleton (KongFatt-style connectivity evolution); without
    them the network falls back to a random recurrent matrix (self-contained).

    `n_trials` independent noise realisations are simulated in one vectorised
    batch (paper Eqn. 3 trials). The RRN rng is re-seeded per call, so every
    candidate is evaluated on the SAME noise streams (common random numbers):
    objectives are deterministic in the parameters, which stabilises the GA.

    Returns (activity, amplitudes, frange): activity is (n_trials, T) per-trial
    population activity; amplitudes is (n_trials, T, K) resonator amplitudes.
    """
    rrn = ReservoirNetwork(
        Fs=1.0 / ENV_BIN_S, sigma=float(params.get('sigma', TWIN_SIGMA)),
        sparsity=TWIN_SKELETON_SPARSITY,
        spectral_radius=params['spectral_radius'],
        base_geometric_ratio=params['base_geometric_ratio'],
        random_state=TWIN_SEED, skeleton=skeleton)
    if skeleton is not None and W_values is not None:
        rrn.build_W_res(W_values, params['spectral_radius'])
    # Washout scales with the resonators' memory: an AR(2) pole at radius r
    # decays with time constant ~1/(1-r) bins, so lightly damped (ringing)
    # candidates need a longer equilibration than the flat TWIN_BURNIN_S.
    # 4 time constants, floored at TWIN_BURNIN_S and capped at 600 s.
    r_damp = float(params.get('base_geometric_ratio', 0.93))
    burn_s = min(600.0, max(float(TWIN_BURNIN_S), 4.0 / max(1e-6, 1.0 - r_damp)))
    burn = int(round(burn_s / ENV_BIN_S))
    if input_env is not None:
        e = np.asarray(input_env, dtype=np.float64)[:T]
        e = (e - e.mean()) / (e.std() + 1e-12)
        u_core = float(params.get('sync_gain', 1.0)) * e
        head = u_core[0] if u_core.size else 0.0
        u = np.concatenate([np.full(burn, head), u_core]).reshape(-1, 1)
    else:
        t = np.arange(T + burn) * ENV_BIN_S
        u = (params['drive_amp'] * np.sin(2 * np.pi * params['drive_freq'] * t)
             ).reshape(-1, 1)
    _, amplitudes = rrn.collect_states(u, n_trials=n_trials)  # (B, T+burn, K)
    amplitudes = amplitudes[:, burn:]  # discard zero-state equilibration
    activity = amplitudes.mean(axis=2)  # (B, T) population activity per trial
    return activity, amplitudes, rrn.frange


def twin_rate_traces(params, activity):
    """Twin population-rate traces (Hz): doubly-stochastic burst readout.

    The organoid population rate at 1 s resolution is a quiescent baseline
    punctuated by rare, near-impulsive network bursts (1-2 bins tall spikes to
    ~100 Hz). A smooth transform of the reservoir activity - the previous
    rate = gain * exp(beta*z) readout - can match the rate HISTOGRAM but is
    structurally unable to produce that temporal character: the resonator
    bank lives at 0.02-0.5 Hz, so its readout is red-spectrum wandering,
    while impulsive bursting has a near-flat spectrum. Readout model:

        z_t      = z-scored mean resonator amplitude   (reservoir excitability)
        p_t      = clip(burst_rate * exp(kappa*z_t) / mean_t exp(kappa*z), 0, 1)
        event_t  ~ Bernoulli(p_t)                       (burst timing)
        A_t      = burst_amp * exp(shape*G_t + beta*z_t) / mean_t exp(...)
                                                        (log-normal amplitudes)
        y_t      = event_t*A_t + burst_decay * y_{t-1}  (1-2 s burst kernel)
        rate_t   = base_rate + y_t

    The reservoir modulates burst timing (kappa) and size (beta), so an
    oscillatory reservoir yields rhythmic bursting (SO1) while a weakly
    modulated one yields aperiodic shot noise with a flat spectrum (BO14).
    Dense-event limit (burst_rate -> 1, kappa=0) recovers the smooth
    log-normal readout. Normalising by mean exp(.) keeps burst_rate the mean
    event probability and burst_amp the mean burst height independently of
    kappa/shape/beta. The noise stream is drawn from a FIXED seed distinct
    from the reservoir's, so all candidates see identical draws (common
    random numbers) and objectives stay deterministic in the parameters.
    """
    act = np.asarray(activity, dtype=np.float64)
    act2 = np.atleast_2d(act)
    B, T = act2.shape
    mu = act2.mean(axis=-1, keepdims=True)
    sd = act2.std(axis=-1, keepdims=True)
    z = (act2 - mu) / (sd + 1e-12)
    rng = np.random.default_rng(TWIN_SEED + 1)
    U = rng.random((B, T))
    G = rng.standard_normal((B, T))
    kappa = float(params.get('burst_kappa', 0.0))
    lam = np.exp(np.clip(kappa * z, -30.0, 30.0))
    lam /= lam.mean(axis=-1, keepdims=True) + 1e-300
    p = np.clip(float(params.get('burst_rate', 1.0)) * lam, 0.0, 1.0)
    events = (U < p).astype(np.float64)
    shape = float(params.get('burst_shape', 0.0))
    beta = float(params.get('burst_beta', 0.0))
    A = np.exp(np.clip(shape * G + beta * z, -30.0, 30.0))
    A /= A.mean(axis=-1, keepdims=True) + 1e-300
    s = float(params.get('burst_amp', 0.0)) * events * A
    d = float(params.get('burst_decay', 0.0))
    y = lfilter([1.0], [1.0, -d], s, axis=-1)
    rates = float(params.get('base_rate', 0.0)) + y
    return rates if act.ndim > 1 else rates[0]


def synchronised_twin(params, pop_env, skeleton=None, W_values=None,
                      n_trials=None):
    """Teacher-forced (driven) twin: generalized-synchronisation pass.

    The SAME reservoir is driven by the z-scored organoid envelope
    (master-slave coupling, gain = params['sync_gain']); a ridge readout of
    the resonator amplitudes is fit on the first TWIN_SYNC_TRAIN_FRAC of the
    recording (pooled across trials) and evaluated on the held-out tail.
    This is both the F_sync objective and the pipeline's demonstration that
    the twin can lock onto its organoid rather than only match statistics.

    Returns {'preds': (B, T) readout predictions over the FULL recording,
    'amps': (B, T, K) driven resonator amplitudes, 'split': train/test index,
    'rs': (B,) per-trial held-out Pearson r}.
    """
    if n_trials is None:
        n_trials = TWIN_N_TRIALS
    T = len(pop_env)
    env = np.asarray(pop_env, dtype=np.float64)
    _, amps_c, _ = simulate_twin(params, T, skeleton, W_values,
                                 n_trials=n_trials, input_env=pop_env)
    split = min(max(int(TWIN_SYNC_TRAIN_FRAC * T), 2), T - 2)
    K = amps_c.shape[2]
    reader = Ridge(alpha=1.0).fit(
        amps_c[:, :split].reshape(-1, K).astype(np.float64),
        np.tile(env[:split], n_trials))
    preds = np.stack([reader.predict(amps_c[b].astype(np.float64))
                      for b in range(n_trials)])
    yte = env[split:]
    rs = np.zeros(n_trials)
    if np.std(yte) > 1e-12:
        for b in range(n_trials):
            pred = preds[b, split:]
            if np.std(pred) > 1e-12:
                r = float(np.corrcoef(pred, yte)[0, 1])
                rs[b] = r if np.isfinite(r) else 0.0
    return {'preds': preds, 'amps': amps_c, 'split': split, 'rs': rs}


def _twin_metrics(params, pop_env, targets, skeleton=None, W_values=None,
                  n_trials=None):
    """Return metric dict or None if the twin diverged/died.

    Paper-faithful objective form (Sethi, Faraz & Wong-Lin, arXiv:2605.25224):
    each candidate is simulated for `n_trials` independent noise realisations
    and each objective is the RMSE over trials (Eqn. 3),

        RMSE_x = sqrt( (1/N) * sum_i (x_i - x_target)^2 ),

    normalised by its target, F_x = RMSE_x / (x_target + eps). FIVE normalised
    RMSEs stay SEPARATE as the NSGA-III objectives:
      F_rate: time-averaged population firing rate.
      F_osc : dominant-oscillation mismatch. With a real organoid peak:
              per-trial peak-frequency error (a peakless twin trial scores the
              full band width). With NO organoid peak (bursty 1/f state): the
              fraction of trials showing a spurious dominant peak.
      F_spec: 1 - spectral containment (normalised band-PSD overlap).
      F_sync: teacher-forced synchronisation loss 1 - r: the same reservoir,
              driven by the organoid envelope (master-slave coupling with
              evolvable gain), must track it through a ridge readout scored on
              held-out data - generalized-synchronisation capacity.
      F_dist: Wasserstein-1 distance between organoid and twin firing-rate
              distributions (Hz) / target rate - matches fluctuation/burst
              amplitude, which mean rate + normalised PSD shape both miss.
    `overall` is the composite - the RMS of the normalised RMSEs,
    sqrt(mean_j F_j^2) (paper Sec. III) - used afterwards to pick one twin
    off the Pareto front.
    """
    if n_trials is None:
        n_trials = TWIN_N_TRIALS
    T = len(pop_env)
    try:
        activity, _, _ = simulate_twin(params, T, skeleton, W_values,
                                       n_trials=n_trials)  # (B, T)
        if not np.all(np.isfinite(activity)) or np.any(activity.std(axis=1) < 1e-12):
            return None  # diverged or dead twin in any trial
        rate_traces = twin_rate_traces(params, activity)  # (B, T) in Hz
        rates = rate_traces.mean(axis=1)  # (B,)
        # spectrum/oscillation/distribution all measured on the RATE traces,
        # matching what the organoid side (its rate envelope) provides
        x = detrend(rate_traces, axis=1)
        freqs, psd = welch(x, fs=1.0 / ENV_BIN_S, nperseg=targets['nperseg'],
                           axis=1)  # psd is (B, n_freqs)
        band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
        band_psd = psd[:, band]
        band_tot = band_psd.sum(axis=1)
        if not np.any(band) or not np.all(band_tot > 0):
            return None
        band_freqs = freqs[band]
        twin_psd_norm = band_psd / band_tot[:, None]
        # per-trial dominant peak, same calibrated significance criterion as
        # the organoid target (otherwise flat-spectrum twin trials trip the
        # spurious-peak objective by chance and the GA avoids honest twins)
        twin_freqs = [_dominant_freq(band_freqs, row,
                                     targets.get('peak_min_ratio'))
                      for row in twin_psd_norm]
        org_psd = targets['psd_norm']
        m_bins = min(org_psd.size, twin_psd_norm.shape[1])
        containments = np.minimum(org_psd[None, :m_bins],
                                  twin_psd_norm[:, :m_bins]).sum(axis=1)  # (B,)

        # --- Eqn. 3 RMSE over trials, target-normalised, per objective ---
        rmse_rate = np.sqrt(np.mean((rates - targets['target_rate']) ** 2))
        F_rate = rmse_rate / (targets['target_rate'] + TWIN_RMSE_EPS)

        band_span = TWIN_PEAK_BAND[1] - TWIN_PEAK_BAND[0]
        if targets['target_freq'] is not None:
            # organoid oscillates: per-trial peak error; a peakless twin
            # trial scores the full band width
            errs = np.array([abs(f - targets['target_freq']) if f is not None
                             else band_span for f in twin_freqs])
            F_osc = (np.sqrt(np.mean(errs ** 2))
                     / (targets['target_freq'] + TWIN_RMSE_EPS))
        else:
            # no organoid oscillation: penalise spurious twin peaks
            spurious = np.array([0.0 if f is None else 1.0 for f in twin_freqs])
            F_osc = float(np.sqrt(np.mean(spurious ** 2)))

        F_spec = float(np.sqrt(np.mean((1.0 - containments) ** 2)))

        # --- teacher-forced synchronisation loss (reservoir observer) ---
        # per-trial held-out r with target 1, Eqn. 3 form (see
        # synchronised_twin; the same routine draws panel C of the activity
        # figure, so the objective and the shown trace are one computation).
        rs = synchronised_twin(params, pop_env, skeleton, W_values,
                               n_trials=n_trials)['rs']
        F_sync = float(np.sqrt(np.mean((1.0 - rs) ** 2)))

        # --- burst-statistics loss: Wasserstein-1 between rate histograms ---
        # Time-shift invariant, so the AUTONOMOUS twin can satisfy it; this is
        # the term that forbids a flat twin at the right mean rate.
        wass = np.array([wasserstein_distance(pop_env, r)
                         for r in rate_traces])
        F_dist = float(np.sqrt(np.mean(wass ** 2))
                       / (targets['target_rate'] + TWIN_RMSE_EPS))

        F = np.array([F_rate, F_osc, F_spec, F_sync, F_dist], dtype=float)
        if not np.all(np.isfinite(F)):
            return None
        F = np.minimum(F, TWIN_F_CAP)
        overall = float(np.sqrt(np.mean(F ** 2)))  # RMS of the normalised RMSEs
        tf = [f for f in twin_freqs if f is not None]
        peak_frac = len(tf) / max(1, len(twin_freqs))
        # Report a twin frequency only when it is a property of the twin, not
        # of estimator noise: always when the organoid oscillates (peakless
        # trials are already punished in F_osc), but for a no-oscillation
        # target only if the MAJORITY of trials shows a peak - "twin 0.33 Hz"
        # in a legend, when 2 of 15 trials tripped the 5% test by chance,
        # misreads as an oscillating twin.
        if targets['target_freq'] is not None:
            twin_freq = float(np.median(tf)) if tf else None
        else:
            twin_freq = float(np.median(tf)) if peak_frac >= 0.5 else None
        return {'pred_rate': float(rates.mean()),
                'twin_freq': twin_freq,
                'twin_peak_frac': float(peak_frac),
                'containment': float(containments.mean()),
                'sync_r': float(rs.mean()),
                'wasserstein_hz': float(wass.mean()),
                'rmse_rate': float(F[0]), 'rmse_freq': float(F[1]),
                'rmse_spec': float(F[2]), 'rmse_sync': float(F[3]),
                'rmse_dist': float(F[4]), 'overall': overall}
    except Exception:
        return None


# Global twin parameters searched by NSGA-III (the recurrent-weight VALUES are
# added as extra free variables on a fixed skeleton; see run_twinning). The
# log-uniform variables are searched in log10 space then exponentiated back.
# 'sigma' (reservoir noise) sets autonomous excitability fluctuations;
# 'sync_gain' is the master-slave coupling gain of the teacher-forced pass;
# the burst-* genes parameterise the doubly-stochastic readout
# (see twin_rate_traces): baseline + reservoir-modulated burst events.
TWIN_PARAM_SPECS = [
    ('drive_freq', 0.03, 0.45, None),
    ('drive_amp', 1e-3, 5.0, 'log'),
    ('base_geometric_ratio', 0.70, 0.995, None),  # resonator damping r: up to
                                                  # near-critical ringing
                                                  # (Kramer runs r=0.99999)
    ('spectral_radius', 0.05, 1.2, None),
    ('sigma', 1e-4, 1.0, 'log'),
    ('sync_gain', 1e-2, 1e1, 'log'),
    ('base_rate', 1e-3, 50.0, 'log'),    # quiescent baseline (Hz)
    ('burst_rate', 1e-3, 1.0, 'log'),    # mean burst events / 1 s bin
    ('burst_amp', 1e-2, 300.0, 'log'),   # mean burst height (Hz)
    ('burst_shape', 0.0, 1.5, None),     # log-normal amplitude spread
    ('burst_kappa', 0.0, 5.0, None),     # reservoir->burst-TIMING modulation
    ('burst_beta', 0.0, 3.0, None),      # reservoir->burst-SIZE modulation
    ('burst_decay', 0.0, 0.9, None),     # burst kernel decay / bin (width)
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
    five objectives stay separate for NSGA-III (Das-Dennis reference
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
          f"dom. freq={_fmt_hz(targets['target_freq'])}"
          + ("" if targets['target_freq'] is not None
             else " (1/f-type spectrum: matching profile, bursts and sync)"))

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
    ref_dirs = get_reference_directions("das-dennis", 5,
                                        n_partitions=TWIN_REF_PARTITIONS)

    # Running best-ever composite individual, recorded at EVALUATION time:
    # NSGA-III niching can drop it from survivor populations, and pymoo's
    # `res.opt` holds only the niche representatives of the first front.
    best_seen = {'comp': float('inf'), 'X': None, 'F': None}

    class _TwinProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=5, n_ieq_constr=0, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            params = _vec_to_params(X[:n_g])
            W_values = X[n_g:]
            m = _twin_metrics(params, pop_env, targets, skeleton, W_values)
            if m is not None:
                Fv = np.array([m['rmse_rate'], m['rmse_freq'], m['rmse_spec'],
                               m['rmse_sync'], m['rmse_dist']])
                comp = float(np.sqrt(np.mean(Fv ** 2)))
                if comp < best_seen['comp']:
                    best_seen.update(comp=comp, X=np.array(X, dtype=float), F=Fv)
                out["F"] = Fv
            else:
                # penalise failed sims with a dominated vector
                out["F"] = np.full(5, TWIN_FAIL_F)

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
            comp = np.sqrt((F ** 2).mean(axis=1))  # same RMS form as selection
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

    # Select the Pareto-front member with the lowest composite RMSE, the RMS
    # of the normalised RMSEs (paper Sec. III). Pool res.opt (only the niche
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
    agg = np.sqrt(np.mean(F ** 2, axis=1))
    best_i = int(np.argmin(agg))
    # Oscillation-honesty gate, then minimum composite: within 25% of the
    # front's minimum composite, exclude members with a spurious twin peak
    # when the organoid has none (F_osc = sqrt(spurious trial fraction)), or
    # a far-off peak when it does - otherwise the RMS composite can buy
    # rate/distribution accuracy with an invented low-frequency bump (e.g.
    # the old 0.167 Hz hump on a 1/f state) - and take the lowest composite
    # among the remainder. (A Chebyshev worst-objective pick inside the band
    # was evaluated on the saved real-data fronts and dropped: it doubled
    # the median rate error while improving the median worst objective by
    # under 3% - rate accuracy is the visibly load-bearing objective.)
    osc_tol = 0.4 if targets['target_freq'] is None else 0.5
    near = agg <= 1.25 * float(agg.min())
    honest = F[:, 1] <= osc_tol
    cand = near & honest
    if not np.any(cand):
        cand = near
    sel = int(np.argmin(np.where(cand, agg, np.inf)))
    if sel != best_i:
        print("      Selection: oscillation-honest front member "
              f"(F_osc {F[sel, 1]:.3f} <= {osc_tol}; composite "
              f"{agg[sel]:.4f} vs {agg[best_i]:.4f} at unconstrained min)")
    best_i = sel
    best_X = X[best_i]
    best_params = _vec_to_params(best_X[:n_g])
    best_W = best_X[n_g:]
    # Common random numbers make this re-evaluation identical to the GA's.
    m = _twin_metrics(best_params, pop_env, targets, skeleton, best_W)
    if m is None:
        print("      Optimisation failed to find a valid twin")
        return None
    details = {k: m[k] for k in ('pred_rate', 'twin_freq', 'twin_peak_frac',
                                 'containment', 'sync_r', 'wasserstein_hz',
                                 'rmse_rate', 'rmse_freq', 'rmse_spec',
                                 'rmse_sync', 'rmse_dist', 'overall')}
    optimiser = {
        'algorithm': 'NSGA-III (Sethi, Faraz & Wong-Lin, arXiv:2605.25224)',
        'pop_size': TWIN_POP_SIZE, 'n_gen': TWIN_N_GEN,
        'n_trials_per_eval': TWIN_N_TRIALS,
        'ref_dirs': f'das-dennis p={TWIN_REF_PARTITIONS} ({ref_dirs.shape[0]} dirs)',
        'crossover': 'SBX(eta=30, prob=1.0)', 'mutation': 'PM(eta=20, prob=1/n_var)',
        'n_var': int(len(xl)), 'n_recurrent_weights': int(n_w),
        'objectives': ('F = RMSE_over_trials/(target+eps): rate, dominant-osc '
                       '(spurious-peak fraction when organoid has none), '
                       '1-containment, sync (1-r teacher-forced ridge readout), '
                       'Wasserstein rate-distribution / target rate'),
        'organoid_has_oscillation': bool(targets['target_freq'] is not None),
        'composite': 'sqrt(mean_j F_j^2) (RMS of normalised RMSEs)',
        'seed': RANDOM_SEED,
    }
    print(f"      Best composite RMSE sqrt(mean F^2): {m['overall']:.4f} | "
          f"rate {m['pred_rate']:.4f} vs {targets['target_rate']:.4f} Hz | "
          f"freq {_fmt_hz(m['twin_freq'])} vs {_fmt_hz(targets['target_freq'])} | "
          f"containment {m['containment']:.3f} | sync r {m['sync_r']:.3f} | "
          f"wass {m['wasserstein_hz']:.3f} Hz | front size {len(F)} | "
          f"{TWIN_POP_SIZE}x{TWIN_N_GEN} gens, {TWIN_N_TRIALS} trials/eval")
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
    rate_traces = twin_rate_traces(params, activity)  # (B, T) in Hz

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    t = np.arange(len(pop_env)) * ENV_BIN_S
    rep = int(np.argmin([wasserstein_distance(pop_env, row)
                         for row in rate_traces]))
    ax.plot(t, pop_env, 'b-', lw=0.8, alpha=0.8, label='Organoid rate envelope')
    ax.plot(t, rate_traces[rep], 'r-', lw=0.8, alpha=0.8,
            label=f'RRN twin rate (realisation {rep + 1}/{TWIN_N_TRIALS})')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Population rate (Hz)')
    ax.set_title('Population dynamics (autonomous twin, one realisation)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(targets['psd_freqs'], targets['psd_norm'], 'b-', lw=1.5,
                label='Organoid')
    x = detrend(rate_traces, axis=1)
    freqs, psd = welch(x, fs=1.0 / ENV_BIN_S, nperseg=targets['nperseg'], axis=1)
    band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
    band_psd = psd[:, band]
    band_norm = band_psd / (band_psd.sum(axis=1, keepdims=True) + 1e-30)
    for row in band_norm:  # per-trial spectra (paper Eqn. 3 trials)
        ax.semilogy(freqs[band], row, 'r-', lw=0.5, alpha=0.25)
    ax.semilogy(freqs[band], band_norm.mean(axis=0), 'r-', lw=1.5,
                label=f'Twin (mean of {TWIN_N_TRIALS} trials)')
    if targets['target_freq'] is not None:
        ax.axvline(targets['target_freq'], color='blue', ls='--', lw=1,
                   label=f"target {targets['target_freq']:.3f} Hz")
    if details['twin_freq'] is not None:
        ax.axvline(details['twin_freq'], color='red', ls='--', lw=1,
                   label=f"twin {details['twin_freq']:.3f} Hz")
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Normalised PSD')
    ax.set_title('Dominant oscillation (KongFatt objective)'
                 if targets['target_freq'] is not None else
                 'Spectral profile (no dominant oscillation in organoid)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    recruitment = amplitudes.mean(axis=(0, 1))  # mean over trials and time
    ax.stem(frange, recruitment, linefmt='g-', markerfmt='go', basefmt=' ')
    ax.set_xscale('log')
    ax.set_xlabel('Resonator frequency (Hz)'); ax.set_ylabel('Mean amplitude')
    ax.set_title('Resonator recruitment (interpretability: which rhythms make the state)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"Objectives, RMSE over {TWIN_N_TRIALS} trials / target (Eqn. 3)\n"
        f"  Population rate:  {targets['target_rate']:.4f} -> {details['pred_rate']:.4f} Hz "
        f"(F {details['rmse_rate']:.3f})\n"
        f"  Dominant osc:     {_fmt_hz(targets['target_freq'])} -> "
        f"{_fmt_hz(details['twin_freq'])} (F {details['rmse_freq']:.3f})\n"
        f"  PSD containment:  {details['containment']:.3f} "
        f"(F {details['rmse_spec']:.3f})\n"
        f"  Sync (ridge r):   {details['sync_r']:.3f} "
        f"(F {details['rmse_sync']:.3f})\n"
        f"  Rate Wasserstein: {details['wasserstein_hz']:.3f} Hz "
        f"(F {details['rmse_dist']:.3f})\n"
        f"  COMPOSITE sqrt(mean F^2): {details['overall']:.4f}\n"
        f"  NSGA-III: pop {TWIN_POP_SIZE} x {TWIN_N_GEN} gens, 5 objectives\n\n"
        f"Twin parameters\n"
        f"  drive_freq={params['drive_freq']:.4f} Hz, drive_amp={params['drive_amp']:.3f}\n"
        f"  bgr={params['base_geometric_ratio']:.3f}, sr={params['spectral_radius']:.3f}, "
        f"sigma={params.get('sigma', TWIN_SIGMA):.4f}, "
        f"sync_gain={params.get('sync_gain', 1.0):.3f}\n"
        f"  burst readout: base={params.get('base_rate', 0.0):.3f} Hz + events "
        f"p={params.get('burst_rate', 0.0):.3f}/bin x {params.get('burst_amp', 0.0):.2f} Hz\n"
        f"  shape={params.get('burst_shape', 0.0):.2f}, "
        f"kappa={params.get('burst_kappa', 0.0):.2f} (timing), "
        f"beta={params.get('burst_beta', 0.0):.2f} (size), "
        f"decay={params.get('burst_decay', 0.0):.2f}\n"
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
    """NSGA-III optimisation report a la the paper's Figs. 3D/5C, extended to
    the five-objective problem: parallel-coordinates view of the Pareto front
    coloured by composite RMSE (darker = lower), the two most informative
    pairwise projections with the selected twin starred, and the
    per-generation convergence trace that justifies the generation count."""
    F = np.asarray(twin.get('pareto_F', []), dtype=float)
    agg = np.asarray(twin.get('pareto_agg', []), dtype=float)
    conv = twin.get('convergence', {}) or {}
    if F.ndim != 2 or F.size == 0 or agg.size != F.shape[0]:
        return
    best_i = int(twin.get('best_index', int(np.argmin(agg))))
    n_obj = F.shape[1]
    names = ['F rate', 'F dom osc', 'F spectral', 'F sync', 'F dist'][:n_obj]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    order = np.argsort(-agg)  # draw worse (lighter) first so best stays on top
    cmap = plt.get_cmap('Purples_r')
    a_lo, a_hi = float(agg.min()), float(agg.max())
    span = (a_hi - a_lo) or 1.0

    ax = axes[0, 0]
    xs = np.arange(n_obj)
    for k in order:
        ax.plot(xs, F[k], color=cmap((agg[k] - a_lo) / span), lw=1.0,
                alpha=0.65)
    ax.plot(xs, F[best_i], color='purple', lw=2.6, marker='o',
            label='lowest composite RMSE')
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Normalised RMSE F')
    ax.set_title('Pareto front, parallel coordinates (darker = lower composite)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax, (i, j) in zip((axes[0, 1], axes[1, 0]), ((0, 1), (3, 4))):
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
        plt.colorbar(sc, ax=ax, label='composite sqrt(mean F^2), darker = lower')

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
    ax.set_ylabel('Composite RMSE sqrt(mean F^2)')
    ax.set_title(f'Convergence (pop {TWIN_POP_SIZE}, {TWIN_N_TRIALS} trials/eval)')
    ax.grid(True, alpha=0.3, which='both')

    plt.suptitle(f'{title}\nPareto front, {F.shape[0]} non-dominated solutions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _safe_savefig(fig, output_path)
    plt.close(fig)


# Computer Modern (LaTeX) styling for the paper figure. cmr10 is the actual
# LaTeX text font bundled with matplotlib; glyphs it lacks (e.g. underscore)
# fall back per-glyph to STIX/DejaVu serif. Scoped via rc_context so only the
# twin activity figure is affected.
_TWIN_TEX_RC = {
    'font.family': ['cmr10', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
}


def _with_tex_style(fn):
    def wrapped(*args, **kwargs):
        with plt.rc_context(_TWIN_TEX_RC):
            return fn(*args, **kwargs)
    return wrapped


@_with_tex_style
def plot_twin_activity(twin, pop_env, title, output_path, skeleton=None,
                       W_values=None):
    """Paper-style activity traces (arXiv:2605.25224 Figs. 3B-C / 4A-B).

    (A) organoid vs autonomous twin firing-rate timecourses in Hz. The bold
    twin trace is ONE representative realisation (the trial closest to the
    organoid rate distribution in Wasserstein-1) - a trial MEAN of an
    aperiodic process is flat by construction and misrepresents the twin.
    A marginal panel overlays the rate histograms (the statistic the
    autonomous twin is supposed to reproduce). (B) normalised PSDs with the
    dominant frequencies marked against target. (C) the DRIVEN twin: the
    reservoir teacher-forced by the organoid envelope, its ridge readout
    tracking the recording through time - generalized synchronisation,
    scored on the held-out tail (this is the F_sync objective, visualised).
    """
    params, targets, details = twin['params'], twin['targets'], twin['details']
    activity, _, _ = simulate_twin(params, len(pop_env), skeleton, W_values,
                                   n_trials=TWIN_N_TRIALS)
    rate_traces = twin_rate_traces(params, activity)  # (n_trials, T), in Hz
    rep = int(np.argmin([wasserstein_distance(pop_env, row)
                         for row in rate_traces]))
    t = np.arange(len(pop_env)) * ENV_BIN_S

    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(3, 2, width_ratios=[4.0, 1.0],
                          height_ratios=[1.15, 1.0, 1.0],
                          hspace=0.38, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])

    # --- (A) autonomous twin: one realisation against the organoid ---
    for row in rate_traces:  # individual trials (paper Eqn. 3 trials)
        ax_a.plot(t, row, color='lightcoral', lw=0.4, alpha=0.16, zorder=1)
    ax_a.plot(t, pop_env, color='tab:blue', lw=1.0, zorder=3,
              label='Organoid population rate')
    ax_a.plot(t, rate_traces[rep], color='crimson', lw=1.1, zorder=4,
              label=f'RRN twin rate (realisation {rep + 1}/{TWIN_N_TRIALS})')
    ax_a.set_xlabel('Time (s)')
    ax_a.set_ylabel('Population firing rate (Hz)')
    ax_a.set_title(f"(A) Autonomous twin - organoid "
                   f"{targets['target_rate']:.2f} Hz, twin "
                   f"{details['pred_rate']:.2f} Hz")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(True, alpha=0.3)

    # marginal rate histograms: the distribution the F_dist objective matches
    bins = np.histogram_bin_edges(
        np.concatenate([np.asarray(pop_env, float), rate_traces.ravel()]), 40)
    ax_h.hist(pop_env, bins=bins, orientation='horizontal', density=True,
              color='tab:blue', alpha=0.55, label='organoid')
    ax_h.hist(rate_traces.ravel(), bins=bins, orientation='horizontal',
              density=True, color='crimson', alpha=0.45,
              label=f'twin ({TWIN_N_TRIALS} trials)')
    ax_h.set_xscale('log')
    ax_h.set_title(f"rate distribution\nW1 = {details['wasserstein_hz']:.2f} Hz",
                   fontsize=9)
    ax_h.tick_params(labelleft=False, labelsize=7)
    ax_h.legend(fontsize=7, loc='upper right')
    ax_h.grid(True, alpha=0.3)

    # --- (B) spectral profile: organoid PSD over the twin's trial range ---
    x = detrend(rate_traces, axis=1)
    freqs, psd = welch(x, fs=1.0 / ENV_BIN_S, nperseg=targets['nperseg'], axis=1)
    band = (freqs >= TWIN_PEAK_BAND[0]) & (freqs <= TWIN_PEAK_BAND[1])
    band_psd = psd[:, band]
    band_norm = band_psd / (band_psd.sum(axis=1, keepdims=True) + 1e-30)
    ax_b.fill_between(freqs[band], band_norm.min(axis=0), band_norm.max(axis=0),
                      color='lightcoral', alpha=0.35, lw=0, zorder=1,
                      label=f'RRN twin, {TWIN_N_TRIALS} trials (min-max range)')
    ax_b.plot(targets['psd_freqs'], targets['psd_norm'], color='tab:blue',
              lw=1.6, zorder=3, label='Organoid PSD')
    if targets['target_freq'] is not None:
        ax_b.axvline(targets['target_freq'], color='black', ls='--', lw=1.2,
                     label=f"target {targets['target_freq']:.3f} Hz")
        ax_b.set_title('(B) Spectral profile (dominant oscillation vs target)')
    else:
        ax_b.set_title('(B) Spectral profile (no dominant oscillation)')
        spur = int(round(details.get('twin_peak_frac', 0.0) * TWIN_N_TRIALS))
        ax_b.text(0.98, 0.82, 'no dominant oscillation detected\n'
                  '(prominence vs permutation null, 5% level)\n'
                  f'twin trials tripping the same test: {spur}/{TWIN_N_TRIALS}',
                  transform=ax_b.transAxes,
                  ha='right', fontsize=10, color='dimgray', style='italic')
    if details['twin_freq'] is not None:
        ax_b.axvline(details['twin_freq'], color='red', ls='--', lw=1.2,
                     label=f"twin {details['twin_freq']:.3f} Hz")
    ax_b.set_xlabel('Frequency (Hz)')
    ax_b.set_ylabel('Normalised PSD')
    # the no-oscillation annotation owns the top-right corner - keep the
    # legend away from it ('best' placement can park it on the text)
    ax_b.legend(fontsize=9,
                loc=('upper left' if targets['target_freq'] is None else 'best'))
    ax_b.grid(True, alpha=0.3)

    # --- (C) driven twin: synchronisation to THIS recording ---
    sync = synchronised_twin(params, pop_env, skeleton, W_values,
                             n_trials=TWIN_N_TRIALS)
    preds, split = sync['preds'], sync['split']
    ax_c.axvspan(t[0], t[split], color='gray', alpha=0.12, zorder=0,
                 label='ridge readout fit (train)')
    for row in preds:
        ax_c.plot(t, row, color='lightcoral', lw=0.4, alpha=0.16, zorder=1)
    ax_c.plot(t, pop_env, color='tab:blue', lw=1.0, zorder=3,
              label='Organoid population rate')
    ax_c.plot(t, preds.mean(axis=0), color='crimson', lw=1.2, zorder=4,
              label='Driven twin readout (trial mean)')
    ax_c.axvline(t[split], color='black', ls='--', lw=1.0, zorder=2)
    ax_c.text(0.99, 0.93, f"held-out sync r = {details['sync_r']:.3f}",
              transform=ax_c.transAxes, ha='right', fontsize=11,
              color='darkred', fontweight='bold')
    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylabel('Population firing rate (Hz)')
    ax_c.set_title("(C) Driven twin: teacher-forced by the organoid envelope")
    ax_c.legend(fontsize=8, ncol=2, loc='upper left')
    ax_c.grid(True, alpha=0.3)

    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.07, right=0.97)
    # cmr10 (OT1 encoding) has no underscore glyph - the slot renders as a
    # dot accent - so sanitise it out of the displayed title only
    plt.suptitle(title.replace('_', ' '), fontsize=14, fontweight='bold')
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
        ax.set_title(f'Driven RRN twin latent - {label}')
        plt.suptitle('CEBRA latents (organoid channels vs reservoir driven by '
                     f'the organoid) | ridge R2 org->twin={r2_ot:.3f}, '
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
            plot_twin_activity(twin_result, pop_env,
                               f"{subject} - {label}: RRN twin vs organoid",
                               os.path.join(tw_dir, 'twin_activity_traces.png'),
                               skeleton=skel, W_values=Wv)
            # CEBRA compares the DRIVEN reservoir's amplitudes (the twin
            # synchronised to this recording) with the organoid's channel
            # envelopes: does the assimilated twin state carry the organoid's
            # latent geometry? An autonomous realisation cannot align
            # trajectory-wise with the recording by construction (its bursts
            # fall at unrelated times), so it is the wrong object to embed.
            twin_amps = synchronised_twin(twin_result['params'], pop_env,
                                          skel, Wv, n_trials=1)['amps'][0]
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
                              'sync_r': twin_result['details']['sync_r'],
                              'wasserstein_hz': twin_result['details']['wasserstein_hz'],
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
        freqs_ok = [t['target_freq'] for t in twins
                    if t.get('target_freq') is not None]
        freq_t.append(np.mean(freqs_ok) if freqs_ok else np.nan)
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
