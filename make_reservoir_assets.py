"""Paper assets for one twinned state, from saved provenance (no re-fitting):

  * reservoir_graph.png - the evolved RRN as a directed graph: 15 resonator
    nodes on a circle ordered by centre frequency (golden ladder), coloured
    by frequency (log scale), edges = the evolved recurrent weights after
    the spectral-radius rescale (blue positive, red negative, width ~ |w|).
  * twin_params_table.tex - booktabs LaTeX table of the fitted parameters.

Usage:
    .venv/bin/python make_reservoir_assets.py [subject] [label]
    .venv/bin/python make_reservoir_assets.py BO14 BO14_20uA
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

import shell_mea_twin_pipeline as pipe


def draw_reservoir(saved, out_png, title):
    p = saved['rrn_params']
    skel = saved['recurrent_skeleton']
    rows, cols = np.array(skel['rows']), np.array(skel['cols'])
    W_values = np.array(saved['recurrent_weights'])
    net = pipe.ReservoirNetwork(Fs=1.0 / pipe.ENV_BIN_S,
                                random_state=pipe.TWIN_SEED,
                                sparsity=pipe.TWIN_SKELETON_SPARSITY,
                                base_geometric_ratio=p['base_geometric_ratio'],
                                skeleton=(rows, cols))
    net.build_W_res(W_values, p['spectral_radius'])
    W = net.W_res.toarray()
    frange = net.frange
    K = len(frange)

    ang = np.pi / 2 - 2 * np.pi * np.arange(K) / K  # clockwise from the top
    xy = np.c_[np.cos(ang), np.sin(ang)]

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    wmax = np.abs(W).max() + 1e-12
    for k, j in zip(*np.nonzero(W)):  # W[k, j]: node j -> node k
        w = W[k, j]
        frac = abs(w) / wmax
        ax.add_patch(FancyArrowPatch(
            xy[j], xy[k], connectionstyle='arc3,rad=0.14',
            arrowstyle='-|>', mutation_scale=8, zorder=1,
            shrinkA=13, shrinkB=13,
            lw=0.4 + 2.2 * frac, alpha=0.20 + 0.55 * frac,
            color='#2166ac' if w > 0 else '#b2182b'))

    norm = LogNorm(vmin=frange.min(), vmax=frange.max())
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=frange, cmap='viridis', norm=norm,
                    s=950, edgecolors='black', linewidths=1.0, zorder=3)
    for k in range(K):
        ax.annotate(f'{frange[k]:.3f}'.rstrip('0').rstrip('.'),
                    1.17 * xy[k], ha='center', va='center', fontsize=11,
                    zorder=4)

    ticks = [float(frange.min()), 0.03, 0.06, 0.12, 0.25, float(frange.max())]
    cb = fig.colorbar(sc, ax=ax, fraction=0.043, pad=0.02, ticks=ticks)
    cb.set_label('resonator centre frequency (Hz)')
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels([f'{t:.3f}'.rstrip('0').rstrip('.') for t in ticks])

    ax.legend(handles=[
        Line2D([], [], color='#2166ac', lw=2, label='positive weight'),
        Line2D([], [], color='#b2182b', lw=2, label='negative weight')],
        loc='lower left', fontsize=10, frameon=False)
    ax.set_title(title)
    ax.set_xlim(-1.42, 1.42)
    ax.set_ylim(-1.36, 1.36)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    pipe._safe_savefig(fig, out_png)
    plt.close(fig)


TABLE_ROWS = [
    ('Reservoir', r'resonators $K$ (golden ladder, 0.015--0.435\,Hz)',
     None, '{:d}'),
    ('', r'damping $r$', 'base_geometric_ratio', '{:.3f}'),
    ('', r'spectral radius $\rho$', 'spectral_radius', '{:.3f}'),
    ('', r'state noise $\sigma$', 'sigma', '{:.4f}'),
    ('Drive', r'frequency $f_{d}$ (Hz)', 'drive_freq', '{:.3f}'),
    ('', r'amplitude $a_{d}$', 'drive_amp', '{:.4f}'),
    ('Coupling', r'sync gain $g$ (driven mode)', 'sync_gain', '{:.3f}'),
    ('Readout', r'quiescent baseline $b$ (Hz)', 'base_rate', '{:.3f}'),
    ('', r'burst rate $p_{0}$ (events/bin)', 'burst_rate', '{:.3f}'),
    ('', r'burst amplitude $A$ (Hz)', 'burst_amp', '{:.2f}'),
    ('', r'burst shape $s$ (lognormal)', 'burst_shape', '{:.3f}'),
    ('', r'timing gain $\kappa$', 'burst_kappa', '{:.3f}'),
    ('', r'size gain $\beta$', 'burst_beta', '{:.3f}'),
    ('', r'kernel decay $d$ (per bin)', 'burst_decay', '{:.3f}'),
    ('Recurrent', r'nonzero weights (density 0.35, evolved)', None, '{:d}'),
]


def _state_name(subject, label):
    name = label if label.startswith(subject) else f'{subject} {label}'
    return name.replace('_', ' ')


def write_table(saved, out_tex, subject, label):
    p = saved['rrn_params']
    n_w = len(saved['recurrent_weights'])
    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        (r'  \caption{Fitted RRN twin parameters for '
         f'{_state_name(subject, label)}' r'.}'),
        r'  \label{tab:rrn-params}',
        r'  \begin{tabular}{llr}',
        r'    \toprule',
        r'    Group & Parameter & Value \\',
        r'    \midrule',
    ]
    for group, desc, key, fmt in TABLE_ROWS:
        if key is None:
            val = fmt.format(pipe.TWIN_K if 'resonators' in desc else n_w)
        else:
            val = fmt.format(p[key])
        lines.append(f'    {group} & {desc} & {val} ' + r'\\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    with open(out_tex, 'w') as f:
        f.write('\n'.join(lines))
    return '\n'.join(lines)


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else 'BO14'
    label = sys.argv[2] if len(sys.argv) > 2 else 'BO14_20uA'
    tw_dir = os.path.join(str(pipe.main_output_dir), subject, 'twinning', label)
    with open(os.path.join(tw_dir, 'twin_params.json')) as f:
        saved = json.load(f)

    out_png = os.path.join(tw_dir, 'reservoir_graph.png')
    pipe._with_tex_style(draw_reservoir)(
        saved, out_png,
        f"{_state_name(subject, label)}: evolved RRN "
        "(nodes coloured by centre frequency)")
    print(f'Wrote {out_png}')

    out_tex = os.path.join(tw_dir, 'twin_params_table.tex')
    tex = write_table(saved, out_tex, subject, label)
    print(f'Wrote {out_tex}\n')
    print(tex)


if __name__ == '__main__':
    main()
