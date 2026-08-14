"""Regenerate one state's twin_activity_traces.png WITHOUT re-running the GA.

Rebuilds the twin dict from the saved provenance (twin_params.json) and the
cached population envelope, then calls plot_twin_activity. Use after figure
cosmetic changes:

    .venv/bin/python replot_twin_activity.py [subject] [label]
    .venv/bin/python replot_twin_activity.py BO14 BO14_20uA
"""
import json
import os
import sys

import numpy as np

import shell_mea_twin_pipeline as pipe


def _load_pop_env(subject, label):
    """Cached envelope; falls back to the run_twinning_only loader path."""
    candidates = [
        os.path.join('data', 'env_cache', f'{subject}_{label}_env.npz'),
        os.path.join(str(pipe.DATA_ROOT), 'env_cache',
                     f'{subject}_{label}_env.npz'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return np.load(c, allow_pickle=False)['pop_env']
    import run_twinning_only as rto
    for rec in pipe.discover_recordings(pipe.DATA_ROOT):
        if rec['subject'] == subject and rto._label(rec) == label:
            cache_dir = os.path.join(str(pipe.DATA_ROOT), 'env_cache')
            os.makedirs(cache_dir, exist_ok=True)
            pop_env, _ = rto._load_env(rec, cache_dir)
            return pop_env
    raise SystemExit(f'No cached envelope or recording found for '
                     f'{subject} / {label}')


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else 'BO14'
    label = sys.argv[2] if len(sys.argv) > 2 else 'BO14_20uA'
    tw_dir = os.path.join(str(pipe.main_output_dir), subject, 'twinning', label)
    with open(os.path.join(tw_dir, 'twin_params.json')) as f:
        saved = json.load(f)

    pop_env = _load_pop_env(subject, label)
    twin = {'params': saved['rrn_params'],
            'details': saved['achieved'],
            'targets': pipe.compute_twin_targets(pop_env)}
    skel_json = saved['recurrent_skeleton']
    skeleton = ((np.array(skel_json['rows']), np.array(skel_json['cols']))
                if skel_json is not None else None)
    W_values = (np.array(saved['recurrent_weights'])
                if saved.get('recurrent_weights') is not None else None)

    out = os.path.join(tw_dir, 'twin_activity_traces.png')
    pipe.plot_twin_activity(
        twin, pop_env,
        f"{subject} - {label}: RRN twin vs organoid",
        out, skeleton=skeleton, W_values=W_values)
    print(f'Rewrote {out}')


if __name__ == '__main__':
    main()
