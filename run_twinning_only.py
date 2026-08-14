"""Twinning-only runner for the shell MEA pipeline.

The PP-GLM / CEBRA / longitudinal stages are stable and slow; when iterating
on the digital twin, run THIS script instead of the full pipeline. It reuses
shell_mea_twin_pipeline for everything (loader, spike detection, envelopes,
NSGA-III twinning, figures) and adds:

  * an envelope cache (data/env_cache/<subject>_<label>_env.npz): the first
    run per state pays the NWB read + spike detection once, every later run
    starts straight at the GA;
  * optional state filtering by substring, and --skip-done to leave states
    that already have a twin_activity_traces.png untouched.

Usage:
    .venv/bin/python run_twinning_only.py                    # all states
    .venv/bin/python run_twinning_only.py BO14_50uA SO1      # only matches
    .venv/bin/python run_twinning_only.py --skip-done        # resume a sweep

Figures land in the same figures/<subject>/twinning/<label>/ folders as the
full pipeline; a twin-only overview is written to figures/twinning_summary.json
(run_summary.json is left to the full pipeline).
"""
import gc
import json
import os
import sys

import numpy as np

import shell_mea_twin_pipeline as pipe


def _label(rec):
    return (rec['state'] if rec['session'] == 'unknown'
            else f"{rec['session']}_{rec['state']}")


def _load_env(rec, cache_dir):
    """Population/channel envelopes for one recording, cached on disk."""
    full = f"{rec['subject']}_{_label(rec)}"
    cache = os.path.join(cache_dir, full + '_env.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return z['pop_env'], str(z['session_start'])
    loaded = pipe.load_recording(rec['path'])
    if loaded is None:
        return None, None
    raw, fs = loaded['raw'], loaded['fs']
    session_start = str(loaded['session_start'])
    print(f"      Stream: {fs:.0f} Hz, shape {raw.shape}")
    spk, counts = pipe.detect_spikes(raw, fs)
    del raw, loaded
    gc.collect()
    print(f"      Spikes: {int(counts.sum())} total across "
          f"{(counts >= pipe.MIN_SPIKES_PER_CHANNEL).sum()} active channels")
    pop_env, ch_env = pipe.compute_envelopes(spk)
    del spk
    gc.collect()
    np.savez_compressed(cache, pop_env=pop_env, ch_env=ch_env,
                        session_start=np.str_(session_start))
    return pop_env, session_start


def main():
    argv = sys.argv[1:]
    skip_done = '--skip-done' in argv
    filters = [a for a in argv if not a.startswith('--')]

    records = pipe.discover_recordings(pipe.DATA_ROOT)
    if not records:
        print(f"No NWB files found in {pipe.DATA_ROOT}")
        return
    out_base = pipe.main_output_dir
    cache_dir = os.path.join(str(pipe.DATA_ROOT), 'env_cache')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Twinning-only run | {len(records)} recordings | "
          f"filters={filters or 'none'} | skip_done={skip_done}")
    summary = []
    for rec in records:
        label = _label(rec)
        full = f"{rec['subject']}_{label}"
        if filters and not any(f in full for f in filters):
            continue
        tw_dir = os.path.join(out_base, rec['subject'], 'twinning', label)
        done_marker = os.path.join(tw_dir, 'twin_activity_traces.png')
        if skip_done and os.path.exists(done_marker):
            print(f"\n   [{rec['subject']}] {label}: twin exists, skipping")
            continue
        print(f"\n   [{rec['subject']}] {label} ({rec['path'].name})")
        try:
            pop_env, session_start = _load_env(rec, cache_dir)
            if pop_env is None:
                print("      Could not load recording; skipping")
                continue
            tw = pipe.run_twinning(pop_env)
            entry = {'subject': rec['subject'], 'session': rec['session'],
                     'state': rec['state'], 'session_start': session_start,
                     'twin': None}
            if tw is not None:
                os.makedirs(tw_dir, exist_ok=True)
                skel, Wv = tw.get('skeleton'), tw.get('W_values')
                pipe.plot_twin_report(
                    tw, pop_env, f"{rec['subject']} - {label} - RRN Digital Twin",
                    os.path.join(tw_dir, 'twin_report.png'),
                    skeleton=skel, W_values=Wv)
                pipe.plot_ga_report(
                    tw, f"{rec['subject']} - {label} - NSGA-III optimisation",
                    os.path.join(tw_dir, 'ga_pareto_convergence.png'))
                pipe.plot_twin_activity(
                    tw, pop_env,
                    f"{rec['subject']} - {label} - Twin vs organoid activity",
                    os.path.join(tw_dir, 'twin_activity_traces.png'),
                    skeleton=skel, W_values=Wv)
                params_json = {
                    'state': label, 'overall_rmse': tw['overall'],
                    'targets': {'rate_hz': tw['targets']['target_rate'],
                                'dom_freq_hz': tw['targets']['target_freq']},
                    'achieved': tw['details'],
                    'rrn_params': {k: float(v) for k, v in tw['params'].items()},
                    'recurrent_skeleton': ({'rows': list(skel[0]),
                                            'cols': list(skel[1])}
                                           if skel is not None else None),
                    'recurrent_weights': (np.asarray(Wv).tolist()
                                          if Wv is not None else None),
                    'optimiser': tw.get('optimiser'),
                    'convergence': tw.get('convergence'),
                    'pareto_front_F': tw.get('pareto_F'),
                }
                with open(os.path.join(tw_dir, 'twin_params.json'), 'w') as f:
                    json.dump(params_json, f, indent=2)
                entry['twin'] = {
                    'overall': tw['overall'],
                    'target_rate': tw['targets']['target_rate'],
                    'target_freq': tw['targets']['target_freq'],
                    'pred_rate': tw['details']['pred_rate'],
                    'twin_freq': tw['details']['twin_freq'],
                    'sync_r': tw['details']['sync_r'],
                    'wasserstein_hz': tw['details']['wasserstein_hz'],
                    'params': {k: float(v) for k, v in tw['params'].items()}}
            summary.append(entry)
        except Exception as e:
            print(f"      Error on {full}: {e}")
            import traceback
            traceback.print_exc()
        gc.collect()

    # Merge into any existing summary so a FILTERED run updates only its
    # states instead of clobbering the full-sweep record.
    sum_path = os.path.join(out_base, 'twinning_summary.json')
    merged = []
    if os.path.exists(sum_path):
        try:
            with open(sum_path) as f:
                merged = json.load(f)
        except Exception:
            merged = []
    key = lambda s: (s['subject'], s['session'], s['state'])
    by_key = {key(s): s for s in merged}
    for s in summary:
        by_key[key(s)] = s
    merged = sorted(by_key.values(), key=key)
    with open(sum_path, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    n_ok = sum(1 for s in summary if s['twin'] is not None)
    print(f"\nTwinning-only run complete: {n_ok}/{len(summary)} states twinned "
          f"this run; summary now holds {len(merged)} states: {sum_path}")


if __name__ == '__main__':
    main()
