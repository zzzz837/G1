import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, norm


HIGHER_BETTER = {
    'stoi': True,
    'estoi': True,
    'si_sdr': True,
    'lsd': False,
    'hf_lsd': False,
}

DISPLAY_NAMES = {
    'stoi': 'STOI',
    'estoi': 'ESTOI',
    'si_sdr': 'SI-SDR',
    'lsd': 'LSD',
    'hf_lsd': 'HF-LSD',
}


def load_sentence_rows(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def percentile_ci(values, lo=2.5, hi=97.5):
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def bootstrap_mean_ci(values, rng, n_boot=2000):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def effect_size_r_from_wilcoxon(p_value, n):
    if n <= 0:
        return float('nan')
    p_safe = max(float(p_value), 1e-300)
    z = abs(norm.isf(p_safe / 2.0))
    return float(z / np.sqrt(n))


def plot_delta_hist(values, metric_key, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=60, color='#6aaed6', edgecolor='white', alpha=0.9)
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1.2)
    ax.set_title(f"Delta Distribution: {DISPLAY_NAMES[metric_key]}")
    ax.set_xlabel(f"Delta {DISPLAY_NAMES[metric_key]} (Predicted-0.75 - Base)")
    ax.set_ylabel('Count')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_delta_box(values, metric_key, output_path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.boxplot(values, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='#d9e8fb', edgecolor='#4f81bd'),
               medianprops=dict(color='#c0504d'))
    ax.axhline(0.0, color='red', linestyle='--', linewidth=1.0)
    ax.set_xticklabels([DISPLAY_NAMES[metric_key]])
    ax.set_ylabel(f"Delta {DISPLAY_NAMES[metric_key]}")
    ax.set_title(f"Delta Boxplot: {DISPLAY_NAMES[metric_key]}")
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Statistical analysis for Predicted-0.75 vs Base')
    parser.add_argument('--sentence-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--predicted-method', default='v1_s0p75')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    rows = load_sentence_rows(Path(args.sentence_csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for row in rows:
        grouped.setdefault(row['sample_id'], {})[row['method']] = row

    metric_keys = ['stoi', 'estoi', 'si_sdr', 'lsd', 'hf_lsd']
    deltas = {k: [] for k in metric_keys}

    for sample_id, methods in grouped.items():
        if 'base' not in methods or args.predicted_method not in methods:
            continue
        base = methods['base']
        pred = methods[args.predicted_method]
        for k in metric_keys:
            base_v = to_float(base[k])
            pred_v = to_float(pred[k])
            if np.isnan(base_v) or np.isnan(pred_v):
                continue
            deltas[k].append(pred_v - base_v)

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    json_summary = {}

    for metric_key in metric_keys:
        values = np.asarray(deltas[metric_key], dtype=np.float64)
        if values.size == 0:
            continue

        stat, p_value = wilcoxon(values, alternative='two-sided', zero_method='wilcox')
        mean_delta = float(np.mean(values))
        median_delta = float(np.median(values))
        mean_ci_lo, mean_ci_hi = bootstrap_mean_ci(values, rng)
        dist_ci_lo, dist_ci_hi = percentile_ci(values)

        if HIGHER_BETTER[metric_key]:
            improved_ratio = float(np.mean(values > 0))
            severe_negative_ratio = float(np.mean(values < -0.5)) if metric_key == 'si_sdr' else float(np.mean(values < 0))
        else:
            improved_ratio = float(np.mean(values < 0))
            severe_negative_ratio = float(np.mean(values > 0))

        effect_r = effect_size_r_from_wilcoxon(p_value, len(values))

        summary_rows.append({
            'metric': DISPLAY_NAMES[metric_key],
            'n_pairs': int(len(values)),
            'mean_delta': mean_delta,
            'median_delta': median_delta,
            'improved_ratio': improved_ratio,
            'wilcoxon_stat': float(stat),
            'p_value': float(p_value),
            'effect_size_r': effect_r,
            'mean_ci_lo': mean_ci_lo,
            'mean_ci_hi': mean_ci_hi,
            'delta_p2_5': dist_ci_lo,
            'delta_p97_5': dist_ci_hi,
            'severe_negative_ratio': severe_negative_ratio,
        })

        json_summary[metric_key] = summary_rows[-1]

        plot_delta_hist(values, metric_key, out_dir / f'delta_hist_{metric_key}.png')
        plot_delta_box(values, metric_key, out_dir / f'delta_box_{metric_key}.png')

    csv_path = out_dir / 'predicted_s075_significance.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = out_dir / 'predicted_s075_significance.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)

    print(f'[DONE] Wrote significance summary to {csv_path}')
    print(f'[DONE] Wrote significance JSON to {json_path}')


if __name__ == '__main__':
    main()
