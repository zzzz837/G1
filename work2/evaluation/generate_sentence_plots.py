"""
Generate sentence-level boxplots and improvement-ratio plots from evaluation CSVs.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def boxplot_metric(rows, method_key, metric_key, savepath, title, labels_order=None):
    data = defaultdict(list)
    for r in rows:
        method = r[method_key]
        val = to_float(r[metric_key])
        if not np.isnan(val):
            data[method].append(val)
    labels = labels_order or list(data.keys())
    values = [data[l] for l in labels]
    fig, ax = plt.subplots(figsize=(7,4))
    bp = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#d9e8fb')
        patch.set_edgecolor('#4f81bd')
    for med in bp['medians']:
        med.set_color('#c0504d')
    ax.set_title(title)
    ax.set_ylabel(metric_key.upper())
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def improvement_ratio_plot(rows, methods, metric_key, lower_better, savepath, title):
    totals = []
    labels = []
    # rows contain per-sample method comparisons relative to base if delta columns exist
    base_metric = 'base_' + metric_key if ('base_' + metric_key) in rows[0] else None
    for method in methods:
        improved = 0
        count = 0
        for r in rows:
            if r['method'] != method:
                continue
            val = to_float(r[metric_key])
            if np.isnan(val):
                continue
            if base_metric is not None:
                base = to_float(r[base_metric])
                if np.isnan(base):
                    continue
                better = val < base if lower_better else val > base
                improved += 1 if better else 0
                count += 1
        if count > 0:
            totals.append(improved / count)
            labels.append(method)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, totals, color=['#7aa6c2', '#9ccc65', '#ffb74d'][:len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Improved sample ratio')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)
    for i, v in enumerate(totals):
        ax.text(i, min(v + 0.02, 0.98), f'{v:.2%}', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhancement-csv', type=str, default='outputs/paper_results_debug/sentence_level_metrics.csv')
    parser.add_argument('--condition-v1-csv', type=str, default='outputs/paper_results_condition_v1/sentence_level_condition_metrics.csv')
    parser.add_argument('--condition-v2-csv', type=str, default='outputs/paper_results_condition_v2/sentence_level_condition_metrics.csv')
    parser.add_argument('--output-dir', type=str, default='outputs/paper_results_extra')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enh_rows = load_csv(Path(args.enhancement_csv)) if Path(args.enhancement_csv).exists() else []
    if enh_rows:
        boxplot_metric(enh_rows, 'method', 'si_sdr', out_dir / 'enhancement_si_sdr_boxplot.png', 'System-level SI-SDR distribution', ['degraded','base','v1','v2'])
        boxplot_metric(enh_rows, 'method', 'lsd', out_dir / 'enhancement_lsd_boxplot.png', 'System-level LSD distribution', ['degraded','base','v1','v2'])
        boxplot_metric(enh_rows, 'method', 'hf_lsd', out_dir / 'enhancement_hf_lsd_boxplot.png', 'System-level HF-LSD distribution', ['degraded','base','v1','v2'])
        if 'stoi' in enh_rows[0]:
            boxplot_metric(enh_rows, 'method', 'stoi', out_dir / 'enhancement_stoi_boxplot.png', 'System-level STOI distribution', ['degraded','base','v1','v2'])
            boxplot_metric(enh_rows, 'method', 'estoi', out_dir / 'enhancement_estoi_boxplot.png', 'System-level ESTOI distribution', ['degraded','base','v1','v2'])
        improvement_ratio_plot(enh_rows, ['v1','v2'], 'lsd', True, out_dir / 'improvement_ratio_lsd.png', 'Improved sample ratio vs Base (LSD)')
        improvement_ratio_plot(enh_rows, ['v1','v2'], 'hf_lsd', True, out_dir / 'improvement_ratio_hf_lsd.png', 'Improved sample ratio vs Base (HF-LSD)')
        improvement_ratio_plot(enh_rows, ['v1','v2'], 'si_sdr', False, out_dir / 'improvement_ratio_si_sdr.png', 'Improved sample ratio vs Base (SI-SDR)')

    for csv_path, prefix in [(args.condition_v1_csv, 'condition_v1'), (args.condition_v2_csv, 'condition_v2')]:
        p = Path(csv_path)
        if not p.exists():
            continue
        rows = load_csv(p)
        order = ['zero','shuffled','predicted','oracle']
        boxplot_metric(rows, 'method', 'si_sdr', out_dir / f'{prefix}_si_sdr_boxplot.png', f'{prefix} SI-SDR distribution', order)
        boxplot_metric(rows, 'method', 'lsd', out_dir / f'{prefix}_lsd_boxplot.png', f'{prefix} LSD distribution', order)
        boxplot_metric(rows, 'method', 'hf_lsd', out_dir / f'{prefix}_hf_lsd_boxplot.png', f'{prefix} HF-LSD distribution', order)
        if 'stoi' in rows[0]:
            boxplot_metric(rows, 'method', 'stoi', out_dir / f'{prefix}_stoi_boxplot.png', f'{prefix} STOI distribution', order)
            boxplot_metric(rows, 'method', 'estoi', out_dir / f'{prefix}_estoi_boxplot.png', f'{prefix} ESTOI distribution', order)

    print(f'[DONE] Extra plots saved to {out_dir}')


if __name__ == '__main__':
    main()
