import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_csv(path: Path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def add_value_labels(ax, x, y, fmt='{:.4f}', color='#222222'):
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.03
    for xi, yi in zip(x, y):
        ax.text(xi, yi + pad, fmt.format(yi), ha='center', va='bottom', fontsize=8, color=color)


def style_axis(ax, ylabel=None, title=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def build_scale_figure(scale_csv: Path, out_dir: Path):
    rows = load_csv(scale_csv)
    keep = [r for r in rows if r['Scale'] in ('0.75', '0.90')]
    x_labels = [r['Scale'] for r in keep]
    x = np.arange(len(x_labels))

    metrics = [
        ('STOI', [to_float(r['STOI']) for r in keep], '#4f81bd'),
        ('SI-SDR', [to_float(r['SI-SDR']) for r in keep], '#c0504d'),
        ('HF-LSD', [to_float(r['HF-LSD']) for r in keep], '#70ad47'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    for ax, (name, vals, color) in zip(axes, metrics):
        ax.plot(x, vals, marker='o', markersize=7, linewidth=2.2, color=color)
        ax.scatter(x, vals, color=color, s=36, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'scale={label}' for label in x_labels])
        style_axis(ax, ylabel=name, title=name)
        ymin = min(vals)
        ymax = max(vals)
        pad = max((ymax - ymin) * 0.4, 1e-3)
        ax.set_ylim(ymin - pad, ymax + pad)
        add_value_labels(ax, x, vals)

    fig.suptitle('Residual Scale Calibration', y=1.02)
    fig.tight_layout()
    combo_path = out_dir / '图_残差强度校准_组合图.png'
    fig.savefig(combo_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_condition_figure(condition_csv: Path, out_dir: Path):
    rows = load_csv(condition_csv)
    order = ['zero', 'shuffled', 'predicted', 'oracle']
    row_map = {r['Condition']: r for r in rows}
    labels = [m for m in order if m in row_map]
    x = np.arange(len(labels))

    metrics = [
        ('SI-SDR', [to_float(row_map[m]['SI-SDR']) for m in labels], '#4f81bd'),
        ('HF-LSD', [to_float(row_map[m]['HF-LSD']) for m in labels], '#70ad47'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, (name, vals, color) in zip(axes, metrics):
        bars = ax.bar(x, vals, color=color, edgecolor='#333333', linewidth=0.8, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        style_axis(ax, ylabel=name, title=name)
        add_value_labels(ax, x, vals)
    fig.suptitle('Condition Effectiveness under Predicted-0.75', y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / '图_条件有效性_组合图.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_stability_figure(by_severity_csv: Path, out_dir: Path):
    rows = load_csv(by_severity_csv)
    severities = ['clean', 'light', 'medium', 'heavy']
    target_method = 'v1_s0p75'
    row_map = {(r['Severity'], r['Method']): r for r in rows}

    si_sdr_vals = [to_float(row_map[(s, target_method)]['SI-SDR']) for s in severities]
    hf_lsd_vals = [to_float(row_map[(s, target_method)]['HF-LSD']) for s in severities]
    x = np.arange(len(severities))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

    axes[0].plot(x, si_sdr_vals, marker='o', color='#c0504d', linewidth=2.2)
    axes[0].scatter(x, si_sdr_vals, color='#c0504d', s=36, zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(severities)
    style_axis(axes[0], ylabel='SI-SDR', title='SI-SDR')
    add_value_labels(axes[0], x, si_sdr_vals)

    axes[1].bar(x, hf_lsd_vals, color='#70ad47', edgecolor='#333333', linewidth=0.8, width=0.62)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(severities)
    style_axis(axes[1], ylabel='HF-LSD', title='HF-LSD')
    add_value_labels(axes[1], x, hf_lsd_vals)

    fig.suptitle('Predicted-0.75 by Degradation Severity', y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / '图_最终模型按退化强度稳定性_组合图.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def combine_spectrograms(case_dir: Path, out_path: Path):
    import matplotlib.image as mpimg
    files = [
        case_dir / 'spectrogram_case_01.png',
        case_dir / 'spectrogram_case_02.png',
        case_dir / 'spectrogram_case_03.png',
    ]
    imgs = [mpimg.imread(str(p)) for p in files if p.exists()]
    if len(imgs) != 3:
        raise RuntimeError('Expected 3 spectrogram case images')
    fig, axes = plt.subplots(3, 1, figsize=(12, 19))
    titles = ['(a) Light degradation', '(b) Medium degradation', '(c) Heavy degradation']
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate thesis internal figures from existing result CSVs (without Base as the main subject)')
    parser.add_argument('--scale-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\02_Scale对比_0_0.75_0.90\\Scale对比_0_0.75_0.90_总体表.csv')
    parser.add_argument('--condition-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\03_条件有效性_完整版\\条件有效性_Zero_Shuffled_Predicted_Oracle_总体表.csv')
    parser.add_argument('--by-severity-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\01_系统级主结果\\系统级主结果_按退化强度分组.csv')
    parser.add_argument('--spectrogram-dir', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\06_典型样本频谱图\\原始结果目录_spectrogram_cases_scale075')
    parser.add_argument('--output-dir', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\09_论文内部实验最终图')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_scale_figure(Path(args.scale_csv), out_dir)
    build_condition_figure(Path(args.condition_csv), out_dir)
    build_stability_figure(Path(args.by_severity_csv), out_dir)
    combine_spectrograms(Path(args.spectrogram_dir), out_dir / '图_典型频谱图_轻中重组合.png')

    print(f'[DONE] Thesis internal figures saved to {out_dir}')


if __name__ == '__main__':
    main()
