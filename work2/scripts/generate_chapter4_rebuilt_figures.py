import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


MODEL_ORDER = ['base', 'edcr_predicted_0p75', 'fullsubnet_plus', 'mpsenet_vb']
MODEL_LABELS = ['GTCRN', 'EDCR', 'FullSubNet+', 'MP-SENet-VB']
MODEL_COLORS = ['#5b9bd5', '#ed7d31', '#70ad47', '#a5a5a5']
SEVERITIES = ['clean', 'light', 'medium', 'heavy']
SEVERITIES_COMPARE = ['light', 'medium', 'heavy']


def load_csv(path: Path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def style_axis(ax, ylabel=None, title=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def load_image(path: Path):
    import matplotlib.image as mpimg
    return mpimg.imread(str(path))


def figure_4_3_internal_absolute_performance(by_severity_csv: Path, out_dir: Path):
    rows = load_csv(by_severity_csv)
    target = 'v1_s0p75'
    row_map = {(r['Severity'], r['Method']): r for r in rows}

    stoi_vals = [to_float(row_map[(s, target)]['STOI']) for s in SEVERITIES]
    estoi_vals = [to_float(row_map[(s, target)]['ESTOI']) for s in SEVERITIES]
    sisdr_vals = [to_float(row_map[(s, target)]['SI-SDR']) for s in SEVERITIES]
    lsd_vals = [to_float(row_map[(s, target)]['LSD']) for s in SEVERITIES]
    hflsd_vals = [to_float(row_map[(s, target)]['HF-LSD']) for s in SEVERITIES]

    x = np.arange(len(SEVERITIES))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))

    axes[0].bar(x - width/2, stoi_vals, width, color='#5b9bd5', edgecolor='#333333', linewidth=0.8, label='STOI')
    axes[0].bar(x + width/2, estoi_vals, width, color='#ed7d31', edgecolor='#333333', linewidth=0.8, label='ESTOI')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SEVERITIES)
    style_axis(axes[0], ylabel='Score', title='(a) STOI and ESTOI')
    axes[0].legend(frameon=True)

    axes[1].bar(x, sisdr_vals, width=0.55, color='#c0504d', edgecolor='#333333', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SEVERITIES)
    style_axis(axes[1], ylabel='dB', title='(b) SI-SDR')

    axes[2].bar(x - width/2, lsd_vals, width, color='#70ad47', edgecolor='#333333', linewidth=0.8, label='LSD')
    axes[2].bar(x + width/2, hflsd_vals, width, color='#a5a5a5', edgecolor='#333333', linewidth=0.8, label='HF-LSD')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(SEVERITIES)
    style_axis(axes[2], ylabel='Distance', title='(c) LSD and HF-LSD')
    axes[2].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / '图4-3_EDCR在不同退化强度下的绝对性能.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def grouped_compare_figure(compare_csv: Path, metrics, savepath: Path, title_prefix: str):
    rows = load_csv(compare_csv)
    row_map = {(r['Severity'], r['Method']): r for r in rows}
    width = 0.18
    x = np.arange(len(SEVERITIES_COMPARE))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric_key, title, higher_better) in zip(axes, metrics):
        best_positions = []
        best_values = []
        all_vals = []
        for s_idx, sev in enumerate(SEVERITIES_COMPARE):
            vals = [to_float(row_map[(sev, m)][metric_key]) for m in MODEL_ORDER]
            all_vals.extend(vals)
            best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
            best_positions.append(x[s_idx] + (best_idx - 1.5) * width)
            best_values.append(vals[best_idx])
            for m_idx, v in enumerate(vals):
                ax.bar(x[s_idx] + (m_idx - 1.5) * width, v, width=width*0.94, color=MODEL_COLORS[m_idx], edgecolor='#333333', linewidth=0.8)
        ymin = min(all_vals)
        ymax = max(all_vals)
        pad = max((ymax - ymin) * 0.18, 0.02)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITIES_COMPARE)
        style_axis(ax, ylabel=metric_key, title=title)
        for xi, yi in zip(best_positions, best_values):
            ax.text(xi, yi + pad * 0.08, '★', ha='center', va='bottom', fontsize=12, color='black')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec='#333333') for c in MODEL_COLORS]
    fig.legend(handles, MODEL_LABELS, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(savepath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def figure_4_4_perceptual_comparison(compare_csv: Path, out_dir: Path):
    metrics = [
        ('PESQ', 'PESQ', True),
        ('STOI', 'STOI', True),
        ('ESTOI', 'ESTOI', True),
    ]
    grouped_compare_figure(compare_csv, metrics, out_dir / '图4-4_四种模型的感知与可懂度指标.png', 'Perceptual')


def figure_4_5_time_freq_comparison(compare_csv: Path, out_dir: Path):
    metrics = [
        ('SI-SDR', 'SI-SDR', True),
        ('LSD', 'LSD', False),
        ('HF-LSD', 'HF-LSD', False),
    ]
    grouped_compare_figure(compare_csv, metrics, out_dir / '图4-5_四种模型的时域与频谱指标.png', 'Time-Frequency')


def figure_4_10_scale_tradeoff(scale_csv: Path, out_dir: Path):
    rows = load_csv(scale_csv)
    keep = [r for r in rows if r['Scale'] in ('0 (Base)', '0.75', '0.90')]
    labels = ['0（GTCRN，无残差）' if r['Scale'] == '0 (Base)' else r['Scale'] for r in keep]
    x = np.arange(len(labels))

    base = keep[0]
    stoi = [to_float(r['STOI']) - to_float(base['STOI']) for r in keep]
    estoi = [to_float(r['ESTOI']) - to_float(base['ESTOI']) for r in keep]
    sisdr = [to_float(r['SI-SDR']) - to_float(base['SI-SDR']) for r in keep]
    lsd_drop = [(to_float(base['LSD']) - to_float(r['LSD'])) / max(to_float(base['LSD']), 1e-8) * 100 for r in keep]
    hflsd_drop = [(to_float(base['HF-LSD']) - to_float(r['HF-LSD'])) / max(to_float(base['HF-LSD']), 1e-8) * 100 for r in keep]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    axes[0].plot(x, stoi, marker='o', linewidth=2.0, color='#5b9bd5', label='ΔSTOI')
    axes[0].plot(x, estoi, marker='s', linewidth=2.0, color='#ed7d31', label='ΔESTOI')
    axes[0].plot(x, sisdr, marker='^', linewidth=2.0, color='#70ad47', label='ΔSI-SDR')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    style_axis(axes[0], ylabel='Relative change', title='(a) STOI / ESTOI / SI-SDR')
    axes[0].legend(frameon=True)

    axes[1].plot(x, lsd_drop, marker='o', linewidth=2.0, color='#5b9bd5', label='LSD reduction rate')
    axes[1].plot(x, hflsd_drop, marker='s', linewidth=2.0, color='#ed7d31', label='HF-LSD reduction rate')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    style_axis(axes[1], ylabel='Reduction rate (%)', title='(b) LSD / HF-LSD reduction')
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / '图4-10_残差缩放系数多指标权衡.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def figure_4_12_edcr_vs_gtcrn_statistics(stats_csv: Path, out_dir: Path):
    rows = load_csv(stats_csv)
    metrics = [r['metric'] for r in rows]
    mean_delta = [to_float(r['mean_delta']) for r in rows]
    ci_lo = [to_float(r['mean_ci_lo']) for r in rows]
    ci_hi = [to_float(r['mean_ci_hi']) for r in rows]
    improved = [to_float(r['improved_ratio']) * 100 for r in rows]
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    bars1 = axes[0].bar(x, mean_delta, color='#5b9bd5', edgecolor='#333333', linewidth=0.8)
    axes[0].errorbar(x, mean_delta, yerr=[np.array(mean_delta) - np.array(ci_lo), np.array(ci_hi) - np.array(mean_delta)], fmt='none', ecolor='black', capsize=4)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD'])
    style_axis(axes[0], ylabel='Mean delta', title='(a) Mean delta and 95% CI')

    bars2 = axes[1].bar(x, improved, color='#70ad47', edgecolor='#333333', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD'])
    style_axis(axes[1], ylabel='Improved ratio (%)', title='(b) Improved sample ratio')

    fig.tight_layout()
    fig.savefig(out_dir / '图4-12_EDCR相对GTCRN的逐句统计.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_multimodel_spectrogram(case_name: str, clean_dir: Path, noisy_dir: Path, gtcrn_dir: Path, fullsubnet_dir: Path, mpsenet_dir: Path, edcr_dir: Path, out_path: Path):
    import soundfile as sf
    import torch
    from work2.data.stft_utils import stft_to_ri

    def read_wav(path):
        wav, fs = sf.read(str(path), dtype='float32')
        if fs != 16000:
            raise RuntimeError(f'Expected 16kHz, got {fs} for {path}')
        return wav

    def spec_mag_db(wav):
        window = torch.hann_window(512).pow(0.5)
        spec = stft_to_ri(torch.from_numpy(wav), n_fft=512, hop_length=256, win_length=512, window=window).numpy()
        mag = np.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-12)
        return 20 * np.log10(mag + 1e-12)

    clean = read_wav(clean_dir / f'{case_name}.wav')
    degraded = read_wav(noisy_dir / f'{case_name}.wav')
    gtcrn = read_wav(gtcrn_dir / f'{case_name}.wav')
    fullsubnet = read_wav(fullsubnet_dir / f'{case_name}.wav')
    mpsenet = read_wav(mpsenet_dir / f'{case_name}.wav')
    edcr = read_wav(edcr_dir / f'{case_name}.wav')

    clean_db = spec_mag_db(clean)
    degraded_db = spec_mag_db(degraded)
    gtcrn_db = spec_mag_db(gtcrn)
    fullsubnet_db = spec_mag_db(fullsubnet)
    mpsenet_db = spec_mag_db(mpsenet)
    edcr_db = spec_mag_db(edcr)

    display_specs = [clean_db, degraded_db, gtcrn_db, fullsubnet_db, mpsenet_db, edcr_db]
    error_specs = [np.abs(gtcrn_db - clean_db), np.abs(fullsubnet_db - clean_db), np.abs(mpsenet_db - clean_db), np.abs(edcr_db - clean_db)]
    vmax = max(float(np.max(s)) for s in display_specs)
    vmin = min(float(np.min(s)) for s in display_specs)
    err_vmax = np.percentile(np.concatenate([e.ravel() for e in error_specs]), 99)

    titles = ['Clean', 'Degraded', 'GTCRN', 'FullSubNet+', 'MP-SENet-VB', 'EDCR',
              'GTCRN Error', 'FullSubNet+ Error', 'MP-SENet-VB Error', 'EDCR Error']
    specs = display_specs + error_specs

    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16.5))
    for ax, title, spec in zip(axes.flat, titles, specs):
        if 'Error' in title:
            ax.imshow(spec, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=err_vmax)
        else:
            ax.imshow(spec, origin='lower', aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Frequency Bin')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def figure_4_7_8_9_multimodel_spectrograms(manifest_csv: Path, compare_sentence_csv: Path, gtcrn_dir: Path, fullsubnet_dir: Path, mpsenet_dir: Path, edcr_dir: Path, out_dir: Path):
    manifest_rows = load_csv(manifest_csv)
    compare_rows = load_csv(compare_sentence_csv)
    clean_map = {r['sample_id']: Path(r['clean_wav']) for r in manifest_rows}
    noisy_map = {r['sample_id']: Path(r['noisy_wav']) for r in manifest_rows}

    for sev, out_name in [('light', '图4-7_light多模型典型频谱对比.png'), ('medium', '图4-8_medium多模型典型频谱对比.png'), ('heavy', '图4-9_heavy多模型典型频谱对比.png')]:
        group = [r for r in compare_rows if r['method'] == 'edcr_predicted_0p75' and r['severity'] == sev]
        if not group:
            continue
        deltas = np.array([to_float(r['hf_lsd']) for r in group], dtype=np.float64)
        median = float(np.median(deltas))
        picked = min(group, key=lambda r: abs(to_float(r['hf_lsd']) - median))
        sample_id = picked['sample_id']
        build_multimodel_spectrogram(
            sample_id,
            clean_map[sample_id].parent,
            noisy_map[sample_id].parent,
            gtcrn_dir,
            fullsubnet_dir,
            mpsenet_dir,
            edcr_dir,
            out_dir / out_name,
        )


def main():
    parser = argparse.ArgumentParser(description='Generate rebuilt Chapter 4 figures from existing results')
    parser.add_argument('--internal-scale-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\02_Scale对比_0_0.75_0.90\\Scale对比_0_0.75_0.90_总体表.csv')
    parser.add_argument('--condition-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\03_条件有效性_完整版\\条件有效性_Zero_Shuffled_Predicted_Oracle_总体表.csv')
    parser.add_argument('--edcr-by-severity-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\01_系统级主结果\\系统级主结果_按退化强度分组.csv')
    parser.add_argument('--stats-csv', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\05_统计显著性_改善比例_分布\\统计显著性_总体表.csv')
    parser.add_argument('--compare-csv', default='D:\\workshop\\G1\\outputs\\edcr_fullsubnet_mpsenet_smallset\\table_by_severity.csv')
    parser.add_argument('--compare-sentence-csv', default='D:\\workshop\\G1\\outputs\\edcr_fullsubnet_mpsenet_smallset\\sentence_level_metrics.csv')
    parser.add_argument('--smallset-manifest-csv', default='D:\\workshop\\FullSubNet-plus-master\\data\\test_noisy_small\\manifest_small_test.csv')
    parser.add_argument('--gtcrn-enhanced-dir', default='D:\\workshop\\G1\\outputs\\smallset_multimodel_wavs\\gtcrn')
    parser.add_argument('--fullsubnet-enhanced-dir', default='D:\\workshop\\FullSubNet-plus-master\\outputs\\inference_test_small\\enhanced_0194')
    parser.add_argument('--mpsenet-enhanced-dir', default='D:\\workshop\\MP-SENet-main\\MP-SENet-main\\outputs\\inference_test_small_vb')
    parser.add_argument('--edcr-enhanced-dir', default='D:\\workshop\\G1\\outputs\\smallset_multimodel_wavs\\edcr')
    parser.add_argument('--output-dir', default='D:\\workshop\\G1\\outputs\\00_正式主结果_Predicted-0.75_论文当前使用\\09_论文重构图_对比与消融分离')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_4_3_internal_absolute_performance(Path(args.edcr_by_severity_csv), out_dir)
    figure_4_4_perceptual_comparison(Path(args.compare_csv), out_dir)
    figure_4_5_time_freq_comparison(Path(args.compare_csv), out_dir)
    figure_4_10_scale_tradeoff(Path(args.internal_scale_csv), out_dir)
    figure_4_12_edcr_vs_gtcrn_statistics(Path(args.stats_csv), out_dir)
    figure_4_7_8_9_multimodel_spectrograms(
        Path(args.smallset_manifest_csv),
        Path(args.compare_sentence_csv),
        Path(args.gtcrn_enhanced_dir),
        Path(args.fullsubnet_enhanced_dir),
        Path(args.mpsenet_enhanced_dir),
        Path(args.edcr_enhanced_dir),
        out_dir,
    )

    print(f'[DONE] Chapter 4 rebuilt figures saved to {out_dir}')


if __name__ == '__main__':
    main()
