"""
Compile current experiment outputs into paper-friendly tables and figures.
"""
import json
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs")
OUT = ROOT / "paper_results"
OUT.mkdir(parents=True, exist_ok=True)

# Load estimator results
est_train = json.load(open(ROOT / "degradation_estimator_fast" / "train_summary.json", encoding="utf-8"))
est_metrics = json.load(open(ROOT / "degradation_estimator_fast" / "evaluation" / "metrics.json", encoding="utf-8"))

# Load adaptive results (current available)
adapt_train = json.load(open(ROOT / "adaptive_residual_shardwise" / "train_summary.json", encoding="utf-8"))

# Table 1: model summary
with open(OUT / "table_model_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Module", "Trainable Params", "Key Output/Role"])
    writer.writerow(["Frozen GTCRN", 0, "Base enhancement + bottleneck stats"])
    writer.writerow(["Degradation Estimator", 1803, "noise / SNR / bandwidth / bit-depth"])
    writer.writerow(["Adaptive Residual Module", adapt_train["residual_params"], "conditioned spectral residual refinement"])

# Table 2: estimator overall metrics
with open(OUT / "table_estimator_overall_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Noise Accuracy", est_metrics["noise"]["accuracy"]])
    writer.writerow(["Noise Precision", est_metrics["noise"]["precision"]])
    writer.writerow(["Noise Recall", est_metrics["noise"]["recall"]])
    writer.writerow(["Noise F1", est_metrics["noise"]["f1"]])
    writer.writerow(["Bandwidth Accuracy", est_metrics["bandwidth"]["accuracy"]])
    writer.writerow(["Bandwidth Macro-F1", est_metrics["bandwidth"]["macro_f1"]])
    writer.writerow(["Bit-depth Accuracy", est_metrics["bit"]["accuracy"]])
    writer.writerow(["Bit-depth Macro-F1", est_metrics["bit"]["macro_f1"]])
    writer.writerow(["SNR MAE (dB)", est_metrics["snr"]["mae_db"]])
    writer.writerow(["SNR RMSE (dB)", est_metrics["snr"]["rmse_db"]])
    writer.writerow(["SNR Pearson r", est_metrics["snr"]["pearson_r"]])

# Table 3: severity metrics
with open(OUT / "table_estimator_by_severity.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["severity", "n_samples", "noise_accuracy", "bandwidth_accuracy", "bit_accuracy"])
    writer.writeheader()
    for row in est_metrics["by_severity"]:
        writer.writerow(row)

# Table 4: training summary
with open(OUT / "table_training_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Epochs", "Total Time (s)", "Best Valid Loss", "Notes"])
    writer.writerow(["Degradation Estimator", est_train["final_epoch"], est_train["total_training_time_s"], est_train["best_valid_loss"], "Formal full-data training"])
    writer.writerow(["Adaptive Residual (debug run)", adapt_train["epochs"], adapt_train["total_training_time_s"], adapt_train["best_valid_loss"], "Pipeline verified; full training pending"])

# Figure 1: estimator overview bar chart
fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Noise Acc", "BW Acc", "Bit Acc", "1-SNR MAE/10"]
values = [
    est_metrics["noise"]["accuracy"],
    est_metrics["bandwidth"]["accuracy"],
    est_metrics["bit"]["accuracy"],
    1 - min(est_metrics["snr"]["mae_db"] / 10.0, 1.0),
]
ax.bar(labels, values)
ax.set_ylim(0, 1.05)
ax.set_title("Estimator Overall Performance")
ax.set_ylabel("Normalized score")
fig.tight_layout()
fig.savefig(OUT / "fig_estimator_overview.png", dpi=150)
plt.close(fig)

# Figure 2: severity grouped bar chart
sev = [x["severity"] for x in est_metrics["by_severity"]]
noise = [x["noise_accuracy"] for x in est_metrics["by_severity"]]
bw = [x["bandwidth_accuracy"] for x in est_metrics["by_severity"]]
bit = [x["bit_accuracy"] for x in est_metrics["by_severity"]]
fig, ax = plt.subplots(figsize=(8, 4))
import numpy as np
x = np.arange(len(sev))
width = 0.25
ax.bar(x - width, noise, width, label="Noise")
ax.bar(x, bw, width, label="Bandwidth")
ax.bar(x + width, bit, width, label="Bit-depth")
ax.set_xticks(x)
ax.set_xticklabels(sev)
ax.set_ylim(0, 1.05)
ax.set_title("Estimator Performance by Severity")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_estimator_by_severity.png", dpi=150)
plt.close(fig)

# Markdown summary
md = []
md.append("# 当前实验结果汇总\n")
md.append("## 1. 退化估计器正式结果\n")
md.append(f"- Noise accuracy: **{est_metrics['noise']['accuracy']:.4f}**\n")
md.append(f"- Bandwidth accuracy: **{est_metrics['bandwidth']['accuracy']:.4f}**\n")
md.append(f"- Bit-depth accuracy: **{est_metrics['bit']['accuracy']:.4f}**\n")
md.append(f"- SNR MAE: **{est_metrics['snr']['mae_db']:.4f} dB**\n")
md.append("\n## 2. 当前 adaptive residual 状态\n")
md.append(f"- Residual params: **{adapt_train['residual_params']}**\n")
md.append(f"- Current run: **debug validation only** ({adapt_train['epochs']} epoch)\n")
md.append(f"- Best valid loss: **{adapt_train['best_valid_loss']:.4f}**\n")
md.append("\n## 3. 已生成表格与图像\n")
md.append("- table_model_summary.csv\n")
md.append("- table_estimator_overall_metrics.csv\n")
md.append("- table_estimator_by_severity.csv\n")
md.append("- table_training_summary.csv\n")
md.append("- fig_estimator_overview.png\n")
md.append("- fig_estimator_by_severity.png\n")
md.append("- 以及 estimator evaluation 目录下的 confusion matrices / snr scatter / training curves\n")
(Path(OUT / "results_summary.md")).write_text("".join(md), encoding="utf-8")
print(f"[DONE] Compiled paper-friendly results to {OUT}")
