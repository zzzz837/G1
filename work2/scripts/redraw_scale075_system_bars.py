import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEVERITIES = ["clean", "light", "medium", "heavy"]


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value: str):
    try:
        return float(value)
    except Exception:
        return float("nan")


def collect_values(rows, metric_key: str, methods):
    by_method = {m: [] for m in methods}
    for sev in SEVERITIES:
        group = [r for r in rows if r["Severity"] == sev]
        for method in methods:
            hit = next((r for r in group if r["Method"] == method), None)
            by_method[method].append(to_float(hit[metric_key]) if hit is not None else float("nan"))
    return by_method


def plot_severity_metric(rows, methods, metric_key, ylabel, output_path: Path):
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(SEVERITIES))

    n_methods = len(methods)
    total_width = 0.72
    width = total_width / max(n_methods, 1)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    values = collect_values(rows, metric_key, methods)

    for idx, method in enumerate(methods):
        ax.bar(x + offsets[idx], values[method], width * 0.88, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(SEVERITIES)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Severity")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Redraw scale0.75 system bar charts in original style")
    parser.add_argument("--csv", required=True, help="Path to by-severity CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    methods = []
    for row in rows:
        method = row["Method"]
        if method not in methods:
            methods.append(method)

    preferred_order = ["degraded", "base", "v1_s0p75"]
    ordered_methods = [m for m in preferred_order if m in methods]

    plot_severity_metric(
        rows,
        ordered_methods,
        "SI-SDR",
        "SI-SDR",
        out_dir / "系统级对比_Base-V1_scale0p75_SI-SDR曲线_重绘.png",
    )
    plot_severity_metric(
        rows,
        ordered_methods,
        "HF-LSD",
        "HF-LSD",
        out_dir / "系统级对比_Base-V1_scale0p75_HF-LSD曲线_重绘.png",
    )

    print(f"[DONE] Redrawn figures saved to {out_dir}")


if __name__ == "__main__":
    main()
