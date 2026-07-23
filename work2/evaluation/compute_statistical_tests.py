"""
Statistical significance analysis from sentence-level evaluation CSVs.

Computes:
- Mean / Std for each method-metric pair
- Paired Wilcoxon signed-rank test (V1 vs Base, etc.)
- Bootstrap 95% CI for improvement ratios
- Overall improvement counts
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(x):
    try: return float(x)
    except: return float("nan")


def bootstrap_ci(samples, n_boot=2000, alpha=0.05):
    """Bootstrap 95% CI for mean."""
    samples = np.array(samples)
    means = []
    n = len(samples)
    rng = np.random.RandomState(2026)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        means.append(float(np.mean(samples[idx])))
    means = np.sort(means)
    lo = int(alpha / 2 * n_boot)
    hi = int((1 - alpha / 2) * n_boot)
    return float(means[lo]), float(means[hi])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhancement-csv", default="outputs/paper_results_full_stoi/sentence_level_metrics.csv")
    parser.add_argument("--unconditional-csv", default="outputs/paper_results_unconditional/sentence_level_unconditional.csv")
    parser.add_argument("--output-dir", default="outputs/paper_results_stats")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Enhancement system-level ---
    enh_path = Path(args.enhancement_csv)
    if enh_path.exists():
        rows = load_csv(enh_path)
        metrics = ["stoi", "estoi", "si_sdr", "lsd", "hf_lsd"]
        methods = ["base", "v1"]
        # per-method stats
        for method in methods:
            for metric in metrics:
                vals = [to_float(r[metric]) for r in rows if r["method"] == method and not np.isnan(to_float(r[metric]))]
                if len(vals) < 5: continue
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals))
                ci_lo, ci_hi = bootstrap_ci(vals)
                results[f"{method}_{metric}"] = {"mean": round(mean_v,4), "std": round(std_v,4), "ci_lo": round(ci_lo,4), "ci_hi": round(ci_hi,4), "n": len(vals)}
        # paired test V1 vs Base
        paired_tests = []
        for metric in metrics:
            base_vals, v1_vals = {}, {}
            for r in rows:
                v = to_float(r[metric])
                if np.isnan(v): continue
                if r["method"] == "base": base_vals[r["sample_id"]] = v
                elif r["method"] == "v1": v1_vals[r["sample_id"]] = v
            common = sorted(set(base_vals) & set(v1_vals))
            if len(common) < 10: continue
            d = np.array([v1_vals[k] - base_vals[k] for k in common])
            try:
                w, p = wilcoxon(d, zero_method="zsplit", alternative="two-sided")
            except Exception:
                w, p = float("nan"), float("nan")
            impro = int(np.sum(d > 0)) / len(d) if metric in ["stoi","estoi","si_sdr"] else int(np.sum(d < 0)) / len(d)
            paired_tests.append({"metric": metric, "n_pairs": len(common), "mean_diff": round(float(np.mean(d)),4),
                                 "wilcoxon_stat": round(float(w),2) if not np.isnan(w) else None, "p_value": round(float(p),6) if not np.isnan(p) else None,
                                 "improved_ratio": round(float(impro),4)})
        with open(out_dir / "stats_enhancement.json", "w") as f:
            json.dump({"per_method": results, "paired_tests": paired_tests}, f, indent=2)
        print(f"[INFO] Enhancement stats saved: {out_dir/'stats_enhancement.json'}")

    # --- Unconditional residual ---
    unc_path = Path(args.unconditional_csv)
    if unc_path.exists():
        rows = load_csv(unc_path)
        metrics = ["si_sdr", "lsd", "hf_lsd"]
        methods = ["base", "v1_uncond", "v1_cond"]
        results_unc = {}
        for method in methods:
            for metric in metrics:
                vals = [to_float(r[metric]) for r in rows if r["method"] == method and not np.isnan(to_float(r[metric]))]
                if len(vals) < 5: continue
                results_unc[f"{method}_{metric}"] = {"mean": round(float(np.mean(vals)),4), "std": round(float(np.std(vals)),4), "n": len(vals)}
        # paired: v1_cond vs v1_uncond
        paired = []
        for metric in metrics:
            cond_vals, unc_vals = {}, {}
            for r in rows:
                v = to_float(r[metric]); id_ = r["sample_id"]
                if np.isnan(v): continue
                if r["method"] == "v1_cond": cond_vals[id_] = v
                elif r["method"] == "v1_uncond": unc_vals[id_] = v
            common = sorted(set(cond_vals) & set(unc_vals))
            if len(common) < 10: continue
            d = np.array([cond_vals[k] - unc_vals[k] for k in common])
            try:
                w, p = wilcoxon(d, zero_method="zsplit", alternative="two-sided")
            except Exception:
                w, p = float("nan"), float("nan")
            paired.append({"metric": metric, "n_pairs": len(common), "mean_diff": round(float(np.mean(d)),4),
                           "p_value": round(float(p),6) if not np.isnan(p) else None,
                           "interpretation": "cond ≈ uncond (p>=0.05)" if (p is not None and not np.isnan(p) and p >= 0.05) else "cond ≠ uncond (p<0.05)"})
        with open(out_dir / "stats_unconditional.json", "w") as f:
            json.dump({"per_method": results_unc, "paired_cond_vs_uncond": paired}, f, indent=2)
        print(f"[INFO] Unconditional stats saved: {out_dir/'stats_unconditional.json'}")

    print("[DONE] Statistical analysis complete")


if __name__ == "__main__":
    main()
