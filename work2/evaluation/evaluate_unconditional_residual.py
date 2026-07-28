"""
Evaluate unconditional residual baseline.

Compares:
- Base Enhancer (no residual)
- Base + unconditional residual (same module, zero condition)
- Base + conditional residual (oracle condition)

If conditional > unconditional, it proves the degradation condition is useful.
If unconditional ≈ base, it proves the module alone does not create artificial gain.
"""
import argparse
import csv
import json
import numpy as np
import torch
from pathlib import Path
from pystoi import stoi

from work2.data.stft_utils import ri_to_istft
from work2.models.adaptive_residual import AdaptiveResidualModule


def load_index(index_path: Path, split: str):
    shard_to_entries = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            if e.get("split") != split: continue
            shard_to_entries.setdefault(e["shard_file"], []).append(e["offset"])
    return shard_to_entries

def si_sdr(ref, est, eps=1e-8):
    n = min(len(ref), len(est))
    ref = ref[:n].astype(np.float64) - np.mean(ref[:n])
    est = est[:n].astype(np.float64) - np.mean(est[:n])
    proj = np.dot(est, ref) / (np.sum(ref**2)+eps)
    s_tgt = proj * ref
    e_noise = est - s_tgt
    return float(10*np.log10((np.sum(s_tgt**2)+eps)/(np.sum(e_noise**2)+eps)))

def lsd(clean, test, hf=False):
    cm = np.sqrt(clean[...,0]**2+clean[...,1]**2+1e-12)
    tm = np.sqrt(test[...,0]**2+test[...,1]**2+1e-12)
    if hf:
        start = round(4000 / (16000 / 512))
        cm=cm[start:]
        tm=tm[start:]
    return float(np.sqrt(np.mean((20*np.log10(cm+1e-12)-20*np.log10(tm+1e-12))**2,axis=0)).mean())

def build_oracle_condition(noise_t, snr_t, bw_t, bit_t):
    B = noise_t.shape[0]
    bw_oh = torch.zeros(B,3,dtype=torch.float32).scatter_(1,bw_t.view(-1,1),1)
    bit_oh = torch.zeros(B,4,dtype=torch.float32).scatter_(1,bit_t.view(-1,1),1)
    snr_n = torch.clamp(snr_t/40.,0,1)
    return torch.cat([noise_t, snr_n, bw_oh, bit_oh], dim=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="outputs/adaptive_cache_sharded")
    parser.add_argument("--v1-checkpoint", default="outputs/adaptive_residual_cached_v1_oracle/best_model.pt")
    parser.add_argument("--output-dir", default="outputs/paper_results_unconditional")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--fast-metrics-only", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.75)
    parser.add_argument("--disable-pesq", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cd = Path(args.cache_dir)
    sm = load_index(cd/"index.jsonl", args.split)
    shard_files = list(sm.keys())
    if args.max_shards: shard_files = shard_files[:args.max_shards]

    v1 = AdaptiveResidualModule(n_freqs=257, cond_dim=9, hidden_dim=32).to(device)
    ckpt = torch.load(args.v1_checkpoint, map_location="cpu")
    print(f"[CONFIG] residual_scale={args.residual_scale}")
    print(f"[CHECKPOINT] V1={Path(args.v1_checkpoint).resolve()}")
    print(f"[EPOCH] V1={ckpt.get('epoch')}")
    v1.load_state_dict(ckpt["residual_state_dict"])
    v1.eval()

    window = torch.hann_window(512).pow(0.5)
    rows = []
    with torch.inference_mode():
        for si, sf in enumerate(shard_files, start=1):
            shard = torch.load(cd/sf, map_location="cpu", weights_only=False)
            offs = sorted(sm[sf])
            clean_b = shard["clean_spec"][offs].float()
            base_b = shard["enhanced_base"][offs].float()
            noise_b = shard["noise_target"][offs].float()
            snr_b = shard["snr_target"][offs].float()
            bw_b = shard["bandwidth_target"][offs].long()
            bit_b = shard["bit_target"][offs].long()
            sev_b = [shard["severity"][i] for i in offs]

            oracle = build_oracle_condition(noise_b, snr_b, bw_b, bit_b).to(device)
            zero = torch.zeros_like(oracle)
            bp = base_b.permute(0,3,2,1).contiguous().to(device)

            out_ora = v1(bp, oracle)
            out_unc = v1(bp, zero)
            v1_ora = (bp + args.residual_scale * out_ora["residual"]).permute(0,3,2,1).contiguous().cpu()
            v1_unc = (bp + args.residual_scale * out_unc["residual"]).permute(0,3,2,1).contiguous().cpu()

            for li, off in enumerate(offs):
                clean_s = clean_b[li].numpy()
                base_s = base_b[li].numpy()
                sev = sev_b[li]

                cw = ri_to_istft(torch.tensor(clean_s), window=window).numpy()
                bw_w = ri_to_istft(torch.tensor(base_s), window=window).numpy()

                def metrics(est_w, est_s):
                    m = {"si_sdr": si_sdr(cw, est_w), "lsd": lsd(clean_s, est_s, False), "hf_lsd": lsd(clean_s, est_s, True)}
                    if not args.fast_metrics_only:
                        m["stoi"] = float(stoi(cw, est_w, 16000, extended=False))
                        m["estoi"] = float(stoi(cw, est_w, 16000, extended=True))
                    return m

                base_m = metrics(bw_w, base_s)

                for name, spec in [("base", base_s), ("v1_uncond", v1_unc[li].numpy()), ("v1_cond", v1_ora[li].numpy())]:
                    wav = bw_w if name == "base" else ri_to_istft(torch.tensor(spec), window=window).numpy()
                    mm = metrics(wav, spec) if name != "base" else base_m
                    rows.append({"sample_id": f"{sf}:{off}", "severity": sev, "method": name,
                                 "si_sdr": mm["si_sdr"], "lsd": mm["lsd"], "hf_lsd": mm["hf_lsd"],
                                 "stoi": mm.get("stoi", float("nan")), "estoi": mm.get("estoi", float("nan"))})
            if si % 50 == 0 or si == 1:
                print(f"[INFO] Processed shard {si}/{len(shard_files)}")

    methods = ["base", "v1_uncond", "v1_cond"]
    metric_keys = ["stoi", "estoi", "si_sdr", "lsd", "hf_lsd"]
    with open(out_dir/"table_unconditional_residual.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method"] + [k.upper() for k in metric_keys])
        for m in methods:
            g = [r for r in rows if r["method"] == m]
            vals = {k: float(np.mean([r[k] for r in g])) for k in metric_keys}
            w.writerow([m] + [vals[k] for k in metric_keys])

    with open(out_dir/"sentence_level_unconditional.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"[DONE] Unconditional residual results saved to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
