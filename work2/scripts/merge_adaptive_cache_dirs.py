import argparse
import json
import shutil
from pathlib import Path


def load_index(index_path: Path):
    entries = []
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(description='Merge train/valid/test adaptive shard caches into one unified directory')
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--valid-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    src_dirs = [Path(args.train_dir), Path(args.valid_dir), Path(args.test_dir)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # clean old merged files if they exist
    for p in out_dir.glob('adaptive_shard_*.pt'):
        p.unlink()
    index_out = out_dir / 'index.jsonl'
    if index_out.exists():
        index_out.unlink()

    merged_entries = []
    counter = 0

    for src in src_dirs:
        if not src.exists():
            print(f'[WARN] Skip missing source dir: {src}')
            continue
        index_path = src / 'index.jsonl'
        if not index_path.exists():
            print(f'[WARN] Skip missing index: {index_path}')
            continue

        print(f'[INFO] Processing {src}')
        entries = load_index(index_path)

        name_map = {}
        shard_files = sorted(src.glob('adaptive_shard_*.pt'))
        for i, shard_path in enumerate(shard_files, start=1):
            new_name = f'adaptive_shard_{counter:05d}.pt'
            shutil.copy2(shard_path, out_dir / new_name)
            name_map[shard_path.name] = new_name
            counter += 1
            if i % 20 == 0:
                print(f'[INFO]   copied {i}/{len(shard_files)} shards from {src.name}')

        for obj in entries:
            obj['shard_file'] = name_map[obj['shard_file']]
            merged_entries.append(obj)

        print(f'[INFO] Finished {src.name}: entries={len(entries)}, shards={len(shard_files)}')

    with open(index_out, 'w', encoding='utf-8') as f:
        for obj in merged_entries:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    for obj in merged_entries:
        split = obj.get('split', '')
        if split in split_counts:
            split_counts[split] += 1

    summary = {
        'output_dir': str(out_dir),
        'total_entries': len(merged_entries),
        'total_shards': counter,
        'split_counts': split_counts,
    }
    with open(out_dir / 'merge_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('[DONE] Merge complete')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
