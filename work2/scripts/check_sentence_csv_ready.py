import argparse
import csv
from pathlib import Path


def read_sentence_csv(path: Path):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def main():
    parser = argparse.ArgumentParser(description='Lightweight sanity check for sentence-level CSV used in significance analysis')
    parser.add_argument('--sentence-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    csv_path = Path(args.sentence_csv)
    rows, fieldnames = read_sentence_csv(csv_path)

    required = [
        'sample_id', 'severity', 'method',
        'stoi', 'estoi', 'si_sdr', 'lsd', 'hf_lsd'
    ]
    missing = [k for k in required if k not in fieldnames]

    method_counts = {}
    severity_counts = {}
    for row in rows:
        method = row.get('method', '')
        severity = row.get('severity', '')
        method_counts[method] = method_counts.get(method, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['item', 'value'])
        writer.writerow(['csv_path', str(csv_path)])
        writer.writerow(['n_rows', len(rows)])
        writer.writerow(['n_fields', len(fieldnames)])
        writer.writerow(['missing_required_fields', ';'.join(missing) if missing else ''])
        for key, value in sorted(method_counts.items()):
            writer.writerow([f'method_count::{key}', value])
        for key, value in sorted(severity_counts.items()):
            writer.writerow([f'severity_count::{key}', value])

    print(f'[DONE] Wrote sentence CSV sanity summary to {out_path}')


if __name__ == '__main__':
    main()
