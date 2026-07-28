import argparse
import csv
from pathlib import Path


def read_overall_csv(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_method(rows, method_name: str):
    for row in rows:
        if row['Method'] == method_name:
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description='Build unified scale ablation table from existing result CSVs')
    parser.add_argument('--pred075', required=True)
    parser.add_argument('--pred090', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows075 = read_overall_csv(Path(args.pred075))
    rows090 = read_overall_csv(Path(args.pred090))

    base = find_method(rows075, 'base')
    pred075 = find_method(rows075, 'v1_s0p75')
    pred090 = find_method(rows090, 'v1_s0p9')

    if base is None or pred075 is None or pred090 is None:
        raise RuntimeError('Required methods not found in provided CSV files')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['Scale', 'Method', 'STOI', 'ESTOI', 'SI-SDR', 'LSD', 'HF-LSD']
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'Scale': '0 (Base)',
            'Method': 'Base',
            'STOI': base['STOI'],
            'ESTOI': base['ESTOI'],
            'SI-SDR': base['SI-SDR'],
            'LSD': base['LSD'],
            'HF-LSD': base['HF-LSD'],
        })
        writer.writerow({
            'Scale': '0.75',
            'Method': 'Predicted-0.75',
            'STOI': pred075['STOI'],
            'ESTOI': pred075['ESTOI'],
            'SI-SDR': pred075['SI-SDR'],
            'LSD': pred075['LSD'],
            'HF-LSD': pred075['HF-LSD'],
        })
        writer.writerow({
            'Scale': '0.90',
            'Method': 'Predicted-0.90',
            'STOI': pred090['STOI'],
            'ESTOI': pred090['ESTOI'],
            'SI-SDR': pred090['SI-SDR'],
            'LSD': pred090['LSD'],
            'HF-LSD': pred090['HF-LSD'],
        })

    print(f'[DONE] Wrote scale ablation table to {out_path}')


if __name__ == '__main__':
    main()
