from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


LOAD_CODE = r"""
import sys
import torch

path = sys.argv[1]

obj = torch.load(
    path,
    map_location='cpu',
    weights_only=False,
)


def check_value(value, name='root'):
    if torch.is_tensor(value):
        if value.numel() > 0:
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f'Non-finite tensor: {name}, '
                    f'shape={tuple(value.shape)}'
                )
        return

    if isinstance(value, dict):
        for key, item in value.items():
            check_value(item, f'{name}.{key}')
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            check_value(item, f'{name}[{index}]')

check_value(obj)
print('OK', flush=True)
"""


def collect_shard_paths(cache_dir: Path) -> list[Path]:
    index_path = cache_dir / 'index.jsonl'
    paths: list[Path] = []

    if index_path.exists():
        with index_path.open('r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f'Invalid JSON at {index_path}:{line_number}'
                    ) from exc

                candidate = None
                for key in ('shard_file', 'shard_path', 'shard', 'file', 'path', 'cache_path'):
                    if key in record:
                        candidate = record[key]
                        break

                if candidate is not None:
                    candidate_path = Path(str(candidate))
                    if not candidate_path.is_absolute():
                        candidate_path = cache_dir / candidate_path
                    paths.append(candidate_path)

    if not paths:
        for pattern in ('*.pt', '*.pth', '*.tar'):
            paths.extend(cache_dir.rglob(pattern))

    return sorted({path.resolve() for path in paths})


def main() -> None:
    parser = argparse.ArgumentParser(description='Check adaptive shard cache files in isolated subprocesses')
    parser.add_argument('cache_dirs', nargs='+', type=Path)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--report-path', type=Path, default=Path('bad_shards.txt'))
    args = parser.parse_args()

    all_paths: list[Path] = []
    for cache_dir in args.cache_dirs:
        if not cache_dir.exists():
            raise FileNotFoundError(cache_dir)
        paths = collect_shard_paths(cache_dir)
        print(f'[INFO] {cache_dir}: found {len(paths)} candidate shards', flush=True)
        all_paths.extend(paths)

    all_paths = sorted(set(all_paths))
    bad_files: list[tuple[Path, int, str]] = []

    for index, path in enumerate(all_paths, start=1):
        if not path.exists():
            bad_files.append((path, -1, 'file does not exist'))
            print(f'[BAD] {index}/{len(all_paths)} missing: {path}', flush=True)
            continue

        command = [
            sys.executable,
            '-X',
            'faulthandler',
            '-c',
            LOAD_CODE,
            str(path),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            bad_files.append((path, -2, 'timeout'))
            print(f'[BAD] {index}/{len(all_paths)} timeout: {path}', flush=True)
            continue

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or 'no error output'
            bad_files.append((path, result.returncode, detail))
            print(f'\n[BAD] {index}/{len(all_paths)}', flush=True)
            print(f'file={path}', flush=True)
            print(f'size={path.stat().st_size} bytes', flush=True)
            print(f'returncode={result.returncode}', flush=True)
            print(detail, flush=True)
        else:
            print(f'[OK] {index}/{len(all_paths)} {path.name}', flush=True)

    report_path = args.report_path
    with report_path.open('w', encoding='utf-8') as file:
        for path, returncode, detail in bad_files:
            file.write(
                f'FILE: {path}\n'
                f'RETURN_CODE: {returncode}\n'
                f'DETAIL:\n{detail}\n'
                f"{'=' * 80}\n"
            )

    print()
    print(f'[SUMMARY] checked={len(all_paths)}, bad={len(bad_files)}', flush=True)
    print(f'[SUMMARY] report={report_path.resolve()}', flush=True)

    if bad_files:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
