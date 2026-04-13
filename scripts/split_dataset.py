#!/usr/bin/env python3
"""
Utility to split the Spotify dataset into reproducible train/dev/test CSV files.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the Spotify dataset into train/dev/test subsets with a fixed seed "
            "for reproducibility."
        )
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("data/datasets/Spotify_Dataset_V3.csv"),
        help="Path to the full dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/datasets/splits"),
        help="Directory where the split CSVs will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of rows to allocate to the training split.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Proportion of rows to allocate to the development split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of rows to allocate to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=51,
        help="Random seed used when shuffling rows prior to splitting.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    total = train_ratio + dev_ratio + test_ratio
    if not 0.999 <= total <= 1.001:
        raise ValueError(
            f"Ratios must sum to 1.0, received {train_ratio}, {dev_ratio}, {test_ratio}"
        )
    for name, ratio in [
        ("train", train_ratio),
        ("dev", dev_ratio),
        ("test", test_ratio),
    ]:
        if ratio <= 0 or ratio >= 1:
            raise ValueError(f"{name} ratio must be within (0, 1); received {ratio}")


def load_rows(csv_path: pathlib.Path) -> tuple[List[str], List[List[str]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.reader(fp, delimiter=";")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    header, data = rows[0], rows[1:]
    return header, data


def split_rows(
    rows: Sequence[Sequence[str]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[List[Sequence[str]], List[Sequence[str]], List[Sequence[str]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_rows = shuffled[:train_end]
    dev_rows = shuffled[train_end:dev_end]
    test_rows = shuffled[dev_end:]
    return train_rows, dev_rows, test_rows


def write_split(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.dev_ratio, args.test_ratio)

    header, rows = load_rows(args.input)
    if len(rows) < 3:
        raise ValueError("Need at least 3 rows to perform a meaningful split.")

    train_rows, dev_rows, test_rows = split_rows(
        rows,
        args.train_ratio,
        args.dev_ratio,
        args.seed,
    )

    out_dir = args.output_dir
    write_split(header, train_rows, out_dir / "train.csv")
    write_split(header, dev_rows, out_dir / "dev.csv")
    write_split(header, test_rows, out_dir / "test.csv")

    print(
        "Split complete:\n"
        f"  Train: {len(train_rows)} rows -> {out_dir / 'train.csv'}\n"
        f"  Dev:   {len(dev_rows)} rows -> {out_dir / 'dev.csv'}\n"
        f"  Test:  {len(test_rows)} rows -> {out_dir / 'test.csv'}"
    )


if __name__ == "__main__":
    main()
