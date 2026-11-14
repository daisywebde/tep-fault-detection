#!/usr/bin/env python3
"""
convert_rdata_to_csv.py - cut a big TEP .RData file down to a small CSV.

Run this ONCE on a machine with enough RAM (e.g. Google Colab), then copy the small CSV
into data/ and use it everywhere. The pipeline prefers CSV over RData when both exist.

    python convert_rdata_to_csv.py data/TEP_Faulty_Testing.RData
    python convert_rdata_to_csv.py data/TEP_Faulty_Testing.RData --max-run 10 --faults 1 2 4 5 6
"""
import argparse
from pathlib import Path

import tep_fd as T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help=".RData file to convert")
    ap.add_argument("--max-run", type=int, default=5, help="keep simulation runs 1..N (default 5)")
    ap.add_argument("--faults", type=int, nargs="+", default=None, help="keep only these fault numbers (default: all)")
    ap.add_argument("--out", default=None, help="output CSV (default: <name>_subset.csv next to the input)")
    args = ap.parse_args()

    src = Path(args.path)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_subset.csv")
    df = T.load_tep(src, faults=args.faults, max_run=args.max_run)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df['faultNumber'].nunique()} fault ids, "
          f"{df['simulationRun'].nunique()} runs each) to {out}")


if __name__ == "__main__":
    main()
