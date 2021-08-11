import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Normalization")
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("scaler_path", type=str, help="scaler path")
    parser.add_argument("in_dir", type=str, help="in directory")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--inverse", action="store_true", help="Inverse transform")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")

    return parser


def process(path, scaler, inverse, out_dir):
    x = np.load(path)
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)
    assert x.dtype == y.dtype
    np.save(out_dir / path.name, y, allow_pickle=False)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    paths = [Path(in_dir / f"{utt_id}-feats.npy") for utt_id in utt_ids]
    scaler = joblib.load(args.scaler_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(process, path, scaler, args.inverse, out_dir)
            for path in paths
        ]
        for future in tqdm(futures):
            future.result()
