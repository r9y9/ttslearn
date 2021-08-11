import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(description="Fit scalers")
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("in_dir", type=str, help="in directory")
    parser.add_argument("out_path", type=str, help="Output path")
    parser.add_argument("--external_scaler", type=str, help="External scaler")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    if args.external_scaler is not None:
        scaler = joblib.load(args.external_scaler)
    else:
        scaler = StandardScaler()
    with open(args.utt_list) as f:
        for utt_id in tqdm(f):
            c = np.load(in_dir / f"{utt_id.strip()}-feats.npy")
            scaler.partial_fit(c)
        joblib.dump(scaler, args.out_path)
