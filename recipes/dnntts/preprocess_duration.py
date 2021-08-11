import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for duration models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("lab_root", type=str, help="label directory")
    parser.add_argument("qst_file", type=str, help="HTS style question file")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    return parser


def preprocess(lab_file, binary_dict, numeric_dict, in_dir, out_dir):
    labels = hts.load(lab_file)
    in_feats = fe.linguistic_features(labels, binary_dict, numeric_dict)
    out_feats = fe.duration_features(labels)

    utt_id = lab_file.stem
    np.save(
        in_dir / f"{utt_id}-feats.npy", in_feats.astype(np.float32), allow_pickle=False
    )
    np.save(
        out_dir / f"{utt_id}-feats.npy",
        out_feats.astype(np.float32),
        allow_pickle=False,
    )


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    lab_files = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]
    binary_dict, numeric_dict = hts.load_question_set(args.qst_file)

    in_dir = Path(args.out_dir) / "in_duration"
    out_dir = Path(args.out_dir) / "out_duration"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess, lab_file, binary_dict, numeric_dict, in_dir, out_dir
            )
            for lab_file in lab_files
        ]
        for future in tqdm(futures):
            future.result()
