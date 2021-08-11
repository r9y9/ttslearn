import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.io import hts
from scipy.io import wavfile
from tqdm import tqdm
from ttslearn.dsp import world_log_f0_vuv


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for log-F0 prediction models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("qst_file", type=str, help="HTS style question file")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    return parser


def preprocess(wav_file, lab_file, binary_dict, numeric_dict, sr, in_dir, out_dir):
    assert wav_file.stem == lab_file.stem
    # フルコンテキストラベルの読み込み
    labels = hts.load(lab_file)

    # フレーム単位の言語特徴量の抽出
    in_feats = fe.linguistic_features(
        labels,
        binary_dict,
        numeric_dict,
        add_frame_features=True,
        subphone_features="coarse_coding",
    )

    # 音声ファイルの読み込み
    _sr, x = wavfile.read(wav_file)
    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    x = librosa.resample(x, _sr, sr)

    # 連続対数基本周波数と有声 / 無声フラグを結合した特徴量の計算
    out_feats = world_log_f0_vuv(x.astype(np.float64), sr)

    # フレーム数の調整
    minL = min(in_feats.shape[0], out_feats.shape[0])
    in_feats, out_feats = in_feats[:minL], out_feats[:minL]

    # 冒頭と末尾の非音声区間の長さを調整
    assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
    start_frame = int(labels.start_times[1] / 50000)
    end_frame = int(labels.end_times[-2] / 50000)

    # 冒頭：50 ミリ秒、末尾：100 ミリ秒
    start_frame = max(0, start_frame - int(0.050 / 0.005))
    end_frame = min(minL, end_frame + int(0.100 / 0.005))

    in_feats = in_feats[start_frame:end_frame]
    out_feats = out_feats[start_frame:end_frame]

    # NumPy 形式でファイルに保存
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
    wav_files = [Path(args.wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_files = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]
    binary_dict, numeric_dict = hts.load_question_set(args.qst_file)

    in_dir = Path(args.out_dir) / "in_logf0"
    out_dir = Path(args.out_dir) / "out_logf0"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                wav_file,
                lab_file,
                binary_dict,
                numeric_dict,
                args.sample_rate,
                in_dir,
                out_dir,
            )
            for wav_file, lab_file in zip(wav_files, lab_files)
        ]
        for future in tqdm(futures):
            future.result()
