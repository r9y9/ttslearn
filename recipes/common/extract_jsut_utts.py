import argparse
import sys

import pyopenjtalk
import yaml
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract utt_list.txt for JSUT basic 5000",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text_kana", type=str, help="text_kana/basic5000.yaml")
    parser.add_argument("out_path", type=str, help="Output path")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.text_kana) as f:
        d = yaml.safe_load(f)

    cnt = 0
    allow_pau_mismatch = False
    with open(args.out_path, "w") as of:
        for k, v in tqdm(d.items()):
            text_level0 = v["text_level0"]
            text_level2 = v["text_level2"]
            phone_level3 = v["phone_level3"].replace("-", " ").lower()

            ps = pyopenjtalk.g2p(text_level2).lower()
            if allow_pau_mismatch:
                phone_level3 = phone_level3.replace(" pau ", " ")
                ps = ps.replace(" pau ", " ")

            # 正解の音素列と、Open JTalkのG2Pの結果が異なる場合
            if ps != phone_level3:
                cnt += 1
            else:
                of.write(f"{k}\n")

    print("G2Pの推定誤りを含む発話数:", cnt)
    print("有効な発話数:", 5000 - cnt)
