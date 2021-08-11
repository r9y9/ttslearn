import argparse
import tempfile

import numpy as np
import streamlit as st
import torch
import ttslearn
from scipy.io import wavfile
from stqdm import stqdm
from ttslearn.logger import getLogger
from ttslearn.pretrained import get_available_model_ids


def get_parser():
    parser = argparse.ArgumentParser(description="TTS Demo with streamlit")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbose level")

    return parser


@st.cache
def create_tts_engine(model_id, device):
    from ttslearn.pretrained import create_tts_engine

    engine = create_tts_engine(model_id)
    engine.set_device(device)
    return engine


if __name__ == "__main__":
    args = get_parser().parse_args()
    logger = getLogger(verbose=args.verbose, add_stream_handler=False)
    logger.info("Using device: {}".format(args.device))

    st.title("Pythonで学ぶ音声合成のデモ")
    st.text(f"ttslearn's version: {ttslearn.__version__}")

    with st.expander("利用方法を見る"):
        st.text(
            """
使い方:
1. 左メニューの Model ID から、学習済みモデルの名前を選択してください。
   学習済みモデルの説明は、 https://r9y9.github.io/ttslearn/pretrained.html を参照してください。
2. (Optional) 多話者音声合成の場合は、左メニューに、話者を選択するプルダウンメニューが表示されます。
   好きな話者を選択してください。
3. 「日本語テキスト入力」から、合成したい日本語テキストを入力してください
4. 合成ボタンを押して下さい。
5. 音声を再生するボタンを押すと、合成音声が再生されます。
"""
        )

    # 音声合成エンジンのインスタンス化
    model_id = st.sidebar.selectbox("Model ID", get_available_model_ids())
    engine = create_tts_engine(model_id, args.device)

    # Multi-speaker TTSの場合、話者情報が必要
    if hasattr(engine, "spks") and engine.spks is not None:
        assert engine.spk2id is not None
        spk = st.sidebar.selectbox("Speaker", engine.spks, index=0)
    else:
        spk = None

    text = st.text_area("日本語テキスト入力", value="ここに、好きな日本語テキストを入力してください。").strip()

    if st.button("合成") and len(text) > 0:
        logger.info(f"Input text: {text}")
        if spk is None:
            wav, sr = engine.tts(text, tqdm=stqdm)
        else:
            logger.info(f"Speaker: {spk}")
            wav, sr = engine.tts(text, tqdm=stqdm, spk_id=engine.spk2id[spk])

        # 音量を正規化
        assert wav.dtype == np.int16
        wav = (wav / np.abs(wav).max() * 32767.0).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            wavfile.write(f.name, sr, wav)
            with open(f.name, "rb") as wav_file:
                st.audio(wav_file.read(), format="audio/wav")
