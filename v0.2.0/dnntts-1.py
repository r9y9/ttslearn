from ttslearn.dnntts import DNNTTS
import matplotlib.pyplot as plt

engine = DNNTTS()
wav, sr = engine.tts("日本語音声合成のデモです。")

fig, ax = plt.subplots(figsize=(8,2))
librosa.display.waveplot(wav.astype(np.float32), sr, ax=ax)