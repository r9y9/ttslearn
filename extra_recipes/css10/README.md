# css10 recipes

## dnntts

パラメトリック音声合成のレシピです。

### Data

https://www.kaggle.com/bryanpark/japanese-single-speaker-speech-dataset

CSS 10 Japanese を利用します。

### Label

Juliusで音素アライメント、Open JTalkでフルコンテキストを推定し、フルコンテキストラベルとしてまとめています。
https://github.com/r9y9/css10-lab

Julilusで音素アライメントを求めるのに失敗した発話、およびOpen JTalkによるG2Pに誤りが多く含まれる可能性が高い発話は、フルコンテキストラベルはありません。フルコンテキストラベルがある発話のリストは、data/utt_list.txt を参照してください。

レシピでは、非音声区間の切り詰め、音素継続長を求めるために利用しています。

## tacotron2_pwg

Tacotron 2 + Parallel WaveGAN を用いた音声合成のレシピです。

### Data

dnntts レシピと同じ

### Label

非音声区間の切り詰め、韻律記号付き音素列を求めるために利用しています。

## Links

- CSS 10 Japanese: https://www.kaggle.com/bryanpark/japanese-single-speaker-speech-dataset