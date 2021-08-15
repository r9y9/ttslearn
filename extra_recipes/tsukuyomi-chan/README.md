# tsukuyomi-chan recipes

[つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/)を用いた音声合成のレシピです。

## dnntts

統計的パラメトリック音声合成のレシピです。

### Label

Juliusで音素アライメント、Open JTalkでフルコンテキストを推定し、フルコンテキストラベルとしてまとめています。
https://github.com/r9y9/tsukuyomi-chan-lab

レシピでは、非音声区間の切り詰め、音素継続長の教師データを作るのに利用しています。


## tacotron2_pwg

Tacotron 2 + Parallel WaveGAN による音声合成のレシピです。

### Label

https://github.com/r9y9/tsukuyomi-chan-lab

レシピでは、非音声区間の切り詰め、韻律記号付き音素列を求めるために利用しています。

## ライセンス

ソースコードのライセンスはMITです。
ただし、レシピを通して学習したモデルを利用する場合、または本レポジトリのリリースページで配布する学習済みモデルを利用する場合、学習済みモデルの利用規約 [LICENSE.md](LICENSE.md)をご確認ください。

## 注意書き

このディレクトリに含まれるレシピの実行のためには、つくよみちゃんコーパスを**事前に**ダウンロードする必要があります。
レシピの実行の前に、つくよみちゃんコーパスの利用規約をご確認の上で、本リポジトリのソースコードをご利用ください。

■つくよみちゃんコーパス（CV.夢前黎）
https://tyc.rei-yumesaki.net/material/corpus/
© Rei Yumesaki