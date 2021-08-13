# commonvoice recipes

## multspk_tacotron2_pwg_20spks

Multi-speaker Tacotron 2 + Parallel WaveGAN (PWG) による、多話者音声合成のレシピです。

### Data

Common Voice JP corpus (`cv-corpus-7.0-2021-07-21`) を利用します。

https://commonvoice.mozilla.org/ja/datasets

学習データには、validated.tsv に記述された発話を用います。
ただし、すべての発話を利用するのではなく、発話数が100を超える話者の発話のみを利用します。
理由は、発話数があまりに少ない（5文以下など）場合は、話者のimbalance性のため、学習が困難になると考えられるからです。
発話数が100を超えるデータのみに絞ると、話者数は20でした。[multspk_tacotron2_pwg/data/spks](multspk_tacotron2_pwg/data/spks) に話者 (正確には、client_id) のリストが書かれています。
話者毎の発話リストは、[multspk_tacotron2_pwg/data](multspk_tacotron2_pwg/data) にまとまっています。
Julusによる音素アライメントに失敗した発話は、発話リストから除かれています。

train.tsv, dev.tsv, test.tsv は使用しせず、独自にtrain/dev/test のsplitを行っています（※強い理由はありません。単なる便宜上の理由によるものです）。

## multspk_tacotron2_pwg_368spks

上述のように、発話数が100を超える話者のみに限定せず、使えるデータはすべて使った場合のレシピです。
話者数は368です。発話数が1の話者が多く学習データに含まれていることに注意してください。

## Links

- https://commonvoice.mozilla.org/ja