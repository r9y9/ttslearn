# ttslearn: Library for Pythonで学ぶ音声合成 (Text-to-speech with Python)

[![][docs-latest-img]][docs-latest-url]
[![PyPI](https://img.shields.io/pypi/v/ttslearn.svg)](https://pypi.python.org/pypi/ttslearn)
![Python package](https://github.com/r9y9/ttslearn/workflows/Python%20package/badge.svg)
[![Docker Hub](https://github.com/r9y9/ttslearn/actions/workflows/docker-hub-ci.yaml/badge.svg)](https://github.com/r9y9/ttslearn/actions/workflows/docker-hub-ci.yaml)
![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://r9y9.github.io/ttslearn/

<div align="left">
<a href="https://book.impress.co.jp/books/1120101073">
<img src="docs/_static/image/ttslearn.jpg" alt="drawing" width="260"/>
</a>

</div>

日本語は以下に続きます (Japanese follows)

English: This book is written in Japanese and primarily focuses on Japanese TTS. Some of the functionality (e.g., neural network implementations) in this codebase can be used for other languages. However, we didn't prepare any guide or code for non-Japanese TTS systems.
We may extend the codebase for other languages in the future but cannot guarantee if we would work on it.

## Installation

```
pip install ttslearn
```

## リポジトリの構成

- [ttslearn](ttslearn): 「Pythonで学ぶ音声合成」のために作成された、音声合成のコアライブラリです。 `pip install ttslearn` としてインストールされるライブラリの実体です。書籍のサンプルコードとしてだけでなく、汎用的な音声合成のライブラリとしてもご利用いただけます。
- [notebooks](notebooks): 第4章から第10章までの、Jupyter notebook形式のソースコードです。
- [hydra](hydra): 第6章で解説している hydra のサンプルコードです。
- [recipes](recipes): 第6章、第8章、第10章で解説している、日本語音声合成のレシピです。[JSUTコーパス](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)を利用した日本語音声合成システムの実装が含まれています。
- [extra_recipes](extra_recipes): 発展的な音声合成のレシピです。書籍では解説していませんが、`ttslearn` ライブラリの利用例として、JSUTコーパス、JVSコーパスを用いた音声合成のレシピをリポジトリに含めています。

詳細なドキュメントは、https://r9y9.github.io/ttslearn/ を参照してください。

## ライセンス

ソースコードのライセンスはMITです。商用・非商用問わずに、お使いいただけます。
詳細は [LICENSEファイル](LICENSE)を参照してください。

### 学習済みモデルの利用規約

本リポジトリのリリースページでは、[JSUTコーパス](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)・[JVSコーパス](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)を用いて学習した、学習済みモデルを配布しています。それらの学習済みモデルは、「非商用目的」でのみ利用可能です。
学習済みモデルを利用する際は、各コーパスの利用規約も併せてご確認ください。

また、作者は、学習済みモデルの利用による一切の請求、損害、その他の義務について何らの責任も負わないものとします。

## 付録

付録として、日本語音声合成のフルコンテキストラベルの仕様をまとめています。
詳細は、[docs/appendix.pdf](docs/appendix.pdf) を参照してください。

## 問い合わせ

書籍の内容、ソースコードに関する質問などありましたら、GitHub issue にてお問い合わせをいただければ、可能な限り返答します。

## お詫びと訂正

本書の正誤表を以下のリンク先でまとめています。

[本書の正誤表](https://docs.google.com/spreadsheets/d/185pTXTzCI3l4kkJTXVa4fsu6yhAwd8aury2PnLol55Q/edit?usp=sharing)

もし、正誤表に記載されていない誤植などの間違いを見つけた場合は、GitHub issue にてご連絡ください。

## 謝辞

- Tacotron 2の一部ソースコードは、[ESPnet](https://github.com/espnet/espnet)を元に作られました。(thanks to [@kan-bayashi](https://github.com/kan-bayashi))
- 発展的なレシピの実装のほとんどにおいて、[kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)を利用しました。
- 日本語音声合成のテキスト処理には、[Open JTalk](https://open-jtalk.sp.nitech.ac.jp/) およびその[Pythonラッパー](https://github.com/r9y9/pyopenjtalk)を利用しました。

## リンク

- Amazon: https://www.amazon.co.jp/dp/4295012270/
- インプレス書籍情報: https://book.impress.co.jp/books/1120101073
