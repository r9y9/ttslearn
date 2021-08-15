Extra recipes
==============

ttslearn ライブラリは、「Pythonで学ぶ音声合成」の執筆に伴って開発されましたが、その応用は書籍で解説した音声合成に限りません。
書籍で解説しなかった発展的な音声合成の例として、第11章で少し触れた「非自己回帰型ニューラルボコーダ」を用いたレシピを用意しています。
音声サンプルは、左メニューの「Advanced TTS demos」で示したとおりです。

発展的なレシピのソースコードは、GitHubの ``extra_recipes`` ディレクトリに保存されています。
また、発展的なレシピに必要な機能は、``ttslearn.contrib`` モジュールにまとまっています。
書籍に解説はありませんが、興味のある読者は、ソースコードを読んで試してみて下さい。

以下、コーパスごとにレシピの概要を示します。

JSUT corpus
^^^^^^^^^^^^

https://sites.google.com/site/shinnosuketakamichi/publication/jsut


- ``dnntts``: DNN音声合成（24kHz, 48kHz）
- ``tacotron2_pwg``: Tacotron 2 with Parallel WaveGAN (16kHz, 24kHz)

JVS corpus
^^^^^^^^^^^^

https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus

- ``dnntts``: DNN音声合成（24kHz）
- ``tacotron2_pwg``: Multi-speaker Tacotron 2 with Parallel WaveGAN (16kHz, 24kHz)

Common voice (ja)
^^^^^^^^^^^^^^^^^^

https://commonvoice.mozilla.org/ja/datasets

- ``multispk_tacotron2_pwg_20spks``: Multi-speaker Tacotron 2 with Parallel WaveGAN (16kHz, 24kHz)
- ``multispk_tacotron2_pwg_386spks``: Multi-speaker Tacotron 2 with Parallel WaveGAN (16kHz, 24kHz)


その他のコーパスに同様の音声合成の仕組みを応用することは容易ですが、それらは読者に委ねます。

ttslearnのGitHubリポジトリにレシピを追加したい場合は、積極的に検討しますので、お問い合わせください。