.. ttslearn documentation master file, created by
   sphinx-quickstart on Thu May  6 15:44:21 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pythonで学ぶ音声合成 (Text-to-speech with Python)
========================================================

- Amazon: https://www.amazon.co.jp/dp/4295012270/
- インプレス書籍情報: https://book.impress.co.jp/books/1120101073
- GitHub: https://github.com/r9y9/ttslearn

.. image:: _static/image/ttslearn.jpg
   :alt: Pythonで学ぶ音声合成
   :width: 260 px

このサイトは何？
------------------

これは「Pythonで学ぶ音声合成」のドキュメントサイトです。内容は以下の通りです。

- 学習済みモデルを利用した音声合成のデモ
- 書籍に付属のソースコードのうち、著者 (山本) が実行した結果を保存した Jupyter ノートブック (第4章から第10章まで)
- 音声合成のためのコアライブラリ ttslearn のドキュメント

書籍と併せて、学習の参考にしていただければ幸いです。

Python環境
----------

Pythonの実行環境には、PyTorchやNumPyのインストールのしやすさから、Anacondaを推奨します。
ただし、Pythonの環境管理に長けた読者であれば、任意のPython環境を使っていただいても問題ありません。

動作環境
---------

- Linux
- Mac OS X

Windowsは、動作確認をしておりません [#f1]_ 。動作環境には、CUDA/cuDNNがセットアップされたLinux環境を推奨します。

Google Colab を利用する場合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Linux環境を自前で用意するのが難しい読者のために、すべてのJupyter notebook をGoogle Colabで実行できるように配慮しています。
ただし、Google Colabは無償利用に制限があること、Python環境が予告なく変更されることから、可能であれば読者のローカル環境で実行することを推奨します。

Google Colabの基本的な使い方は、https://colab.research.google.com/ を参照してください。

インストール
-------------

用途によって、インストール方法が異なります。詳細は以下の通りです。

書籍のサンプルコードをすべて利用する場合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

音声合成のコアライブラリ ttslearn、レシピ、Jupyterノートブックのすべてが必要です。
GitHubリポジトリにすべて含まれていますので、リポジトリをクローンした後、必要なライブラリをインストールしてください。
Python環境の準備は、前もって行って下さい。

.. code::

   git clone https://github.com/r9y9/ttslearn.git && cd ttslearn
   pip install -e ".[recipes]"

学習済みモデルを用いた音声合成のみを利用する場合
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ttslearn ライブラリをインストールすれば十分です。GitHubリポジトリのソースコードをダウンロードする必要はありません。

.. code::

   pip install ttslearn

インストールが完了すれば、左メニューのデモに示すように、学習済みモデルを利用したテキスト音声合成が可能になります。


.. toctree::
   :maxdepth: 1
   :caption: Demo

   notebooks/ch00_Quick-start.ipynb
   notebooks/ch11_Advanced-demos.ipynb
   demo_server

.. toctree::
   :maxdepth: 1
   :caption: Chapters

   notebooks/ch04_Python-SP.ipynb
   notebooks/ch05_DNNTTS.ipynb
   notebooks/ch06_Recipe-DNNTTS.ipynb
   notebooks/ch07_WaveNet.ipynb
   notebooks/ch08_Recipe-WaveNet.ipynb
   notebooks/ch09_Tacotron.ipynb
   notebooks/ch10_Recipe-Tacotron.ipynb

.. toctree::
  :maxdepth: 1
  :caption: Core modules

  dsp
  dnntts
  wavenet
  tacotron

.. toctree::
  :maxdepth: 1
  :caption: Helpers

  pretrained
  util
  train_util

.. toctree::
    :maxdepth: 1
    :caption: Advanced topics

    extra_recipes
    contrib

.. toctree::
    :maxdepth: 1
    :caption: Meta information

    changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#f1] Windowsで原理的に動かないわけではありません。もしWindowsサポートに協力してくれる方がいれば、貢献を歓迎します。