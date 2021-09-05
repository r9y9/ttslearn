Demo server (with streamlit)
==============================

音声合成を行うデモサーバのソースコードをGitHubリポジトリに含めています。
ソースコードは、 ``demo_server.py`` です。

オンラインデモ
------------------

https://share.streamlit.io/r9y9/ttslearn/demo_server.py


.. warning::
    上記URLのオンラインデモは、steamlit shareの都合でサーバに負荷がかかると停止する場合があります。
    安定した動作のためには、後述するようにローカルで ``demo_server.py`` を実行してください。

デモサーバをローカルで立ち上げる場合
---------------------------------------

デモサーバの実行には、`streamlit <https://streamlit.io/>`_ が必要です。
streamlitを含め、デモサーバの実行に必要なパッケージをインストールするには、下記のコマンドを実行してください。

.. code::

    pip install "ttslearn[demo]"

必要な依存関係をインストールした後、下記コマンドを実行すれば、デモサーバが立ち上がります。

.. code::

    streamlit run demo_server.py -- --device cpu

CUDAを利用する場合は、次のように実行してください。

.. code::

    streamlit run demo_server.py -- --device cuda

以下のような画面がブラウザ上に表示されれば、音声合成の準備はOKです。

.. image:: _static/image/demo_server.png
   :alt: デモサーバ

左メニューから、学習済みモデルを切り替えられます。
右側の日本語テキスト入力から、任意のテキストを入力し、音声合成の結果をブラウザ上から確認できます。
なお、ノートブック形式で提供している機能と本質的には同じです。

プログラマブルなインタフェースが好みであればJupyterノートブックを、単に音声合成の機能が必要なだけであればデモサーバを利用すると良いでしょう。
