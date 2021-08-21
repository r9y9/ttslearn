ttslearn.dnntts
================

DNN音声合成のためのモジュールです。

TTS
---

The TTS functionality is accessible from ``ttslearn.dnntts.*``

.. automodule:: ttslearn.dnntts.tts

.. autoclass:: DNNTTS
    :members:


Models
------

The following models are acceible from ``ttslearn.dnntts.*``

.. automodule:: ttslearn.dnntts.model

Feed-forward DNN
^^^^^^^^^^^^^^^^

.. autoclass:: DNN
    :members:

LSTM-RNN
^^^^^^^^

.. autoclass:: LSTMRNN
    :members:

Multi-stream functionality
--------------------------

.. automodule:: ttslearn.dnntts.multistream

.. autosummary::
    :toctree: generated/
    :nosignatures:

    split_streams
    multi_stream_mlpg
    get_windows
    get_static_stream_sizes
    get_static_features


Generation utility
-------------------

.. automodule:: ttslearn.dnntts.gen


.. autosummary::
    :toctree: generated/
    :nosignatures:

    predict_duration
    predict_acoustic
    gen_waveform
