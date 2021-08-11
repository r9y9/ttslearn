ttslearn.wavenet
================

WaveNet音声合成のためのモジュールです。

TTS
---

The TTS functionality is accessible from ``ttslearn.wavenet.*``

.. automodule:: ttslearn.wavenet.tts

.. autoclass:: WaveNetTTS
    :members:

Upsampling networks
-------------------

.. automodule:: ttslearn.wavenet.upsample

Repeat upsampling
^^^^^^^^^^^^^^^^^

.. autoclass:: RepeatUpsampling
    :members:


Nearest neighbor upsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: UpsampleNetwork
    :members:


Conv1d + nearest neighbor upsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ConvInUpsampleNetwork
    :members:


Convolution block
------------------

.. automodule:: ttslearn.wavenet.modules

.. autoclass:: ResSkipBlock
    :members:

WaveNet
------------------

.. automodule:: ttslearn.wavenet.wavenet

.. autoclass:: WaveNet
    :members:

Generation utility
-------------------

.. automodule:: ttslearn.wavenet.gen

.. autosummary::
    :toctree: generated/
    :nosignatures:

    gen_waveform


Utility
-------

.. automodule:: ttslearn.wavenet

.. autosummary::
    :toctree: generated/
    :nosignatures:

    receptive_field_size

