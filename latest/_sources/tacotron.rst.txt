ttslearn.tacotron
=================

Tacotron 2に基づく音声合成のためのモジュールです。

TTS
---

The TTS functionality is accessible from ``ttslearn.tacotron.*``

.. automodule:: ttslearn.tacotron.tts

.. autoclass:: Tacotron2TTS
    :members:


Text processing frontend
-------------------------

Open JTalk (Japanese)
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ttslearn.tacotron.frontend.openjtalk


.. autosummary::
    :toctree: generated/
    :nosignatures:

    pp_symbols
    text_to_sequence
    sequence_to_text
    num_vocab

Text (English)
^^^^^^^^^^^^^^^

.. automodule:: ttslearn.tacotron.frontend.text


.. autosummary::
    :toctree: generated/
    :nosignatures:

    text_to_sequence
    sequence_to_text
    num_vocab


Encoder
--------

.. automodule:: ttslearn.tacotron.encoder

.. autoclass:: Encoder
    :members:

Attention
-----------

.. automodule:: ttslearn.tacotron.attention

.. autoclass:: BahdanauAttention
    :members:

.. autoclass:: LocationSensitiveAttention
    :members:

Decoder
-----------

.. automodule:: ttslearn.tacotron.decoder


.. autoclass:: Prenet
    :members:

.. autoclass:: Decoder
    :members:


Post-Net
-----------

.. automodule:: ttslearn.tacotron.postnet


.. autoclass:: Postnet
    :members:


Tacotron 2
-----------

.. automodule:: ttslearn.tacotron.tacotron2


.. autoclass:: Tacotron2
    :members:



Generation utility
-------------------

.. automodule:: ttslearn.tacotron.gen


.. autosummary::
    :toctree: generated/
    :nosignatures:

    synthesis_griffin_lim
    synthesis
