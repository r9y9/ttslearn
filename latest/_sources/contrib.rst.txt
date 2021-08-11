ttslearn.contrib
================

発展的な実装のためのモジュールです。

TTS
---

.. automodule:: ttslearn.contrib.tacotron2_pwg

.. autoclass:: Tacotron2PWGTTS
    :members:


Multi-speaker Tacotron 2
-------------------------

.. automodule:: ttslearn.contrib.multispk_tacotron2

.. autoclass:: MultiSpkTacotron2
    :members:


Utility for multi-speaker training
-----------------------------------

.. automodule:: ttslearn.contrib.multispk_util

.. autoclass:: Dataset
    :members: __getitem__, __len__

DataLoader
^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    collate_fn_ms_tacotron
    get_data_loaders


Helper for training
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:

    setup
