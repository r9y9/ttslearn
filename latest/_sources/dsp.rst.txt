ttslearn.dsp
============

音声信号処理の機能を提供するモジュールです。

.. automodule:: ttslearn.dsp

Acoustic feature extraction
----------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    world_spss_params
    world_log_f0_vuv
    logspectrogram
    logmelspectrogram

F0 conversion
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    f0_to_lf0
    lf0_to_f0


Dynamic features
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    compute_delta


Compression and de-compression
------------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    mulaw
    mulaw_quantize
    inv_mulaw
    inv_mulaw_quantize

Inversion
-----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    logmelspectrogram_to_audio
