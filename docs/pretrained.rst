ttslearn.pretrained
====================

学習済みモデルを管理するためのモジュールです。

.. automodule:: ttslearn.dnntts

Pre-trained models will be automatically downloaded if you run TTS functionality (e.g., :py:class:`ttslearn.dnntts.tts.DNNTTS`) at the first time.
The models are saved in ``$HOME/.cache/ttslearn/`` by default.
To control the save location, you can manually set it by the environmental variable ``TTSLEARN_CACHE_DIR``.

Pretrained models
-----------------

All the models listed here were trained using JSUT corpus.

+----------------+------------------------------------------------+------------------------------------------------------------+
| Model ID       | Class                                          | Details of the model                                       |
+----------------+------------------------------------------------+------------------------------------------------------------+
| ``dnntts``     | :py:class:`ttslearn.dnntts.tts.DNNTTS`         | DNN-based statistical parametric speech synthesis (sec. 6) |
+----------------+------------------------------------------------+------------------------------------------------------------+
| ``wavenettts`` | :py:class:`ttslearn.wavenet.tts.WaveNetTTS`    | WaveNet TTS (sec. 8)                                       |
+----------------+------------------------------------------------+------------------------------------------------------------+
| ``tacotron2``  | :py:class:`ttslearn.tacotron.tts.Tacotron2TTS` | An end-to-end TTS based on Tacotron 2 (sec. 10)            |
+----------------+------------------------------------------------+------------------------------------------------------------+

Extra pretrained models
------------------------

Note that the following models are not explained in our book.
Those were trained using extra recipes found in our GitHub repository.

+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| Model ID                             | Corpus       | Class                                                      | Details of the model                                                                                |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``tacotron2_pwg_jsut16k``            | JSUT         | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Tacotron 2 with Parallel WaveGAN (PWG). Trained on JSUT corpus. Sampling rate: 16 kHz.              |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``tacotron2_pwg_jsut24k``            | JSUT         | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Tacotron 2 with PWG. Trained on JSUT corpus. Sampling rate: 24 kHz.                                 |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``tacotron2_hifipwg_jsut24k``        | JSUT         | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Tacotron 2 with HiFi-GAN. Trained on JSUT corpus. Sampling rate: 24 kHz.                            |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``multspk_tacotron2_pwg_jvs16k``     | JVS          | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Multi-speaker Tacotron 2 with PWG. Trained on JVS corpus. Sampling rate: 16 kHz.                    |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``multspk_tacotron2_pwg_jvs24k``     | JVS          | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Multi-speaker Tacotron 2 with Parallel WaveGAN (PWG). Trained on JVS corpus. Sampling rate: 24 kHz. |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``multspk_tacotron2_hifipwg_jvs24k`` | JVS          | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Multi-speaker Tacotron 2 with HiFi-GAN. Trained on JVS corpus. Sampling rate: 24 kHz.               |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``multspk_tacotron2_pwg_cv16k``      | common voice | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Multi-speaker Tacotron 2 with PWG. Trained on common voice (ja) corpus. Sampling rate: 16 kHz.      |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+
| ``multspk_tacotron2_pwg_cv24k``      | common voice | :py:class:`ttslearn.contrib.tacotron2_pwg.Tacotron2PWGTTS` | Multi-speaker Tacotron 2 with PWG. Trained on common voice (ja) corpus. Sampling rate: 24 kHz.      |
+--------------------------------------+--------------+------------------------------------------------------------+-----------------------------------------------------------------------------------------------------+

Helpers
--------

.. automodule:: ttslearn.pretrained

.. autosummary::
    :toctree: generated/
    :nosignatures:

    create_tts_engine
    get_available_model_ids
    retrieve_pretrained_model
