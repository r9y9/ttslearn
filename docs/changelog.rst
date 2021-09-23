Change log
==========

v0.2.2 <2021-xx-xx>
-------------------

- `#17`_: Add conv1d test to ensure forward/incremental_forward correctness
- `#14`_: windows: use expanduser instread of os.environ["HOME"]
- `#13`_: Fix: 毎回JSUTダウンロードをやり直す
- `#12`_: Fix `#10`_ 書籍 p.82 code4.9 関数stftの誤植
- `#11`_: Add warning for streamlit online demo

v0.2.1 <2021-08-21>
-------------------

- pretrained: add PWG TTS models for common voice (ja)
- pretrained: add HiFi-GAN based TTS models using JVS and JSUT corpus
- Add HiFi-GAN configs for JVS and JSUT extra recipes
- `#7`_: Add script to generate ground-truth aligned (GTA) features
- `#5`_: [docker] Push docker image to Docker Hub
- `#4`_: [docker] fix docker build fail because no 'gcc' command
- `#2`_: [extra_recipes] Fix the suffix of the script; s/sh/bash/
- `#1`_: Add common voice jp recipes

v0.2.0 <2021-08-12>
-------------------

The first public release!

.. _#1: https://github.com/r9y9/ttslearn/pull/1
.. _#2: https://github.com/r9y9/ttslearn/pull/2
.. _#4: https://github.com/r9y9/ttslearn/pull/4
.. _#5: https://github.com/r9y9/ttslearn/pull/5
.. _#7: https://github.com/r9y9/ttslearn/pull/7
.. _#10: https://github.com/r9y9/ttslearn/issues/10
.. _#11: https://github.com/r9y9/ttslearn/pull/11
.. _#12: https://github.com/r9y9/ttslearn/pull/12
.. _#13: https://github.com/r9y9/ttslearn/pull/13
.. _#14: https://github.com/r9y9/ttslearn/pull/14
.. _#17: https://github.com/r9y9/ttslearn/pull/17

