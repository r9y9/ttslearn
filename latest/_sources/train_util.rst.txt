ttslearn.train_util
====================

Specific utilities for training

.. automodule:: ttslearn.train_util

Dataset
-------

.. autoclass:: Dataset
    :members: __getitem__, __len__


DataLoader
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    collate_fn_dnntts
    collate_fn_wavenet
    collate_fn_tacotron
    get_data_loaders


Model parameters
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    moving_average_
    num_trainable_params


Helper for training
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    setup
    save_checkpoint
    num_trainable_params
    get_epochs_with_optional_tqdm


Plotting
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    plot_attention
    plot_2d_feats
