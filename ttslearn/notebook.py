import os


def get_cmap():
    """Gets the colormap (default: ``viridis``)

    The colormap can be set by the environment variable ``TTSLEARN_CMAP``
    for convienence.

    Returns:
        str: The name of the current colormap.


    Examples:

    .. ipython::

        In [1]: from ttslearn.notebook import get_cmap

        In [2]: get_cmap()
        Out[2]: 'viridis'
    """
    return os.environ.get("TTSLEARN_CMAP", "viridis")


def init_plot_style():
    """Initializes the plotting style."""
    import matplotlib.pyplot as plt

    if get_cmap() == "gray":
        plt.style.use("grayscale")


def savefig(name, dpi=350, *args, **kwargs):
    """Saves the figure to a file.

    By default, a figure is saved as a .png file. The extension of the file can be set by
    the environment variable ``TTSLEARN_EXT`` for convienence.
    If the environment variable ``TTSLEARN_SAVEFIG`` is set to 0, figures will not be saved
    and this function do nothing.

    Args:
        name (str): The name of the file.
        dpi (int): The resolution of the image.
        args: Additional arguments for plt.savefig.
        kwargs: Additional keyword arguments for plt.savefig.
    """
    import matplotlib.pyplot as plt

    if os.environ.get("TTSLEARN_NO_SAVEFIG", 0):
        return  # no op

    fig_ext = os.environ.get("TTSLEARN_FIG_EXT", ".png")

    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    plt.savefig(name + fig_ext, *args, dpi=dpi, bbox_inches="tight", **kwargs)
