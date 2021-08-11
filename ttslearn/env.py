def is_colab():
    """If running on Google Colab, return True."""
    try:
        import google.colab  # NOQA

        return True
    except ImportError:
        return False


def is_notebook():
    """If running in a Jupyter Notebook, return True."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type
    except NameError:
        return False
