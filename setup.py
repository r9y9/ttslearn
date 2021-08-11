import codecs
import re
from os.path import exists, join

from setuptools import find_packages, setup


def find_version(*file_paths: str) -> str:
    with codecs.open(join(*file_paths), "r") as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if exists("README.md"):
    with open("README.md", encoding="utf8") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""

setup(
    name="ttslearn",
    version=find_version("ttslearn", "__init__.py"),
    description="ttslearn: Text-to-speech with Python",
    author="Ryuichi Yamamoto",
    author_email="zryuichi@gmail.com",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    package_data={"": ["_example_data/*"]},
    packages=find_packages(include=["ttslearn*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19",
        "scipy",
        "torch>=1.0.0",
        "pysptk",
        "nnmnkwii>=0.0.23",
        "pyworld",
        "librosa>=0.8.1",
        "tqdm",
        "joblib",
        "hydra-core >= 1.1.0",
        "pyopenjtalk>=0.1.0",
        "matplotlib>=1.5",
        "parallel_wavegan>=0.5.1",
    ],
    extras_require={
        "recipes": [
            "scikit-learn",
            "pandas",
            "tensorboard>=2.3.0",
            "ipython",
            "jupyterlab",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx_rtd_theme",
            "nbsphinx>=0.8.6",
            "Jinja2>=3.0.1",
            "pandoc",
            "ipython",
            "jupyter",
            "gdown",
        ],
        "lint": [
            "pysen",
            "types-setuptools",
            "mypy<=0.910",
            "black>=19.19b0,<=20.8",
            "flake8>=3.7,<4",
            "flake8-bugbear",
            "isort>=4.3,<5.2.0",
        ],
        "test": ["pytest"],
        "demo": ["streamlit>=0.86.0", "stqdm"],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
