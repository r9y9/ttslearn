#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

# cd to the top of ttslearn
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $script_dir/..

n_jobs=8

cd recipes/dnntts
if [ ! -e dnntts_exp.tar.gz ]; then
    gdown "https://drive.google.com/uc?id=171gGoH3H4PJ-9cMQES-l6KpTu9n0udGD"
fi
if [ ! -d exp ]; then
    tar zxvf dnntts_exp.tar.gz
fi
cd -

cd recipes/wavenet
if [ ! -e wavenet_exp.tar.gz ]; then
    gdown "https://drive.google.com/uc?id=1Z09yCCAKyKOUU3Zsdxs0mZ1Q-TCPqFHA"
fi
if [ ! -d exp ]; then
    tar zxvf wavenet_exp.tar.gz
fi
cd -

cd recipes/tacotron
if [ ! -e tacotron_exp.tar.gz ]; then
    gdown "https://drive.google.com/uc?id=1LoIGkwTLUZmkJkxbTR1S7yyaWexn-Wfp"
fi
if [ ! -d exp ]; then
    tar zxvf tacotron_exp.tar.gz
fi
cd -


cd recipes/dnntts
# $HOME/data/jsut_ver1 と、コーパスのディレクトリ内にラベルファイルがあることを仮定
ln -sf ~/data downloads
if [ ! -d dump/jsut_sr16000/norm ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 1 --stop-stage 3 --n-jobs $n_jobs
fi
if [ ! -d tts_models ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 99 --stop-stage 99
fi
cd -

cd recipes/wavenet
# $HOME/data/jsut_ver1 と、コーパスのディレクトリ内にラベルファイルがあることを仮定
ln -sf ~/data downloads
if [ ! -d dump/jsut_sr16000/norm ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 1 --stop-stage 4 --n-jobs $n_jobs
fi
if [ ! -d tts_models ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 99 --stop-stage 99
fi
cd -

cd recipes/tacotron
# $HOME/data/jsut_ver1 と $HOME/data/jsut-label があることを仮定
ln -sf ~/data downloads
if [ ! -d dump/jsut_sr16000/norm ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 1 --stop-stage 2 --n-jobs $n_jobs
fi
if [ ! -d tts_models ]; then
    CUDA_VISIBLE_DEVICES= ./run.sh --stage 99 --stop-stage 99
fi
cd -

# ノートブックのコンパイル！
for n in notebooks/ch0[4-9]*.ipynb notebooks/ch10*.ipynb notebooks/ch11*.ipynb notebooks/ch00*.ipynb;
do
    name=$(basename $n)
    echo "Processing $name.... (this will take some time)"
    xrun jupyter nbconvert --to notebook --execute $n
    # mv notebooks/ch04_Python-SP.nbconvert.ipynb
    filename="${name%.*}"
    mv -v notebooks/${filename}.nbconvert.ipynb docs/notebooks/$name
    echo "Finished processing $name"
done