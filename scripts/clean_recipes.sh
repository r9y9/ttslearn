#!/bin/bash

set -e

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $script_dir/..

for name in dnntts wavenet tacotron;
do
    echo "Removing all intermediate features in $name"
    cd recipes/$name
    rm -rvf dump exp fig outputs
    rm -rvf tensorboard/jsut_*
    rm -rvf tts_models
    cd -
done