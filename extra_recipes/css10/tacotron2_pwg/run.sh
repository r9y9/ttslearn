#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
COMMON_ROOT=../../../recipes/common
. $COMMON_ROOT/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

stage=0
stop_stage=0

. $COMMON_ROOT/parse_options.sh || exit 1;

dumpdir=dump
dump_org_dir=$dumpdir/${spk}_sr${sample_rate}/org
dump_norm_dir=$dumpdir/${spk}_sr${sample_rate}/norm

vocoder_model=$(basename $parallel_wavegan_config)
vocoder_model=${vocoder_model%.*}

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"
    exit -1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "train/dev/eval split"

    mkdir -p data/
    n=$(( $(wc -l < data/utt_list.txt) -300 ))
    head -n $n data/utt_list.txt > data/train.list
    tail -300 data/utt_list.txt > data/deveval.list
    head -n 200 data/deveval.list > data/dev.list
    tail -n 100 data/deveval.list > data/eval.list
    rm -f data/deveval.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation"
    for s in ${datasets[@]}; do
        xrun python preprocess.py data/$s.list $wav_root $lab_root \
            $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature normalization"
    xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
        $dump_org_dir/$train_set/out_tacotron $dump_org_dir/out_tacotron_scaler.joblib

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
            $dump_org_dir/out_tacotron_scaler.joblib \
            $dump_org_dir/$s/out_tacotron/ \
            $dump_norm_dir/$s/out_tacotron/ --n_jobs $n_jobs
        # 波形データは手動でコピー
        find $dump_org_dir/$s/out_tacotron/ -name "*-wave.npy" -exec cp "{}" $dump_norm_dir/$s/out_tacotron \;
        # 韻律記号付き音素列は手動でコピー
        rm -rf $dump_norm_dir/$s/in_tacotron
        cp -r $dump_org_dir/$s/in_tacotron $dump_norm_dir/$s/
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training Tacotron"
    xrun python train_tacotron.py model=$acoustic_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_tacotron/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_tacotron/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_tacotron/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_tacotron/ \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.max_train_steps=$tacotron_train_max_train_steps \
        data.batch_size=$tacotron_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training Parallel WaveGAN"
    xrun parallel-wavegan-train --config $parallel_wavegan_config \
        --train-dumpdir $dump_norm_dir/$train_set/out_tacotron \
        --dev-dumpdir $dump_norm_dir/$dev_set/out_tacotron/ \
        --outdir $expdir/$vocoder_model
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis waveforms by parallel_wavegan"
    if [ -z "${vocoder_eval_checkpoint}" ]; then
        vocoder_eval_checkpoint="$(ls -dt "${expdir}/${vocoder_model}"/*.pkl | head -1 || true)"
    fi
    outdir="${expdir}/$vocoder_model/wav/$(basename "${vocoder_eval_checkpoint}" .pkl)"
    for s in ${testsets[@]}; do
        xrun parallel-wavegan-decode --dumpdir $dump_norm_dir/$s/out_tacotron/ \
            --checkpoint $vocoder_eval_checkpoint \
            --outdir $outdir
    done
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${acoustic_model}_${vocoder_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
acoustic_model: ${acoustic_model}
vocoder_model: ${vocoder_model}
EOL

    # Acoustic model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
        $dst_dir/acoustic_model.pth
    cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

    # parallel wavegan
    if [ -z "${vocoder_eval_checkpoint}" ]; then
        vocoder_eval_checkpoint="$(ls -dt "$expdir/$vocoder_model"/*.pkl | head -1 || true)"
    fi
    python $COMMON_ROOT/clean_checkpoint_state.py $vocoder_eval_checkpoint \
        $dst_dir/vocoder_model.pth
    cp $expdir/${vocoder_model}/config.yml $dst_dir/vocoder_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
 fi