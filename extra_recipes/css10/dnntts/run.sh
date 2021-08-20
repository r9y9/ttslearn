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

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"
    mkdir -p downloads

    echo "Please download data manually!"
    exit 1
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
    echo "stage 1: Feature generation for duration model"
    for s in ${datasets[@]}; do
        xrun python preprocess_duration.py data/$s.list $lab_root/lab/ $qst_path \
            $dump_org_dir/$s --n_jobs $n_jobs
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Feature generation for acoustic model"
    for s in ${datasets[@]}; do
        xrun python preprocess_acoustic.py data/$s.list $db_root/$spk/ $lab_root/lab/ \
            $qst_path $dump_org_dir/$s --sample_rate $sample_rate \
            --n_jobs $n_jobs
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: feature normalization"
    for typ in "duration" "acoustic"; do
       for inout in "in" "out"; do
            xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
                $dump_org_dir/$train_set/${inout}_${typ} \
                $dump_org_dir/${inout}_${typ}_scaler.joblib
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "duration" "acoustic"; do
            for inout in "in" "out"; do
                xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
                    $dump_org_dir/${inout}_${typ}_scaler.joblib \
                    $dump_org_dir/$s/${inout}_${typ}/ \
                    $dump_norm_dir/$s/${inout}_${typ}/ --n_jobs $n_jobs
            done
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training duration model"
    xrun python train_dnntts.py model=$duration_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_duration/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_duration/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_duration/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_duration/ \
        train.out_dir=$expdir/${duration_model} \
        train.log_dir=tensorboard/${expname}_${duration_model} \
        train.nepochs=$dnntts_train_nepochs \
        data.batch_size=$dnntts_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Training acoustic model"
    xrun python train_dnntts.py model=$acoustic_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.nepochs=$dnntts_train_nepochs \
        data.batch_size=$dnntts_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=$lab_root/lab/ \
            out_dir=$expdir/synthesis_${duration_model}_${acoustic_model}/$s \
            sample_rate=$sample_rate qst_path=$qst_path \
            duration.checkpoint=$expdir/${duration_model}/$duration_eval_checkpoint \
            duration.in_scaler_path=$dump_norm_dir/in_duration_scaler.joblib \
            duration.out_scaler_path=$dump_norm_dir/out_duration_scaler.joblib \
            duration.model_yaml=$expdir/${duration_model}/model.yaml \
            acoustic.checkpoint=$expdir/${acoustic_model}/$acoustic_eval_checkpoint \
            acoustic.in_scaler_path=$dump_norm_dir/in_acoustic_scaler.joblib \
            acoustic.out_scaler_path=$dump_norm_dir/out_acoustic_scaler.joblib \
            acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
            post_filter=$post_filter reverse=$reverse num_eval_utts=$num_eval_utts
    done
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${duration_model}_${acoustic_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
duration_model: ${duration_model}
acoustic_model: ${acoustic_model}
EOL

    # Hed file
    cp -v $qst_path $dst_dir/qst.hed

    # Stats
    for typ in "duration" "acoustic"; do
        for inout in "in" "out"; do
            python $COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/${inout}_${typ}_scaler.joblib $dst_dir
        done
    done

    # Duration model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${duration_model}/$duration_eval_checkpoint \
        $dst_dir/duration_model.pth
    cp $expdir/${duration_model}/model.yaml $dst_dir/duration_model.yaml

    # Acoustic model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
        $dst_dir/acoustic_model.pth
    cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
 fi