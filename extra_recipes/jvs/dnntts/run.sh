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
    echo "JVS corpus: https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus"
    echo "After downloading the corpus, please run audio.bash from https://github.com/r9y9/jvs_r9y9"
    echo "to remove some wrong/missing-label utterances."
    echo "Please make sure to have JVS corpus in 'db_root' in config.yaml."
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "train/dev/eval split"

    # make utt_list.txt
    mkdir -p data/$spk
    find $db_root/$spk/parallel100/wav24kHz16bit/ -type f -name "*.wav" -exec basename {} \; | sed -e 's/.wav//' | sort > data/$spk/utt_list.txt

    head -n 90 data/$spk/utt_list.txt > data/$spk/train.list
    tail -10 data/$spk/utt_list.txt > data/$spk/deveval.list
    head -n 5 data/$spk/deveval.list > data/$spk/dev.list
    tail -n 5 data/$spk/deveval.list > data/$spk/eval.list
    rm -f data/$spk/deveval.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for duration model"
    for s in ${datasets[@]}; do
        xrun python preprocess_duration.py data/$spk/$s.list $db_root/${spk}/parallel100/lab/ful/ $qst_path \
            $dump_org_dir/$s --n_jobs $n_jobs
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Feature generation for acoustic model"

    for s in ${datasets[@]}; do
        xrun python preprocess_acoustic.py data/$spk/$s.list $db_root/$spk/parallel100/wav24kHz16bit \
        $db_root/${spk}/parallel100/lab/ful/ $qst_path $dump_org_dir/$s \
        --sample_rate $sample_rate --n_jobs $n_jobs
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: feature normalization"

    # NOTE: JVSコーパスは話者毎のデータ量が少ないので、JSUTコーパスで計算した統計量をベースに利用する
    for typ in "duration" "acoustic"; do
       for inout in "in" "out"; do
            xrun python $COMMON_ROOT/fit_scaler.py data/$spk/train.list \
                $dump_org_dir/$train_set/${inout}_${typ} \
                $dump_org_dir/${inout}_${typ}_scaler.joblib \
                --external_scaler ../../jsut/dnntts/dump/jsut_sr${sample_rate}/org/${inout}_${typ}_scaler.joblib
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "duration" "acoustic"; do
            for inout in "in" "out"; do
                xrun python $COMMON_ROOT/preprocess_normalize.py data/$spk/$s.list \
                    $dump_org_dir/${inout}_${typ}_scaler.joblib \
                    $dump_org_dir/$s/${inout}_${typ}/ \
                    $dump_norm_dir/$s/${inout}_${typ}/ --n_jobs $n_jobs
            done
        done
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training duration model"
    if [ -z ${pretrained_duration_checkpoint} ]; then
        pretrained_duration_checkpoint=$PWD/../../jsut/dnntts/exp/jsut_sr${sample_rate}/${duration_model}/${duration_eval_checkpoint}
        if [ ! -e $pretrained_duration_checkpoint ]; then
            echo "Please first train a duration model for JSUT corpus!"
            echo "Expected model path: $pretrained_duration_checkpoint"
            exit 1
        fi
    fi

    xrun python train_dnntts.py model=$duration_model tqdm=$tqdm \
        data.train.utt_list=data/$spk/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_duration/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_duration/ \
        data.dev.utt_list=data/$spk/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_duration/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_duration/ \
        train.out_dir=$expdir/${duration_model} \
        train.log_dir=tensorboard/${expname}_${duration_model} \
        train.nepochs=$dnntts_train_nepochs \
        data.batch_size=$dnntts_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        train.pretrained.checkpoint=$pretrained_duration_checkpoint
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Training acoustic model"
    if [ -z ${pretrained_acoustic_checkpoint} ]; then
        pretrained_acoustic_checkpoint=$PWD/../../jsut/dnntts/exp/jsut_sr${sample_rate}/${acoustic_model}/${acoustic_eval_checkpoint}
        if [ ! -e $pretrained_acoustic_checkpoint ]; then
            echo "Please first train a acoustic model for JSUT corpus!"
            echo "Expected model path: $pretrained_acoustic_checkpoint"
            exit 1
        fi
    fi

    xrun python train_dnntts.py model=$acoustic_model tqdm=$tqdm \
        data.train.utt_list=data/$spk/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_acoustic/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_acoustic/ \
        data.dev.utt_list=data/$spk/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_acoustic/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_acoustic/ \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.nepochs=$dnntts_train_nepochs \
        data.batch_size=$dnntts_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic \
        train.pretrained.checkpoint=$pretrained_acoustic_checkpoint
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$spk/$s.list tqdm=$tqdm \
            in_dir=$db_root/${spk}/parallel100/lab/ful/ \
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