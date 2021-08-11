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
COMMON_ROOT=../common
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
    if [ ! -d downloads/jsut_ver1 ]; then
        cd downloads
        curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
        unzip -o jsut_ver1.1.zip
        cd -
    fi
    if [ ! -d downloads/jsut-lab ]; then
        cd downloads
        curl -LO https://github.com/r9y9/jsut-lab/archive/v0.1.1.zip
        unzip -o v0.1.1.zip
        cp -r jsut-lab-0.1.1/basic5000/lab jsut_ver1.1/basic5000/
        cd -
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "train/dev/eval split"
    head -n 3133 data/utt_list.txt > data/train.list
    tail -300 data/utt_list.txt > data/deveval.list
    head -n 200 data/deveval.list > data/dev.list
    tail -n 100 data/deveval.list > data/eval.list
    rm -f data/deveval.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for duration model"
    for s in ${datasets[@]}; do
        xrun python preprocess_duration.py data/$s.list $db_root/lab/ $qst_path \
            $dump_org_dir/$s --n_jobs $n_jobs
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Feature generation for log-F0 prediction model"
    for s in ${datasets[@]}; do
        xrun python preprocess_logf0.py data/$s.list $db_root/wav/ $db_root/lab/ \
            $qst_path $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Feature generation for WaveNet"
    for s in ${datasets[@]}; do
        xrun python preprocess_wavenet.py data/$s.list $db_root/wav/ $db_root/lab/ \
            $qst_path $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate --mu $mu
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: feature normalization"
    for typ in "duration" "logf0" "wavenet"; do
       for inout in "in" "out"; do
            if [ $typ == "wavenet" ] && [ $inout == "out" ]; then
                # 波形は正規化しないため、スキップ
                continue
            fi
            xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
                $dump_org_dir/$train_set/${inout}_${typ} \
                $dump_org_dir/${inout}_${typ}_scaler.joblib
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "duration" "logf0" "wavenet"; do
            for inout in "in" "out"; do
                if [ $typ == "wavenet" ] && [ $inout == "out" ]; then
                    continue
                fi
                xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
                    $dump_org_dir/${inout}_${typ}_scaler.joblib \
                    $dump_org_dir/$s/${inout}_${typ}/ \
                    $dump_norm_dir/$s/${inout}_${typ}/ --n_jobs $n_jobs
            done
        done
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Training duration model"
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

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Training log-f0 prediction model"
    xrun python train_dnntts.py model=$logf0_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_logf0/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_logf0/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_logf0/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_logf0/ \
        train.out_dir=$expdir/${logf0_model} \
        train.log_dir=tensorboard/${expname}_${logf0_model} \
        train.nepochs=$dnntts_train_nepochs \
        data.batch_size=$dnntts_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Training WaveNet"
    xrun python train_wavenet.py model=${wavenet_model} tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_wavenet/ \
        data.train.out_dir=$dump_org_dir/$train_set/out_wavenet/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_wavenet/ \
        data.dev.out_dir=$dump_org_dir/$dev_set/out_wavenet/ \
        train.out_dir=$expdir/${wavenet_model} \
        train.log_dir=tensorboard/${expname}_${wavenet_model} \
        train.max_train_steps=$wavenet_train_max_train_steps \
        data.batch_size=$wavenet_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Synthesis waveforms"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=$db_root/lab/ \
            out_dir=$expdir/synthesis_${duration_model}_${logf0_model}_${wavenet_model}/$s \
            sample_rate=$sample_rate qst_path=$qst_path \
            duration.checkpoint=$expdir/${duration_model}/$duration_eval_checkpoint \
            duration.in_scaler_path=$dump_norm_dir/in_duration_scaler.joblib \
            duration.out_scaler_path=$dump_norm_dir/out_duration_scaler.joblib \
            duration.model_yaml=$expdir/${duration_model}/model.yaml \
            logf0.checkpoint=$expdir/${logf0_model}/$logf0_eval_checkpoint \
            logf0.in_scaler_path=$dump_norm_dir/in_logf0_scaler.joblib \
            logf0.out_scaler_path=$dump_norm_dir/out_logf0_scaler.joblib \
            logf0.model_yaml=$expdir/${logf0_model}/model.yaml \
            wavenet.checkpoint=$expdir/${wavenet_model}/$wavenet_eval_checkpoint \
            wavenet.in_scaler_path=$dump_norm_dir/in_wavenet_scaler.joblib \
            wavenet.model_yaml=$expdir/${wavenet_model}/model.yaml \
            reverse=$reverse num_eval_utts=$num_eval_utts
    done
fi

if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ]; then
    echo "Create tar.gz to share experiments"
    rm -rf tmp/exp
    mkdir -p tmp/exp/$expname
    for model in $duration_model $logf0_model $wavenet_model; do
        rsync -avr $expdir/$model tmp/exp/$expname/ --exclude "epoch*.pth"
    done
    rsync -avr $expdir/synthesis_${duration_model}_${logf0_model}_${wavenet_model} tmp/exp/$expname/ --exclude "epoch*.pth"
    cd tmp
    tar czvf wavenet_exp.tar.gz exp/
    mv wavenet_exp.tar.gz ..
    cd -
    rm -rf tmp
    echo "Please check wavenet_exp.tar.gz"
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${duration_model}_${logf0_model}_${wavenet_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
mu: ${mu}
duration_model: ${duration_model}
logf0_model: ${logf0_model}
wavenet_model: ${wavenet_model}
EOL

    # Hed file
    cp -v $qst_path $dst_dir/qst.hed

    # Stats
    for typ in "duration" "logf0" "wavenet"; do
        for inout in "in" "out"; do
            if [ $typ == "wavenet" ] && [ $inout == "out" ]; then
                continue
            fi
            python $COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/${inout}_${typ}_scaler.joblib $dst_dir
        done
    done

    # Duration model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${duration_model}/$duration_eval_checkpoint \
        $dst_dir/duration_model.pth
    cp $expdir/${duration_model}/model.yaml $dst_dir/duration_model.yaml

    # Log f0 model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${logf0_model}/$logf0_eval_checkpoint \
        $dst_dir/logf0_model.pth
    cp $expdir/${logf0_model}/model.yaml $dst_dir/logf0_model.yaml

    # WaveNet
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${wavenet_model}/$wavenet_eval_checkpoint \
        $dst_dir/wavenet_model.pth
    cp $expdir/${wavenet_model}/model.yaml $dst_dir/wavenet_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
 fi