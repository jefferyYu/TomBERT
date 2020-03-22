#!/usr/bin/env bash
for i in 'twitter' 'twitter2015' # 'twitter' 'twitter2015' --bertlayer
do
    echo ${i}
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=6 python run_classifier.py --data_dir ./absa_data/${i} \
    --task_name ${i} --output_dir ./output/${i}_noBL_output/ --bert_model bert-base-uncased --do_train --do_eval \
    --train_batch_size 32
done
