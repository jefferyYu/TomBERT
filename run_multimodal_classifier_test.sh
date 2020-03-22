#!/usr/bin/env bash
for i in 'twitter' 'twitter2015' # 'twitter'
do
    echo ${i}
    for k in 'TomBert' # 'TomBert' 'MBert' 'TomBertNoPooling' 'MBertNoPooling' 'ResBert'
    do
        echo ${k}
        for j in 'first' # 'first' 'cls' 'both'
        do
            echo ${j}
            PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=6 python run_multimodal_classifier.py --data_dir \
            ./absa_data/${i} --task_name ${i} --output_dir ./output/${i}_${k}_${j}_mm_output/ \
            --bert_model bert-base-uncased --do_eval --mm_model ${k} --pooling ${j}
        done
    done
done