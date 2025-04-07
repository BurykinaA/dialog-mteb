#!/bin/bash

#evaluate/scripts/sim_check.sh

MODEL_DIR='.'
model_name='metricscontrastive/stella_en_400M_v5.dse_training.tsv.lr3e-06.lrscale100.bs512.tmp0.05.decay1.seed1.turn1'
DATA_DIR='data/_downstream_data'
OUTPUT_DIR='metrics_check'

# Find all epoch directories (modify pattern if needed)
epochs=$(find ${MODEL_DIR}/${model_name} -type d -name "*" | sort -t '_' -k 2 -n)

# If no epochs found, evaluate just the model directory
if [ -z "$epochs" ]; then
    epochs="${MODEL_DIR}/${model_name}"
fi

for epoch_dir in $epochs; do
    epoch_name=$(basename $epoch_dir)
    
    echo "Evaluating epoch: $epoch_name"
    
    python evaluate/run_similarity.py \
        --model_dir $epoch_dir \
        --data_root_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/intent_sim/${model_name}/${epoch_name} \
        --TASK intent \
        --num_runs 10 \
        --max_seq_length 64
    
    python evaluate/run_similarity.py \
        --model_dir $epoch_dir \
        --data_root_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/oos_sim/${model_name}/${epoch_name} \
        --TASK oos \
        --num_runs 10 \
        --max_seq_length 64

    python evaluate/run_similarity.py \
        --model_dir $epoch_dir \
        --data_root_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/rs_ubuntu_sim/${model_name}/${epoch_name} \
        --TASK rs_ubuntu \
        --max_seq_length 128

    python evaluate/run_similarity.py \
        --model_dir $epoch_dir \
        --data_root_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR}/rs_sim/${model_name}/${epoch_name} \
        --TASK rs_amazon \
        --max_seq_length 128

done