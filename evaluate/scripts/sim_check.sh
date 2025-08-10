#!/bin/bash

#evaluate/scripts/sim_check.sh

MODEL_DIR='.'
model_name='checkpoint-epoch-130'

# Optional: set to a Hugging Face Hub model id to evaluate a remote model
# You can set it here or export an env var: export HF_MODEL_ID='AndrewZeng/futuretod-base-v1.0'
HF_MODEL_ID=${HF_MODEL_ID:-''}

DATA_DIR='data/_downstream_data'
OUTPUT_DIR='metrics_short_fuhuretod_130epohs'

# Determine what to evaluate: local epochs or a HF model id
if [ -n "$HF_MODEL_ID" ]; then
    # Evaluate the single HF model id
    epochs="$HF_MODEL_ID"
else
    # Find all epoch directories (modify pattern if needed)
    if [ -d "${MODEL_DIR}/${model_name}" ]; then
        epochs=$(find "${MODEL_DIR}/${model_name}" -type d -name "*" | sort -t '_' -k 2 -n)
        # If no sub-epochs found, evaluate just the model directory
        if [ -z "$epochs" ]; then
            epochs="${MODEL_DIR}/${model_name}"
        fi
    else
        # If the directory does not exist, treat model_name as a single identifier (e.g., a HF id)
        epochs="$model_name"
    fi
fi

# Name to use for output paths (avoid slashes for HF ids)
MODEL_OUTPUT_NAME="$model_name"
if [ -n "$HF_MODEL_ID" ]; then
    MODEL_OUTPUT_NAME="${HF_MODEL_ID//\//_}"
fi

for epoch_dir in $epochs; do
    epoch_name=$(basename "$epoch_dir")
    
    echo "Evaluating epoch: $epoch_name"
    
    python evaluate/run_similarity.py \
        --model_dir "$epoch_dir" \
        --data_root_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/intent_sim/${MODEL_OUTPUT_NAME}/${epoch_name}" \
        --TASK intent \
        --num_runs 10 \
        --max_seq_length 64
    
    python evaluate/run_similarity.py \
        --model_dir "$epoch_dir" \
        --data_root_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/oos_sim/${MODEL_OUTPUT_NAME}/${epoch_name}" \
        --TASK oos \
        --num_runs 10 \
        --max_seq_length 64

    python evaluate/run_similarity.py \
        --model_dir "$epoch_dir" \
        --data_root_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/rs_ubuntu_sim/${MODEL_OUTPUT_NAME}/${epoch_name}" \
        --TASK rs_ubuntu \
        --max_seq_length 128

    python evaluate/run_similarity.py \
        --model_dir "$epoch_dir" \
        --data_root_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/rs_sim/${MODEL_OUTPUT_NAME}/${epoch_name}" \
        --TASK rs_amazon \
        --max_seq_length 128

done
