#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash evaluate/scripts/ft.sh /abs/path/to/checkpoint-epoch-XX /abs/path/to/data_root /abs/path/to/out_root
#
# Example:
#   bash evaluate/scripts/ft.sh /home/coder/project/jasper_model_checkpointS/checkpoint-epoch-45 /home/coder/project/data/_downstream_data /home/coder/project/down_stream/metrics_jasper_45

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${1:?Pass MODEL_DIR (checkpoint path) as arg 1}"
DATA_DIR="${2:?Pass DATA_DIR (root of downstream datasets) as arg 2}"
OUTPUT_DIR="${3:?Pass OUTPUT_DIR as arg 3}"

# Sanity check
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "config.json not found in ${MODEL_DIR}"; exit 1;
fi
if [ ! -f "${MODEL_DIR}/model.safetensors" ] && [ ! -f "${MODEL_DIR}/pytorch_model.bin" ]; then
  echo "No model weights found in ${MODEL_DIR}"; exit 1;
fi

RUN_ID="$(basename "${MODEL_DIR}")"

echo "Using checkpoint: ${MODEL_DIR}"
echo "Run id: ${RUN_ID}"
echo "Data root: ${DATA_DIR}"
echo "Output root: ${OUTPUT_DIR}"

# ---------------------------
# Intent classification (seq)
# ---------------------------
for dataset in bank77 hwu64 clinc150 snips; do
  for data_ratio in 1 5; do
    python evaluate/run_finetune.py \
      --data_dir "${DATA_DIR}/intent/${dataset}" \
      --model_type "${MODEL_DIR}" \
      --TASK seq \
      --output_dir "${OUTPUT_DIR}/intent_ft/${RUN_ID}/${dataset}/${data_ratio}" \
      --bert_lr 2e-5 \
      --epoch 50 \
      --max_seq_length 64 \
      --per_gpu_batch_size 64 \
      --gradient_accumulation_steps 1 \
      --data_ratio "${data_ratio}" \
      --num_runs 10 \
      --patience 5 \
      --classification_pooling average \
      --early_stop_type metric
  done
done

# ---------------------------
# Response selection (rs)
# ---------------------------
for data_ratio in 500 1000; do
  python evaluate/run_finetune.py \
    --data_dir "${DATA_DIR}/rs/amazonqa" \
    --model_type "${MODEL_DIR}" \
    --TASK rs \
    --output_dir "${OUTPUT_DIR}/rs_ft/${RUN_ID}/amazonqa/${data_ratio}" \
    --bert_lr 2e-5 \
    --epoch 50 \
    --max_seq_length 128 \
    --per_gpu_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --data_ratio "${data_ratio}" \
    --num_runs 2 \
    --patience 3 \
    --eval_steps 50 \
    --concatenate
done

# ---------------------------------
# Dialogue action prediction (da)
# ---------------------------------
for dataset in dstc2 sim_joint; do
  for data_ratio in 10 20; do
    python evaluate/run_finetune.py \
      --data_dir "${DATA_DIR}/da/${dataset}" \
      --model_type "${MODEL_DIR}" \
      --TASK da \
      --output_dir "${OUTPUT_DIR}/da_concat_ft/${RUN_ID}/${dataset}/${data_ratio}" \
      --bert_lr 5e-5 \
      --epoch 100 \
      --max_seq_length 32 \
      --per_gpu_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --data_ratio "${data_ratio}" \
      --num_runs 5 \
      --patience 3 \
      --eval_steps 30 \
      --num_turn 1 \
      --concatenate \
      --save_model \
      --early_stop_type metric
  done
done

echo "All fine-tuning runs finished."