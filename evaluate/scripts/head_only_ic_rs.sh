#!/usr/bin/env bash
set -euo pipefail


#chmod +x evaluate/scripts/head_only_ic_rs.sh


# Usage:
#   bash evaluate/scripts/head_only_ic_rs.sh <MODEL_OR_HF_ID> <DATA_ROOT> <OUT_ROOT>


# bash evaluate/scripts/head_only_ic_rs.sh \
#   bert-base-uncased \
#   /home/coder/project/data/_downstream_data \
#   /home/coder/project/down_stream/head_only_metrics

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${1:?Pass MODEL_DIR or HF model id as arg 1}"
DATA_DIR="${2:?Pass DATA_DIR as arg 2}"
OUT_DIR="${3:?Pass OUT_DIR as arg 3}"

RUN_ID="$(basename "${MODEL_DIR}")"
echo "Model: ${MODEL_DIR}"
echo "Data:  ${DATA_DIR}"
echo "Out:   ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# ---------------------------
# Intent classification (head-only fine-tuning)
# Datasets we have: bank77, clinc150, snips, hwu64
# Shots: detect 1 and 5; if not present, skip that shot
# ---------------------------
for dataset in bank77 clinc150 snips hwu64; do
  base="${DATA_DIR}/intent/${dataset}"
  if [ ! -f "${base}/seq_test.txt" ]; then
    echo "Skip ${dataset}: ${base}/seq_test.txt not found"
    continue
  fi

  shots_to_run=()
  [ -f "${base}/1/0/seq_train.txt" ]  && shots_to_run+=("1")
  [ -f "${base}/5/0/seq_train.txt" ]  && shots_to_run+=("5")
  if [ ${#shots_to_run[@]} -eq 0 ]; then
    echo "No few-shot splits for ${dataset}; skipping"
    continue
  fi

  for shots in "${shots_to_run[@]}"; do
    tag="${shots}-shot"
    python evaluate/run_finetune.py \
      --data_dir "${base}" \
      --model_type "${MODEL_DIR}" \
      --TASK seq \
      --output_dir "${OUT_DIR}/intent_head_only/${RUN_ID}/${dataset}/${tag}" \
      --bert_lr 3e-5 \
      --head_lr 1e-3 \
      --scheduler cosine \
      --warmup_ratio 0.1 \
      --epoch 200 \
      --max_seq_length 64 \
      --per_gpu_batch_size 256 \
      --gradient_accumulation_steps 1 \
      --data_ratio "${shots}" \
      --num_runs 1 \
      --patience 20 \
      --classification_pooling cls \
      --early_stop_type loss \
      --freeze_encoder_epochs 1000000 \
      --fp16
  done
done

# ---------------------------
# Zero-shot response selection (AmazonQA, Ubuntu)
# No training; compute Top-1/Top-3/Top-10 with 99 random negatives
# ---------------------------
for dataset in amazonqa ubuntu; do
  base="${DATA_DIR}/rs/${dataset}"
  if [ ! -f "${base}/test.txt" ]; then
    echo "Skip RS ${dataset}: ${base}/test.txt not found"
    continue
  fi
  python evaluate/run_response_selection_100.py \
    --data_dir "${base}" \
    --model_type "${MODEL_DIR}" \
    --output_dir "${OUT_DIR}/rs_zeroshot/${RUN_ID}/${dataset}" \
    --bert_lr 2e-5 \
    --epoch 0 \
    --max_seq_length 128 \
    --max_resp_length 32 \
    --per_gpu_batch_size 256 \
    --eval_batch_size 2048
done

echo "Done: head-only intent (1/5-shot) and zero-shot RS."