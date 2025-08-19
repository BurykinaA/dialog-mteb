#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash evaluate/scripts/tod_eval.sh /abs/path/to/checkpoint /abs/path/to/data_root /abs/path/to/output_root
# Data layout expected under ${DATA_ROOT}:
#  - intent/oos/{seq_train.txt, seq_val.txt, seq_test.txt}
#  - dst/mwoz21/{train.json, dev.json, test.json, ontology.json}
#  - da/mwoz/{train.json, dev.json, test.json}
#  - da/dstc2/{train.json, dev.json, test.json}
#  - rs/mwoz/{train.txt, dev.txt, test.txt}
#  - rs/dstc2/{train.txt, dev.txt, test.txt}

# bash /home/coder/project/evaluate/scripts/tod_eval.sh \
#   /home/coder/project/jasper_model_checkpointS/checkpoint-epoch-45 \
#   /home/coder/project/data/_downstream_data \
#   /home/coder/project/down_stream/tod/metrics_jasper_45

# bash /home/coder/project/evaluate/scripts/tod_eval.sh \
#   aws-ai/dse-bert-base \
#   /home/coder/project/data/_downstream_data \
#   /home/coder/project/down_stream/tod/metrics_dse


# bash /home/coder/project/evaluate/scripts/tod_eval.sh \
#   TODBERT/TOD-BERT-MLM-V1 \
#   /home/coder/project/data/_downstream_data \
#   /home/coder/project/down_stream/tod/metrics_tod_bert

# bash /home/coder/project/evaluate/scripts/tod_eval.sh \
#   google-bert/bert-base-uncased \
#   /home/coder/project/data/_downstream_data \
#   /home/coder/project/down_stream/tod/metrics_bert



export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

MODEL_DIR="${1:?Pass MODEL_DIR as arg 1}"
DATA_DIR="${2:?Pass DATA_DIR as arg 2}"
OUT_DIR="${3:?Pass OUT_DIR as arg 3}"

RUN_ID="$(basename "${MODEL_DIR}")"
echo "Model: ${MODEL_DIR}"
echo "Data: ${DATA_DIR}"
echo "Out: ${OUT_DIR}"

mkdir -p "${OUT_DIR}"

# ---------------------------
# 1) Intent recognition (OOS on CLINC150_ALL)
#    Few-shot: detect 1- and 5-shot if present; always run full otherwise
# ---------------------------
base_dir="${DATA_DIR}/intent/clinc150_all"
if [ ! -f "${base_dir}/seq_test.txt" ]; then
  echo "ERROR: CLINC150_ALL not found at ${base_dir}/seq_test.txt"; exit 1;
fi

shots_to_run=("-1") # full
if [ -d "${base_dir}/1/0" ] && [ -f "${base_dir}/1/0/seq_train.txt" ]; then
  shots_to_run=("1" "${shots_to_run[@]}")
fi
if [ -d "${base_dir}/5/0" ] && [ -f "${base_dir}/5/0/seq_train.txt" ]; then
  shots_to_run=("5" "${shots_to_run[@]}")
fi

for shots in "${shots_to_run[@]}"; do
  tag="full"; [ "$shots" != "-1" ] && tag="${shots}-shot"
  python evaluate/run_finetune.py \
    --data_dir "${base_dir}" \
    --model_type "${MODEL_DIR}" \
    --TASK oos \
    --output_dir "${OUT_DIR}/oos/${RUN_ID}/${tag}" \
    --bert_lr 3e-5 \
    --epoch 400 \
    --max_seq_length 64 \
    --per_gpu_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --data_ratio "${shots}" \
    --num_runs 1 \
    --patience 20 \
    --classification_pooling cls \
    --early_stop_type loss
done

# ---------------------------
# 2) Dialogue State Tracking (MWOZ 2.1)
#    Few-shot percents: 1,5,10,25 and full (-1)
#    Metrics: Joint Acc, Slot Acc
# ---------------------------
for pct in 1 5 10 25 -1; do
  tag="full"; [ "$pct" != "-1" ] && tag="${pct}pct"
    python evaluate/run_dst_mwoz.py \
    --data_dir "${DATA_DIR}/dst/mwoz21" \
    --model_type "${MODEL_DIR}" \
    --output_dir "${OUT_DIR}/dst/${RUN_ID}/mwoz21/${tag}" \
    --bert_lr 3e-5 \
    --epoch 100 \
    --max_seq_length 256 \
    --per_gpu_batch_size 64 \
    --eval_steps 200 \
    --patience 5 \
    --data_ratio "${pct}"
done

# ---------------------------
# 3) Dialogue Act Prediction (MWOZ, DSTC2), CLS pooling
#    Few-shot percents: 1,10 and full (-1)
#    Metrics: micro-F1, macro-F1
# ---------------------------
for dataset in mwoz dstc2; do
  for pct in 1 10 -1; do
    tag="full"; [ "$pct" != "-1" ] && tag="${pct}pct"
    python evaluate/run_da_cls.py \
      --data_dir "${DATA_DIR}/da/${dataset}" \
      --model_type "${MODEL_DIR}" \
      --output_dir "${OUT_DIR}/da/${RUN_ID}/${dataset}/${tag}" \
      --bert_lr 5e-5 \
      --epoch 100 \
      --max_seq_length 128 \
      --per_gpu_batch_size 128 \
      --eval_steps 200 \
      --patience 5 \
      --data_ratio "${pct}"
  done
done

# ---------------------------
# 4) Response Selection (MWOZ, DSTC2), k-to-100
#    Few-shot percents: 1,10 and full (-1)
#    Metrics: 1-to-100, 3-to-100
# ---------------------------
for dataset in mwoz dstc2; do
  for pct in 1 10 -1; do
    tag="full"; [ "$pct" != "-1" ] && tag="${pct}pct"
    python evaluate/run_response_selection_100.py \
      --data_dir "${DATA_DIR}/rs/${dataset}" \
      --model_type "${MODEL_DIR}" \
      --output_dir "${OUT_DIR}/rs/${RUN_ID}/${dataset}/${tag}" \
      --bert_lr 2e-5 \
      --epoch 50 \
      --max_seq_length 128 \
      --max_resp_length 32 \
      --per_gpu_batch_size 256 \
      --eval_batch_size 1024 \
      --data_ratio "${pct}"
  done
done

echo "All TOD evaluations finished."

