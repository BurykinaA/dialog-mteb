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

# bash /Users/alina_burykina/git_repos/dialog-mteb/evaluate/scripts/tod_eval.sh \
#   /abs/path/to/your_model_checkpoint_or_hf_name \
#   /abs/path/to/data_root \
#   /abs/path/to/output_root

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
# 1) Intent recognition (OOS) — Acc(all), Acc(in), Acc(out), Recall(out)
#    Few-shot: 1-shot, 10-shot; Full: -1 (full)
# ---------------------------
for shots in 1 10 -1; do
  tag="full"; [ "$shots" != "-1" ] && tag="${shots}-shot"
  python evaluate/run_finetune.py \
    --data_dir "${DATA_DIR}/intent/oos" \
    --model_type "${MODEL_DIR}" \
    --TASK oos \
    --output_dir "${OUT_DIR}/oos/${RUN_ID}/${tag}" \
    --bert_lr 3e-5 \
    --epoch 200 \
    --max_seq_length 64 \
    --per_gpu_batch_size 16 \
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
    --epoch 30 \
    --max_seq_length 256 \
    --per_gpu_batch_size 16 \
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
      --epoch 30 \
      --max_seq_length 128 \
      --per_gpu_batch_size 16 \
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
      --epoch 5 \
      --max_seq_length 128 \
      --max_resp_length 32 \
      --per_gpu_batch_size 32 \
      --data_ratio "${pct}"
  done
done

echo "All TOD evaluations finished."

