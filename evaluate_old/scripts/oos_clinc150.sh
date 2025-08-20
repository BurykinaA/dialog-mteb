export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set the model and data paths
MODEL_DIR='jasper_model_1/checkpoint-epoch-48'
DATA_DIR='/home/coder/project/data/_downstream_data'
OUTPUT_DIR='down_stream_old/metrics_jasper1_48'

echo "Using MODEL_DIR: $MODEL_DIR"
echo "Using DATA_DIR: $DATA_DIR"
echo "Using OUTPUT_DIR: $OUTPUT_DIR"

# Out-of-scope detection on Clinc150 - note: should be binary classification
for data_ratio in 1 5
do
    python evaluate_old/run_finetune.py \
        --data_dir ${DATA_DIR}/intent/clinc150_all \
        --model_type ${MODEL_DIR} \
        --TASK oos \
        --output_dir ${OUTPUT_DIR}/oos_ft/${MODEL_DIR}/clinc150/${data_ratio} \
        --bert_lr 5e-6 \
        --epoch 50 \
        --max_seq_length 64 \
        --per_gpu_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --data_ratio ${data_ratio} \
        --num_runs 1 \
        --patience 5 \
        --eval_steps 20 \
        --classification_pooling average \
        --early_stop_type metric
done 