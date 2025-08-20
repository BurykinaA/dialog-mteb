export CUDA_VISIBLE_DEVICES=0

# Set default values if no arguments provided
MODEL_DIR=${1:-"AndrewZeng/futuretod-base-v1.0"}
DATA_DIR=${2:-"/home/coder/project/data/_downstream_data"}
OUTPUT_DIR=${3:-"down_stream_old/metrics_futuretod"}

echo "Using MODEL_DIR: $MODEL_DIR"
echo "Using DATA_DIR: $DATA_DIR"
echo "Using OUTPUT_DIR: $OUTPUT_DIR"

# Example usage
# chmod +x evaluate_old/scripts/oos_clinc150.sh
# ./evaluate_old/scripts/oos_clinc150.sh "bert-base-uncased" "/path/to/your/data" "/path/to/output"

# Out-of-scope detection on Clinc150
for data_ratio in 1 5
do
    python evaluate_old/run_finetune.py \
        --data_dir ${DATA_DIR}/intent/clinc150_all \
        --model_type ${MODEL_DIR} \
        --TASK oos \
        --output_dir ${OUTPUT_DIR}/oos_ft/${MODEL_DIR}/clinc150/${data_ratio} \
        --bert_lr 2e-5 \
        --epoch 50 \
        --max_seq_length 64 \
        --per_gpu_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --data_ratio ${data_ratio} \
        --num_runs 10 \
        --patience 5 \
        --eval_steps 50 \
        --classification_pooling average \
        --early_stop_type metric
done 