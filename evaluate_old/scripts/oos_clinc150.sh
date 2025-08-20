export CUDA_VISIBLE_DEVICES=0

# MODEL_DIR='AndrewZeng/futuretod-base-v1.0'
# DATA_DIR='/home/coder/project/data/_downstream_data'
# OUTPUT_DIR='down_stream_old/metrics_futuretod'

MODEL_DIR='jasper_model_1/checkpoint-epoch-48'
DATA_DIR='/home/coder/project/data/_downstream_data'
OUTPUT_DIR='down_stream_old/metrics_jasper1_48'

# Example usage
# chmod +x evaluate_old/scripts/oos_clinc150.sh
# ./evaluate_old/scripts/oos_clinc150.sh "bert-base-uncased" "/path/to/your/data" "/path/to/output"


# Out-of-scope detection on Clinc150 5
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