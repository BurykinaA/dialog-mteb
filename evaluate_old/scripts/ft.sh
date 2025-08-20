export CUDA_VISIBLE_DEVICES=0

# MODEL_DIR='jasper_model_1/checkpoint-epoch-48'
# DATA_DIR='/home/coder/project/data/_downstream_data'
# OUTPUT_DIR='down_stream_old/metrics_jasper1_48'


MODEL_DIR='AndrewZeng/futuretod-base-v1.0'
DATA_DIR='/home/coder/project/data/_downstream_data'
OUTPUT_DIR='down_stream_old/metrics_futuretod'


#3e-5
#intent classification
# for dataset in bank77 hwu64  clinc150 snips
# do  
#     for data_ratio in 1 5
#     do
#         python evaluate/run_finetune.py \
#             --data_dir ${DATA_DIR}/intent/${dataset} \
#             --model_type ${MODEL_DIR} \
#             --TASK seq \
#             --output_dir ${OUTPUT_DIR}/intent_ft/${MODEL_DIR}/${dataset}/${data_ratio} \
#             --bert_lr 1e-3 \
#             --epoch 150 \
#             --max_seq_length 64 \
#             --per_gpu_batch_size 64 \
#             --gradient_accumulation_steps 1 \
#             --data_ratio ${data_ratio} \
#             --num_runs 1 \
#             --patience 5 \
#             --classification_pooling average \
#             --early_stop_type metric
#     done
# done




#response selection 500
for data_ratio in 500 1000 
do
    python evaluate/run_finetune.py \
        --data_dir ${DATA_DIR}/rs/amazonqa \
        --model_type ${MODEL_DIR} \
        --TASK rs \
        --output_dir ${OUTPUT_DIR}/rs_ft/${MODEL_DIR}/amazonqa/${data_ratio} \
        --bert_lr 5e-4 \
        --epoch 50 \
        --max_seq_length 128 \
        --per_gpu_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --data_ratio ${data_ratio} \
        --num_runs 1 \
        --patience 3 \
        --eval_steps 50 \
        --concatenate
done

# dialogue action prediction
# for dataset in dstc2 sim_joint
# do
#     for data_ratio in 10 20
#     do
#         python evaluate/run_finetune.py \
#             --data_dir ${DATA_DIR}/da/${dataset} \
#             --model_type ${MODEL_DIR} \
#             --TASK da \
#             --output_dir ${OUTPUT_DIR}/da_concat_ft/${MODEL_DIR}/${dataset}/${data_ratio} \
#             --bert_lr 5e-5 \
#             --epoch 100 \
#             --max_seq_length 32 \
#             --per_gpu_batch_size 16 \
#             --gradient_accumulation_steps 1 \
#             --data_ratio ${data_ratio} \
#             --num_runs 5 \
#             --patience 3 \
#             --eval_steps 30 \
#             --num_turn 1 \
#             --concatenate \
#             --save_model \
#             --early_stop_type metric
#     done
# done
