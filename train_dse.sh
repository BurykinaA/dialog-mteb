export DATA_DIR=/home/coder/project/data/data_todbert_pretrain
export FILE_NAME="dse_training.tsv"
export OUTPUT_DIR=/home/coder/project/metrics
export MODEL_TYPE=dunzhang/stella_en_400M_v5 # choose from [bertbase, bertlarge, robertabase, robertalarge, distilbertbase]
cd pretrain

# [None, bf16, fp16]
# 1024
# epochs 15
# max_length 32

python main.py \
    --resdir ${OUTPUT_DIR} \
    --datapath ${DATA_DIR} \
    --dataname dse_training.tsv \
    --mode contrastive \
    --bert ${MODEL_TYPE} \
    --contrast_type HardNeg \
    --lr 3e-06 \
    --lr_scale 100 \
    --batch_size 1024 \
    --max_length 32 \
    --temperature 0.05 \
    --epochs 15 \
    --mixed_precision None --max_iter 10000000 \
    --logging_step 400 \
    --feat_dim 128 \
    --num_turn 1 \
    --seed 1 \
    --save_model_every_epoch 
