export DATA_DIR=/home/coder/project/data/data_todbert_pretrain
export FILE_NAME="dse_training.tsv"
export OUTPUT_DIR=/home/coder/project/metrics
export MODEL_TYPE=bertbase #dunzhang/stella_en_400M_v5 # choose from [bertbase, bertlarge, robertabase, robertalarge, distilbertbase]
cd pretrain

# [None, bf16, fp16]
# 1024
# epochs 15
# max_length 32

# =====================================================================
# LEARNING TYPE CONTROL:
# The system uses the same data fields (text1/text2) for both contrastive 
# learning and distillation. Control the learning type with --mode:
#
# 1. For COMBINED learning (contrastive + distillation):
#    --mode combined
#    - The system will use text1 as context and text2 as both response and future
#    - Adjust --distill_weight to control the balance between losses
#
# 2. For CONTRASTIVE learning only:
#    --mode contrastive
#    - The system will only compute contrastive loss between text1 and text2
#
# 3. For DISTILLATION learning only:
#    --mode distill
#    - The system will only compute distillation loss
# =====================================================================

python main.py \
    --resdir ${OUTPUT_DIR} \
    --datapath ${DATA_DIR} \
    --dataname dse_training.tsv \
    --mode combined \
    --bert ${MODEL_TYPE} \
    --contrast_type HardNeg \
    --lr 3e-06 \
    --lr_scale 100 \
    --batch_size 512 \
    --max_length 32 \
    --temperature 0.05 \
    --epochs 15 \
    --mixed_precision None --max_iter 10000000 \
    --logging_step 400 \
    --feat_dim 128 \
    --num_turn 1 \
    --seed 1 \
    --save_model_every_epoch \
    --update_teacher_interval 4 \
    --distill_weight 1.0
