export DATA_DIR=/home/coder/project/data/data_todbert_pretrain
export FILE_NAME="dse_training_short.tsv"
export OUTPUT_DIR=/home/coder/project/short_futuretod_bertbase_pretrain
export MODEL_TYPE=bertbase #dunzhang/stella_en_400M_v5 # choose from [bertbase, bertlarge, robertabase, robertalarge, distilbertbase]
cd pretrain

# Training parameters based on FutureTOD paper:
# - Batch Size: 32
# - Max Input Length: 512
# - Learning Rate: 5e-5
# - MLM Probability (Mask Ratio): 0.15
# - Number of Distillation Layers: 12
# - Teacher Update Frequency: 10 epochs
# - Optimizer: Adam (usually default, ensure main.py uses it)
# - Scheduler: Linear (usually default, ensure main.py uses it)
# - Dropout: 0.2 (This is typically set in the model's config, e.g., when loading BERTModel.from_pretrained)

# =====================================================================
# LEARNING TYPE CONTROL:
# The system uses the same data fields (text1/text2) for both contrastive 
# learning and distillation. Control the learning type with --mode:
#
# 1. For COMBINED learning (FutureTOD + contrastive):
#    --mode combined
#    - Adjust --distill_weight to control the balance. Contrastive params like
#      --contrast_type, --temperature, --feat_dim will be active.
#
# 2. For CONTRASTIVE learning only:
#    --mode contrastive
#
# 3. For DISTILLATION learning only (FutureTOD: L = Ldis + Lmlm):
#    --mode distill
#    - The system will only compute distillation loss
# =====================================================================

python main.py \
    --resdir ${OUTPUT_DIR} \
    --datapath ${DATA_DIR} \
    --dataname dse_training_short.tsv \
    --mode distill \
    --bert ${MODEL_TYPE} \
    --contrast_type HardNeg \
    --lr 5e-5 \
    --lr_scale 100 \
    --batch_size 32 \
    --max_length 512 \
    --temperature 0.05 \
    --epochs 100 \
    --mixed_precision None --max_iter 10000000 \
    --logging_step 400 \
    --feat_dim 128 \
    --num_turn 1 \
    --seed 1 \
    --save_model_every_epoch \
    --update_teacher_interval 10 \
    --num_distill_layers 9 \
    --mlm_probability 0.15 
    
    # --dropout 0.2 # Add if your main.py uses this to set model config dropout

    # Contrastive specific (only if mode is 'combined' or 'contrastive'):
    # --contrast_type HardNeg \
    # --temperature 0.05 \
    # --feat_dim 128 \
    # --distill_weight 1.0 # Only if mode is 'combined'

    # Other parameters to consider from your original script:
    # --lr_scale 100 # Removed, assuming standard Adam optimizer
    # --num_turn 1 # Keep if relevant for your data processing
