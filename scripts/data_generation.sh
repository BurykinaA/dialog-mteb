#!/bin/bash

# Установить переменную окружения для директории вывода
export OUTPUT_DIR="./data/_downstream_data"

mkdir -p $OUTPUT_DIR

echo "1. Downloading raw data for SNIPS and HWU64..."

wget https://raw.githubusercontent.com/clinc/nlu-datasets/master/nlu_datasets/intent_classification/snips_train.json
wget https://raw.githubusercontent.com/clinc/nlu-datasets/master/nlu_datasets/intent_classification/snips_test.json
wget https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/master/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv
 

echo "2. Processing raw data and generating evaluation data..."

python data/process_evaluate.py --output_dir $OUTPUT_DIR --task clinc150
python data/process_evaluate.py --output_dir $OUTPUT_DIR --task bank77
python data/process_evaluate.py --output_dir $OUTPUT_DIR --task snips
python data/process_evaluate.py --output_dir $OUTPUT_DIR --task hwu64

echo "Processing complete. Evaluation data saved in $OUTPUT_DIR."
