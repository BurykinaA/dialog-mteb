export WORK_PLACE="./"
export OUTPUT_DIR="./data/dse_training.tsv"

# 1. Clone the repos

cd $WORK_PLACE
git clone https://github.com/jasonwu0731/ToD-BERT
git clone https://github.com/amazon-research/dse

# 2. Downloads the raw datasets from https://drive.google.com/file/d/1EnGX0UF4KW6rVBKMF3fL-9Q2ZyFKNOIy/view?usp=sharing
#    and put the "TODBERT_dialog_datasets.zip" file at current directory

# 3. Unzip the downloaded file 

unzip TODBERT_dialog_datasets.zip -d $WORK_SPACE

# 4. Modify "ToD-BERT/my_tod_pretraining.py" to acquire the data processed by TOD-BERT's codebase
### 4.1 Change line 745 to default='/PATH/TO/dialog_datasets' (e.g., '/home/dialog_datasets')
### 4.2 Add the following line after line 951

        with open("pre_train.pkl", "wb") as f:
            pickle.dump(datasets, f) 

        raise ValueError("Done")

### 4.3 Run the following script, once it stops, a file named "pre_train.pkl" should appear in this folder

cd $WORK_PLACE/ToD-BERT
./run_tod_lm_pretraining.sh 0 bert bert-base-uncased save/pretrain/ToD-BERT-MLM --only_last_turn


# 5. Run our script to generate positive pairs from TOD-BERT's training data
cd $WORK_PLACE/dse/data
python process_pretrain.py --data_dir $WORK_PLACE/ToD-BERT/pre_train.pkl --output_dir $OUTPUT_DIR


# A tsv file should appear at OUTPUT_DIR, which can be directly used for model pre-training
