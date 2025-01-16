export WORK_SPACE="/home/coder/project"
export OUTPUT_DIR="/home/coder/project/data/_downstream_data"


# echo "1. Install ParlAi Package"

# cd $WORK_SPACE
# # git clone https://github.com/facebookresearch/ParlAI.git 
# cd ParlAI; python setup.py develop


# echo "2. Use ParlAi to generate raw data"

# cd $WORK_SPACE/dse/data
# parlai convert_to_parlai --task amazon_qa --datatype train --outfile amazonqa.txt
# parlai convert_to_parlai --task dstc7 --datatype test --outfile ubuntu_test.txt
# parlai convert_to_parlai --task dstc7 --datatype valid --outfile ubuntu_valid.txt


# echo "3. Use our scripts to generate evaluation data"

python /home/coder/project/data/process_evaluate.py --output_dir $OUTPUT_DIR --task amazonqa
python /home/coder/project/data/process_evaluate.py --output_dir $OUTPUT_DIR --task ubuntu
