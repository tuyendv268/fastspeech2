python3 synthesize.py \
        --source preprocessed_data/val.txt \
        --restore_step 20000 \
        --mode batch \
        -p config/preprocess.yaml \
        -m config/model.yaml \
        -t config/train.yaml