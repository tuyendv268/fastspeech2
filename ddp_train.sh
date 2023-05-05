torchrun --nnodes=1 \
        --nproc_per_node=2 \
        train.py \
            -p config/preprocess.yaml \
            -m config/model.yaml \
            -t config/train.yaml \
            -fp16 False \
            --distributed_training True \
            --restore_step 76000 
