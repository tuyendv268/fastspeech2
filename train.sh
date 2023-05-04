# python3 prepare_align.py config/preprocess.yaml
# python3 preprocess.py config/preprocess.yaml
# python3 train.py \
#     -p config/preprocess.yaml \
#     -m config/model.yaml \
#     -t config/train.yaml \
#     -fp16 False \
#     -ddp True

torchrun --nnodes=1 \
        --nproc_per_node=2 \
        train.py \
            -p config/preprocess.yaml \
            -m config/model.yaml \
            -t config/train.yaml \
            -fp16 False \
            -ddp True \
            --restore_step 55000 
