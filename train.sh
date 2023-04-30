# python3 prepare_align.py config/preprocess.yaml
# python3 preprocess.py config/preprocess.yaml
# python3 train.py \
#     -p config/preprocess.yaml \
#     -m config/model.yaml \
#     -t config/train.yaml \
#     -fp16 False \
#     -ddp True

python -m torch.distributed.launch \
        --nnode=1 \
        --node_rank=0 \
        --nproc_per_node=1 \
        train.py \
            --local_world_size=1 \
            --local_rank=0 \
            -p config/preprocess.yaml \
            -m config/model.yaml \
            -t config/train.yaml \
            -fp16 False \
            -ddp True
