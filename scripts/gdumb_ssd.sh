mode=0
tag=gdumb-ssd
device=4

CUDA_VISIBLE_DEVICES=$device python general_main.py \
    --data cifar100 \
    --seed 10086 \
    --cl_type nc \
    --agent GDUMBSSD \
    --tag $tag-1000mem \
    --retrieve random \
    --update summarize \
    --mem_size 1000 \
    --images_per_class 10 \
    --head mlp \
    --temp 0.07 \
    --eps_mem_batch 100 \
    --lr_img 2e-4 \
    --summarize_interval 1 \
    --queue_size 64 \
    --mem_weight 1 \
    --num_runs 5 \
    --estimator_update_mode $mode