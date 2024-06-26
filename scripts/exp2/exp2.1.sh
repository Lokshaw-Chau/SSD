tag=exp2_1
device=4

CUDA_VISIBLE_DEVICES=$device python general_main.py \
    --data cifar100 \
    --seed 10086 \
    --cl_type nc \
    --agent SSCR \
    --tag $tag-100mem \
    --retrieve random \
    --update summarize \
    --mem_size 100 \
    --images_per_class 1 \
    --head mlp \
    --temp 0.07 \
    --eps_mem_batch 100 \
    --lr_img 2e-4 \
    --summarize_interval 6 \
    --queue_size 64 \
    --mem_weight 1 \
    --num_runs 3 \
    --reverse_match_loss True


CUDA_VISIBLE_DEVICES=$device python general_main.py \
    --data cifar100 \
    --seed 10086 \
    --cl_type nc \
    --agent SSCR \
    --tag $tag-500mem \
    --retrieve random \
    --update summarize \
    --mem_size 500 \
    --images_per_class 5 \
    --head mlp \
    --temp 0.07 \
    --eps_mem_batch 100 \
    --lr_img 1e-3 \
    --summarize_interval 6 \
    --queue_size 64 \
    --mem_weight 1 \
    --num_runs 3 \
    --reverse_match_loss True

CUDA_VISIBLE_DEVICES=$device python general_main.py \
    --data cifar100 \
    --seed 10086 \
    --cl_type nc \
    --agent SSCR \
    --tag $tag-1000mem \
    --retrieve random \
    --update summarize \
    --mem_size 1000 \
    --images_per_class 10 \
    --head mlp \
    --temp 0.07 \
    --eps_mem_batch 100 \
    --lr_img 4e-3 \
    --summarize_interval 6 \
    --queue_size 64 \
    --mem_weight 1 \
    --num_runs 3 \
    --reverse_match_loss True