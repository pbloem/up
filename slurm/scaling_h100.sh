#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --time 120:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
##SBATCH --output=job_%A.out

source /home/pbloem/.bashrc

# export CUDA_LAUNCH_BLOCKING=1

i=$SLURM_ARRAY_TASK_ID
l=3e-4 #$SLURM_ARRAY_TASK_ID
w=0.0 #1e-$SLURM_ARRAY_TASK_ID
h=(1152 1536 2048 2432 2560)
b=(48 30 14 2 1)     # batch sizes for min_heads=16
m=(48 60 42 42 42)   # min sizes for min_heads=16
o=4 #$SLURM_ARRAY_TASK_ID
d=0.1 #$SLURM_ARRAY_TASK_ID # dropout

/home/pbloem/miniconda3/bin/python -u /home/pbloem/git/up/experiments/gpt-mup.py go \
     --width ${h[i]} --target-microbatch-size ${b[i]} --name scaling-${h[i]}-o$o-l$l-m${m[i]} \
     --warmup 1_000_000 --mbwarmup 1_000_000 --mb-min ${m[i]} --mb-start 500_000 --base-lr $l --init-factor 1 \
     --weight-mult1 2 --weight-mult2 4 --weight-multb 4 --source-mask --temperature 1 --macrobatch-size 500 \
     --out_factor $o --min-heads 16 --eval-ood --nl-target relu --weight-decay $w --dropout $d --depth-factor 1.0 --attn-factor 1.0 \
     --freeze-blocks 8 --unfreeze-time 10_000 \
#    --weight-decay $w --skip-mup --old-init --sqrt-attn-scale
#    --debug --skip-eval --print-every 10

