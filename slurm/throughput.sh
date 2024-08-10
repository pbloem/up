#!/bin/bash
#SBATCH --job-name=gpt-throughput
#SBATCH --time 2:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --output=job_%A.out

source /home/pbloem/.bashrc

# export CUDA_LAUNCH_BLOCKING=1

#d=12
#c=512
#b=116
#l=3e-5
#a=128

/home/pbloem/miniconda3/bin/python -u /home/pbloem/git/up/experiments/gpt-mup.py throughput
