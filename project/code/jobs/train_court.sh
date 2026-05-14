#!/bin/bash
#SBATCH --job-name=train_court
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_court_%j.out
#SBATCH --error=../logs/train_court_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy.jalloh@student.uliege.be
#SBATCH --partition=quadro

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep

cd /home/andyjalloh/andy/INFO8010-1_Project/project/code/src/court_detection/

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 # trying to avoid out of memory crash
python train.py