#!/bin/bash
#SBATCH --job-name=train_player
#SBATCH --partition=2080ti
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/train_player_%j.out

conda activate deep
export PROJECT_ENV=alan

python code/src/player_tracking/train.py