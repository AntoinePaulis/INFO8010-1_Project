#!/bin/bash
#SBATCH --job-name=train_player
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/train_player_%j.out
#SBATCH --error=../logs/train_player_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy.jalloh@student.uliege.be
#SBATCH --partition=all

source /home/andyjalloh/anaconda3/etc/profile.d/conda.sh
conda activate deep

cd /home/andyjalloh/antoine/INFO8010-1_Project/project/code/src/player_tracking/

python model.py