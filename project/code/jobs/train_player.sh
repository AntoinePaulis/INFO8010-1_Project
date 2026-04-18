#!/bin/bash
#SBATCH --job-name=train_player
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=log.out
#SBATCH --error=err.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoine.paulis@student.uliege.be
#SBATCH --partition=all

source /home/andyjalloh/anaconda3/etc/profile.d/conda.sh
conda activate deep

cd /home/andyjalloh/INFO8010-1_Project/project/code/src/player_tracking/

python model.py