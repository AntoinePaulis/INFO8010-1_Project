#!/bin/bash
#SBATCH --job-name=train_ball
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

python train_test.py