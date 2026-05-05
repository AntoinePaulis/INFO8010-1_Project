#!/bin/bash
#SBATCH --job-name=inference_court
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/inference_court_%j.out
#SBATCH --error=../logs/inference_court_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=andy.jalloh@student.uliege.be
#SBATCH --partition=all

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep

cd /home/andyjalloh/antoine/INFO8010-1_Project/project/code/src/court_detection/

python inference.py