#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=1:00:00
#SBATCH --partition=debug
#SBATCH --job-name=crystal_gnn_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/jlaw/gpu_debug.%j.out

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
module load gcc
conda activate rlmol

srun python train_model.py
