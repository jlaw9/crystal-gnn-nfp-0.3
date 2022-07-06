#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=crystal_gnn_lc
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=gpu_model.%j.out
#SBATCH --qos=high

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
module load gcc
conda activate /home/jlaw/.conda-envs/crystals_nfp0_3
run_id=1

srun -l hostname

for ((i = 0 ; i < 10 ; i++)); do
    srun -l -n 1 --gres=gpu:1 --nodes=1 python train_model.py $run_id $i &
done

wait
