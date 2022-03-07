conda_env="/projects/rlmolecule/jlaw/envs/crystals_nfp0_3"
#run_id="model_b64_dist_class_0_1"
run_id="model_b64_dist_class_0_05"
out_dir="outputs/20220205_cos_dist_bins/$run_id"
mkdir -p $out_dir

cp train_model.py $out_dir

submit_script="""#!/bin/bash
#SBATCH --account=rlmolecule
#SBATCH --time=2-00
#SBATCH --job-name=$run_id
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=$out_dir/run.out
#SBATCH --mail-user=jlaw@nrel.gov
#SBATCH --mail-type=END

source ~/.bashrc
module load cudnn/8.1.1/cuda-11.2
conda activate $conda_env

echo \"\$PWD\"
echo \"Job started at: \$(date)\"
srun python $out_dir/train_model.py
echo \"Job finished at: \$(date)\"
"""

echo "$submit_script" > $out_dir/submit.sh
echo "sbatch $out_dir/submit.sh"
sbatch $out_dir/submit.sh
