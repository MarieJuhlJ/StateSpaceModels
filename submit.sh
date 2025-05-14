#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J s6_smnist[1-5]
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -B
#BSUB -N
#BSUB -o ../hpc_files/%J.out
#BSUB -e ../hpc_files/%J.err

# 1) Activate the virtual environment
export WANDB_API_KEY=a27d22f917a84b2d0c5627aab5fce573af80fb17
. /work3/s194572/miniconda3/etc/profile.d/conda.sh
conda activate VDTU

python src/ssm/train.py experiment=s6_smnist dataset=smnist wandb=True k_folds=5 idx_fold=$LSB_JOBINDEX