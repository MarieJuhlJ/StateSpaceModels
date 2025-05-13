#!/bin/sh
#BSUB -J trainConv512[1-5]
#BSUB -o logs/trainConv512_%J.out
#BSUB -e logs/trainConv512_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1080
####BSUB -u mariejuhljorgensen@gmail.com
####BSUB -B
####BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment	
# module load scipy/VERSION

nvidia-smi
module load cuda/11.6

# activate the virtual environment 
# NOTE: needs to have been built with the same SciPy version above!
source /zhome/e3/b/155491/miniconda3/etc/profile.d/conda.sh
conda activate ssm

python src/ssm/train.py experiment=conv experiment.hyperparameters.N=512 dataset=smnist k_folds=5 idx_fold=$LSB_JOBINDEX wandb=True experiment.hyperparameters.max_epochs=50
