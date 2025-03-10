#!/bin/sh
#BSUB -J train2
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1440
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

python -m cProfile -s cumulative src/ssm/train.py experiment=exp2 experiment.hyperparameters.num_blocks=2
