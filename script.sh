#!/bin/bash

# Request ten hours of runtime:
#SBATCH -c 4
#SBATCH --time=17:00:00
#SBATCH -p gpu

# Use more memory (2096GB) (CPU RAM):
#SBATCH --mem=160G

# Specify a job name:
#SBATCH -J concepttest

# Specify an output file


# Set up the environment by loading modules
module load anaconda/3-5.2.0
source /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate sherlock_env

# Run a script
python experiments.py oai Sequential_CtoY --name XtoChat_ChatToY_PRE-XtoC_C0.1_FC50_Opt2-_Opt2 --pretrained outputs/XtoC_C0.1_FC50_Opt2/model_weights.pth --front_fc_layers_to_freeze 1 --fc_layers 10 50 50 1 --C_fc_name fc1 --y_fc_name fc4 --C_weight 0.1 --seed 841538 --lr 0.0005

