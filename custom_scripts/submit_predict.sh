#!/bin/bash
#SBATCH -p gpu
#SBATCH -J CSPOT
#SBATCH -o run.o
#SBATCH -e run.e
#SBATCH -t 0-1:00
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --gres=gpu:teslaM40
#SBATCH --mail-type=END
#SBATCH --mail-user=anirmal@bwh.harvard.edu

source ~/.bashrc
conda init
conda activate 3dunet

train3dunet --config "/n/scratch/users/a/ajn16/cspot_new/config_files/test_config_o2.yml"

