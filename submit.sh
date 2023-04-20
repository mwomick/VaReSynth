#!/bin/bash

## This is an example of an sbatch script to run a pytorch script
## using Singularity to run the pytorch image.
##
## Set the DATA_PATH to the directory you want the job to run in.
##
## On the singularity command line, replace ./test.py with your program
##
## Change reserved resources as needed for your job.
##

#SBATCH --job-name=varesynth
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00-06:00:00
#SBATCH --partition=volta-gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu_access

unset OMP_NUM_THREADS

# Set SIMG path
SIMG_PATH=/pine/scr/m/w/rwomick/Singularity

# Set SIMG name
SIMG_NAME=huggingface.sif

# Set data path
DATA_PATH=/pine/scr/m/w/rwomick/VaReSynth

# GPU with Singularity
singularity exec --nv -B /pine $SIMG_PATH/$SIMG_NAME bash -c "cd $DATA_PATH; python train.py"