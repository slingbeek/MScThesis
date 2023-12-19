#!/bin/bash

#SBATCH -c 16
#SBATCH -t 00:05:00
#SBATCH -p rome
#SBATCH -o %x-%j
#SBATCH -e %x-%j

source ~/.bashrc
conda activate geo
#python fieldaverages.py */EM/*.nc
python timeaverages.py */EM/*.nc
