#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=splitSVDtest
#SBATCH --mail-type=END
#SBATCH --mail-user=nakayh01@nyumc.org
#SBATCH --output=slurm_%j.out
#SBATCH --array=0-60

module purge
module load matlab/R2018a

cd /gpfs/home/nakayh01/matlab

name='/gpfs/scratch/nakayh01/2P_Data/HN1953/190603/aligned/HN1953_190603_field2_00001_00001.tif'

{
    echo $name
    matlab -nodisplay -r "SVD_2p_cluster('$name'); exit"
} > $name.log 2>&1


exit
