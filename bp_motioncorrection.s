#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=motionCorrectionTest
#SBATCH --mail-type=END
#SBATCH --mail-user=jds814@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --array=0-46
# Expects two input parameters:
#   $1: Working directory containing tifs to correct
#   $2: File name pattern to select tifs, i.e. Run0034_00*.tif.
#        If omitted, uses all .tif that are not Ref.tif.

if [[ ! -z $2 ]]
then
    pattern=$2
else
    shopt -s extglob
    pattern="!(Ref).tif"
fi

module purge
module load matlab/R2018a

cd $1

tifs=(`pwd`/$pattern)

numTifs=${#tifs[@]}
echo $numTifs

tif=${tifs[$SLURM_ARRAY_TASK_ID]}

{
    echo $tif
    matlab -nodisplay -r "normcorremotioncorrection_single('$tif','Ref.tif'); exit"
} > $tif.log 2>&1

exit
