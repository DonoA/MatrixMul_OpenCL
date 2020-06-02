#!/bin/bash
#
#SBATCH --job-name=mm_omp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=96
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=openmp_test.out
#SBATCH --time=4:00:00
#

FOLDER=COEN_145_GROUP_PRJ

nrows=( 1000 5000 5000 10000)
ncols=( 1000 5000 9000 10000)
ncols2=(1000 5000 2000 10000)

for setup in 0 1 2 3
do
    echo "rows=${nrows[setup]}"
    echo "cols=${ncols[setup]}"
    echo "cols2=${ncols2[setup]}"
    ~/$FOLDER/matmult_omp ${nrows[setup]} ${ncols[setup]} ${ncols2[setup]} 96
    echo "===================================="
done