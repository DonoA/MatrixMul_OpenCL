#!/bin/bash
#
#SBATCH --job-name=mm_ocl
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=opencl_test.out
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
    ~/$FOLDER/matmul_cl ${nrows[setup]} ${ncols[setup]} ${ncols2[setup]}
    echo "===================================="
done
