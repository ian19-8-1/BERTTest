#!/bin/bash
#SBATCH --nodes=1                      	    # Get one node
#SBATCH --cpus-per-task=2              	    # Two cores per task
#SBATCH --ntasks=1                     	    # But only one task
#SBATCH --gpus=1                   	        # And one GPU
#SBATCH --gres-flags=enforce-binding   	    # Insist on good CPU/GPU alignment
#SBATCH --time=1-00:00:00  		            # Run for 7 days, at most
#SBATCH --job-name=cifar-koh		        # Name the job so I can see it in squeue
#SBATCH --output=dropout-%J.out
#SBATCH --mail-type=END,FAIL	            # Send me email for various states
#SBATCH --mail-user iskoh2000@gmail.com 	# Use this address

# Load modules
module load gcc/gcc-8.3.0
module load anaconda/anaconda3
module load cuda/cuda-11.0

module list                            # Have Nvidia tell us the GPU/CPU mapping so we know

nvidia-smi topo -m

source activate reu2021

python3 --version

CUDA_VISIBLE_DEVICES=0 python3 -u Main.py