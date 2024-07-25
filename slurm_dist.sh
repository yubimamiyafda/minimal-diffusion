#!/bin/bash -l
#SBATCH --job-name=minimal-diffusion-test
#SBATCH --output=sysout/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8			# 2 = 1
#SBATCH --gres=gpu:4  # Placeholder, will be overridden by main script
#SBATCH --time=03:00:00 # change to 3 hrs for test
#SBATCH --mem=10G


SECONDS=0
echo "====" `date +%Y%m%d-%H%M%S` "start of job $SLURM_JOB_NAME ($SLURM_JOB_ID) on node $SLURMD_NODENAME on cluster $SLURM_CLUSTER_NAME"


# Print the allocated nodes and partition
echo "Allocated nodes:" $SLURM_JOB_NODELIST
echo "Partition:" $SLURM_JOB_PARTITION


nvidia-smi

conda activate minimal-diffusion-env

# CUDA_VISIBLE_DEVICES=0
echo -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
NP=$((SLURM_CPUS_PER_TASK/2))
# replace nproc_per_node w $NP
time python -m torch.distributed.launch --nproc_per_node=1 --master_port 55200 main_dist.py \
	--arch UNet --dataset melanoma --epochs 250 \
	--batch-size 128 --sampling-steps 50 \
	--data_dir /projects01/VICTRE/yubi_mamiya/datasets/melanoma/


EXIT_STATUS=$?
echo "Duration: $SECONDS seconds elapsed."
echo "====" `date +%Y%m%d-%H%M%S` "end of job $SLURM_JOB_NAME ($SLURM_JOB_ID) on node $SLURMD_NODENAME on cluster $SLURM_CLUSTER_NAME: EXIT_STATUS=$EXIT_STATUS"
