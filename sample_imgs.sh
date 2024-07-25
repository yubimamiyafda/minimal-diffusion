#!/bin/bash -l
#SBATCH --job-name=minimal-diffusion-test
#SBATCH --output=sysout/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1  # Placeholder, will be overridden by main script
#SBATCH --time=048:00:00
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

DIR="/projects01/VICTRE/yubi.mamiya/minimal-diffusion/trained_models/UNet_melanoma-epoch_250-timesteps_1000-class_condn_False_ema_0.9995.pt"


time python main.py --arch UNet --dataset melanoma \
	--sampling-only --sampling-steps 250 --num-sampled-images 5 \
	--pretrained-ckpt $DIR \
	--save-dir "/projects01/VICTRE/yubi.mamiya/minimal-diffusion/sample_imgs_output_test/"


EXIT_STATUS=$?
echo "Duration: $SECONDS seconds elapsed."
echo "====" `date +%Y%m%d-%H%M%S` "end of job $SLURM_JOB_NAME ($SLURM_JOB_ID) on node $SLURMD_NODENAME on cluster $SLURM_CLUSTER_NAME: EXIT_STATUS=$EXIT_STATUS"
