#!/bin/bash -e
#SBATCH --job-name=mask2for
#SBATCH --output=/lustre/scratch/client/vinai/users/tuannd42/3dis_ws/slurm_out/slurm_%A.out
#SBATCH --error=/lustre/scratch/client/vinai/users/tuannd42/3dis_ws/slurm_out/slurm_%A.err

#SBATCH --gpus=1
#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-gpu=32

#SBATCH --partition=applied
#SBATCH --mail-type=all
#SBATCH --mail-user=v.tuannd42@vinai.io

srun --container-image=/lustre/scratch/client/vinai/users/tuannd42/docker_images/spformer.sqsh \
--container-mounts=/lustre/scratch/client/vinai/users/tuannd42/3dis_ws/SPFormer:/home/ubuntu/SPFormer \
--container-workdir=/home/ubuntu/SPFormer/ \
python3 tools/train.py configs/spf_scannet_gp_ps_prob_levelset_preduncertainty.yaml --exp_name head_best_label_probbceloss_preduncertainty