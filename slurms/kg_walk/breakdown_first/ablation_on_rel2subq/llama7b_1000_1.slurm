#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=96gb
#SBATCH --time=320:00:00

module load Python/3.8.6-GCCcore-10.2.0 
module load CUDA/11.7.0
source /mnt/vstor/CSE_CSDS_VXC204/sxz517/venv_vault/mh_model_edit2/bin/activate

export TRANSFORMERS_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/model_zoo/HF_transformer_cache/.cache/
export HF_HOME=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/
export HUGGINGFACE_HUB_CACHE=/mnt/vstor/CSE_CSDS_VXC204/sxz517/cache_zoo/HF_cache/.cache/

CUDA_VISIBLE_DEVICES=0 python /mnt/vstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/code_git/mh_model_edit/slurms/kg_walk/breakdown_first/ablation_on_rel2subq/ablation_model_edit_main.py \
--model_name llama-7b \
--device cuda \
--file_path /mnt/vstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/code_git/mh_model_edit/ \
--seed 100 \
--output_dir /mnt/vstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/output/ \
--delete_duplicate_output_file True \
--kg_walk True \
--breakdown_first True \
--edit_num 1000 \
--start 0 \
--end 3000 \
--ablation_code 1
