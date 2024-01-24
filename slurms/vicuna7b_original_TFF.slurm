#!/bin/bash
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=96gb
#SBATCH --time=320:00:00

module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.7.0
source /mnt/rstor/CSE_CSDS_VXC204/sxz517/venv_vault/mh_model_edit/bin/activate

export TRANSFORMERS_CACHE=/mnt/rstor/CSE_CSDS_VXC204/sxz517/
export HF_HOME=/mnvicuna7b_original_TFF.slurmt/rstor/CSE_CSDS_VXC204/sxz517/
export HUGGINGFACE_HUB_CACHE=/mnt/rstor/CSE_CSDS_VXC204/sxz517/

CUDA_VISIBLE_DEVICES=0 python /mnt/rstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/code/model_edit_main.py \
--model_name vicuna-7b \
--device cuda \
--file_path /mnt/rstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/code/ \
--seed 100 \
--subquestion_breakdown True \
--fact_query_on subquestion \
--output_dir /mnt/rstor/CSE_CSDS_VXC204/sxz517/mh_model_edit/output/ \
--delete_duplicate_output_file True \
--start 0 \
--end 3000