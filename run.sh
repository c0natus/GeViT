#!/bin/sh  

#SBATCH -J vit_24
#SBATCH -o ./output/vit_24.out
#SBATCH -t 3-00:00:00

#SBATCH --nodelist=n37
#SBATCH -p A5000
#SBATCH --gres=gpu:2

#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR
echo "Start"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deepvit
torchrun --nproc_per_node=2 train.py --world-size 2 --workers 10 --batch-size 64 --sync-bn --layer-num 24 --model-type vit --data-path ./dataset/CIFAR100 --dataset-name cifar
conda deactivate

echo "Done!"