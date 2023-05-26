#!/bin/sh  

#SBATCH -J deepvit_32
#SBATCH -o ./output/deepvit_32.out
#SBATCH -t 3-00:00:00 

#SBATCH --nodelist=n46
#SBATCH -p A6000
#SBATCH --gres=gpu:4

#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10

cd $SLURM_SUBMIT_DIR
echo "Start"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate deepvit
torchrun --nproc_per_node=4 train.py --world-size 4 --workers 10 --batch-size 32 --sync-bn --layer-num 32 --model-type deepvit --data-path ./dataset/CIFAR100 --dataset-name cifar
conda deactivate

echo "Done!"