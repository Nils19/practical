#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -J ggnn_depth2_8
#SBATCH -o ggnn_%j.out
#SBATCH -e ggnn_%j.err

# Enable conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate adgn    # the env created from env.yml

cd "$HOME/paper-05/nils/bottleneck"

# If this env doesn't have a tiny dependency like attrdict or sklearn yet:
pip install attrdict3 scikit-learn

python run-ggnn-2-8.py >> batch_results_ggnn_conda.txt 2>&1
