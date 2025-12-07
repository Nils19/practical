#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00
#SBATCH -J gin_depth2_8
#SBATCH -o gin_%j.out
#SBATCH -e gin_%j.err

# Enable conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate adgn    # the env created from env.yml

cd "$HOME/paper-05/nils/bottleneck"

# If this env doesn't have a tiny dependency like attrdict or sklearn yet:
pip install attrdict3 scikit-learn

python run-gin-2-8.py >> batch_results_gin_conda.txt 2>&1
