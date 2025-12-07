#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4     # smallest suitable GPU partition (2x V100 16
#SBATCH --gres=gpu:1           # one GPU
#SBATCH -t 8:00:00             # time limit (adjust if needed)
#SBATCH -J gat_depth2_8       # job name
#SBATCH -o gat_%j.out         # stdout log
#SBATCH -e gat_%j.err         # stderr log

# Go to your project directory
cd ~/paper-05/nils/bottleneck

# Create a container for this job, name it with the job id to avoid clashes
CONTAINER_NAME=gnn_env_$SLURM_JOB_ID
enroot create --name $CONTAINER_NAME nvidia+pytorch+23.10-py3.sqsh

# Run your script inside the container, with your project mounted at /workspace
enroot start \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  -m /dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck:/workspace \
  "$CONTAINER_NAME" \
  bash -lc "
    # Install Python dependencies inside the container
    pip install -q attrdict3 scikit-learn &&
    pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html &&
    pip install -q torch-sparse  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html &&
    pip install -q torch-geometric &&

    cd /workspace &&
    python run-gat-2-8.py >> batch_results_gat_seed.txt 2>&1
  "


