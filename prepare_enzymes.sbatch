#!/bin/bash
#SBATCH --job-name=enzymes_exp
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=enzymes_exp_%j.out
#SBATCH --error=enzymes_exp_%j.err

cd ~/paper-05/nils/bottleneck

CONTAINER_NAME=gnn_env_$SLURM_JOB_ID
enroot create --name $CONTAINER_NAME nvidia+pytorch+23.10-py3.sqsh

enroot start \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  -m /dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck:/workspace \
  "$CONTAINER_NAME" \
  bash -lc "
    pip install -q attrdict3 scikit-learn pyyaml &&
    pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html &&
    pip install -q torch-sparse  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html &&
    pip install -q torch-geometric &&

    cd /workspace/gnn-comparison &&
    export PYTHONPATH=/workspace/gnn-comparison:\$PYTHONPATH &&
    
    echo 'Preparing ENZYMES dataset...' &&
    python PrepareDatasets.py DATA/CHEMICAL --dataset-name ENZYMES --outer-k 10 &&
    
    echo 'Running ENZYMES experiments...' &&
    python Launch_Experiments.py --config-file config_BaselineENZYMES.yml \
        --dataset-name ENZYMES \
        --outer-folds 10 \
        --outer-processes 1 \
        --inner-folds 5 \
        --inner-processes 1 \
        --result-folder RESULTS
  "
