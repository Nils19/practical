#!/bin/bash
#SBATCH -J debug_varmisuse
#SBATCH -o debug_varmisuse_%j.out
#SBATCH -e debug_varmisuse_%j.err
#SBATCH -t 00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --container-image=/dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck/nvidia+tf-23.01-tf1-py3.sqsh
#SBATCH --container-mounts=/dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck/tf-gnn-samples:/workspace
#SBATCH --container-workdir=/workspace

echo "Installing dependencies..."
python -m pip install -q dpu_utils docopt more_itertools 2>&1

echo "Testing train.py with verbose output..."
python train.py --run-test RGAT VarMisuse --model-param-overrides '{"random_seed": 1,"last_layer_fa":true}'
