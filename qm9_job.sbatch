#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4                   # GPU partition (H100)
#SBATCH --gres=gpu:1                           # 1 GPU
#SBATCH -t 48:00:00                            # time limit (increased to 48 hours)
#SBATCH -J qm9_fa                              # job name
#SBATCH -o qm9_%j.out                          # stdout log
#SBATCH -e qm9_%j.err                          # stderr log
#SBATCH --container-image=nvcr.io/nvidia/tensorflow:23.01-tf1-py3
#SBATCH --container-mounts=/dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck:/workspace

cd /workspace

# Install missing dependencies
pip install -q dpu_utils docopt

# Install extra deps, but SKIP tensorflow-gpu (already in the container, removed from PyPI)
if [ -f tf-gnn-samples/requirements.txt ]; then
  grep -v 'tensorflow-gpu' tf-gnn-samples/requirements.txt > /tmp/qm9_req_no_tfgpu.txt
  pip install -q -r /tmp/qm9_req_no_tfgpu.txt
fi

# Make sure log directory exists
mkdir -p /workspace/qm9_logs

# Run QM9 experiment
cd /workspace/tf-gnn-samples
python run_qm9_benchs_fa.py /workspace/qm9_logs >> /workspace/qm9_results.txt 2>&1
