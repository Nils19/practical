#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -J varmisuse
#SBATCH -o varmisuse_%j.out
#SBATCH -e varmisuse_%j.err
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --container-image=nvcr.io/nvidia/tensorflow:23.01-tf1-py3
#SBATCH --container-mounts=/dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck:/workspace

cd /workspace

# Install missing dependencies (inside the container)
python -m pip install -q dpu_utils docopt

if [ -f tf-gnn-samples/requirements.txt ]; then
  grep -v 'tensorflow-gpu' tf-gnn-samples/requirements.txt > /tmp/req_no_tfgpu.txt
  python -m pip install -q -r /tmp/req_no_tfgpu.txt
fi

# Optional: less chatty TF logs
export TF_CPP_MIN_LOG_LEVEL=1

# First, prepare the chunked data if it doesn't exist
if [ ! -d "/workspace/tf-gnn-samples/data/varmisuse-chunked/train" ]; then
    echo "Preparing VarMisuse chunked data..."
    mkdir -p /workspace/tf-gnn-samples/data/varmisuse-chunked/train
    mkdir -p /workspace/tf-gnn-samples/data/varmisuse-chunked/valid
    mkdir -p /workspace/tf-gnn-samples/data/varmisuse-chunked/test

    # Process train data
    echo "Processing training data..."
    python /workspace/tf-gnn-samples/utils/varmisuse_data_splitter.py \
        --chunk-size 100 \
        --window-size 5000 \
        --num-workers 8 \
        /workspace/tf-gnn-samples/data/varmisuse/graphs-train \
        /workspace/tf-gnn-samples/data/varmisuse-chunked/train

    # Process validation data
    echo "Processing validation data..."
    python /workspace/tf-gnn-samples/utils/varmisuse_data_splitter.py \
        --chunk-size 100 \
        --window-size 5000 \
        --num-workers 8 \
        /workspace/tf-gnn-samples/data/varmisuse/graphs-valid \
        /workspace/tf-gnn-samples/data/varmisuse-chunked/valid

    # Process test data
    echo "Processing test data..."
    python /workspace/tf-gnn-samples/utils/varmisuse_data_splitter.py \
        --chunk-size 100 \
        --window-size 5000 \
        --num-workers 8 \
        /workspace/tf-gnn-samples/data/varmisuse/graphs-test \
        /workspace/tf-gnn-samples/data/varmisuse-chunked/test
fi

# Now run the VarMisuse experiments
mkdir -p /workspace/varmisuse_logs

cd /workspace/tf-gnn-samples
python run_varmisuse_benchs_fa.py /workspace/varmisuse_logs >> /workspace/varmisuse_results.txt 2>&1
