#!/bin/bash
# Script to prepare VarMisuse data chunks for training

cd /dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck/tf-gnn-samples

# Create output directories for chunked data
mkdir -p data/varmisuse-chunked/train
mkdir -p data/varmisuse-chunked/valid
mkdir -p data/varmisuse-chunked/test

# Process train data
echo "Processing training data..."
python utils/varmisuse_data_splitter.py \
    --chunk-size 100 \
    --window-size 5000 \
    data/varmisuse/graphs-train \
    data/varmisuse-chunked/train

# Process validation data
echo "Processing validation data..."
python utils/varmisuse_data_splitter.py \
    --chunk-size 100 \
    --window-size 5000 \
    data/varmisuse/graphs-valid \
    data/varmisuse-chunked/valid

# Process test data
echo "Processing test data..."
python utils/varmisuse_data_splitter.py \
    --chunk-size 100 \
    --window-size 5000 \
    data/varmisuse/graphs-test \
    data/varmisuse-chunked/test

echo "Done! Chunked data is in data/varmisuse-chunked/"
