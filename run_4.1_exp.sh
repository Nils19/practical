#!/bin/bash
# run_4.1_exp.sh

echo "=========================================="
echo "Starting all GNN experiments"
echo "Started at: $(date)"
echo "=========================================="
echo ""

echo "Running GAT (depth 2-8)..."
python run-gat-2-8.py 2>&1 | tee gat_latest.txt
echo ""
echo "✓ GAT completed"
echo ""

echo "Running GIN (depth 2-8)..."
python run-gin-2-8.py 2>&1 | tee gin_latest.txt
echo ""
echo "✓ GIN completed"
echo ""

echo "Running GCN (depth 2-8)..."
python run-gcn-2-8.py 2>&1 | tee gcn_latest.txt
echo ""
echo "✓ GCN completed"
echo ""

echo "Running GGNN (depth 2-8)..."
python run-ggnn-2-8.py 2>&1 | tee ggnn_latest.txt
echo ""
echo "✓ GGNN completed"
echo ""

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE!"
echo "Finished at: $(date)"
echo "=========================================="