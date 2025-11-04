#!/bin/bash
#
# Script to run TEMI clustering on CIFAR100 with optimal settings.
#
# This script runs the complete clustering pipeline with default
# hyperparameters tuned for CIFAR100 with k=100 clusters.

echo "=================================================="
echo "TEMI Clustering on CIFAR100 with DINOv2"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Dataset: CIFAR100"
echo "  - Clusters: 100"
echo "  - Model: DINOv2-base"
echo "  - Epochs: 100"
echo ""

# Run the main script with optimal settings
python main.py \
    --num_clusters 100 \
    --dinov2_model facebook/dinov2-base \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --temperature 0.1 \
    --hidden_dim 2048 \
    --projection_dim 256 \
    --save_features \
    --device cuda

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Experiment completed successfully!"
    echo "Results saved in ./results/"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "Experiment failed. Check error messages above."
    echo "=================================================="
    exit 1
fi
