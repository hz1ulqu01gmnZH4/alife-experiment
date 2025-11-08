#!/bin/bash
# Example training and visualization script for Petri Dish NCA

set -e

echo "=== Petri Dish NCA Example ==="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "=== Training Petri Dish NCA ==="
echo "Configuration:"
echo "  - Grid size: 64x64"
echo "  - Agents: 3"
echo "  - Epochs: 500"
echo "  - Device: cuda"
echo ""

# Train on GPU for 500 epochs
python train.py \
    --grid-size 64 \
    --n-agents 3 \
    --epochs 500 \
    --steps-per-epoch 100 \
    --device cuda \
    --save-interval 100 \
    --vis-interval 50

echo ""
echo "=== Visualizing Results ==="
echo ""

# Visualize results
python visualize.py \
    --checkpoint checkpoints/latest.pt \
    --n-steps 500 \
    --device cuda \
    --output-dir visualizations

echo ""
echo "=== Complete! ==="
echo ""
echo "Outputs:"
echo "  - Checkpoints: checkpoints/"
echo "  - Logs: logs/"
echo "  - Visualizations: visualizations/"
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir logs"
echo ""
