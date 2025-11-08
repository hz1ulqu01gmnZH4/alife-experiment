# Quick Start Guide

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (CPU)

For a quick test on CPU (slower but works without GPU):

```bash
python train.py --device cpu --n-agents 2 --epochs 100 --grid-size 32
```

## Full Training (GPU)

For full training on your RTX 3090:

```bash
python train.py \
    --device cuda \
    --n-agents 3 \
    --epochs 1000 \
    --grid-size 64 \
    --steps-per-epoch 100
```

### Training Options

- `--grid-size`: Size of the petri dish grid (default: 64)
- `--n-agents`: Number of competing NCA agents (default: 3)
- `--epochs`: Number of training epochs (default: 1000)
- `--steps-per-epoch`: Simulation steps per epoch (default: 100)
- `--learning-rate`: Learning rate (default: 2e-3)
- `--device`: Device to use - cuda or cpu (default: cuda)

### Advanced Options

```bash
python train.py \
    --grid-size 128 \
    --n-agents 5 \
    --epochs 2000 \
    --hidden-channels 256 \
    --num-layers 5 \
    --device cuda
```

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## Visualization

After training, visualize the results:

```bash
python visualize.py \
    --checkpoint checkpoints/latest.pt \
    --n-steps 500 \
    --device cuda \
    --output-dir visualizations
```

This creates:
- `visualizations/simulation.gif` - Animated GIF of the simulation
- `visualizations/montage.png` - Grid of frames showing evolution
- `visualizations/statistics.png` - Plots of agent populations over time

## All-in-One Script

For a complete run (train + visualize), use the example script:

```bash
bash run_example.sh
```

## Expected Results

After training, you should see:

1. **Multiple agents coexisting**: Different colored regions representing different NCA agents
2. **Competitive dynamics**: Agents expanding, competing for space, and defending territory
3. **Emergent patterns**: Complex morphologies emerging from simple local rules
4. **Population dynamics**: Agents growing, shrinking, and potentially going extinct

## Performance Notes

On your RTX 3090:
- **Grid 64x64, 3 agents**: ~2-3 seconds per epoch
- **Grid 128x128, 5 agents**: ~8-10 seconds per epoch
- **Grid 256x256, 10 agents**: ~30-40 seconds per epoch

Training for 1000 epochs on a 64x64 grid should take about 40-50 minutes.

## Troubleshooting

### CUDA out of memory

If you get OOM errors, try:
- Reducing `--grid-size`
- Reducing `--n-agents`
- Reducing `--hidden-channels`

### Training instability

If agents die too quickly or grow uncontrollably, try adjusting:
- `--learning-rate` (try 1e-3 or 5e-4)
- Attack/defense parameters in `config.py`

### No variation in behavior

If all agents behave identically, try:
- Increasing `--n-agents` for more diversity
- Training for more epochs
- Using different random seeds: `--seed 123`
