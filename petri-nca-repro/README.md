# Petri Dish Neural Cellular Automata - Reproduction

A reproduction of Sakana AI's Petri Dish Neural Cellular Automata (PD-NCA), implementing multi-agent open-ended growth with continual backpropagation.

## Overview

Petri Dish NCA replaces the fixed, non-adaptive morphogenesis of conventional Neural Cellular Automata with multi-agent open-ended growth. Multiple independent NCA agents coexist, compete, and adapt within a shared "petri dish" environment.

## Key Features

- **Multi-agent simulation**: Multiple NCA agents with independent neural parameters
- **Continual learning**: Agents adapt through ongoing gradient-based optimization during simulation
- **Competitive dynamics**: Agents interact through differentiable attack and defense channels
- **Differentiable substrate**: Fully differentiable environment enabling gradient flow

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train on CPU
python train.py --device cpu --n-agents 3 --epochs 1000

# Train on GPU
python train.py --device cuda --n-agents 5 --epochs 10000 --grid-size 128

# Visualize results
python visualize.py --checkpoint checkpoints/latest.pt
```

## Project Structure

```
petri-nca-repro/
├── models/
│   ├── nca.py              # NCA model architecture
│   └── petri_dish.py       # Petri dish environment
├── train.py                # Training script
├── visualize.py            # Visualization utilities
├── config.py               # Configuration
└── requirements.txt        # Dependencies
```

## Architecture

### NCA Model
Each NCA agent is a small convolutional neural network that:
- Takes local grid neighborhood as input
- Outputs state updates (growth, death, attack, defense)
- Maintains independent learnable parameters

### Petri Dish Environment
- Shared 2D grid substrate
- Multi-occupancy support (multiple agents per cell)
- Differentiable interaction channels
- Aliveness thresholding

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (RTX 3060+)
- **Tested on**: RTX 3090 (24GB VRAM)

## References

- [Petri Dish NCA Paper](https://pub.sakana.ai/pdnca/)
- [Original Sakana AI Repository](https://github.com/SakanaAI/petri-dish-nca)
