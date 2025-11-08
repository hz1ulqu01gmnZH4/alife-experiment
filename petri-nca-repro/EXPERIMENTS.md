# Experiment Results

This document summarizes experiments run with the Petri Dish NCA implementation.

## Experiment 1: Quick Validation (10 epochs, CPU)

**Configuration:**
- Device: CPU
- Agents: 2
- Grid size: 32×32
- Epochs: 10
- Steps per epoch: 20

**Purpose:** Verify implementation correctness after code review fixes.

**Results:**
- ✅ Training completed without errors
- ✅ Gradient flow working correctly (detach fix applied)
- ✅ Multi-agent coexistence observed
- Population growth: 12 → 1,175 cells over 100 simulation steps
- Agent balance: 629 (red) vs 546 (green) - relatively balanced

**Files:** `checkpoints/`, `visualizations/`

---

## Experiment 2: Long Evolution (2000 epochs, GPU)

**Configuration:**
- Device: CUDA (RTX 3090)
- Agents: 3 (red, green, blue)
- Grid size: 64×64
- Epochs: 2000
- Steps per epoch: 100
- Total training steps: 200,000

**Hardware Performance:**
- Training speed: ~2.4 epochs/second
- Total training time: ~14 minutes
- GPU utilization: 48%
- Peak power: 176W
- VRAM usage: ~2.5GB / 24GB

**Purpose:** Observe long-term evolutionary dynamics and strategy development.

### Key Findings:

#### 1. Open-Ended Evolution
The system **never converged to a stable equilibrium** over 2000 epochs. Agents continued adapting and counter-adapting throughout training, demonstrating genuine co-evolutionary dynamics.

#### 2. Competitive Asymmetry Emerged
After extended training, **red agent developed superior competitive strategies**:
- Consistently maintained higher population counts
- More effective attack/defense balance
- Better territorial control

#### 3. Population Dynamics Shifted
- **Short training (10 epochs):** Dense populations (1000+ cells), stable coexistence
- **Long training (2000 epochs):** Sparse populations (200-400 cells), unstable dynamics
- Agents evolved more aggressive strategies leading to lower carrying capacity

#### 4. Boom-Bust Cycles
Population exhibited cyclic dynamics:
- **Step 0-50:** Explosive growth (15 → 1,436 cells)
- **Step 50-200:** Population crash (1,436 → 511 cells)
- **Step 200-500:** Oscillating between 200-500 cells with periodic invasions

#### 5. No Extinction Despite Asymmetry
Even after 2000 epochs with red dominance:
- All three agents survived
- Weaker agents (green, blue) maintained small populations
- Periodic invasions by weaker agents prevented complete takeover
- Diversity loss remained negative (high population variance)

### Observed Behaviors:

**Early Phase (Epochs 0-500):**
- Rapid territorial expansion
- All agents viable and competing

**Mid Phase (Epochs 500-1200):**
- Red agent gains advantage
- Increased aggression in all agents
- Population density decreases

**Late Phase (Epochs 1200-2000):**
- Unstable equilibrium established
- Red maintains ~60% population share
- Blue and green persist through guerrilla-style survival
- No further significant strategy shifts

### Population Statistics:

| Timestep | Total Alive | Red | Green | Blue | Diversity Score |
|----------|-------------|-----|-------|------|-----------------|
| 0        | 15          | 5   | 5     | 5    | 5.0             |
| 50       | 1,436       | 867 | 392   | 177  | -7,831.9        |
| 100      | 954         | 631 | 244   | 79   | -5,034.2        |
| 250      | 397         | 296 | 57    | 44   | -1,209.8        |
| 450      | 246         | 143 | 37    | 66   | -118.1          |

*Note: Negative diversity scores indicate high population variance (imbalanced populations)*

### Implications:

1. **Continual Learning Works:** Agents successfully adapted during simulation through gradient-based optimization

2. **No Convergence Guarantee:** Extended training doesn't necessarily lead to stable equilibria in competitive multi-agent systems

3. **Emergent Complexity:** Simple local rules + competition → complex cyclic dynamics

4. **Strategy Evolution:** Agents evolved from "grow fast" (early) to "attack aggressively" (late)

5. **Persistent Coexistence:** Even with strong asymmetry, biodiversity maintained through niche exploitation

**Files:** `checkpoints_long/`, `logs_long/`, `visualizations_long/`

---

## Checkpoints Available

### Short Training (10 epochs):
- `checkpoints/checkpoint_epoch_0.pt` (653 KB)
- `checkpoints/latest.pt` (650 KB)
- `checkpoints/final.pt` (650 KB)

### Long Training (2000 epochs):
- `checkpoints_long/checkpoint_epoch_200.pt` (979 KB)
- `checkpoints_long/checkpoint_epoch_400.pt` (979 KB)
- `checkpoints_long/checkpoint_epoch_600.pt` (979 KB)
- `checkpoints_long/checkpoint_epoch_800.pt` (979 KB)
- `checkpoints_long/checkpoint_epoch_1000.pt` (979 KB)
- `checkpoints_long/checkpoint_epoch_1200.pt` (980 KB)
- `checkpoints_long/checkpoint_epoch_1400.pt` (980 KB)
- `checkpoints_long/checkpoint_epoch_1600.pt` (980 KB)
- `checkpoints_long/checkpoint_epoch_1800.pt` (980 KB)
- `checkpoints_long/final.pt` (975 KB)

---

## Visualizations

### Short Training:
- `visualizations/simulation.gif` - 100-step animation (253 KB)
- `visualizations/montage.png` - 16-frame overview (48 KB)
- `visualizations/statistics.png` - Population plots (120 KB)

### Long Training:
- `visualizations_long/simulation.gif` - 500-step animation showing evolved dynamics
- `visualizations_long/montage.png` - 16-frame overview of cyclic behavior
- `visualizations_long/statistics.png` - Long-term population dynamics

---

## Future Experiments

**Suggested explorations:**

1. **Parameter Sensitivity:**
   - Vary `attack_strength` / `defense_strength` to test competitive intensity
   - Test different `aliveness_threshold` values
   - Experiment with `cell_update_rate` for stability

2. **Scaling Studies:**
   - Larger grids (128×128, 256×256) for spatial pattern analysis
   - More agents (5-10) to study biodiversity dynamics
   - Longer training (5000+ epochs) to test for eventual convergence

3. **Architectural Variations:**
   - Toroidal topology (circular boundaries) vs. zero-padding
   - Deeper NCA networks (5-10 layers)
   - Larger hidden dimensions (256, 512 channels)

4. **Analysis Studies:**
   - Checkpoint comparison across epochs to track strategy evolution
   - Fitness landscape analysis
   - Spatial pattern quantification (clustering, territory size)
   - Attack/defense strategy visualization

---

## Reproduction

To reproduce these experiments:

```bash
# Short training (CPU validation)
python train.py --device cpu --n-agents 2 --epochs 10 --grid-size 32 --steps-per-epoch 20

# Long evolution (GPU)
python train.py --device cuda --n-agents 3 --epochs 2000 --grid-size 64 --steps-per-epoch 100 \
    --checkpoint-dir checkpoints_long --log-dir logs_long --vis-interval 100 --save-interval 200

# Visualize results
python visualize.py --checkpoint checkpoints_long/final.pt --n-steps 500 --device cuda
```

---

**Last Updated:** 2025-11-08
**Hardware:** NVIDIA RTX 3090 (24GB VRAM)
**Implementation:** Based on Sakana AI Petri Dish NCA architecture
