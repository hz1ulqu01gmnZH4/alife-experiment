"""Configuration for Petri Dish NCA."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PetriNCAConfig:
    """Configuration for Petri Dish Neural Cellular Automata."""

    # Grid settings
    grid_size: int = 64
    state_channels: int = 16

    # NCA model settings
    hidden_channels: int = 128
    num_layers: int = 3
    kernel_size: int = 3

    # Multi-agent settings
    n_agents: int = 3
    max_agents_per_cell: int = 3

    # Training settings
    epochs: int = 1000
    steps_per_epoch: int = 100
    batch_size: int = 1
    learning_rate: float = 2e-3

    # Dynamics settings
    aliveness_threshold: float = 0.4
    attack_strength: float = 0.1
    defense_strength: float = 0.1

    # Device settings
    device: str = "cuda"
    seed: Optional[int] = 42

    # Checkpoint settings
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Visualization settings
    vis_interval: int = 50
    save_video: bool = True


def get_default_config() -> PetriNCAConfig:
    """Get default configuration."""
    return PetriNCAConfig()
