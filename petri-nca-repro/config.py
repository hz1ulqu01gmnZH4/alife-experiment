"""Configuration for Petri Dish NCA."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PetriNCAConfig:
    """Configuration for Petri Dish Neural Cellular Automata.

    This configuration follows the Sakana AI PD-NCA architecture with
    continual backpropagation and multi-agent competition.
    """

    # Grid settings
    grid_size: int = 64  # Size of square petri dish (64x64 default)
    state_channels: int = 16  # Number of state channels per cell

    # NCA model settings
    hidden_channels: int = 128  # Hidden layer size in NCA networks
    num_layers: int = 3  # Number of fully-connected layers
    kernel_size: int = 3  # Perception kernel size (must be 3 for Sobel filters)

    # Multi-agent settings
    n_agents: int = 3  # Number of competing NCA agents
    max_agents_per_cell: int = 3  # Maximum agents that can occupy one cell

    # Training settings
    epochs: int = 1000  # Number of training epochs
    steps_per_epoch: int = 100  # Simulation steps per epoch
    batch_size: int = 1  # Batch size (typically 1 for continual learning)
    learning_rate: float = 2e-3  # Adam learning rate

    # Dynamics settings
    aliveness_threshold: float = 0.4  # Threshold for cell aliveness (first channel)
    attack_strength: float = 0.1  # Strength multiplier for attack damage
    defense_strength: float = 0.1  # Strength multiplier for defense protection
    cell_update_rate: float = 0.5  # Probability of updating a cell per step (stochastic updates)

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
