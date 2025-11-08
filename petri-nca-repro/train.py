"""Training script for Petri Dish NCA."""

import argparse
import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import PetriNCAConfig, get_default_config
from models.nca import MultiAgentNCA
from models.petri_dish import PetriDish


class PetriNCATrainer:
    """Trainer for Petri Dish NCA with continual learning."""

    def __init__(self, config: PetriNCAConfig):
        """Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Set random seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create models
        self.nca = MultiAgentNCA(
            n_agents=config.n_agents,
            state_channels=config.state_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
        ).to(self.device)

        self.petri_dish = PetriDish(
            grid_size=config.grid_size,
            state_channels=config.state_channels,
            n_agents=config.n_agents,
            max_agents_per_cell=config.max_agents_per_cell,
            aliveness_threshold=config.aliveness_threshold,
            attack_strength=config.attack_strength,
            defense_strength=config.defense_strength,
            device=self.device,
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.nca.parameters(),
            lr=config.learning_rate,
        )

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def compute_loss(self, states: torch.Tensor, alive_masks: torch.Tensor) -> dict:
        """Compute training loss.

        The loss encourages:
        1. Agent survival (alive cells)
        2. Agent diversity (multiple agents coexisting)
        3. Growth without explosion

        Args:
            states: Agent states
            alive_masks: Alive masks

        Returns:
            Dictionary of losses
        """
        # Survival loss: encourage agents to stay alive
        alive_counts = alive_masks.sum(dim=[1, 2, 3, 4])  # [n_agents]
        survival_loss = -alive_counts.mean()

        # Diversity loss: encourage multiple agents to coexist
        diversity_reward = self.petri_dish.compute_diversity_reward()
        diversity_loss = -diversity_reward

        # Overflow loss: penalize extremely high state values
        state_magnitude = states.abs().mean()
        overflow_loss = torch.relu(state_magnitude - 10.0)

        # Total loss
        total_loss = survival_loss + 0.5 * diversity_loss + 0.1 * overflow_loss

        return {
            'total': total_loss,
            'survival': survival_loss,
            'diversity': diversity_loss,
            'overflow': overflow_loss,
        }

    def train_epoch(self) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary of metrics
        """
        epoch_losses = {
            'total': 0.0,
            'survival': 0.0,
            'diversity': 0.0,
            'overflow': 0.0,
        }

        # Reset environment
        self.petri_dish.reset()

        # Training loop
        for step in range(self.config.steps_per_epoch):
            # Get current state
            states = self.petri_dish.get_state()
            alive_masks = self.petri_dish.get_alive_masks(states)

            # NCA forward pass (agents update their states)
            new_states, attacks, defenses = self.nca(states, alive_masks)

            # Environment step (apply interactions)
            final_states, final_alive_masks = self.petri_dish.step(
                new_states, attacks, defenses
            )

            # Compute loss
            losses = self.compute_loss(final_states, final_alive_masks)

            # Backpropagation
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.nca.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            # Log to tensorboard
            if step % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'loss/{key}', value.item(), self.global_step)

                stats = self.petri_dish.compute_stats()
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'stats/{key}', value, self.global_step)

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= self.config.steps_per_epoch

        return epoch_losses

    def save_checkpoint(self, filename: str):
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'nca_state_dict': self.nca.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }

        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.nca.load_state_dict(checkpoint['nca_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded checkpoint: {path}")

    def visualize(self, epoch: int):
        """Save visualization of current state.

        Args:
            epoch: Current epoch number
        """
        with torch.no_grad():
            rgb = self.petri_dish.get_visualization()

            # Add to tensorboard
            self.writer.add_image(
                'petri_dish',
                rgb[0],
                epoch,
            )

    def train(self):
        """Main training loop."""
        print(f"\nTraining Petri Dish NCA")
        print(f"Agents: {self.config.n_agents}")
        print(f"Grid size: {self.config.grid_size}x{self.config.grid_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Steps per epoch: {self.config.steps_per_epoch}\n")

        # Training loop
        for epoch in tqdm(range(self.config.epochs), desc="Training"):
            self.epoch = epoch

            # Train one epoch
            losses = self.train_epoch()

            # Log epoch metrics
            tqdm.write(
                f"Epoch {epoch}: "
                f"loss={losses['total']:.4f}, "
                f"survival={losses['survival']:.4f}, "
                f"diversity={losses['diversity']:.4f}"
            )

            # Visualize
            if epoch % self.config.vis_interval == 0:
                self.visualize(epoch)

            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint("latest.pt")

        # Final checkpoint
        self.save_checkpoint("final.pt")
        print("\nTraining complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Petri Dish NCA")

    # Model arguments
    parser.add_argument("--grid-size", type=int, default=64, help="Grid size")
    parser.add_argument("--state-channels", type=int, default=16, help="State channels")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--learning-rate", type=float, default=2e-3, help="Learning rate")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval")
    parser.add_argument("--vis-interval", type=int, default=50, help="Visualization interval")

    args = parser.parse_args()

    # Create config
    config = get_default_config()

    # Update config from args
    config.grid_size = args.grid_size
    config.state_channels = args.state_channels
    config.hidden_channels = args.hidden_channels
    config.num_layers = args.num_layers
    config.n_agents = args.n_agents
    config.epochs = args.epochs
    config.steps_per_epoch = args.steps_per_epoch
    config.learning_rate = args.learning_rate
    config.device = args.device
    config.seed = args.seed
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.save_interval = args.save_interval
    config.vis_interval = args.vis_interval

    # Create trainer and train
    trainer = PetriNCATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
