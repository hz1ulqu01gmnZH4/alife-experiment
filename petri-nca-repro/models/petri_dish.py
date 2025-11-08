"""Petri dish environment for multi-agent NCA simulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PetriDish(nn.Module):
    """Petri dish environment with multi-agent support.

    Provides a shared 2D grid where multiple NCA agents can coexist,
    compete, and adapt through differentiable interactions.
    """

    def __init__(
        self,
        grid_size: int = 64,
        state_channels: int = 16,
        n_agents: int = 3,
        max_agents_per_cell: int = 3,
        aliveness_threshold: float = 0.4,
        attack_strength: float = 0.1,
        defense_strength: float = 0.1,
        device: str = "cuda",
    ):
        """Initialize petri dish environment.

        Args:
            grid_size: Size of square grid
            state_channels: Number of state channels per agent
            n_agents: Number of agents
            max_agents_per_cell: Maximum agents that can occupy one cell
            aliveness_threshold: Threshold for cell aliveness
            attack_strength: Strength of attack interactions
            defense_strength: Strength of defense interactions
            device: Device to run on
        """
        super().__init__()

        self.grid_size = grid_size
        self.state_channels = state_channels
        self.n_agents = n_agents
        self.max_agents_per_cell = max_agents_per_cell
        self.aliveness_threshold = aliveness_threshold
        self.attack_strength = attack_strength
        self.defense_strength = defense_strength
        self.device = device

        # Initialize agent states
        # Shape: [n_agents, batch=1, channels, height, width]
        self.states = None
        self.reset()

    def reset(self):
        """Reset environment with random agent placements."""
        # Initialize states to zeros
        self.states = torch.zeros(
            self.n_agents,
            1,  # batch size
            self.state_channels,
            self.grid_size,
            self.grid_size,
            device=self.device,
        )

        # Place each agent at a random location with initial seed
        for i in range(self.n_agents):
            # Random spawn location
            x = torch.randint(0, self.grid_size, (1,)).item()
            y = torch.randint(0, self.grid_size, (1,)).item()

            # Create small seed (3x3 patch)
            x_start = max(0, x - 1)
            x_end = min(self.grid_size, x + 2)
            y_start = max(0, y - 1)
            y_end = min(self.grid_size, y + 2)

            # Initialize with random values
            self.states[i, 0, :, x_start:x_end, y_start:y_end] = torch.randn(
                self.state_channels,
                x_end - x_start,
                y_end - y_start,
                device=self.device,
            ) * 0.5 + 1.0

    def get_alive_masks(self, states: torch.Tensor) -> torch.Tensor:
        """Compute aliveness masks for all agents.

        A cell is alive if its first channel exceeds the threshold.

        Args:
            states: Agent states [n_agents, batch, channels, height, width]

        Returns:
            Alive masks [n_agents, batch, 1, height, width]
        """
        # Use first channel as aliveness indicator
        aliveness = states[:, :, 0:1, :, :]  # [n_agents, batch, 1, height, width]
        alive_masks = (aliveness > self.aliveness_threshold).float()
        return alive_masks

    def apply_interactions(
        self,
        states: torch.Tensor,
        attacks: torch.Tensor,
        defenses: torch.Tensor,
        alive_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attack/defense interactions between agents.

        Args:
            states: Current states [n_agents, batch, channels, height, width]
            attacks: Attack values [n_agents, batch, 1, height, width]
            defenses: Defense values [n_agents, batch, 1, height, width]
            alive_masks: Alive masks [n_agents, batch, 1, height, width]

        Returns:
            Updated states after interactions
        """
        n_agents = states.shape[0]
        new_states = states.clone()

        # For each agent, compute damage from all other agents
        for i in range(n_agents):
            total_attack = torch.zeros_like(attacks[i])

            # Sum attacks from all other agents
            for j in range(n_agents):
                if i != j:
                    # Only count attacks from alive cells
                    total_attack += attacks[j] * alive_masks[j]

            # Apply damage (attack - defense)
            damage = torch.tanh(total_attack) * self.attack_strength
            protection = torch.sigmoid(defenses[i]) * self.defense_strength
            net_damage = damage - protection

            # Reduce aliveness channel based on net damage
            new_states[i, :, 0:1, :, :] -= net_damage

        return new_states

    def apply_occupancy_limits(self, states: torch.Tensor) -> torch.Tensor:
        """Apply occupancy limits (kill weakest agents in overcrowded cells).

        Args:
            states: Agent states [n_agents, batch, channels, height, width]

        Returns:
            States after applying occupancy limits
        """
        alive_masks = self.get_alive_masks(states)

        # Count agents per cell
        occupancy = alive_masks.sum(dim=0)  # [batch, 1, height, width]

        # Find overcrowded cells
        overcrowded = occupancy > self.max_agents_per_cell

        if overcrowded.any():
            # For overcrowded cells, keep only the strongest agents
            # Strength = aliveness value
            strengths = states[:, :, 0:1, :, :]  # [n_agents, batch, 1, height, width]

            # Sort agents by strength at each position
            sorted_strengths, sorted_indices = torch.sort(strengths, dim=0, descending=True)

            # Create masks for agents to keep
            keep_masks = torch.zeros_like(alive_masks)
            for i in range(self.max_agents_per_cell):
                agent_idx = sorted_indices[i]
                keep_masks.scatter_(0, agent_idx, 1.0)

            # Apply masks only to overcrowded cells
            kill_mask = (~overcrowded).float()  # Keep all in non-overcrowded cells
            keep_masks = keep_masks * overcrowded.float() + kill_mask

            # Kill agents that exceed occupancy limit
            states = states * keep_masks

        return states

    def step(
        self,
        new_states: torch.Tensor,
        attacks: torch.Tensor,
        defenses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one environment step with interactions.

        Args:
            new_states: States after NCA updates [n_agents, batch, channels, height, width]
            attacks: Attack values [n_agents, batch, 1, height, width]
            defenses: Defense values [n_agents, batch, 1, height, width]

        Returns:
            Tuple of (final_states, alive_masks)
        """
        # Get alive masks
        alive_masks = self.get_alive_masks(new_states)

        # Apply agent interactions (attack/defense)
        new_states = self.apply_interactions(new_states, attacks, defenses, alive_masks)

        # Apply occupancy limits
        new_states = self.apply_occupancy_limits(new_states)

        # Update alive masks after interactions
        alive_masks = self.get_alive_masks(new_states)

        # Update internal state
        self.states = new_states

        return new_states, alive_masks

    def get_state(self) -> torch.Tensor:
        """Get current environment state.

        Returns:
            Current states [n_agents, batch, channels, height, width]
        """
        return self.states

    def get_visualization(self) -> torch.Tensor:
        """Get RGB visualization of the petri dish.

        Returns:
            RGB image [batch, 3, height, width]
        """
        # Get alive masks
        alive_masks = self.get_alive_masks(self.states)

        # Create RGB image with different colors per agent
        rgb = torch.zeros(
            1, 3, self.grid_size, self.grid_size,
            device=self.device,
        )

        # Assign colors to agents (RGB)
        colors = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ], device=self.device)

        # Composite agents with their colors
        for i in range(min(self.n_agents, len(colors))):
            agent_mask = alive_masks[i, 0, 0, :, :]  # [height, width]
            for c in range(3):
                rgb[0, c, :, :] += agent_mask * colors[i, c]

        # Normalize to [0, 1]
        rgb = torch.clamp(rgb, 0, 1)

        return rgb

    def compute_diversity_reward(self) -> torch.Tensor:
        """Compute reward based on agent diversity (encourages coexistence).

        Returns:
            Scalar diversity reward
        """
        alive_masks = self.get_alive_masks(self.states)

        # Count alive cells per agent
        alive_counts = alive_masks.sum(dim=[1, 2, 3, 4])  # [n_agents]

        # Reward is higher when all agents are alive
        # Use variance of counts (lower variance = more balanced)
        mean_count = alive_counts.mean()
        variance = ((alive_counts - mean_count) ** 2).mean()

        # Reward: high mean count, low variance
        diversity_reward = mean_count - variance * 0.1

        return diversity_reward

    def compute_stats(self) -> dict:
        """Compute statistics about current state.

        Returns:
            Dictionary of statistics
        """
        alive_masks = self.get_alive_masks(self.states)
        alive_counts = alive_masks.sum(dim=[1, 2, 3, 4])  # [n_agents]

        stats = {
            'total_alive': alive_counts.sum().item(),
            'agent_counts': alive_counts.cpu().numpy(),
            'diversity': self.compute_diversity_reward().item(),
        }

        return stats
