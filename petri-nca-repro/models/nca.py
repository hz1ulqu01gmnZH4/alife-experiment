"""Neural Cellular Automata model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NCAModel(nn.Module):
    """Neural Cellular Automata with learnable update rules.

    Each NCA agent is a small convolutional network that processes
    local neighborhoods and outputs state updates.
    """

    def __init__(
        self,
        state_channels: int = 16,
        hidden_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
    ):
        """Initialize NCA model.

        Args:
            state_channels: Number of channels in cell state
            hidden_channels: Number of hidden channels in network
            num_layers: Number of convolutional layers
            kernel_size: Size of perception kernel (must be 3 for Sobel filters)
        """
        super().__init__()

        # Validate inputs
        assert kernel_size == 3, f"kernel_size must be 3 for Sobel filters, got {kernel_size}"
        assert state_channels > 0, f"state_channels must be positive, got {state_channels}"
        assert num_layers >= 1, f"num_layers must be at least 1, got {num_layers}"

        self.state_channels = state_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        # Register fixed Sobel filters for perception (as per original PD-NCA paper)
        # Sobel filters for gradient estimation in x and y directions
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0

        # Laplacian filter for second derivative
        laplacian = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 16.0

        # Identity filter to preserve original values
        identity = torch.zeros((3, 3), dtype=torch.float32)
        identity[1, 1] = 1.0

        # Stack filters: [4, 1, 3, 3] - identity, sobel_x, sobel_y, laplacian
        filters = torch.stack([identity, sobel_x, sobel_y, laplacian])
        filters = filters.unsqueeze(1)  # [4, 1, 3, 3]

        # Register as buffer (not trainable parameter)
        self.register_buffer('perception_filters', filters)

        # Perception outputs: state_channels * 4 filters
        perception_size = state_channels * 4

        # Build network layers
        layers = []
        in_ch = perception_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_ch, hidden_channels),
                nn.ReLU(),
            ])
            in_ch = hidden_channels

        # Final layer outputs state delta
        # Output channels: state update + attack + defense
        output_channels = state_channels + 2  # +2 for attack/defense
        layers.append(nn.Linear(in_ch, output_channels))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Extract local neighborhood features using fixed Sobel filters.

        Efficiently applies 4 perception filters (identity, sobel_x, sobel_y, laplacian)
        to each state channel using vectorized operations.

        Args:
            state: Cell states [batch, channels, height, width]

        Returns:
            Perceived features [batch, channels*4, height, width]
            (4 filters per channel: identity, sobel_x, sobel_y, laplacian)
        """
        batch, channels, height, width = state.shape

        # Reshape state to treat each channel independently
        # [batch, channels, height, width] -> [batch*channels, 1, height, width]
        state_flat = state.reshape(batch * channels, 1, height, width)

        # Apply all 4 perception filters to all batch*channel instances
        # perception_filters: [4, 1, 3, 3]
        # Output: [batch*channels, 4, height, width]
        perceived_flat = F.conv2d(
            state_flat,
            self.perception_filters,
            padding=1,
        )

        # Reshape back to separate batch and channel dimensions
        # [batch*channels, 4, height, width] -> [batch, channels, 4, height, width]
        perceived = perceived_flat.reshape(batch, channels, 4, height, width)

        # Reorder to interleave filters with channels
        # [batch, channels, 4, height, width] -> [batch, channels*4, height, width]
        # This gives: [ch0_f0, ch0_f1, ch0_f2, ch0_f3, ch1_f0, ...]
        perceived = perceived.reshape(batch, channels * 4, height, width)

        return perceived

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through NCA.

        Args:
            state: Current cell states [batch, channels, height, width]

        Returns:
            Tuple of:
                - state_delta: State updates [batch, state_channels, height, width]
                - attack: Attack values [batch, 1, height, width]
                - defense: Defense values [batch, 1, height, width]
        """
        batch, channels, height, width = state.shape

        # Perceive local neighborhoods using Sobel filters
        perception = self.perceive(state)  # [batch, channels*4, height, width]

        # Reshape for linear layers: [batch, height, width, channels*4]
        perception = perception.permute(0, 2, 3, 1)

        # Apply network
        output = self.network(perception)  # [batch, height, width, state_channels+2]

        # Reshape back: [batch, state_channels+2, height, width]
        output = output.permute(0, 3, 1, 2)

        # Split into state delta, attack, and defense
        state_delta = output[:, :-2, :, :]  # [batch, state_channels, height, width]
        attack = output[:, -2:-1, :, :]      # [batch, 1, height, width]
        defense = output[:, -1:, :, :]       # [batch, 1, height, width]

        return state_delta, attack, defense

    def update(
        self,
        state: torch.Tensor,
        alive_mask: torch.Tensor,
        stochastic: bool = True,
        update_rate: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update cell states with stochastic application.

        Args:
            state: Current cell states
            alive_mask: Binary mask of alive cells
            stochastic: Whether to apply updates stochastically
            update_rate: Probability of updating each cell (only used if stochastic=True)

        Returns:
            Tuple of (new_state, attack, defense)
        """
        # Forward pass
        state_delta, attack, defense = self.forward(state)

        # Apply stochastic updates (only update some cells)
        if stochastic:
            update_mask = torch.rand_like(state_delta[:, 0:1, :, :]) < update_rate
            state_delta = state_delta * update_mask

        # Apply updates only to alive cells
        state_delta = state_delta * alive_mask

        # Gate attack/defense by aliveness (dead cells cannot attack/defend)
        attack = attack * alive_mask
        defense = defense * alive_mask

        # Update state
        new_state = state + state_delta

        return new_state, attack, defense


class MultiAgentNCA(nn.Module):
    """Container for multiple NCA agents with independent parameters."""

    def __init__(
        self,
        n_agents: int,
        state_channels: int = 16,
        hidden_channels: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        cell_update_rate: float = 0.5,
    ):
        """Initialize multi-agent NCA system.

        Args:
            n_agents: Number of independent NCA agents
            state_channels: State channels per agent
            hidden_channels: Hidden channels in NCA networks
            num_layers: Number of layers in NCA networks
            kernel_size: Perception kernel size
            cell_update_rate: Stochastic update rate per cell
        """
        super().__init__()

        self.n_agents = n_agents
        self.state_channels = state_channels
        self.cell_update_rate = cell_update_rate

        # Create independent NCA models for each agent
        self.agents = nn.ModuleList([
            NCAModel(
                state_channels=state_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
            )
            for _ in range(n_agents)
        ])

    def forward(
        self,
        states: torch.Tensor,
        alive_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update all agents.

        Args:
            states: Agent states [n_agents, batch, channels, height, width]
            alive_masks: Alive masks [n_agents, batch, 1, height, width]

        Returns:
            Tuple of (new_states, attacks, defenses) where each is a tensor with shape:
                - new_states: [n_agents, batch, channels, height, width]
                - attacks: [n_agents, batch, 1, height, width]
                - defenses: [n_agents, batch, 1, height, width]
        """
        new_states = []
        attacks = []
        defenses = []

        for i, agent in enumerate(self.agents):
            new_state, attack, defense = agent.update(
                states[i],
                alive_masks[i],
                stochastic=True,
                update_rate=self.cell_update_rate,
            )
            new_states.append(new_state)
            attacks.append(attack)
            defenses.append(defense)

        # Stack results
        new_states = torch.stack(new_states, dim=0)
        attacks = torch.stack(attacks, dim=0)
        defenses = torch.stack(defenses, dim=0)

        return new_states, attacks, defenses
