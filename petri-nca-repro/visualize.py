"""Visualization utilities for Petri Dish NCA."""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np

from config import get_default_config
from models.nca import MultiAgentNCA
from models.petri_dish import PetriDish


class PetriNCAVisualizer:
    """Visualizer for Petri Dish NCA simulations."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """Initialize visualizer.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint (weights_only=False for config objects)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']

        # Create models
        self.nca = MultiAgentNCA(
            n_agents=self.config.n_agents,
            state_channels=self.config.state_channels,
            hidden_channels=self.config.hidden_channels,
            num_layers=self.config.num_layers,
            cell_update_rate=getattr(self.config, 'cell_update_rate', 0.5),  # Default for old checkpoints
        ).to(self.device)

        self.nca.load_state_dict(checkpoint['nca_state_dict'])
        self.nca.eval()

        self.petri_dish = PetriDish(
            grid_size=self.config.grid_size,
            state_channels=self.config.state_channels,
            n_agents=self.config.n_agents,
            max_agents_per_cell=self.config.max_agents_per_cell,
            aliveness_threshold=self.config.aliveness_threshold,
            attack_strength=self.config.attack_strength,
            defense_strength=self.config.defense_strength,
            device=self.device,
        )

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Global step: {checkpoint['global_step']}")

    def run_simulation(self, n_steps: int = 500) -> list:
        """Run simulation and collect frames.

        Args:
            n_steps: Number of simulation steps

        Returns:
            List of RGB frames
        """
        frames = []

        # Reset environment
        self.petri_dish.reset()

        with torch.no_grad():
            for step in range(n_steps):
                # Get current state
                states = self.petri_dish.get_state()
                alive_masks = self.petri_dish.get_alive_masks(states)

                # NCA forward pass
                new_states, attacks, defenses = self.nca(states, alive_masks)

                # Environment step
                final_states, final_alive_masks = self.petri_dish.step(
                    new_states, attacks, defenses
                )

                # Get visualization
                rgb = self.petri_dish.get_visualization()
                frame = rgb[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
                frames.append(frame)

                # Print stats every 50 steps
                if step % 50 == 0:
                    stats = self.petri_dish.compute_stats()
                    print(f"Step {step}: {stats}")

        return frames

    def save_animation(self, frames: list, output_path: str, fps: int = 30):
        """Save frames as animation.

        Args:
            frames: List of RGB frames
            output_path: Output file path
            fps: Frames per second
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')

        im = ax.imshow(frames[0])

        def update(frame_idx):
            im.set_array(frames[frame_idx])
            return [im]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=1000 / fps,
            blit=True,
        )

        anim.save(output_path, writer='pillow', fps=fps)
        print(f"Saved animation to {output_path}")
        plt.close()

    def save_montage(self, frames: list, output_path: str, n_frames: int = 16):
        """Save montage of frames.

        Args:
            frames: List of RGB frames
            output_path: Output file path
            n_frames: Number of frames to include
        """
        # Select evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        selected_frames = [frames[i] for i in indices]

        # Create montage
        rows = int(np.sqrt(n_frames))
        cols = (n_frames + rows - 1) // rows

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        for i, (ax, frame) in enumerate(zip(axes, selected_frames)):
            ax.imshow(frame)
            ax.set_title(f"Step {indices[i]}")
            ax.axis('off')

        # Hide unused subplots
        for i in range(len(selected_frames), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved montage to {output_path}")
        plt.close()

    def plot_statistics(self, output_path: str, n_steps: int = 500):
        """Plot statistics over time.

        Args:
            output_path: Output file path
            n_steps: Number of simulation steps
        """
        stats_history = {
            'total_alive': [],
            'diversity': [],
            'agent_counts': [[] for _ in range(self.config.n_agents)],
        }

        # Reset environment
        self.petri_dish.reset()

        with torch.no_grad():
            for step in range(n_steps):
                # Get current state
                states = self.petri_dish.get_state()
                alive_masks = self.petri_dish.get_alive_masks(states)

                # NCA forward pass
                new_states, attacks, defenses = self.nca(states, alive_masks)

                # Environment step
                final_states, final_alive_masks = self.petri_dish.step(
                    new_states, attacks, defenses
                )

                # Collect stats
                stats = self.petri_dish.compute_stats()
                stats_history['total_alive'].append(stats['total_alive'])
                stats_history['diversity'].append(stats['diversity'])

                for i, count in enumerate(stats['agent_counts']):
                    stats_history['agent_counts'][i].append(count)

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot total alive cells
        axes[0].plot(stats_history['total_alive'], label='Total alive')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Alive cells')
        axes[0].set_title('Total Alive Cells Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot per-agent counts
        colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
        for i in range(self.config.n_agents):
            color = colors[i % len(colors)]
            axes[1].plot(
                stats_history['agent_counts'][i],
                label=f'Agent {i}',
                color=color,
            )

        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Alive cells')
        axes[1].set_title('Per-Agent Alive Cells Over Time')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics plot to {output_path}")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualize Petri Dish NCA")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/latest.pt",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation FPS",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create visualizer
    visualizer = PetriNCAVisualizer(args.checkpoint, args.device)

    # Run simulation
    print(f"\nRunning simulation for {args.n_steps} steps...")
    frames = visualizer.run_simulation(args.n_steps)

    # Save outputs
    print("\nSaving visualizations...")

    # Save animation
    visualizer.save_animation(
        frames,
        str(output_dir / "simulation.gif"),
        fps=args.fps,
    )

    # Save montage
    visualizer.save_montage(
        frames,
        str(output_dir / "montage.png"),
        n_frames=16,
    )

    # Save statistics
    visualizer.plot_statistics(
        str(output_dir / "statistics.png"),
        n_steps=args.n_steps,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
