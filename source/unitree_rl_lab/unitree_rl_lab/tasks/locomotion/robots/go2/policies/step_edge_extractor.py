"""Step Edge Extractor: differentiable explicit feature extraction from height scans.

Extracts four stair-relevant scalars from a 2D height scan grid:
  - step_height      : how high is the next step edge  (metres)
  - step_dist        : distance to the next step edge  (metres, positive = ahead)
  - step_width       : lateral width of the step edge  (metres)
  - edge_confidence  : how reliably a step edge is detected  (dimensionless, ~0-1)

Design principles
-----------------
* No CNN / Transformer — only closed-form operations on the gradient map.
* Fully differentiable via soft-argmax and sigmoid-based soft counting.
* Interpretable: each output has a direct geometric meaning.

edge_confidence rationale
-------------------------
The soft-argmax weights ``w`` (shape ``H-1``) encode which forward row contains
the dominant ascending edge.  When a clear stair edge exists, the weights
concentrate on one row (low entropy).  When the scan is noisy, flat, or in a
flat-to-stair transition zone, the weights spread out (high entropy).

    edge_confidence = 1 - H(w) / H_max
      H(w)    = -Σᵢ wᵢ log wᵢ          (Shannon entropy of w)
      H_max   = log(H - 1)              (entropy of uniform over H-1 rows)

    → near 1: single dominant edge row  (clear stair, policy should trust features)
    → near 0: spread / flat / noise     (ambiguous, policy should hedge)

Height scan sign convention (Isaac Lab RayCaster)
-------------------------------------------------
    value = sensor_z - terrain_z - offset          (positive ⟹ terrain is BELOW sensor)

So when the terrain RISES (ascending stair), the scan value DECREASES.
To recover terrain elevation change we negate the forward finite difference:
    terrain_rise[i] = -(scan[i+1] - scan[i]) > 0  for an ascending step

Grid layout assumption
----------------------
The height scanner in ``stair_env_cfg.py`` is configured with
    GridPatternCfg(resolution=0.1, size=[1.1, 0.7], ordering="yx")

With ``ordering="yx"`` the flattened ray order is:
    outer loop = x (forward, 12 values from −0.55 m to +0.55 m)
    inner loop = y (lateral,  8 values from −0.35 m to +0.35 m)

Reshaped (N, 96) → (N, 12, 8):
    dim 1  —  forward rows   : index 0 ↔ x=−0.55 m, index 11 ↔ x=+0.55 m
    dim 2  —  lateral columns: index 0 ↔ y=−0.35 m, index  7 ↔ y=+0.35 m
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepEdgeExtractor(nn.Module):
    """Differentiable extraction of stair edge features from a flat height scan.

    Args:
        grid_rows:       Number of forward scan rows (x direction). Default 12.
        grid_cols:       Number of lateral scan cols (y direction). Default 8.
        resolution:      Scan resolution in metres. Default 0.1.
        fwd_start:       World x-offset of the first forward row relative to the
                         robot base (negative = behind). Default −0.55 m.
        softmax_temp:    Temperature for soft-argmax distance estimation.
                         Higher → sharper (closer to hard argmax). Default 50.0.
        edge_threshold:  Minimum terrain rise (metres) for a column to count
                         towards step_width. Default 0.02 m.
    """

    def __init__(
        self,
        grid_rows: int = 12,
        grid_cols: int = 8,
        resolution: float = 0.1,
        fwd_start: float = -0.55,
        softmax_temp: float = 50.0,
        edge_threshold: float = 0.02,
    ) -> None:
        super().__init__()

        self.H = grid_rows
        self.W = grid_cols
        self.resolution = resolution
        self.softmax_temp = softmax_temp
        self.edge_threshold = edge_threshold

        # Midpoint x-positions of the (H-1) forward finite differences.
        # diff[i] represents the terrain change between row i and row i+1,
        # centred at x = fwd_start + (i + 0.5) * resolution.
        fwd_midpoints = torch.arange(grid_rows - 1, dtype=torch.float32) * resolution + fwd_start + 0.5 * resolution
        self.register_buffer("fwd_midpoints", fwd_midpoints)  # (H-1,)

        # Maximum possible entropy of the soft-argmax weights (uniform over H-1 rows).
        # Used to normalise edge_confidence to [0, 1].
        self._H_max: float = math.log(grid_rows - 1)  # log(11) ≈ 2.398

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, height_scan: torch.Tensor) -> torch.Tensor:
        """Extract [step_height, step_dist, step_width, edge_confidence].

        Args:
            height_scan: ``(N, H*W)`` flattened height scan observation.

        Returns:
            features: ``(N, 4)`` tensor —
                [step_height (m), step_dist (m), step_width (m), edge_confidence (0-1)].
        """
        N = height_scan.shape[0]

        # ── Reshape to (N, H, W): dim1=forward, dim2=lateral ──────────────
        # Use reshape instead of view to handle non-contiguous tensors
        # (e.g. after split_and_pad_trajectories in recurrent PPO update).
        grid = height_scan.reshape(N, self.H, self.W)

        # ── Forward finite difference → terrain rise map ───────────────────
        # terrain_diff[i] = -(grid[i+1] - grid[i]) > 0 for ascending step
        # Shape: (N, H-1, W)
        terrain_diff = -(grid[:, 1:, :] - grid[:, :-1, :])

        # ── 1. step_height ─────────────────────────────────────────────────
        # Maximum terrain rise across the entire scanned area.
        # Clamp at 0: descending terrain should not suppress a positive step.
        ascent_map = torch.clamp(terrain_diff, min=0.0)           # (N, H-1, W)
        step_height = ascent_map.max(dim=2)[0].max(dim=1)[0]      # (N,)

        # ── 2. step_dist  &  soft-argmax weights ────────────────────────────
        # Distance from the robot to the row with the highest ascending edge.
        # Weights are reused for step_width and edge_confidence.
        edge_per_row = ascent_map.max(dim=2)[0]                    # (N, H-1)
        weights = F.softmax(self.softmax_temp * edge_per_row, dim=1)  # (N, H-1)
        step_dist = (weights * self.fwd_midpoints.unsqueeze(0)).sum(dim=1)  # (N,)

        # ── 3. step_width ───────────────────────────────────────────────────
        # Lateral span of the dominant ascending edge.
        attn = weights                                              # (N, H-1)
        weighted_edge_col = (attn.unsqueeze(2) * ascent_map).sum(dim=1)  # (N, W)
        col_active = torch.sigmoid(10.0 * (weighted_edge_col - self.edge_threshold))
        step_width = col_active.sum(dim=1) * self.resolution       # (N,), metres

        # ── 4. edge_confidence ──────────────────────────────────────────────
        # Measures how reliably the dominant-edge row is identifiable.
        #
        # Intuition:
        #   flat / noisy terrain  → weights ≈ uniform  → high H → confidence ≈ 0
        #   clear stair edge      → weights concentrated → low H → confidence ≈ 1
        #   flat-to-stair transition → intermediate → confidence ≈ 0.3–0.7
        #
        # H(w) = -Σ wᵢ log wᵢ  (Shannon entropy, clamped to avoid log(0))
        H_w = -(weights * torch.log(weights.clamp(min=1e-8))).sum(dim=1)  # (N,)
        edge_confidence = torch.clamp(1.0 - H_w / self._H_max, min=0.0, max=1.0)  # (N,)

        return torch.stack([step_height, step_dist, step_width, edge_confidence], dim=1)  # (N, 4)


class StepEdgeEncoderMLP(nn.Module):
    """Encodes the 4-dim StepEdgeExtractor output into a latent vector.

    Architecture: Linear(4→64) → ELU → Linear(64→64) → ELU
    Output: z_stair of shape (N, 64).
    """

    def __init__(
        self,
        grid_rows: int = 12,
        grid_cols: int = 8,
        resolution: float = 0.1,
        fwd_start: float = -0.55,
        softmax_temp: float = 50.0,
        edge_threshold: float = 0.02,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.extractor = StepEdgeExtractor(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            resolution=resolution,
            fwd_start=fwd_start,
            softmax_temp=softmax_temp,
            edge_threshold=edge_threshold,
        )

        # Input dim 3 → 4 to accommodate the new edge_confidence feature.
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, height_scan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            height_scan: ``(N, H*W)`` flattened height scan.

        Returns:
            z_stair: ``(N, hidden_dim)`` stair latent vector.
        """
        stair_features = self.extractor(height_scan)   # (N, 4)
        return self.mlp(stair_features)                # (N, hidden_dim)
