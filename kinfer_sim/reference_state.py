from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class IdealPositionTracker:
    """Track the ideal (x, y) world position given body-frame velocity commands."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heading_rad: float = 0.0

    def reset(
        self,
        origin_xy: tuple[float, float] | None = None,
        heading_rad: float = 0.0,
    ) -> None:
        """Re-initialise reference state."""
        self.pos[:] = origin_xy if origin_xy is not None else (0.0, 0.0)
        self.heading_rad = float(heading_rad)

    def step(
        self,
        v_cmd_body_xy: tuple[float, float],
        dt: float,
        heading_rad: float | None = None,
    ) -> None:
        """
        Integrate the body-frame (vx, vy) command into world coordinates.

        If ``heading_rad`` is supplied, use it for this step; otherwise the
        heading captured at ``reset`` is used (keeps API minimal).
        """
        h = float(self.heading_rad if heading_rad is None else heading_rad)
        c, s = np.cos(h), np.sin(h)

        # Rotate body-frame velocity → world-frame.
        vx_b, vy_b = v_cmd_body_xy
        vx_w = c * vx_b - s * vy_b
        vy_w = s * vx_b + c * vy_b

        self.pos += np.array([vx_w, vy_w], dtype=np.float32) * dt