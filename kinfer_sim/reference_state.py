from dataclasses import dataclass, field
import numpy as np


@dataclass
class IdealPositionTracker:
    """
    Keeps track of the *ideal* world–space (x, y) position that the robot would
    occupy if it followed the commanded linear velocity perfectly.

    The tracker is intentionally agnostic to where the command originates
    (keyboard, network, RL policy, etc.); callers simply pass the current
    velocity command each control tick.
    """

    # In world coordinates, metres.
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def reset(self, origin_xy: tuple[float, float] | None = None) -> None:
        """Re-initialise to (0, 0) or to a supplied reference point."""
        self.pos[:] = origin_xy if origin_xy is not None else (0.0, 0.0)

    def step(self, v_cmd_xy: tuple[float, float], dt: float) -> None:
        """
        Forward-Euler integration of the velocity command.

        Args
        ----
        v_cmd_xy : (vx, vy) command in **world** frame – metres/second
        dt       : simulation/control time-step in seconds
        """
        self.pos += np.asarray(v_cmd_xy, dtype=np.float32) * dt