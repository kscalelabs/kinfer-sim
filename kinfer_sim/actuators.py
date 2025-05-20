"""Actuator models for kinfer‑sim.

This first implementation keeps the exact PD‑torque behaviour that was
previously hard‑coded in `simulator.py`.  Each joint gets its own Python
object, which means every actuator is stateful by construction.  In later
patches the factory will return specialised subclasses (e.g. Feetech and
Robstride) without touching the simulator again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np

__all__ = [
    "Actuator",
    "PositionActuator",
    "create_actuator",
]


class Actuator(ABC):
    """Abstract per‑joint actuator."""

    Command = Mapping[str, float]

    def __init__(self, *, kp: float, kd: float, max_torque: float | None = None) -> None:
        if kp < 0 or kd < 0:
            raise ValueError("`kp` and `kd` must be non‑negative")
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_torque = None if max_torque is None else float(max_torque)

    @abstractmethod
    def get_ctrl(
        self,
        target_cmd: Command,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        """Return the torque (N·m) to write into ``mj_data.ctrl``."""


class PositionActuator(Actuator):
    """PD actuator that treats the command as a desired position."""

    def get_ctrl(
        self,
        target_cmd: Actuator.Command,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        q_des = target_cmd.get("position", 0.0)
        qdot_des = target_cmd.get("velocity", 0.0)
        tau_add = target_cmd.get("torque", 0.0)

        torque = self.kp * (q_des - qpos) + self.kd * (qdot_des - qvel) + tau_add
        if self.max_torque is not None:
            torque = np.clip(torque, -self.max_torque, self.max_torque)
        return float(torque)


def create_actuator(
    actuator_type: str | None,
    *,
    kp: float,
    kd: float,
    max_torque: float | None = None,
) -> Actuator:
    """Return an actuator instance matching *actuator_type*.

    At this stage the type is ignored – every joint gets a
    :class:`PositionActuator` so that behaviour stays unchanged.  Future
    subclasses will branch on *actuator_type*.
    """
    return PositionActuator(kp=kp, kd=kd, max_torque=max_torque)
