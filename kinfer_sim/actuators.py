"""Actuators for kinfer-sim."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Type, TypedDict

import numpy as np
from kscale.web.gen.api import ActuatorMetadataOutput, JointMetadataOutput

logger = logging.getLogger(__name__)


def _as_float(value: str | float | None, *, default: Optional[float] = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("Numeric metadata field is missing")
        return default
    return float(value)


_actuator_registry: dict[str, Type["Actuator"]] = {}


def register_actuator(*prefixes: str) -> Callable[[Type["Actuator"]], Type["Actuator"]]:
    def decorator(cls: Type["Actuator"]) -> Type["Actuator"]:
        for p in prefixes:
            _actuator_registry[p.lower()] = cls
        return cls

    return decorator


class ActuatorCommandDict(TypedDict, total=False):
    """Keys an actuator may receive each control step."""

    position: float
    velocity: float
    torque: float


class Actuator(ABC):
    """Abstract per-joint actuator."""

    @classmethod
    @abstractmethod
    def from_metadata(
        cls,
        joint_meta: JointMetadataOutput,
        actuator_meta: ActuatorMetadataOutput | None,
        *,
        dt: float,
    ) -> "Actuator":
        """Create an actuator instance from K-Scale metadata."""

    @abstractmethod
    def get_ctrl(
        self,
        cmd: ActuatorCommandDict,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        """Return torque for the current physics step."""

    def configure(self, **kwargs: float) -> None:
        """Update actuator gains/limits at run-time.

        Sub-classes may override; the default implementation silently ignores
        unknown keys so callers don't need to special-case actuator types.
        """
        pass


@register_actuator("robstride", "position")
class PositionActuator(Actuator):
    """Plain PD controller with optional torque saturation."""

    def __init__(self, *, kp: float, kd: float, max_torque: Optional[float] = None) -> None:
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque

    @classmethod
    def from_metadata(
        cls,
        joint_meta: JointMetadataOutput,
        actuator_meta: ActuatorMetadataOutput | None,
        *,
        dt: float,
    ) -> "PositionActuator":
        max_torque = None
        if actuator_meta and actuator_meta.max_torque is not None:
            max_torque = float(actuator_meta.max_torque)
        return cls(kp=_as_float(joint_meta.kp), kd=_as_float(joint_meta.kd), max_torque=max_torque)

    def get_ctrl(
        self,
        cmd: ActuatorCommandDict,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        torque = (
            self.kp * (cmd.get("position", 0.0) - qpos)
            + self.kd * (cmd.get("velocity", 0.0) - qvel)
            + cmd.get("torque", 0.0)
        )
        if self.max_torque is not None:
            torque = float(np.clip(torque, -self.max_torque, self.max_torque))
        return torque

    def configure(self, **kwargs: float) -> None:
        if "kp" in kwargs:
            self.kp = kwargs["kp"]
        if "kd" in kwargs:
            self.kd = kwargs["kd"]
        if "max_torque" in kwargs:
            self.max_torque = kwargs["max_torque"]


@dataclass
class PlannerState:
    position: float
    velocity: float


def trapezoidal_step(
    state: PlannerState,
    target_pos: float,
    *,
    v_max: float,
    a_max: float,
    dt: float,
) -> PlannerState:
    """Scalar trapezoidal velocity planner."""
    pos_err = target_pos - state.position
    direction = 1.0 if pos_err >= 0 else -1.0
    stop_dist = (state.velocity**2) / (2 * a_max) if a_max > 0 else 0.0

    accel = direction * a_max if abs(pos_err) > stop_dist else -direction * a_max
    new_vel = state.velocity + accel * dt
    new_vel = max(-v_max, min(v_max, new_vel))

    # Prevent overshoot when decelerating past zero
    if direction * new_vel < 0:
        new_vel = 0.0

    new_pos = state.position + new_vel * dt
    return PlannerState(position=new_pos, velocity=new_vel)


@register_actuator("feetech")
class FeetechActuator(Actuator):
    """Duty-cycle model for Feetech STS servos."""

    def __init__(
        self,
        *,
        kp: float,
        kd: float,
        max_torque: float,
        max_pwm: float,
        vin: float,
        kt: float,
        r: float,
        error_gain: float,
        v_max: float,
        a_max: float,
        dt: float,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque
        self.max_pwm = max_pwm
        self.vin = vin
        self.kt = kt
        self.R = r
        self.error_gain = error_gain
        self.v_max = v_max
        self.a_max = a_max
        self.dt = dt
        self._state: Optional[PlannerState] = None

    @classmethod
    def from_metadata(
        cls,
        joint_meta: JointMetadataOutput,
        actuator_meta: ActuatorMetadataOutput | None,
        *,
        dt: float,
    ) -> "FeetechActuator":
        if actuator_meta is None:
            raise ValueError("Feetech actuator metadata missing")
        return cls(
            kp=_as_float(joint_meta.kp),
            kd=_as_float(joint_meta.kd),
            max_torque=_as_float(actuator_meta.max_torque),
            max_pwm=_as_float(actuator_meta.max_pwm, default=1.0),
            vin=_as_float(actuator_meta.vin, default=12.0),
            kt=_as_float(actuator_meta.kt, default=1.0),
            r=_as_float(actuator_meta.R, default=1.0),
            error_gain=_as_float(actuator_meta.error_gain, default=1.0),
            v_max=_as_float(actuator_meta.max_velocity, default=5.0),
            a_max=_as_float(actuator_meta.amax, default=17.45),
            dt=dt,
        )

    def get_ctrl(
        self,
        cmd: ActuatorCommandDict,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        if self._state is None:
            self._state = PlannerState(position=qpos, velocity=qvel)
        self._state = trapezoidal_step(
            self._state,
            target_pos=cmd.get("position", qpos),
            v_max=self.v_max,
            a_max=self.a_max,
            dt=self.dt,
        )
        pos_err = self._state.position - qpos
        vel_err = self._state.velocity - qvel
        duty = self.kp * self.error_gain * pos_err + self.kd * vel_err
        duty = float(np.clip(duty, -self.max_pwm, self.max_pwm))
        torque = duty * self.vin * self.kt / self.R
        return float(np.clip(torque, -self.max_torque, self.max_torque))

    def configure(self, **kwargs: float) -> None:
        if "kp" in kwargs:
            self.kp = kwargs["kp"]
        if "kd" in kwargs:
            self.kd = kwargs["kd"]
        if "max_torque" in kwargs:
            self.max_torque = kwargs["max_torque"]


def create_actuator(
    joint_meta: JointMetadataOutput,
    actuator_meta: ActuatorMetadataOutput | None,
    *,
    dt: float,
) -> Actuator:
    act_type = (joint_meta.actuator_type or "").lower()
    for prefix, cls in _actuator_registry.items():
        if act_type.startswith(prefix):
            return cls.from_metadata(joint_meta, actuator_meta, dt=dt)
    logger.warning("Unknown actuator type '%s', defaulting to PD", act_type)
    return PositionActuator.from_metadata(joint_meta, actuator_meta, dt=dt)
