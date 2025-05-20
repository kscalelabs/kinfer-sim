"""Actuators for kinfer-sim."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import numpy as np
from kscale.web.gen.api import ActuatorMetadataOutput, JointMetadataOutput

logger = logging.getLogger(__name__)

def _not_none(value: Any | None) -> Any:
    if value is None:
        raise ValueError("Required metadata field is missing")
    return value


def _as_float(value: str | float | None, *, default: Optional[float] = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("Numeric metadata field is missing")
        return default
    return float(value)




_actuator_registry: Dict[str, Type["Actuator"]] = {}


def register_actuator(*prefixes: str) -> callable:
    def decorator(cls: Type["Actuator"]) -> Type["Actuator"]:
        for p in prefixes:
            _actuator_registry[p.lower()] = cls
        return cls
    return decorator


# Base class

class Actuator(ABC):
    """Abstract per-joint actuator."""

    @abstractmethod
    def get_ctrl(self, cmd: Dict[str, float], *, qpos: float, qvel: float, dt: float) -> float:
        """Return torque for the current physics step."""


# Robstride / PD position actuator

@register_actuator("robstride", "position", "")
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
        return cls(kp=float(_not_none(joint_meta.kp)), kd=float(_not_none(joint_meta.kd)), max_torque=max_torque)

    def get_ctrl(self, cmd: Dict[str, float], *, qpos: float, qvel: float, dt: float) -> float:
        torque = (
            self.kp * (cmd.get("position", 0.0) - qpos)
            + self.kd * (cmd.get("velocity", 0.0) - qvel)
            + cmd.get("torque", 0.0)
        )
        if self.max_torque is not None:
            torque = float(np.clip(torque, -self.max_torque, self.max_torque))
        return torque


# Feetech actuator and planner

@dataclass
class PlannerState:
    position: float
    velocity: float


def trapezoidal_step(state: PlannerState, target_pos: float, *, v_max: float, a_max: float, dt: float) -> PlannerState:
    """Scalar trapezoidal velocity planner matching KSim implementation."""
    pos_err = target_pos - state.position
    direction = np.sign(pos_err)
    stop_dist = (state.velocity**2) / (2 * a_max)

    accel = np.where(np.abs(pos_err) > stop_dist, direction * a_max, -direction * a_max)
    new_vel = np.clip(state.velocity + accel * dt, -v_max, v_max)
    if direction * new_vel < 0:
        new_vel = 0.0
    new_pos = state.position + new_vel * dt
    return PlannerState(position=float(new_pos), velocity=float(new_vel))


@register_actuator("feetech")
class FeetechActuator(Actuator):
    """Duty-cycle model for Feetech STS servos (adapted from KSim)."""

    def __init__(
        self,
        *,
        kp: float,
        kd: float,
        max_torque: float,
        max_pwm: float,
        vin: float,
        kt: float,
        R: float,
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
        self.R = R
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
            kp=float(_not_none(joint_meta.kp)),
            kd=float(_not_none(joint_meta.kd)),
            max_torque=_as_float(actuator_meta.max_torque),
            max_pwm=_as_float(actuator_meta.max_pwm, default=1.0),
            vin=_as_float(actuator_meta.vin, default=12.0),
            kt=_as_float(actuator_meta.kt, default=1.0),
            R=_as_float(actuator_meta.R, default=1.0),
            error_gain=_as_float(actuator_meta.error_gain, default=1.0),
            v_max=_as_float(actuator_meta.max_velocity, default=5.0),
            a_max=_as_float(actuator_meta.amax, default=17.45),
            dt=dt,
        )

    def get_ctrl(self, cmd: Dict[str, float], *, qpos: float, qvel: float, dt: float) -> float:
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


# Factory

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