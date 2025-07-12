"""Actuators for kinfer-sim."""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Type, TypedDict

import numpy as np
from kscale.web.gen.api import ActuatorMetadataOutput, JointMetadataOutput

logger = logging.getLogger(__name__)


def _as_float(value: str | float | None, *, default: float | None = None) -> float:
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

    def __init__(
        self,
        *,
        kp: float,
        kd: float,
        max_torque: float | None = None,
        joint_min: float,
        joint_max: float,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.max_torque = max_torque
        self.joint_min = joint_min
        self.joint_max = joint_max

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
        joint_min, joint_max = get_joint_limits_from_metadata(joint_meta)
        return cls(
            kp=_as_float(joint_meta.kp),
            kd=_as_float(joint_meta.kd),
            max_torque=max_torque,
            joint_min=joint_min,
            joint_max=joint_max,
        )

    def clamp_position_target(self, target_position: float) -> float:
        """Clamp target position to be within joint limits."""
        return float(np.clip(target_position, self.joint_min, self.joint_max))

    def get_ctrl(
        self,
        cmd: ActuatorCommandDict,
        *,
        qpos: float,
        qvel: float,
        dt: float,
    ) -> float:
        # Clamp target position to joint limits
        target_position = cmd.get("position", 0.0)
        clamped_position = self.clamp_position_target(target_position)

        # Log warning if position was clamped
        if target_position != clamped_position:
            logger.warning(
                "Clamped position from %.3f to %.3f (limits: [%.3f, %.3f])",
                target_position,
                clamped_position,
                self.joint_min,
                self.joint_max,
            )

        torque = (
            self.kp * (clamped_position - qpos) + self.kd * (cmd.get("velocity", 0.0) - qvel) + cmd.get("torque", 0.0)
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


def get_servo_deadband() -> tuple[float, float]:
    """Get deadband values based on current servo configuration."""
    encoder_resolution = 0.087 * math.pi / 180  # radians

    pos_deadband = 2 * encoder_resolution
    neg_deadband = 2 * encoder_resolution

    return pos_deadband, neg_deadband


def trapezoidal_step(
    state: PlannerState,
    target_pos: float,
    *,
    v_max: float,
    a_max: float,
    dt: float,
    positive_deadband: float,
    negative_deadband: float,
) -> PlannerState:
    """Scalar trapezoidal velocity planner with deadband logic."""
    position_error = target_pos - state.position

    # Determine which deadband to use based on error direction
    deadband_threshold = positive_deadband if position_error >= 0 else negative_deadband

    in_deadband = abs(position_error) <= deadband_threshold

    if in_deadband:
        # Deadband behavior: gradually decay velocity
        decay_factor = 0.8  # Tunable parameter - could be measured from real servo
        new_velocity = state.velocity * decay_factor
        new_position = state.position + new_velocity * dt
    else:
        # Planning behavior: normal trapezoidal planning
        target_direction = 1.0 if position_error >= 0 else -1.0

        # Calculate stopping distance for current velocity
        stopping_distance = abs(state.velocity**2) / (2 * a_max)

        # Check if velocity is aligned with target direction
        velocity_direction = 1.0 if state.velocity >= 0 else -1.0
        moving_towards_target = velocity_direction * target_direction >= 0

        should_accelerate = moving_towards_target and abs(position_error) > stopping_distance

        # Choose acceleration
        if should_accelerate:
            acceleration = target_direction * a_max  # Accelerate towards target
        else:
            acceleration = -velocity_direction * a_max  # Decelerate (oppose current velocity)

        # Handle zero velocity case
        if abs(state.velocity) < 1e-6:
            acceleration = target_direction * a_max  # If stopped, accelerate towards target

        new_velocity = state.velocity + acceleration * dt
        new_velocity = max(-v_max, min(v_max, new_velocity))
        new_position = state.position + new_velocity * dt

    return PlannerState(position=new_position, velocity=new_velocity)


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
        positive_deadband: float,
        negative_deadband: float,
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
        self._state: PlannerState | None = None
        self.positive_deadband = positive_deadband
        self.negative_deadband = negative_deadband

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
            v_max=_as_float(actuator_meta.max_velocity, default=2.0),
            a_max=_as_float(actuator_meta.amax, default=17.45),
            dt=dt,
            positive_deadband=get_servo_deadband()[0],
            negative_deadband=get_servo_deadband()[1],
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
            positive_deadband=self.positive_deadband,
            negative_deadband=self.negative_deadband,
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


def get_joint_limits_from_metadata(joint_meta: JointMetadataOutput) -> tuple[float, float]:
    """Extract joint limits from joint metadata, converting degrees to radians."""
    if joint_meta.min_angle_deg is None:
        raise ValueError("Joint %s: minimum angle limit not specified" % joint_meta.id)
    if joint_meta.max_angle_deg is None:
        raise ValueError("Joint %s: maximum angle limit not specified" % joint_meta.id)

    joint_min = math.radians(float(joint_meta.min_angle_deg))
    joint_max = math.radians(float(joint_meta.max_angle_deg))

    return joint_min, joint_max


def create_actuator(
    joint_meta: JointMetadataOutput,
    actuator_meta: ActuatorMetadataOutput | None,
    *,
    dt: float,
) -> Actuator:
    """Create an actuator instance from metadata."""
    act_type = (joint_meta.actuator_type or "").lower()
    for prefix, cls in _actuator_registry.items():
        if act_type.startswith(prefix):
            return cls.from_metadata(joint_meta, actuator_meta, dt=dt)

    logger.warning("Unknown actuator type '%s', defaulting to PD", act_type)
    return PositionActuator.from_metadata(joint_meta, actuator_meta, dt=dt)
