"""Defines a K-Infer model provider for the Mujoco simulator."""

import logging
from abc import ABC, abstractmethod
from typing import Sequence, cast

import numpy as np
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata

from kinfer_sim.simulator import MujocoSimulator

logger = logging.getLogger(__name__)


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Converts a quaternion to Euler angles.

    Args:
        quat: The quaternion to convert, shape (*, 4).

    Returns:
        The Euler angles, shape (*, 3).
    """
    eps: float = 1e-6  # small epsilon to avoid division by zero and NaNs

    # Ensure numpy array and normalize the quaternion to unit length
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + eps)

    # Split into components (expects quaternion in (w, x, y, z) order)
    w, x, y, z = np.split(quat, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0 + eps, 1.0 - eps)  # numerical safety
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Concatenate along the last dimension to maintain input shape semantics
    euler = np.concatenate([roll, pitch, yaw], axis=-1)
    return euler.astype(np.float32)


def rotate_vector_by_quat(vector: np.ndarray, quat: np.ndarray, inverse: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Rotates a vector by a quaternion.

    Args:
        vector: The vector to rotate, shape (*, 3).
        quat: The quaternion to rotate by, shape (*, 4).
        inverse: If True, rotate the vector by the conjugate of the quaternion.
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The rotated vector, shape (*, 3).
    """
    # Normalize quaternion
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + eps)
    w, x, y, z = np.split(quat, 4, axis=-1)

    if inverse:
        x, y, z = -x, -y, -z

    # Extract vector components
    vx, vy, vz = np.split(vector, 3, axis=-1)

    # Terms for x component
    xx = (
        w * w * vx
        + 2 * y * w * vz
        - 2 * z * w * vy
        + x * x * vx
        + 2 * y * x * vy
        + 2 * z * x * vz
        - z * z * vx
        - y * y * vx
    )

    # Terms for y component
    yy = (
        2 * x * y * vx
        + y * y * vy
        + 2 * z * y * vz
        + 2 * w * z * vx
        - z * z * vy
        + w * w * vy
        - 2 * w * x * vz
        - x * x * vy
    )

    # Terms for z component
    zz = (
        2 * x * z * vx
        + 2 * y * z * vy
        + z * z * vz
        - 2 * w * y * vx
        + w * w * vz
        + 2 * w * x * vy
        - y * y * vz
        - x * x * vz
    )

    return np.concatenate([xx, yy, zz], axis=-1)


class InputState(ABC):
    """Abstract base class for input state management."""

    value: list[float]

    @abstractmethod
    async def update(self, key: str) -> None:
        """Update the input state based on a key press."""
        pass


class JoystickInputState(InputState):
    """State to hold and modify commands based on joystick input."""

    value: list[float]

    def __init__(self) -> None:
        self.value = [1, 0, 0, 0, 0, 0, 0]

    async def update(self, key: str) -> None:
        if key == "w":
            self.value = [0, 1, 0, 0, 0, 0, 0]
        elif key == "s":
            self.value = [0, 0, 1, 0, 0, 0, 0]
        elif key == "a":
            self.value = [0, 0, 0, 0, 0, 1, 0]
        elif key == "d":
            self.value = [0, 0, 0, 0, 0, 0, 1]
        elif key == "q":
            self.value = [0, 0, 0, 1, 0, 0, 0]
        elif key == "e":
            self.value = [0, 0, 0, 0, 1, 0, 0]


class SimpleJoystickInputState(InputState):
    """State to hold and modify commands based on simple joystick input."""

    value: list[float]

    def __init__(self) -> None:
        self.value = [1, 0, 0, 0]

    async def update(self, key: str) -> None:
        if key == "w":
            self.value = [0, 1, 0, 0]
        elif key == "s":
            self.value = [0, 0, 1, 0]
        elif key == "a":
            self.value = [0, 0, 0, 1]
        elif key == "d":
            self.value = [1, 0, 0, 0]


class ControlVectorInputState(InputState):
    """State to hold and modify control vector commands based on keyboard input."""

    value: list[float]
    STEP_SIZE: float = 0.1

    def __init__(self) -> None:
        self.value = [0.0, 0.0, 0.0]  # x linear, y linear, z angular

    async def update(self, key: str) -> None:
        if key == "w":
            self.value[0] += self.STEP_SIZE
        elif key == "s":
            self.value[0] -= self.STEP_SIZE
        elif key == "a":
            self.value[1] -= self.STEP_SIZE
        elif key == "d":
            self.value[1] += self.STEP_SIZE
        elif key == "q":
            self.value[2] -= self.STEP_SIZE
        elif key == "e":
            self.value[2] += self.STEP_SIZE


class ExpandedControlVectorInputState(InputState):
    """State to hold and modify control vector commands based on keyboard input."""

    value: list[float]
    STEP_SIZE: float = 0.1

    def __init__(self) -> None:
        self.value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x linear, y linear, yaw, base height, roll, pitch

    async def update(self, key: str) -> None:
        if key == "w":
            self.value[0] += self.STEP_SIZE
        elif key == "s":
            self.value[0] -= self.STEP_SIZE
        elif key == "a":
            self.value[1] -= self.STEP_SIZE
        elif key == "d":
            self.value[1] += self.STEP_SIZE
        elif key == "q":
            self.value[2] -= self.STEP_SIZE
        elif key == "e":
            self.value[2] += self.STEP_SIZE
        elif key == "r":
            self.value[4] += self.STEP_SIZE
        elif key == "f":
            self.value[4] -= self.STEP_SIZE
        elif key == "t":
            self.value[5] += self.STEP_SIZE
        elif key == "g":
            self.value[5] -= self.STEP_SIZE


class GenericOHEInputState(InputState):
    """State to hold and modify control vector commands based on keyboard input."""

    value: list[float]

    def __init__(self, num_actions: int) -> None:
        self.value = [0.0] * num_actions

    async def update(self, key: str) -> None:
        if key.isdigit() and int(key) < len(self.value):
            self.value = [0.0] * len(self.value)
            self.value[int(key)] = 1.0


class CombinedInputState(InputState):
    """Multiple input states combined into a single state."""

    input_states: list[InputState]

    def __init__(self, input_states: list[InputState]) -> None:
        self.input_states = input_states

    async def update(self, key: str) -> None:
        for input_state in self.input_states:
            await input_state.update(key)

    @property
    def value(self) -> list[float]:
        return [item for sublist in [input_state.value for input_state in self.input_states] for item in sublist]

    @value.setter
    def value(self, new_value: list[float]) -> None:
        start_idx = 0
        for input_state in self.input_states:
            end_idx = start_idx + len(input_state.value)
            input_state.value = new_value[start_idx:end_idx]
            start_idx = end_idx


class ModelProvider(ModelProviderABC):
    simulator: MujocoSimulator
    quat_name: str
    acc_name: str
    gyro_name: str
    arrays: dict[str, np.ndarray]
    keyboard_state: InputState
    initial_heading: float

    def __new__(
        cls,
        simulator: MujocoSimulator,
        keyboard_state: InputState,
        quat_name: str = "imu_site_quat",
        acc_name: str = "imu_acc",
        gyro_name: str = "imu_gyro",
    ) -> "ModelProvider":
        self = cast(ModelProvider, super().__new__(cls))
        self.simulator = simulator
        self.quat_name = quat_name
        self.acc_name = acc_name
        self.gyro_name = gyro_name
        self.arrays = {}
        self.keyboard_state = keyboard_state
        self.initial_heading = quat_to_euler(self.simulator._data.sensor(self.quat_name).data)[2]
        return self

    def get_inputs(self, input_types: Sequence[str], metadata: PyModelMetadata) -> dict[str, np.ndarray]:
        """Get inputs for the model based on the requested input types.

        Args:
            input_types: List of input type names to retrieve
            metadata: Model metadata containing joint names and other info

        Returns:
            Dictionary mapping input type names to numpy arrays
        """
        inputs = {}
        for input_type in input_types:
            if input_type == "joint_angles":
                inputs[input_type] = self.get_joint_angles(metadata.joint_names)  # type: ignore[attr-defined]
            elif input_type == "joint_angular_velocities":
                inputs[input_type] = self.get_joint_angular_velocities(metadata.joint_names)  # type: ignore[attr-defined]
            elif input_type == "initial_heading":
                inputs[input_type] = np.array([self.initial_heading])
            elif input_type == "quaternion":
                inputs[input_type] = self.get_quaternion()
            elif input_type == "projected_gravity":
                inputs[input_type] = self.get_projected_gravity()
            elif input_type == "accelerometer":
                inputs[input_type] = self.get_accelerometer()
            elif input_type == "gyroscope":
                inputs[input_type] = self.get_gyroscope()
            elif input_type == "command":
                inputs[input_type] = self.get_command()
            elif input_type == "time":
                inputs[input_type] = self.get_time()
            else:
                raise ValueError(f"Unknown input type: {input_type}")
        return inputs

    def get_joint_angles(self, joint_names: Sequence[str]) -> np.ndarray:
        angles = [float(self.simulator._data.joint(joint_name).qpos) for joint_name in joint_names]
        angles_array = np.array(angles, dtype=np.float32)
        angles_array += np.random.normal(
            -self.simulator._joint_pos_noise, self.simulator._joint_pos_noise, angles_array.shape
        )
        self.arrays["joint_angles"] = angles_array
        return angles_array

    def get_joint_angular_velocities(self, joint_names: Sequence[str]) -> np.ndarray:
        velocities = [float(self.simulator._data.joint(joint_name).qvel) for joint_name in joint_names]
        velocities_array = np.array(velocities, dtype=np.float32)
        velocities_array += np.random.normal(
            -self.simulator._joint_vel_noise, self.simulator._joint_vel_noise, velocities_array.shape
        )
        self.arrays["joint_velocities"] = velocities_array
        return velocities_array

    def get_quaternion(self) -> np.ndarray:
        quat_name = self.quat_name
        sensor = self.simulator._data.sensor(quat_name)
        quat = sensor.data
        # quat = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        self.arrays["quaternion"] = quat
        return quat

    def get_projected_gravity(self) -> np.ndarray:
        gravity = self.simulator._model.opt.gravity
        quat = self.get_quaternion()
        proj_gravity = rotate_vector_by_quat(gravity, quat, inverse=True)
        proj_gravity += np.random.normal(
            -self.simulator._projected_gravity_noise, self.simulator._projected_gravity_noise, proj_gravity.shape
        )
        self.arrays["projected_gravity"] = proj_gravity
        return proj_gravity

    def get_accelerometer(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.acc_name)
        acc_array = np.array(sensor.data, dtype=np.float32)
        acc_array += np.random.normal(
            -self.simulator._accelerometer_noise, self.simulator._accelerometer_noise, acc_array.shape
        )
        self.arrays["accelerometer"] = acc_array
        return acc_array

    def get_gyroscope(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.gyro_name)
        gyro_array = np.array(sensor.data, dtype=np.float32)
        gyro_array += np.random.normal(
            -self.simulator._gyroscope_noise, self.simulator._gyroscope_noise, gyro_array.shape
        )
        self.arrays["gyroscope"] = gyro_array
        return gyro_array

    def get_time(self) -> np.ndarray:
        time = self.simulator._data.time
        time_array = np.array([time], dtype=np.float32)
        self.arrays["time"] = time_array
        return time_array

    def get_command(self) -> np.ndarray:
        command_array = np.array(self.keyboard_state.value, dtype=np.float32)
        self.arrays["command"] = command_array
        # logger.info(f"Command: {command_array}")
        return command_array

    def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
        joint_names = metadata.joint_names  # type: ignore[attr-defined]
        assert action.shape == (len(joint_names),)
        self.arrays["action"] = action
        self.simulator.command_actuators({name: {"position": action[i]} for i, name in enumerate(joint_names)})
