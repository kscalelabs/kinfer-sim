"""Defines a K-Infer model provider for the Mujoco simulator."""

import logging
from abc import ABC, abstractmethod
from typing import Sequence, cast

import numpy as np
from kinfer.rust_bindings import ModelProviderABC, PyModelMetadata

from kinfer_sim.simulator import MujocoSimulator

logger = logging.getLogger(__name__)


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


class ModelProvider(ModelProviderABC):
    simulator: MujocoSimulator
    quat_name: str
    acc_name: str
    gyro_name: str
    arrays: dict[str, np.ndarray]
    keyboard_state: InputState

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
        return self

    def get_inputs(self, input_types: list[str], metadata: PyModelMetadata) -> dict[str, np.ndarray]:
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
                inputs[input_type] = self.get_joint_angles(metadata.joint_names)
            elif input_type == "joint_angular_velocities":
                inputs[input_type] = self.get_joint_angular_velocities(metadata.joint_names)
            elif input_type == "projected_gravity":
                inputs[input_type] = self.get_projected_gravity()
            elif input_type == "accelerometer":
                inputs[input_type] = self.get_accelerometer()
            elif input_type == "gyroscope":
                inputs[input_type] = self.get_gyroscope()
            elif input_type == "command":
                inputs[input_type] = self.get_command(metadata)
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

    def get_projected_gravity(self) -> np.ndarray:
        gravity = self.simulator._model.opt.gravity
        quat_name = self.quat_name
        sensor = self.simulator._data.sensor(quat_name)
        proj_gravity = rotate_vector_by_quat(gravity, sensor.data, inverse=True)
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

    def get_command(self, metadata: PyModelMetadata) -> np.ndarray:
        # Get the expected number of commands from metadata
        num_commands = metadata.num_commands or 3  # Default to 3 if not specified

        # Take only the first num_commands values from keyboard state
        command_values = self.keyboard_state.value[:num_commands]

        # Pad with zeros if we don't have enough values
        while len(command_values) < num_commands:
            command_values.append(0.0)

        command_array = np.array(command_values, dtype=np.float32)
        self.arrays["command"] = command_array
        return command_array

    def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
        joint_names = metadata.joint_names
        assert action.shape == (len(joint_names),)
        self.arrays["action"] = action
        self.simulator.command_actuators({name: {"position": action[i]} for i, name in enumerate(joint_names)})
