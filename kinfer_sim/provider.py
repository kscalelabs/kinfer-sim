"""Defines a K-Infer model provider for the Mujoco simulator."""

import logging
from typing import Sequence, cast
from queue import Queue

import numpy as np
from kinfer.rust_bindings import ModelProviderABC

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

def euler_to_quat(euler_3: np.ndarray) -> np.ndarray:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z).

    Args:
        euler_3: The roll, pitch, yaw angles, shape (*, 3).

    Returns:
        The quaternion with shape (*, 4).
    """
    # Extract roll, pitch, yaw from input
    roll, pitch, yaw = np.split(euler_3, 3, axis=-1)

    # Calculate trigonometric functions for each angle
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # Calculate quaternion components using the conversion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Combine into quaternion [w, x, y, z]
    quat = np.concatenate([w, x, y, z], axis=-1)

    # Normalize the quaternion
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat

def rotate_quat(q1, q2):
    """Rotate quaternion q1 by quaternion q2.
    
    Args:
        q1: quaternion to be rotated [w, x, y, z]
        q2: quaternion to rotate by [w, x, y, z]
    
    Returns:
        rotated quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

def quat_to_euler(quat_4: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (np.linalg.norm(quat_4, axis=-1, keepdims=True) + eps)
    w, x, y, z = np.split(quat_4, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)

    # Handle edge cases where |sinp| >= 1
    pitch = np.where(
        np.abs(sinp) >= 1.0,
        np.sign(sinp) * np.pi / 2.0,  # Use 90 degrees if out of range
        np.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.concatenate([roll, pitch, yaw], axis=-1)


class ModelProvider(ModelProviderABC):
    simulator: MujocoSimulator
    quat_name: str
    acc_name: str
    gyro_name: str
    arrays: dict[str, np.ndarray]
    key_queue: Queue | None

    def __new__(
        cls,
        simulator: MujocoSimulator,
        key_queue: Queue | None,
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
        self.key_queue = key_queue
        self.heading = 0.0
        return self
    
    def process_key_queue(self):
        if not hasattr(self, 'command_array'):
            self.command_array = np.zeros(6) # vx vy wz base_height roll pitch
            quat = self.simulator._data.xquat[1] # TODO HACK obs - need heading on real robot
            self.heading = quat_to_euler(quat)[2]

        # READ THE KEYS AND UDATE THE COMMAND ARRAY
        while not self.key_queue.empty():
            key = self.key_queue.get()
            key = key.strip("'")

            # reset commands
            if key == '0' or key == 'key.backspace':
                self.command_array *= 0
            
            # lin vel
            if key == 'w':
                self.command_array[0] += 0.1
            elif key == 's':
                self.command_array[0] -= 0.1
            if key == 'a':
                self.command_array[1] += 0.1
            elif key == 'd':
                self.command_array[1] -= 0.1

            # ang vel
            if key == 'q':
                self.command_array[2] += 0.1
            elif key == 'e':
                self.command_array[2] -= 0.1

            # height
            if key == '=':
                self.command_array[3] += 0.1
            elif key == '-':
                self.command_array[3] -= 0.1
            
            # base orient
            if key == 'r':
                self.command_array[4] += 0.1
            elif key == 'f':
                self.command_array[4] -= 0.1
            if key == 't':
                self.command_array[5] += 0.1
            elif key == 'g':
                self.command_array[5] -= 0.1

    def get_joint_angles(self, joint_names: Sequence[str]) -> np.ndarray:
        angles = [float(self.simulator._data.joint(joint_name).qpos) for joint_name in joint_names]
        angles_array = np.array(angles, dtype=np.float32)
        self.arrays["joint_angles"] = angles_array
        return angles_array

    def get_joint_angular_velocities(self, joint_names: Sequence[str]) -> np.ndarray:
        velocities = [float(self.simulator._data.joint(joint_name).qvel) for joint_name in joint_names]
        velocities_array = np.array(velocities, dtype=np.float32)
        self.arrays["joint_velocities"] = velocities_array
        return velocities_array

    def get_projected_gravity(self) -> np.ndarray:
        gravity = self.simulator._model.opt.gravity
        quat_name = self.quat_name
        sensor = self.simulator._data.sensor(quat_name)
        proj_gravity = rotate_vector_by_quat(gravity, sensor.data, inverse=True)
        self.arrays["projected_gravity"] = proj_gravity
        return proj_gravity

    def get_accelerometer(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.acc_name)
        acc_array = np.array(sensor.data, dtype=np.float32)
        self.arrays["accelerometer"] = acc_array
        return acc_array

    def get_gyroscope(self) -> np.ndarray:
        sensor = self.simulator._data.sensor(self.gyro_name)
        gyro_array = np.array(sensor.data, dtype=np.float32)
        self.arrays["gyroscope"] = gyro_array
        return gyro_array

    def get_time(self) -> np.ndarray:
        time = self.simulator._data.time
        time_array = np.array([time], dtype=np.float32)
        self.arrays["time"] = time_array
        return time_array

    def get_command(self) -> np.ndarray:
        # Process any queued keyboard commands
        if self.key_queue is not None: # TODO this here??
            self.process_key_queue()
            logging.info(f"command array: [{', '.join(f'{x:.2f}' for x in self.command_array)}]")

        self.heading += self.command_array[2] * self.simulator._control_dt
        inv_heading_quat = euler_to_quat(np.array([0, 0, -self.heading]))
        quat = self.simulator._data.xquat[1]
        quat = rotate_quat(quat, inv_heading_quat)

        command_obs = np.concatenate([
            self.command_array[:3],
            quat, # TODO HACK obs - need heading on real robot
            np.zeros_like([self.heading]), # TODO i dont want to feed a useless 0 to model but training code has it
            self.command_array[3:],
        ])

        self.arrays["command"] = command_obs
        return command_obs # this is not the problem!

    def take_action(self, joint_names: Sequence[str], action: np.ndarray) -> None:
        assert action.shape == (len(joint_names),)
        self.arrays["action"] = action
        self.simulator.command_actuators({name: {"position": action[i]} for i, name in enumerate(joint_names)})
