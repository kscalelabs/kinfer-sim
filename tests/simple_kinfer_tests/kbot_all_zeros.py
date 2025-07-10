import argparse
import asyncio
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

# Joint biases - copied from kbot_train.py
# Joint name, target position, penalty weight.
JOINT_BIASES: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),
    ("dof_right_elbow_02", math.radians(90.0), 1.0),
    ("dof_right_wrist_00", 0.0, 1.0),
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),
    ("dof_left_wrist_00", 0.0, 1.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0), 0.01),
    ("dof_right_hip_roll_03", math.radians(-0.0), 2.0),
    ("dof_right_hip_yaw_03", 0.0, 2.0),
    ("dof_right_knee_04", math.radians(-50.0), 0.01),
    ("dof_right_ankle_02", math.radians(30.0), 1.0),
    ("dof_left_hip_pitch_04", math.radians(20.0), 0.01),
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),
    ("dof_left_hip_yaw_03", 0.0, 2.0),
    ("dof_left_knee_04", math.radians(50.0), 0.01),
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),
]

NUM_COMMANDS = 6  # all zeros for this test


def get_mujoco_model() -> mujoco.MjModel:
    """Get the MuJoCo model for the K-Bot."""
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Output path for the kinfer model")
    args = parser.parse_args()

    # Get the mujoco model and joint names
    mujoco_model = get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Remove root joint

    # Extract just the bias values in order
    joint_bias_values = jnp.array([bias for _, bias, _ in JOINT_BIASES])
    
    # Carry shape to store time
    carry_shape = (1,)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        initial_heading: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        # Update time
        time = carry[0] + 0.02  # Increment by ctrl_dt
        
        # Just return the bias values (neutral positions) for all joints
        all_targets = joint_bias_values
        
        new_carry = jnp.array([time])
        return all_targets, new_carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS,
        carry_size=carry_shape,
    )

    print(f"Creating zero action model")
    print(f"All joints will hold their neutral/bias positions")
    
    init_onnx = export_fn(init_fn, metadata)
    step_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kinfer_model)
    print(f"Zero action kinfer model written to {out_path}")


if __name__ == "__main__":
    main()