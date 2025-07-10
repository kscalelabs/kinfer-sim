import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from kbot_train import JOINT_BIASES, HumanoidWalkingTask, HumanoidWalkingTaskConfig


NUM_COMMANDS = 6  # all zeros for this test

# Analyzing JOINT_BIASES for symmetric pairs:
# Right side bias values vs Left side bias values:
# right_hip_yaw (0): 0.0     vs left_hip_yaw (6): 0.0        → SAME
# right_hip_roll (1): -0.1   vs left_hip_roll (7): +0.1      → OPPOSITE
# right_hip_pitch (2): -0.4  vs left_hip_pitch (8): -0.4     → SAME
# right_knee_pitch (3): -0.8 vs left_knee_pitch (9): -0.8    → SAME
# right_ankle_pitch (4): -0.4 vs left_ankle_pitch (10): -0.4 → SAME  
# right_ankle_roll (5): -0.1 vs left_ankle_roll (11): +0.1   → OPPOSITE
# right_shoulder_pitch (16): 0.0 vs left_shoulder_pitch (12): 0.0 → SAME
# right_shoulder_roll (17): -0.2 vs left_shoulder_roll (13): +0.2 → OPPOSITE
# right_elbow_roll (18): +0.2 vs left_elbow_roll (14): -0.2  → OPPOSITE
# right_gripper_roll (19): 0.0 vs left_gripper_roll (15): 0.0 → SAME

# Left and right joint indices
LEFT_JOINT_INDICES = jnp.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
RIGHT_JOINT_INDICES = jnp.array([0, 1, 2, 3, 4, 5, 16, 17, 18, 19])

# Inversion mask for right joints (1 = normal, -1 = invert)
RIGHT_INVERSION_MASK = jnp.array([1, -1, 1, 1, 1, -1, -1, -1, -1, 1])  # hip_roll, ankle_roll, shoulder_roll, elbow_roll inverted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=str, help="Output path for the kinfer model")
    args = parser.parse_args()

    # Create a dummy task to get the mujoco model and joint names
    config = HumanoidWalkingTaskConfig(
        num_envs=1,
        batch_size=1,
        rollout_length_seconds=1.0,
        dt=0.001,
        ctrl_dt=0.02,
        iterations=1,
        ls_iterations=1,
    )
    task = HumanoidWalkingTask(config)
    mujoco_model = task.get_mujoco_model()
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