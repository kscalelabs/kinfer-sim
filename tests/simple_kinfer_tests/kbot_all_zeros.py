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