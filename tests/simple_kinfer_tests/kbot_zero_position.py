"""Simple kinfer model that sends zeros to all the joints."""

import argparse
import asyncio
import logging
from pathlib import Path

import colorlogging
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

logger = logging.getLogger(__name__)


NUM_COMMANDS = 6  # placeholder for test


def get_mujoco_model() -> mujoco.MjModel:
    """Get the MuJoCo model for the K-Bot."""
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")


def main() -> None:
    colorlogging.configure()
    parser = argparse.ArgumentParser()
    # Get the current script name and replace .py with .kinfer for default output
    default_output = Path(__file__).stem + ".kinfer"
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Output path for the kinfer model (default: %(default)s)",
    )
    args = parser.parse_args()

    # Get the mujoco model and joint names
    mujoco_model = get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Remove root joint
    num_joints = len(joint_names)
    logger.info("Number of joints: %s", num_joints)

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

        # Return zeros for all joints
        all_targets = jnp.zeros(num_joints)

        new_carry = jnp.array([time])
        return all_targets, new_carry

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS,
        carry_size=carry_shape,
    )

    logger.info("Creating zero action model")
    logger.info("All joints will be set to 0.0")

    init_onnx = export_fn(init_fn, metadata)
    step_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_onnx, step_onnx, metadata)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(kinfer_model)
    logger.info("Zero action kinfer model written to %s", out_path)


if __name__ == "__main__":
    main()
