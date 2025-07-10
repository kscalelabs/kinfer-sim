"""Simple kinfer model that sends zeros to all the joints."""

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


SIM_DT = 0.02

# Joint biases, these are in the order that they appear in the neural network.
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


NUM_COMMANDS = 6  # placeholder for tests

StepFn = Callable[
    [Array, Array, Array, Array, Array, Array],  # state inputs
    tuple[Array, Array],  # (targets, carry)
]


@dataclass
class Recipe:
    name: str
    init_fn: Callable[[], Array]
    step_fn: StepFn


def get_mujoco_model() -> mujoco.MjModel:
    """Get the MuJoCo model for the K-Bot."""
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")


def get_joint_names() -> list[str]:
    model = get_mujoco_model()
    return ksim.get_joint_names_in_order(model)[1:]  # drop root joint


def make_zero_recipe(num_joints: int, dt: float) -> Recipe:
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
        t = carry[0] + dt
        return jnp.zeros(num_joints), jnp.array([t])

    return Recipe("kbot_zero_position", init_fn, step_fn)


def get_bias_vector(joint_names: list[str]) -> jnp.ndarray:
    """Return an array of neutral/bias angles ordered like `joint_names`."""
    bias_map = {name: bias for name, bias, _ in JOINT_BIASES}
    return jnp.array([bias_map[name] for name in joint_names])


def make_bias_recipe(joint_names: list[str], dt: float) -> Recipe:
    bias_vec = get_bias_vector(joint_names)
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
        t = carry[0] + dt
        return bias_vec, jnp.array([t])

    return Recipe("kbot_bias_position", init_fn, step_fn)


def build_kinfer(recipe: Recipe, joint_names: list[str], out_dir: Path) -> Path:
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=NUM_COMMANDS,
        carry_size=(1,),
    )
    kinfer_blob = pack(
        export_fn(recipe.init_fn, metadata),
        export_fn(recipe.step_fn, metadata),
        metadata,
    )
    out_path = out_dir / f"{recipe.name}.kinfer"
    out_path.write_bytes(kinfer_blob)
    return out_path


def main() -> None:
    colorlogging.configure()
    parser = argparse.ArgumentParser()
    default_output = Path(__file__).parent / "outputs"
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Output path for the kinfer model (default: %(default)s)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = get_joint_names()
    num_joints = len(joint_names)
    logger.info("Number of joints: %s", num_joints)
    logger.info("Joint names: %s", joint_names)

    recipes = [
        make_zero_recipe(num_joints, SIM_DT),
        make_bias_recipe(joint_names, SIM_DT),
    ]
    for recipe in recipes:
        out_path = build_kinfer(recipe, joint_names, out_dir)
        logger.info("kinfer model written to %s", out_path)


if __name__ == "__main__":
    main()
