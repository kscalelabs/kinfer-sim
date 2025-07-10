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

from dataclasses import dataclass
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


NUM_COMMANDS = 6  # placeholder for tests

StepFn = Callable[
    [Array, Array, Array, Array, Array, Array],  # state inputs
    tuple[Array, Array],                        # (targets, carry)
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
    # Get the current script name and replace .py with .kinfer for default output
    default_output = Path(__file__).parent
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Output path for the kinfer model (default: %(default)s)",
    )
    args = parser.parse_args()

    joint_names = get_joint_names()
    num_joints = len(joint_names)
    logger.info("Number of joints: %s", num_joints)
    logger.info("Joint names: %s", joint_names)


    recipes = [
        make_zero_recipe(num_joints, 0.02),
    ]
    for recipe in recipes:
        out_path = build_kinfer(recipe, joint_names, Path(args.output))
        logger.info("kinfer model written to %s", out_path)

if __name__ == "__main__":
    main()
