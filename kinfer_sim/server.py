"""Server and simulation loop for KOS."""

import asyncio
import itertools
import logging
import time
import traceback
from pathlib import Path

import colorlogging
import typed_argparse as tap
from kinfer.rust_bindings import PyModelRunner
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from kscale.web.utils import get_robots_dir, should_refresh_file

from kinfer_sim.provider import ModelProvider
from kinfer_sim.simulator import MujocoSimulator

logger = logging.getLogger(__name__)


class ServerConfig(tap.TypedArgs):
    kinfer_path: str = tap.arg(positional=True, help="Path to the K-Infer model to load")

    # Mujoco settings
    mujoco_model_name: str = tap.arg(positional=True, help="Name of the Mujoco model to simulate")
    mujoco_scene: str = tap.arg(default="smooth", help="Mujoco scene to use")
    no_cache: bool = tap.arg(default=False, help="Don't use cached metadata")
    debug: bool = tap.arg(default=False, help="Enable debug logging")

    # Physics settings
    dt: float = tap.arg(default=0.0001, help="Simulation timestep")
    no_gravity: bool = tap.arg(default=False, help="Enable gravity")
    start_height: float = tap.arg(default=1.5, help="Start height")

    # Rendering settings
    no_render: bool = tap.arg(default=False, help="Enable rendering")
    render_frequency: float = tap.arg(default=1.0, help="Render frequency (Hz)")
    frame_width: int = tap.arg(default=640, help="Frame width")
    frame_height: int = tap.arg(default=480, help="Frame height")
    camera: str | None = tap.arg(default=None, help="Camera to use")

    # Randomization settings
    command_delay_min: float | None = tap.arg(default=None, help="Minimum command delay")
    command_delay_max: float | None = tap.arg(default=None, help="Maximum command delay")
    joint_pos_delta_noise: float = tap.arg(default=0.0, help="Joint position delta noise (degrees)")
    joint_pos_noise: float = tap.arg(default=0.0, help="Joint position noise (degrees)")
    joint_vel_noise: float = tap.arg(default=0.0, help="Joint velocity noise (degrees/second)")


class SimulationServer:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        config: ServerConfig,
    ) -> None:
        self.simulator = MujocoSimulator(
            model_path=model_path,
            model_metadata=model_metadata,
            dt=config.dt,
            gravity=not config.no_gravity,
            render_mode="offscreen" if config.no_render else "window",
            start_height=config.start_height,
            command_delay_min=config.command_delay_min,
            command_delay_max=config.command_delay_max,
            joint_pos_delta_noise=config.joint_pos_delta_noise,
            joint_pos_noise=config.joint_pos_noise,
            joint_vel_noise=config.joint_vel_noise,
            mujoco_scene=config.mujoco_scene,
            camera=config.camera,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
        )
        self._kinfer_path = config.kinfer_path
        self._stop_event = asyncio.Event()
        self._step_lock = asyncio.Semaphore(1)
        self._render_decimation = int(1.0 / config.render_frequency)

    async def _simulation_loop(self) -> None:
        """Run the simulation loop asynchronously."""
        start_time = time.time()
        last_fps_time = start_time
        num_renders = 0
        num_steps = 0
        fps_update_interval = 1.0  # Update FPS every second

        # Initialize the model runner on the simulator.
        model_provider = ModelProvider(self.simulator)
        model_runner = PyModelRunner(str(self._kinfer_path), model_provider)
        carry = model_runner.init()

        try:
            while not self._stop_event.is_set():
                # Runs the simulation for one step.
                async with self._step_lock:
                    for _ in range(self.simulator._sim_decimation):
                        await self.simulator.step()

                # Runs the model runner for one step.
                output, carry = model_runner.step(carry)
                model_runner.take_action(output)

                if num_steps % self._render_decimation == 0:
                    await self.simulator.render()
                    num_renders += 1

                # Sleep until the next control update, to avoid rendering
                # faster than real-time.
                current_time = time.time()
                if current_time < self.simulator._sim_time:
                    await asyncio.sleep(self.simulator._sim_time - current_time)
                num_steps += 1

                # Calculate and log FPS
                if current_time - last_fps_time >= fps_update_interval:
                    fps = num_steps / (current_time - last_fps_time)
                    render_fps = num_renders / (current_time - last_fps_time)
                    logger.info(
                        "FPS: %.2f, Render FPS: %.2f, Simulation time: %f",
                        fps,
                        render_fps,
                        self.simulator._sim_time,
                    )
                    num_steps = 0
                    num_renders = 0
                    last_fps_time = current_time

        except Exception as e:
            logger.error("Simulation loop failed: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())

        finally:
            await self.stop()

    async def start(self) -> None:
        """Start both the gRPC server and simulation loop asynchronously."""
        sim_task = asyncio.create_task(self._simulation_loop())

        try:
            await sim_task
        except asyncio.CancelledError:
            await self.stop()

    async def stop(self) -> None:
        """Stop the simulation and cleanup resources asynchronously."""
        logger.info("Shutting down simulation...")
        self._stop_event.set()
        await self.simulator.close()


async def get_model_metadata(api: K, model_name: str, cache: bool = True) -> RobotURDFMetadataOutput:
    model_path = get_robots_dir() / model_name / "metadata.json"
    if cache and model_path.exists() and not should_refresh_file(model_path):
        return RobotURDFMetadataOutput.model_validate_json(model_path.read_text())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    robot_class = await api.get_robot_class(model_name)
    metadata = robot_class.metadata
    if metadata is None:
        raise ValueError(f"No metadata found for model {model_name}")
    model_path.write_text(metadata.model_dump_json())
    return metadata


async def serve(config: ServerConfig) -> None:
    async with K() as api:
        model_dir, model_metadata = await asyncio.gather(
            api.download_and_extract_urdf(config.mujoco_model_name, cache=(not config.no_cache)),
            get_model_metadata(api, config.mujoco_model_name),
        )

    model_path = next(
        (
            path
            for path in itertools.chain(
                model_dir.glob("*.mjcf"),
                model_dir.glob("*.xml"),
            )
        )
    )

    server = SimulationServer(
        model_path=model_path,
        model_metadata=model_metadata,
        config=config,
    )
    await server.start()


async def run_server(config: ServerConfig) -> None:
    await serve(config=config)


def runner(args: ServerConfig) -> None:
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_server(config=args))


def main() -> None:
    tap.Parser(ServerConfig).bind(runner).run()


if __name__ == "__main__":
    # python -m kinfer_sim.server
    main()
