"""Server and simulation loop for KOS."""

import asyncio
import itertools
import json
import logging
import tarfile
import time
import traceback
from queue import Empty
from pathlib import Path

import colorlogging
import numpy as np
import typed_argparse as tap
from kinfer.rust_bindings import PyModelRunner, metadata_from_json
from kmv.app.viewer import QtViewer
from kmv.utils.logging import VideoWriter, save_logs
from kscale.web.clients.client import WWWClient as K
from kscale.web.gen.api import RobotURDFMetadataOutput
from kscale.web.utils import get_robots_dir, should_refresh_file

from kinfer_sim.keyboard_listener import KeyboardListener
from kinfer_sim.keyboard_controller import KeyboardController
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
    local_model_dir: str | None = tap.arg(
        default=None,
        help="Path to local robot directory containing metadata.json and *.mjcf/*.xml (bypass K API)",
    )

    # Physics settings
    dt: float = tap.arg(default=0.0001, help="Simulation timestep")
    pd_update_frequency: float = tap.arg(default=1000.0, help="PD update frequency for the actuators (Hz)")
    no_gravity: bool = tap.arg(default=False, help="Enable gravity")
    start_height: float = tap.arg(default=1.1, help="Start height")
    initial_quat: str | None = tap.arg(default=None, help="Initial quaternion (w, x, y, z)")
    suspend: bool = tap.arg(default=False, help="Suspend robot base in place to prevent falling")
    quat_name: str = tap.arg(default="imu_site_quat", help="Name of the quaternion sensor")
    acc_name: str = tap.arg(default="imu_acc", help="Name of the accelerometer sensor")
    gyro_name: str = tap.arg(default="imu_gyro", help="Name of the gyroscope sensor")

    # Rendering settings
    no_render: bool = tap.arg(default=False, help="Enable rendering")
    render_frequency: float = tap.arg(default=1.0, help="Render frequency (Hz)")
    frame_width: int = tap.arg(default=640, help="Frame width")
    frame_height: int = tap.arg(default=480, help="Frame height")
    camera: str | None = tap.arg(default=None, help="Camera to use")
    save_path: str = tap.arg(default="logs", help="Path to save logs")
    save_video: bool = tap.arg(default=False, help="Save video")
    save_logs: bool = tap.arg(default=False, help="Save logs")
    free_camera: bool = tap.arg(default=False, help="Free camera")

    # control settings
    use_keyboard: bool = tap.arg(default=False, help="Use keyboard to control the robot")

    # Randomization settings
    command_delay_min: float | None = tap.arg(default=None, help="Minimum command delay")
    command_delay_max: float | None = tap.arg(default=None, help="Maximum command delay")
    drop_rate: float = tap.arg(default=0.0, help="Drop actions with this probability")
    joint_pos_delta_noise: float = tap.arg(default=0.0, help="Joint position delta noise (degrees)")
    joint_pos_noise: float = tap.arg(default=0.0, help="Joint position noise (degrees)")
    joint_vel_noise: float = tap.arg(default=0.0, help="Joint velocity noise (degrees/second)")
    joint_zero_noise: float = tap.arg(default=0.0, help="Joint zero noise (degrees)")
    accelerometer_noise: float = tap.arg(default=0.0, help="Accelerometer noise (m/s^2)")
    gyroscope_noise: float = tap.arg(default=0.0, help="Gyroscope noise (rad/s)")
    projected_gravity_noise: float = tap.arg(default=0.0, help="Projected gravity noise (m/s^2)")


class SimulationServer:
    def __init__(
        self,
        model_path: str | Path,
        model_metadata: RobotURDFMetadataOutput,
        config: ServerConfig,
        command_provider: KeyboardController | None,
        keyboard_listener: KeyboardListener,
    ) -> None:
        # Add control state
        self._paused = False
        self._control_queue = keyboard_listener.get_queue()
        
        initial_quat_str = config.initial_quat
        if initial_quat_str is not None:
            initial_quat = tuple(float(x) for x in initial_quat_str.split(","))
            if len(initial_quat) != 4:
                raise ValueError(f"Invalid initial quaternion: {initial_quat_str}")
        else:
            initial_quat = (1.0, 0.0, 0.0, 0.0)
        self.simulator = MujocoSimulator(
            model_path=model_path,
            model_metadata=model_metadata,
            dt=config.dt,
            gravity=not config.no_gravity,
            render_mode="offscreen" if config.no_render else "window",
            start_height=config.start_height,
            initial_quat=initial_quat,
            suspended=config.suspend,
            command_delay_min=config.command_delay_min,
            command_delay_max=config.command_delay_max,
            drop_rate=config.drop_rate,
            joint_pos_delta_noise=config.joint_pos_delta_noise,
            joint_pos_noise=config.joint_pos_noise,
            joint_vel_noise=config.joint_vel_noise,
            joint_zero_noise=config.joint_zero_noise,
            accelerometer_noise=config.accelerometer_noise,
            gyroscope_noise=config.gyroscope_noise,
            projected_gravity_noise=config.projected_gravity_noise,
            pd_update_frequency=config.pd_update_frequency,
            mujoco_scene=config.mujoco_scene,
            camera=config.camera,
            free_camera=config.free_camera,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
        )
        self._kinfer_path = config.kinfer_path
        self._stop_event = asyncio.Event()
        self._step_lock = asyncio.Semaphore(1)
        self._video_render_decimation = int(1.0 / config.render_frequency)
        self._quat_name = config.quat_name
        self._acc_name = config.acc_name
        self._gyro_name = config.gyro_name
        self._save_path = Path(config.save_path).expanduser().resolve()
        self._save_video = config.save_video
        self._save_logs = config.save_logs
        self._command_provider = command_provider
        self._joint_names: list[str] = load_joint_names(self._kinfer_path)
        self._plots_w_joint_names: frozenset[str] = frozenset({"joint_angles", "joint_velocities", "action"})

        self._video_writer: VideoWriter | None = None
        if self._save_video:
            self._save_path.mkdir(parents=True, exist_ok=True)

            fps = round(self.simulator._control_frequency)
            self._video_writer = VideoWriter(self._save_path / "video.mp4", fps=fps)

    def _to_scalars(self, name: str, arr: np.ndarray) -> dict[str, float]:
        """Convert a 1-D array into `{legend_name: value}` pairs.

        This is useful for including the joint names in the legend.
        """
        flat = arr.flatten()
        use_joint_names = name in self._plots_w_joint_names and len(flat) == len(self._joint_names)

        # Plot with indices if joint names are not needed
        if not use_joint_names:
            return {f"{name}_{idx}": float(v) for idx, v in enumerate(flat)}

        # Plot with joint names
        return {
            f"{name}_{idx} - {joint_name}": float(val)
            for idx, (joint_name, val) in enumerate(zip(self._joint_names, flat))
        }

    async def _simulation_loop(self) -> None:
        """Run the simulation loop asynchronously."""
        ctrl_dt = 1.0 / self.simulator._control_frequency

        # Initialize the model runner on the simulator.
        model_provider = ModelProvider(
            self.simulator,
            quat_name=self._quat_name,
            acc_name=self._acc_name,
            gyro_name=self._gyro_name,
            command_provider=self._command_provider,
        )
        model_runner = PyModelRunner(str(self._kinfer_path), model_provider, pre_fetch_time_ms=None)
        logs = []

        try:
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()

                # Check for control keys
                try:
                    key = self._control_queue.get_nowait()
                    if key == "key.space":
                        self._paused = not self._paused
                        logger.info("Simulation %s", "paused" if self._paused else "resumed")
                    elif key == "key.backspace":
                        logger.info("Resetting simulation...")
                        await self.simulator.reset()
                        if self._command_provider:
                            self._command_provider.reset_cmd()
                        logger.info("Simulation reset complete")
                except Empty:
                    pass

                if not self._paused:
                    # forward pass through policy
                    model_runner.step_and_take_action()

                    # Runs the simulation for one step.
                    async with self._step_lock:
                        for _ in range(self.simulator._sim_decimation):
                            await self.simulator.step()

                # Handle Qt viewer specific operations
                if isinstance(self.simulator._viewer, QtViewer):
                    if not self.simulator._viewer.is_open:
                        break

                    for n, a in model_provider.arrays.items():
                        self.simulator._viewer.push_plot_metrics(scalars=self._to_scalars(n, a), group=n)

                logs.append(model_provider.arrays.copy())
                if self._video_writer is not None and self.simulator.sim_time % self._video_render_decimation < ctrl_dt:
                    self._video_writer.append(self.simulator.read_pixels())

                # Wait until ctrl_dt has elapsed
                await asyncio.sleep(max(0.0, ctrl_dt - (time.perf_counter() - loop_start)))

        except Exception as e:
            logger.error("Simulation loop failed: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())

        finally:
            await self.stop()

            if self._video_writer is not None:
                self._video_writer.close()

            if isinstance(self.simulator._viewer, QtViewer):
                self.simulator._viewer.close()

            if self._save_logs:
                save_logs(logs, self._save_path / "logs")

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
    if config.local_model_dir:
        model_dir = Path(config.local_model_dir).expanduser().resolve()
        model_metadata = load_local_model_metadata(model_dir)
    else:
        async with K() as api:
            model_dir, model_metadata = await asyncio.gather(
                api.download_and_extract_urdf(config.mujoco_model_name, cache=(not config.no_cache)),
                get_model_metadata(api, config.mujoco_model_name),
            )

    model_path = find_mjcf(model_dir)

    keyboard_listener = KeyboardListener()

    server = SimulationServer(
        model_path=model_path,
        model_metadata=model_metadata,
        config=config,
        command_provider=KeyboardController(keyboard_listener.get_queue()) if config.use_keyboard else None,
        keyboard_listener=keyboard_listener,
    )

    await server.start()


def find_mjcf(model_dir: Path) -> Path:
    """Return the primary MJCF/XML file in model_dir."""
    try:
        return next(
            path
            for path in itertools.chain(
                model_dir.glob("*.mjcf"),
                model_dir.glob("*.xml"),
            )
        )
    except StopIteration as exc:
        raise FileNotFoundError(f"No *.mjcf or *.xml found in {model_dir}") from exc


async def run_server(config: ServerConfig) -> None:
    await serve(config=config)


def runner(args: ServerConfig) -> None:
    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(run_server(config=args))


def main() -> None:
    tap.Parser(ServerConfig).bind(runner).run()


def load_joint_names(kinfer_path: str | Path) -> list[str]:
    """Return the ordered joint-name list stored in a .kinfer archive."""
    kinfer_path = Path(kinfer_path)
    try:
        with tarfile.open(kinfer_path, "r:gz") as tar:
            metadata_file = tar.extractfile("metadata.json")
            if metadata_file is None:
                raise FileNotFoundError("'metadata.json' not found in archive")
            metadata = metadata_from_json(metadata_file.read().decode("utf-8"))
    except (tarfile.TarError, FileNotFoundError) as exc:
        raise ValueError(f"Could not read metadata from {kinfer_path}: {exc}") from exc

    joint_names = getattr(metadata, "joint_names", None)
    if not joint_names:
        raise ValueError(f"'joint_names' missing in metadata for {kinfer_path}")

    logger.info("Loaded %d joint names from model metadata", len(joint_names))
    return list(joint_names)


def load_local_model_metadata(model_dir: Path) -> RobotURDFMetadataOutput:
    """Load and validate local model metadata from ``metadata.json``.

    Coerces numeric actuator fields to strings to satisfy the schema.
    """
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")

    raw = json.loads(metadata_path.read_text())
    act_meta = raw.get("actuator_type_to_metadata")
    if isinstance(act_meta, dict):
        for _, v in act_meta.items():
            if isinstance(v, dict):
                for k2, v2 in list(v.items()):
                    if isinstance(v2, (int, float)):
                        v[k2] = str(v2)

    return RobotURDFMetadataOutput.model_validate(raw)


if __name__ == "__main__":
    # python -m kinfer_sim.server
    main()
