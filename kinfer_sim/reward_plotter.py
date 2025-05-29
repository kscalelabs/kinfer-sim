import asyncio
import importlib.util
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
from jaxtyping import Array
import pyqtgraph as pg
import numpy as np
import mujoco

import xax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    """ Simplified trajectory class mimicking the Trajectory class in ksim.types"""

    qpos: Array
    qvel: Array
    xpos: Array
    xquat: Array
    # ctrl: Array
    obs: dict[str, Array]
    command: dict[str, Array]
    # event_state: xax.FrozenDict[str, Array]
    # action: Array
    # done: Array
    # success: Array
    # timestep: Array
    # termination_components: xax.FrozenDict[str, Array]
    # aux_outputs: xax.FrozenDict[str, PyTree]


class RewardPlotter:
    def __init__(self, mujoco_model: mujoco.MjModel):
        path_to_train_file = "/home/bart/kbot-joystick/train.py"
        spec = importlib.util.spec_from_file_location("train", path_to_train_file)
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        
        # Get the actual rewards being used in train.py
        self.rewards = train.HumanoidWalkingTask.get_rewards(self=None, physics_model=mujoco_model)
        print("\n=== Found Reward Classes ===")
        for i, reward in enumerate(self.rewards, 1):
            print(f"\n{i}. {reward.__class__.__name__}")
            print(f"   {reward.__doc__ or 'No description available'}")
        print("\n" + "=" * 30 + "\n")

        # get observations
        self.observations = train.HumanoidWalkingTask.get_observations(self=None, physics_model=mujoco_model)
        self.observations = {obs.__class__.__name__: obs for obs in self.observations}
        print(f"found {len(self.observations)} observations")
        for i, obs in enumerate(self.observations):
            print(f"{i}: {obs}")

        # Initialize PyQtPlot window and widgets
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        
        # Create dictionaries to store plots, curves and data
        self.traj_data = {}
        self.plots = {}
        self.curves = {}
        self.plot_data = {}
        
        # Setup reward plots
        first_plot = None
        for i, reward in enumerate(self.rewards):
            name = reward.__class__.__name__
            self.plots[name] = self.win.addPlot(title=name)
            if first_plot is None:
                first_plot = self.plots[name]
            else:
                self.plots[name].setXLink(first_plot)  # Link x-axis to first plot
            self.plots[name].setLabel('left', 'Reward')
            self.plots[name].setLabel('bottom', 'Time')
            self.plots[name].showGrid(x=True, y=True, alpha=0.3)
            self.curves[name] = self.plots[name].plot(pen='y')
            self.plot_data[name] = []
            self.win.nextRow()

        # command plots
        additional_metrics = ['linvel', 'angvel', 'base_height', 'xyorientation']
        for metric in additional_metrics:
            self.plots[metric] = self.win.addPlot(title=metric.capitalize())
            self.plots[metric].setXLink(first_plot)  # Link x-axis to first plot
            self.plots[metric].setLabel('left', metric.capitalize())
            self.plots[metric].setLabel('bottom', 'Time')
            self.plots[metric].addLegend()
            self.plots[metric].showGrid(x=True, y=True, alpha=0.3)
            
            if metric == 'linvel':
                self.curves[metric] = {
                    'x_cmd': self.plots[metric].plot(pen=pg.mkPen('r', width=2, style=pg.QtCore.Qt.DashLine), name='X Command'),
                    'x_real': self.plots[metric].plot(pen=pg.mkPen('r', width=2), name='X Actual'),
                    'y_cmd': self.plots[metric].plot(pen=pg.mkPen('g', width=2, style=pg.QtCore.Qt.DashLine), name='Y Command'),
                    'y_real': self.plots[metric].plot(pen=pg.mkPen('g', width=2), name='Y Actual')
                }
            elif metric == 'angvel':
                self.curves[metric] = {
                    'wz_cmd': self.plots[metric].plot(pen=pg.mkPen('y', width=2, style=pg.QtCore.Qt.DashLine), name='ωz Command'),
                    # 'wz_real': self.plots[metric].plot(pen=pg.mkPen('y', width=2), name='ωz Actual')
                    'heading_cmd': self.plots[metric].plot(pen=pg.mkPen('y', width=2), name='Heading'),
                    # 'heading_real': self.plots[metric].plot(pen=pg.mkPen('y', width=2), name='Heading Actual')
                }
            elif metric == 'base_height':
                self.curves[metric] = {
                    'base_height_cmd': self.plots[metric].plot(pen=pg.mkPen('b', width=2, style=pg.QtCore.Qt.DashLine), name='Height Command'),
                    'base_height_real': self.plots[metric].plot(pen=pg.mkPen('b', width=2), name='Height Actual')
                }
            elif metric == 'xyorientation':
                self.curves[metric] = {
                    'pitch_cmd': self.plots[metric].plot(pen=pg.mkPen('m', width=2, style=pg.QtCore.Qt.DashLine), name='Pitch Command'),
                    # 'pitch_real': self.plots[metric].plot(pen=pg.mkPen('m', width=2), name='Pitch Actual'),
                    'roll_cmd': self.plots[metric].plot(pen=pg.mkPen('c', width=2, style=pg.QtCore.Qt.DashLine), name='Roll Command'),
                    # 'roll_real': self.plots[metric].plot(pen=pg.mkPen('c', width=2), name='Roll Actual')
                }
            else:
                self.curves[metric] = self.plots[metric].plot(pen='g')
            self.win.nextRow()
            
        self.win.show()
        
        # Create a queue for communication between sim and plot threads
        self.plot_queue = asyncio.Queue()
        
        # Data processing and rendering flags
        self.data_needs_update = False
        self.running = True
        
        # Start the tasks
        self.data_task = None
        self.render_task = None

        self.executor = ThreadPoolExecutor(max_workers=1)

    async def start(self):
        """Start both the data processing and rendering tasks"""
        self.running = True
        self.data_task = asyncio.create_task(self._data_loop())
        self.render_task = asyncio.create_task(self._render_loop())

    async def stop(self):
        """Stop all tasks gracefully"""
        self.running = False
        if self.data_task:
            await self.data_task
        if self.render_task:
            await self.render_task
        self.executor.shutdown(wait=True)

    async def reset(self):
        """Reset all plots"""
        for name in self.plot_data.keys():
            self.plot_data[name] = []
        self.traj_data = {}
        self.data_needs_update = True
        
    async def _data_loop(self):
        """Process incoming data in background"""
        while self.running:
            try:
                await self.collect_and_organize_data()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in data loop: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(1)

    def _process_data_sync(self):
        """Process data from the queue, ran through executor in separate thread to avoid blocking the main thread"""
        new_data = False
        while not self.plot_queue.empty():
            new_data = True
            # Get data from queue synchronously
            mjdata, obs_arrays = self.plot_queue.get_nowait()

            # mjdata
            for key in ['qpos', 'qvel', 'xpos', 'xquat', 'heading']:
                self.traj_data.setdefault(key, []).append(mjdata[key])

            # commands
            if not 'command' in self.traj_data:
                self.traj_data['command'] = {
                    'linear_velocity_command': [],
                    'angular_velocity_command': [],
                    'base_height_command': [],
                    'xyorientation_command': []
                }
            self.traj_data['command']['linear_velocity_command'].append(obs_arrays['command'][0:2])
            ang_vel_cmd = obs_arrays['command'][2:8]
            ang_vel_cmd[-1] = mjdata['heading'][0]
            self.traj_data['command']['angular_velocity_command'].append(ang_vel_cmd)
            self.traj_data['command']['base_height_command'].append(obs_arrays['command'][8:9])
            self.traj_data['command']['xyorientation_command'].append(obs_arrays['command'][9:11])

            # some obs
            if not 'obs' in self.traj_data:
                self.traj_data['obs'] = {
                    'sensor_observation_base_site_linvel': [],
                    'sensor_observation_left_foot_force': [],
                    'sensor_observation_right_foot_force': [],
                    'feet_contact_observation': []
                }
            self.traj_data['obs']['sensor_observation_base_site_linvel'].append(mjdata['base_site_linvel'])
            self.traj_data['obs']['sensor_observation_left_foot_force'].append(mjdata['left_foot_force'])
            self.traj_data['obs']['sensor_observation_right_foot_force'].append(mjdata['right_foot_force'])

            # feet contact obs # TODO should really be done in parallel
            observation_input = self.get_sparse_obs_input(mjdata['contact']['geom'], mjdata['contact']['dist'])
            feet_contact_obs = self.observations['FeetContactObservation'].observe(observation_input, None, None)
            self.traj_data['obs']['feet_contact_observation'].append(feet_contact_obs)

        if not new_data:
            return False

        traj = Trajectory(
            qpos=jnp.stack(self.traj_data['qpos']),
            qvel=jnp.stack(self.traj_data['qvel']),
            xpos=jnp.stack(self.traj_data['xpos']),
            xquat=jnp.stack(self.traj_data['xquat']),
            command={k: jnp.stack(v) for k, v in self.traj_data['command'].items()},
            obs={k: jnp.stack(v) for k, v in self.traj_data['obs'].items()},
        )

        for reward in self.rewards:
            try:
                name = reward.__class__.__name__
                reward_values = reward.get_reward(traj)
                self.plot_data[name] = [float(x) for x in reward_values.flatten()]
            except Exception as e:
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                print(f"Error computing reward for {reward.__class__.__name__}: {e}")

        base_eulers = xax.quat_to_euler(traj.xquat[:, 1, :])
        base_eulers = base_eulers.at[:, :2].set(0.0)
        heading_quats = xax.euler_to_quat(base_eulers)
        local_frame_linvel = xax.rotate_vector_by_quat(traj.obs['sensor_observation_base_site_linvel'], heading_quats, inverse=True)

        self.plot_data['linvel'] = {
            'x_cmd': [float(x[0]) for x in self.traj_data['command']['linear_velocity_command']],
            'x_real': [float(x[0]) for x in local_frame_linvel],
            'y_cmd': [float(x[1]) for x in self.traj_data['command']['linear_velocity_command']],
            'y_real': [float(x[1]) for x in local_frame_linvel]
        }
        self.plot_data['angvel'] = {
            'wz_cmd': [float(x[0]) for x in self.traj_data['command']['angular_velocity_command']],
            # 'wz_real': [float(x[0]) for x in self.traj_data['obs']['sensor_observation_base_site_angvel']]
            'heading_cmd': [float(x[-1]) for x in self.traj_data['command']['angular_velocity_command']],
            # 'heading_real': [float(x[0]) for x in self.traj_data['heading']]
        }
        self.plot_data['base_height'] = {
            'base_height_cmd': [float(x) for x in self.traj_data['command']['base_height_command']],
            'base_height_real': [float(x[1, 2]) for x in self.traj_data['xpos']]
        }
        self.plot_data['xyorientation'] = {
            'pitch_cmd': [float(x[0]) for x in self.traj_data['command']['xyorientation_command']],
            # 'pitch_real': [float(x[0]) for x in self.traj_data['xquat'][:, 0]],
            'roll_cmd': [float(x[1]) for x in self.traj_data['command']['xyorientation_command']],
            # 'roll_real': [float(x[1]) for x in self.traj_data['xquat'][:, 1]]
        }

        self.data_needs_update = True
        return True

    async def collect_and_organize_data(self):
        """Run the entire data processing in an executor"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._process_data_sync
        )

    async def _render_loop(self):
        """Render all plots at a fixed rate"""
        while self.running:
            try:
                if self.data_needs_update:
                    for name, curves in self.curves.items():
                        if isinstance(curves, dict):
                            # Multiple curves per plot
                            for curve_name, curve in curves.items():
                                values = self.plot_data[name][curve_name]
                                x = list(range(len(values)))
                                curve.setData(x, values)
                        else:
                            # Single curve
                            values = self.plot_data[name]
                            x = list(range(len(values)))
                            curves.setData(x, values)
                    self.data_needs_update = False
                
                self.app.processEvents()
                await asyncio.sleep(1/60)
                
            except Exception as e:
                print(f"Error in render loop: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(1)

        
    async def add_data(self, mjdata, obs_arrays, heading):
        """Copy simulation data to be plotted asynchronously"""
        mjdata_copy = {
            'qpos': np.array(mjdata.qpos, copy=True),
            'qvel': np.array(mjdata.qvel, copy=True),
            'xpos': np.array(mjdata.xpos, copy=True),
            'xquat': np.array(mjdata.xquat, copy=True),
            'time': float(mjdata.time),
            'base_site_linvel': np.array(mjdata.sensor('base_site_linvel').data, copy=True),
            'left_foot_force': np.array(mjdata.sensor('left_foot_force').data, copy=True),
            'right_foot_force': np.array(mjdata.sensor('right_foot_force').data, copy=True),
            'heading': np.array([heading]),
            'contact': {
                'geom': np.array(mjdata.contact.geom, copy=True),
                'dist': np.array(mjdata.contact.dist, copy=True)
            }
        }
        obs_arrays_copy = {k: np.array(v, copy=True) for k, v in obs_arrays.items()}
        await self.plot_queue.put((mjdata_copy, obs_arrays_copy))

    @staticmethod
    def get_sparse_obs_input(geom: np.ndarray, dist: np.ndarray):
        """
        Create a minimal mock structure for contact data to call the observe function
        """
        class MinimalContact:
            def __init__(self, geom, dist):
                self.geom = geom
                self.dist = dist

        class MinimalMjData:
            def __init__(self, geom, dist):
                self.contact = MinimalContact(
                    geom,
                    dist
                )

        class MinimalPhysicsState:
            def __init__(self, data):
                self.data = data

        class MinimalObservationInput:
            def __init__(self, physics_state):
                self.physics_state = physics_state

        # Create the minimal observation input with just the contact data
        minimal_mjdata = MinimalMjData(geom, dist)
        minimal_physics_state = MinimalPhysicsState(minimal_mjdata)
        minimal_observation_input = MinimalObservationInput(minimal_physics_state)

        return minimal_observation_input


