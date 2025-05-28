import asyncio
import importlib.util
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array
import pyqtgraph as pg
import numpy as np
import mujoco


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
        self.data = {
            'times': {},
            'values': {}
        }
        
        # Setup reward plots
        for i, reward in enumerate(self.rewards):
            name = reward.__class__.__name__
            self.plots[name] = self.win.addPlot(title=name)
            self.plots[name].setLabel('left', 'Reward')
            self.plots[name].setLabel('bottom', 'Time')
            self.curves[name] = self.plots[name].plot(pen='y')
            self.data['times'][name] = []
            self.data['values'][name] = []
            self.win.nextRow()
            
        # Setup additional metric plots
        additional_metrics = ['commands', 'linvel', 'angvel']
        for metric in additional_metrics:
            self.plots[metric] = self.win.addPlot(title=metric.capitalize())
            self.plots[metric].setLabel('left', metric.capitalize())
            self.plots[metric].setLabel('bottom', 'Time')
            self.curves[metric] = self.plots[metric].plot(pen='g')
            self.data['times'][metric] = []
            self.data['values'][metric] = []
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

    async def reset(self):
        """Reset all plots"""
        for name in self.data['times'].keys():
            self.data['times'][name] = []
            self.data['values'][name] = []
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

    async def collect_and_organize_data(self):
        """Fully run down the data queue and create Trajectory object"""
        new_data = False
        while self.plot_queue.qsize() > 0:
            new_data = True
            mjdata, obs_arrays = await self.plot_queue.get()

            # mjdata
            for key in ['qpos', 'qvel', 'xpos', 'xquat']:
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
            self.traj_data['command']['angular_velocity_command'].append(obs_arrays['command'][2:8])
            self.traj_data['command']['base_height_command'].append(obs_arrays['command'][8])
            self.traj_data['command']['xyorientation_command'].append(obs_arrays['command'][9:10])

            # some obs
            if not 'obs' in self.traj_data:
                self.traj_data['obs'] = {
                    'sensor_observation_base_site_linvel': [],
                    # 'feet_contact_observation': []
                }
            self.traj_data['obs']['sensor_observation_base_site_linvel'].append(mjdata['base_site_linvel'])
            # self.traj_data['obs']['feet_contact_observation'].append(train_obs_arrays['FeetContactObservation'])

        if 'qpos' not in self.traj_data: # quit if we don't have any data
            return

        traj = Trajectory(
            qpos=jnp.stack(self.traj_data['qpos']),
            qvel=jnp.stack(self.traj_data['qvel']),
            xpos=jnp.stack(self.traj_data['xpos']),
            xquat=jnp.stack(self.traj_data['xquat']),
            command={
                k: jnp.stack(v) for k, v in self.traj_data['command'].items()
            },
            obs={
                k: jnp.stack(v) for k, v in self.traj_data['obs'].items()
            },
        )

        for reward in self.rewards:
            try:
                name = reward.__class__.__name__
                reward_values = reward.get_reward(traj)
                self.data['values'][name] = [float(x) for x in reward_values.flatten()]
                self.data['times'][name] = [i for i in range(len(reward_values))]
            except Exception as e:
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                print(f"Error computing reward for {reward.__class__.__name__}: {e}")
        
        # Process additional metrics
        # self.data['values']['commands'] = 0.0 #float(jnp.mean(traj_data.command['desired_speed']))
        # self.data['values']['linvel'] = float(jnp.mean(mjdata['qvel'][:3]))
        # self.data['values']['angvel'] = float(jnp.mean(mjdata['qvel'][3:6]))

        if new_data:
            self.data_needs_update = True
        
    
    async def _render_loop(self):
        """Render all plots at a fixed rate"""
        while self.running:
            try:
                if self.data_needs_update:
                    for name in self.curves.keys():
                        self.curves[name].setData(
                            self.data['times'][name], 
                            self.data['values'][name]
                        )
                    self.data_needs_update = False
                
                self.app.processEvents()
                await asyncio.sleep(1/60)
                
            except Exception as e:
                print(f"Error in render loop: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
                await asyncio.sleep(1)

        
    async def add_data(self, mjdata, obs_arrays):
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
        }
        obs_arrays_copy = {k: np.array(v, copy=True) for k, v in obs_arrays.items()}
        await self.plot_queue.put((mjdata_copy, obs_arrays_copy))




        # # HACK
        # @dataclass(frozen=True)
        # class ObservationInput:
        #     physics_state: mujoco.MjData

        # @dataclass(frozen=True)
        # class PhysicsState:
        #     data: mujoco.MjData


        # observation_input = ObservationInput(
        #     physics_state=PhysicsState(data=mjdata)
        # )

        # feetcontactobs = self.observations['FeetContactObservation'].observe(observation_input, None, None)
        # train_obs_arrays = {'FeetContactObservation': feetcontactobs}




