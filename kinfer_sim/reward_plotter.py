import importlib.util
import asyncio
import pyqtgraph as pg
import numpy as np

import ksim


class RewardPlotter:
    def __init__(self):
        path_to_train_file = "/home/bart/kbot-joystick/train.py"
        spec = importlib.util.spec_from_file_location("train", path_to_train_file)
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        self.train = train
        self.reward_classes = [cls for cls in train.__dict__.values() if isinstance(cls, type) and issubclass(cls, ksim.Reward)]
        print("\n=== Found Reward Classes ===")
        for i, reward_class in enumerate(self.reward_classes, 1):
            print(f"\n{i}. {reward_class.__name__}")
            print(f"   {reward_class.__doc__ or 'No description available'}")
        print("\n" + "=" * 30 + "\n")

        # Initialize PyQtPlot window and widgets
        self.app = pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget()
        self.plot = self.win.addPlot()
        self.win.show()
        
        # Create a queue for communication between sim and plot threads
        self.plot_queue = asyncio.Queue()
        
        # Store historical data
        self.times = []
        self.values = []
        self.curve = self.plot.plot(pen='y')  # Create persistent curve
        
        # Data processing and rendering flags
        self.data_needs_update = False
        self.running = True
        
        # Start the tasks
        self.data_task = None
        self.render_task = None
        self.time_counter = 0

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
        """Reset the plot"""
        self.times = []
        self.values = []
        self.time_counter = 0
        self.data_needs_update = True
        
    async def _data_loop(self):
        """Process incoming data in background"""
        while self.running:
            try:
                sim_data = await self.plot_queue.get()
                
                # Do Jax preprocessing in executor to not block
                processed_data = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._process_data_with_jax, 
                    sim_data
                )
                
                # Store the data
                self.times.append(self.time_counter)
                self.values.append(processed_data)
                self.time_counter += 1
                self.data_needs_update = True
                
            except Exception as e:
                print(f"Error in data loop: {e}")
                await asyncio.sleep(1)  # Wait a bit longer on error
    
    async def _render_loop(self):
        """Render the plot at a fixed rate"""
        while self.running:
            try:
                if self.data_needs_update:
                    self.curve.setData(self.times, self.values)
                    self.data_needs_update = False
                
                self.app.processEvents()  # Always process Qt events
                await asyncio.sleep(1/60)  # Cap at ~60 FPS
                
            except Exception as e:
                print(f"Error in render loop: {e}")
                await asyncio.sleep(1)
        
    def _process_data_with_jax(self, sim_data):
        # Heavy Jax preprocessing here
        print("Processing data with Jax")
        print(sim_data)
        print(sim_data.qpos[:3])
        return float(sim_data.qpos[5])  # Convert to float for plotting
        
    async def add_data(self, sim_data):
        """Add new simulation data to be plotted"""
        await self.plot_queue.put(sim_data)
