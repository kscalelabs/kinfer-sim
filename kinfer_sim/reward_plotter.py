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
        
        # Start the plotting task
        self.plot_task = None
        self.time_counter = 0

    async def start(self):
        """Start the plotting task"""
        self.plot_task = asyncio.create_task(self._plot_loop())

    async def reset(self):
        """Reset the plot"""
        self.times = []
        self.values = []
        self.time_counter = 0
        self.curve.setData([], [])
        self.app.processEvents()
        
    async def _plot_loop(self):
        """Main plotting loop that runs in background"""
        while True:
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
                
                # Update plot (PyQt operations must be in main thread)
                self.curve.setData(self.times, self.values)
                self.app.processEvents()  # Let Qt handle events
            except Exception as e:
                print(f"Error in plot loop: {e}")
            
    def _process_data_with_jax(self, sim_data):
        # Heavy Jax preprocessing here
        print("Processing data with Jax")
        print(sim_data)
        print(sim_data.qpos[:3])
        return float(sim_data.qpos[5])  # Convert to float for plotting
        
    async def add_data(self, sim_data):
        """Add new simulation data to be plotted"""
        await self.plot_queue.put(sim_data)
