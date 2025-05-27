import threading
import logging
from queue import Queue

from pynput import keyboard

logger = logging.getLogger(__name__)

class KeyboardListener:
    """Simple keyboard listener in its own thread, writes to a queue.
    Also maintains a reset queue, which is used to reset the simulator.
    """

    def __init__(self):
        # Create a queue for communication between threads
        self.key_queue = Queue()
        self.reset_queue = Queue()
        self._start_listener()

    def _on_press(self, key):
        self.key_queue.put(str(key).lower())
        if key == keyboard.Key.backspace:
            self.reset_queue.put(True)

    def _start_keyboard_listener(self):
        # Collect events until released
        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()  # Non-blocking
        listener.join()  # Keep the main thread alive

    def _start_listener(self):
        # Create and start the keyboard listener in a separate thread
        keyboard_thread = threading.Thread(target=self._start_keyboard_listener)
        keyboard_thread.daemon = True  # Thread will exit when main program exits
        keyboard_thread.start()

    def get_queues(self):
        # Get queues and read from them in another process
        return self.key_queue, self.reset_queue


