from queue import Queue

from pynput import keyboard

class KeyboardListener:
    """Keyboard listener that writes all key presses to queues."""

    def __init__(self):
        """Initialize keyboard listener with empty queue list and start listening."""
        self.queues: list[Queue] = []
        
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.daemon = True
        self.listener.start()

    def _on_press(self, key):
        """ Write all key presses to all queues. """
        for queue in self.queues:
            queue.put(str(key).lower())

    def get_queue(self):
        """ Get a new queue for a listening process """
        self.queues.append(Queue())
        return self.queues[-1]


