"""Keyboard command provider."""

from queue import Queue, Empty
import sys
import threading
from typing import List


class KeyboardController:
    """Tracks keyboard presses to update the command vector.

    Contains 16 commands that can be modified via keyboard input:
    - [0] x linear velocity [m/s]
    - [1] y linear velocity [m/s]
    - [2] z angular velocity [rad/s]
    - [3] base height offset [m]
    - [4] base roll [rad]
    - [5] base pitkey [rad]
    - [6] right shoulder pitkey [rad]
    - [7] right shoulder roll [rad]
    - [8] right elbow pitkey [rad]
    - [9] right elbow roll [rad]
    - [10] right wrist pitkey [rad]
    - [11] left shoulder pitkey [rad]
    - [12] left shoulder roll [rad]
    - [13] left elbow pitkey [rad]
    - [14] left elbow roll [rad]
    - [15] left wrist pitkey [rad]
    """

    def __init__(self, keyboard_queue: Queue) -> None:
        self.queue = keyboard_queue
        self.cmd = [0.0] * 16

        # Start keyboard reading thread
        self._running = True
        self._thread = threading.Thread(target=self._read_input, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        """Stop the input reading thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self._stop()

    def reset_cmd(self) -> None:
        """Reset all commands to zero."""
        self.cmd = [0.0 for _ in self.cmd]

    def get_cmd(self) -> List[float]:
        """Get current command vector."""
        return self.cmd

    def _read_input(self) -> None:
        """Threaded method that continuously reads keyboard input to update command vector."""
        while self._running:
            try:
                key = self.queue.get(timeout=0.1)
            except Empty:
                continue
        
            key = key.strip("'").lower()

            # base controls
            if key == "0":
                self.reset_cmd()
            elif key == "w":
                self.cmd[0] += 0.1
            elif key == "s":
                self.cmd[0] -= 0.1
            elif key == "a":
                self.cmd[1] += 0.1
            elif key == "d":
                self.cmd[1] -= 0.1
            elif key == "q":
                self.cmd[2] += 0.1
            elif key == "e":
                self.cmd[2] -= 0.1

            # base pose
            elif key == "=":
                self.cmd[3] += 0.05
            elif key == "-":
                self.cmd[3] -= 0.05
            elif key == "r":
                self.cmd[4] += 0.1
            elif key == "f":
                self.cmd[4] -= 0.1
            elif key == "t":
                self.cmd[5] += 0.1
            elif key == "g":
                self.cmd[5] -= 0.1
