"""Keyboard command provider."""

import logging
import threading
from queue import Empty, Queue
from typing import List, Optional, Sequence

from kmotions.motions import MOTIONS, Motion

logger = logging.getLogger(__name__)


class KeyboardController:
    """Tracks keyboard presses to update the command vector.

    Contains 16 commands that can be modified via keyboard input:
    - [0] x linear velocity [m/s]
    - [1] y linear velocity [m/s]
    - [2] z angular velocity [rad/s]
    - [3] base height offset [m]
    - [4] base roll [rad]
    - [5] base pitch [rad]
    - [6] right shoulder pitch [rad]
    - [7] right shoulder roll [rad]
    - [8] right elbow pitch [rad]
    - [9] right elbow roll [rad]
    - [10] right wrist pitch [rad]
    - [11] left shoulder pitch [rad]
    - [12] left shoulder roll [rad]
    - [13] left elbow pitch [rad]
    - [14] left elbow roll [rad]
    - [15] left wrist pitch [rad]
    """

    def __init__(self, keyboard_queue: Queue) -> None:
        self.queue = keyboard_queue
        self.cmd = [0.0] * 16
        self.active_motion: Optional[Motion] = None

        # Start keyboard reading thread
        self._running = True
        self._thread = threading.Thread(target=self._read_input, daemon=True)
        self._thread.start()

    def _stop_and_cleanup(self) -> None:
        """Stop the input reading thread and cleanup resources."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reset_cmd(self) -> None:
        """Reset all commands to zero."""
        self.active_motion = None
        self.cmd = [0.0 for _ in self.cmd]

    def get_cmd(self, command_names: Sequence[str]) -> List[float]:
        """Get current command vector."""
        if self.active_motion and (cmd := self.active_motion.get_next_motion_frame()):
            return [cmd.get(name, 0.0) for name in command_names]
        return self.cmd

    def _set_motion(self, motion_name: str) -> None:
        logger.info("Setting motion to %s", motion_name)
        self.reset_cmd()
        self.active_motion = MOTIONS[motion_name](dt=0.02)

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

            # motions
            elif key == "z":
                self._set_motion("wave")
            elif key == "x":
                self._set_motion("salute")
            elif key == "c":
                self._set_motion("come_at_me")
            elif key == "v":
                self._set_motion("boxing_guard_hold")
            elif key == "b":
                self._set_motion("boxing_left_punch")
            elif key == "n":
                self._set_motion("boxing_right_punch")
            elif key == "m":
                self._set_motion("pickup")
            elif key == "h":
                self._set_motion("pirouette")
            elif key == "j":
                self._set_motion("wild_walk")
            elif key == "k":
                self._set_motion("zombie_walk")
            elif key == "l":
                self._set_motion("squats")
            elif key == "y":
                self._set_motion("backflip")
            elif key == "u":
                self._set_motion("boxing")
            elif key == "i":
                self._set_motion("cone")
