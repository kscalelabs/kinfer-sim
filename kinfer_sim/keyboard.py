import atexit
import select
import sys
import termios
import threading
import tty
from typing import List


class Keyboard:
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

    def __init__(self) -> None:
        # Command vector initialization
        self.length = 16
        self.cmd = [0.0] * self.length

        # Set up stdin for raw input
        self._fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(lambda: termios.tcsetattr(self._fd, termios.TCSADRAIN, old_settings))

        # Start keyboard reading thread
        self._running = True
        self._thread = threading.Thread(target=self._read_input, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        """Stop the input reading thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def __del__(self):
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
            # Use select to check for input with a timeout
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue

            try:
                ch = sys.stdin.read(1).lower()

                # base controls
                if ch == "0":
                    self.reset_cmd()
                elif ch == "w":
                    self.cmd[0] += 0.1
                elif ch == "s":
                    self.cmd[0] -= 0.1
                elif ch == "a":
                    self.cmd[1] += 0.1
                elif ch == "d":
                    self.cmd[1] -= 0.1
                elif ch == "q":
                    self.cmd[2] += 0.1
                elif ch == "e":
                    self.cmd[2] -= 0.1

                # base pose
                elif ch == "=":
                    self.cmd[3] += 0.05
                elif ch == "-":
                    self.cmd[3] -= 0.05
                elif ch == "r":
                    self.cmd[4] += 0.1
                elif ch == "f":
                    self.cmd[4] -= 0.1
                elif ch == "t":
                    self.cmd[5] += 0.1
                elif ch == "g":
                    self.cmd[5] -= 0.1

                # clamp
                self.cmd = [max(-0.3, min(0.3, cmd)) for cmd in self.cmd]

            except (IOError, EOFError):
                continue

