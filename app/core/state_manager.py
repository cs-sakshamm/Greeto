"""
Application State Manager.

Keeps track of the robot's current state.
"""

from enum import Enum, auto
from threading import Lock


class RobotState(Enum):
    """All valid robot states."""

    BOOTING = auto()
    INITIALIZING = auto()
    LOADING_MODELS = auto()
    IDLE = auto()
    VISITOR_DETECTED = auto()
    RECOGNIZING = auto()
    GREETING = auto()
    LISTENING = auto()
    PROCESSING = auto()
    RESPONDING = auto()
    SPEAKING = auto()
    WAITING = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class StateManager:
    """Thread-safe robot state manager."""

    def __init__(self) -> None:
        self._state = RobotState.BOOTING
        self._lock = Lock()

    def get_state(self) -> RobotState:
        """Return the current robot state."""

        with self._lock:
            return self._state

    def set_state(self, state: RobotState) -> None:
        """Change the current robot state."""

        with self._lock:
            self._state = state

    def is_state(self, state: RobotState) -> bool:
        """Check whether the robot is in the given state."""

        with self._lock:
            return self._state == state