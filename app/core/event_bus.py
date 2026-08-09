"""
Thread-safe Event Bus for the AI Admission Robot.

This module allows different parts of the application to communicate
without directly depending on each other.
"""

from collections import defaultdict
from threading import Lock
from typing import Any, Callable


class EventBus:
    """
    A simple thread-safe publish/subscribe event bus.
    """

    def __init__(self) -> None:
        """
        Initialize the event bus.
        """

        # Stores all subscribers.
        #
        # Example:
        # {
        #     "visitor_detected": [func1, func2],
        #     "speech_finished": [func3]
        # }
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        # Makes the event bus thread-safe.
        self._lock = Lock()

    def subscribe(
        self,
        event_name: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Register a callback for an event.

        Parameters
        ----------
        event_name:
            Name of the event.

        callback:
            Function that should run whenever the event occurs.
        """

        with self._lock:
            self._subscribers[event_name].append(callback)

    def unsubscribe(
        self,
        event_name: str,
        callback: Callable[..., Any],
    ) -> None:
        """
        Remove a callback from an event.
        """

        with self._lock:
            if callback in self._subscribers[event_name]:
                self._subscribers[event_name].remove(callback)

    def publish(
        self,
        event_name: str,
        **kwargs: Any,
    ) -> None:
        """
        Publish an event to all subscribers.

        Any keyword arguments are forwarded to the callbacks.
        """

        with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))

        # Execute callbacks outside the lock.
        for callback in callbacks:
            callback(**kwargs)

    def clear(self) -> None:
        """
        Remove all subscribers.
        """

        with self._lock:
            self._subscribers.clear()

    def subscriber_count(
        self,
        event_name: str,
    ) -> int:
        """
        Return the number of subscribers for an event.
        """

        with self._lock:
            return len(self._subscribers.get(event_name, []))