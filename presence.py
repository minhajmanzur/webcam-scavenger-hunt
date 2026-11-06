"""Short-term presence stabilization for intermittent object detections."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _TargetPresence:
    first_seen: float | None = None
    last_seen: float | None = None
    confirmed: bool = False


class PresenceStabilizer:
    """Confirm a target after sustained detections, tolerating brief flickers."""

    def __init__(
        self,
        targets: tuple[str, ...],
        persistence_seconds: float = 1.0,
        flicker_grace_seconds: float = 0.35,
    ) -> None:
        if persistence_seconds <= 0:
            raise ValueError("persistence_seconds must be positive")
        if flicker_grace_seconds < 0:
            raise ValueError("flicker_grace_seconds cannot be negative")
        self.persistence_seconds = persistence_seconds
        self.flicker_grace_seconds = flicker_grace_seconds
        self._states = {target: _TargetPresence() for target in targets}

    def update(
        self,
        detected_targets: set[str],
        now: float | None = None,
    ) -> dict[str, tuple[bool, float]]:
        """Return each target's (confirmed, progress) state for this moment."""
        moment = time.perf_counter() if now is None else now

        for target, state in self._states.items():
            if target in detected_targets:
                if (
                    state.last_seen is None
                    or moment - state.last_seen > self.flicker_grace_seconds
                ):
                    state.first_seen = moment
                    state.confirmed = False

                state.last_seen = moment
                if state.first_seen is not None:
                    state.confirmed = (
                        moment - state.first_seen + 1e-9
                        >= self.persistence_seconds
                    )
            elif (
                state.last_seen is not None
                and moment - state.last_seen > self.flicker_grace_seconds
            ):
                state.first_seen = None
                state.last_seen = None
                state.confirmed = False

        status: dict[str, tuple[bool, float]] = {}
        for target, state in self._states.items():
            recently_seen = (
                state.last_seen is not None
                and moment - state.last_seen <= self.flicker_grace_seconds
            )
            if not recently_seen or state.first_seen is None:
                status[target] = (False, 0.0)
                continue

            elapsed = moment - state.first_seen
            progress = (
                1.0
                if state.confirmed
                else min(1.0, elapsed / self.persistence_seconds)
            )
            status[target] = (state.confirmed, progress)

        return status
