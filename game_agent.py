"""Small local hunt agent and its auditable game tools."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HuntTarget:
    object_name: str
    color: str | None = None

    @property
    def label(self) -> str:
        return f"{self.color} {self.object_name}" if self.color else self.object_name


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]


class HuntAgent:
    """Rule-based agent that uses four small tools to run a webcam hunt."""

    COLOR_RANGES = {
        "orange": ((5, 25),),
        "yellow": ((20, 36),),
        "green": ((36, 86),),
        "blue": ((90, 130),),
        "purple": ((130, 160),),
        "red": ((0, 10), (170, 180)),
    }

    def __init__(
        self,
        targets: list[HuntTarget],
        duration_seconds: int = 120,
        hint_delay_seconds: float = 12.0,
        hint_repeat_seconds: float = 12.0,
    ) -> None:
        if not targets:
            raise ValueError("The hunt needs at least one target")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self.targets = targets
        self.duration_seconds = duration_seconds
        self.hint_delay_seconds = hint_delay_seconds
        self.hint_repeat_seconds = hint_repeat_seconds
        self.reset()

    def reset(self, now: float | None = None) -> None:
        moment = time.perf_counter() if now is None else now
        self.score = 0
        self.found: list[str] = []
        self.target_index = 0
        self.started_at = moment
        self.target_started_at = moment
        self.last_hint_at = moment
        self.last_hint = ""
        self.last_event = "Hunt ready"
        self.hints_given = 0
        self._target_was_visible = False
        self._last_color_match: bool | None = None
        self._record("game reset")

    @property
    def current_target(self) -> HuntTarget | None:
        if self.is_over:
            return None
        return self.targets[self.target_index]

    @property
    def is_time_up(self) -> bool:
        return self.remaining_seconds() <= 0

    @property
    def is_over(self) -> bool:
        return self.target_index >= len(self.targets) or self.is_time_up

    def remaining_seconds(self, now: float | None = None) -> int:
        moment = time.perf_counter() if now is None else now
        elapsed = moment - self.started_at
        return max(0, int(self.duration_seconds - elapsed + 0.999))

    def detect_objects(
        self,
        detections: list[Detection],
        target_name: str,
    ) -> list[Detection]:
        """Tool: return current detections matching the target class."""
        matches = [detection for detection in detections if detection.name == target_name]
        visible = bool(matches)
        if visible != self._target_was_visible:
            self._record(
                f"detect_objects: {target_name} " + ("visible" if visible else "not visible")
            )
            self._target_was_visible = visible
        return matches

    def check_color(
        self,
        frame: np.ndarray,
        detection: Detection,
        expected_color: str,
    ) -> tuple[bool, float]:
        """Tool: estimate the fraction of a detection crop matching a basic hue."""
        if expected_color not in self.COLOR_RANGES:
            raise ValueError(f"Unsupported color: {expected_color}")

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation_mask = cv2.inRange(hsv, (0, 45, 25), (180, 255, 255))
        color_mask = np.zeros(saturation_mask.shape, dtype=np.uint8)
        for hue_start, hue_end in self.COLOR_RANGES[expected_color]:
            hue_mask = cv2.inRange(hsv, (hue_start, 55, 25), (hue_end, 255, 255))
            color_mask = cv2.bitwise_or(color_mask, hue_mask)
        color_mask = cv2.bitwise_and(color_mask, saturation_mask)
        fraction = cv2.countNonZero(color_mask) / float(color_mask.size)
        matched = fraction >= 0.12

        if matched != self._last_color_match:
            self._record(
                f"check_color: {expected_color} "
                f"{'matched' if matched else 'not matched'} ({fraction:.0%})"
            )
            self._last_color_match = matched
        return matched, fraction

    def give_hint(
        self,
        target: HuntTarget,
        *,
        object_visible: bool,
        color_matched: bool | None = None,
        now: float | None = None,
    ) -> str:
        """Tool: provide a contextual hint and record that it was given."""
        moment = time.perf_counter() if now is None else now
        if target.color and color_matched is False:
            hint = f"Show more of the {target.color} part of the {target.object_name}."
        elif object_visible:
            hint = "Hold it steady and keep it fully in the camera view."
        else:
            hint = f"Try bringing the {target.object_name} closer and centering it."
        self.last_hint = hint
        self.last_hint_at = moment
        self.hints_given += 1
        self._record(f"give_hint: {hint}")
        return hint

    def mark_found(self, target: HuntTarget, now: float | None = None) -> int:
        """Tool: award points and advance to the next hunt target."""
        if self.current_target != target:
            return self.score
        self.score += 100
        self.found.append(target.label)
        self.target_index += 1
        moment = time.perf_counter() if now is None else now
        self.target_started_at = moment
        self.last_hint_at = moment
        self.last_hint = ""
        self._target_was_visible = False
        self._last_color_match = None
        self._record(f"mark_found: {target.label} (+100; score {self.score})")
        return self.score

    def observe(
        self,
        detections: list[Detection],
        presence: dict[str, tuple[bool, float]],
        frame: np.ndarray,
        now: float | None = None,
    ) -> None:
        """Apply one frame of detections to the current hunt target."""
        moment = time.perf_counter() if now is None else now
        target = self.current_target
        if target is None:
            return

        matches = self.detect_objects(detections, target.object_name)
        color_matched = True
        if target.color:
            color_matched = any(
                self.check_color(frame, detection, target.color)[0]
                for detection in matches
            )

        stable = presence.get(target.object_name, (False, 0.0))[0]
        if matches and stable and color_matched:
            self.mark_found(target, now=moment)
            return

        hint_due = (
            moment - self.target_started_at >= self.hint_delay_seconds
            and moment - self.last_hint_at >= self.hint_repeat_seconds
        )
        if hint_due:
            self.give_hint(
                target,
                object_visible=bool(matches),
                color_matched=color_matched,
                now=moment,
            )

    def request_hint(self, now: float | None = None) -> str:
        target = self.current_target
        if target is None:
            return "The hunt is over. Press R to play again."
        moment = time.perf_counter() if now is None else now
        return self.give_hint(
            target,
            object_visible=self._target_was_visible,
            color_matched=self._last_color_match,
            now=moment,
        )

    def skip_target(self, now: float | None = None) -> None:
        target = self.current_target
        if target is None:
            return
        self._record(f"skip target: {target.label}")
        self.target_index += 1
        moment = time.perf_counter() if now is None else now
        self.target_started_at = moment
        self.last_hint_at = moment
        self.last_hint = ""
        self._target_was_visible = False
        self._last_color_match = None

    def _record(self, event: str) -> None:
        self.last_event = event
        print(event)
