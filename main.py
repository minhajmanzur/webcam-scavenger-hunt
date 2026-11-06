"""Live webcam object detector for the Webcam Scavenger Hunt project."""

from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

from game_agent import Detection, HuntAgent, HuntTarget
from presence import PresenceStabilizer


# These are all classes in Ultralytics' pretrained COCO model and the hunt list.
TARGET_CLASSES = ("bottle", "cup", "carrot", "potted plant", "book")
COLOR_NAMES = ("red", "orange", "yellow", "green", "blue", "purple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the webcam scavenger hunt. Press Q to quit."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0).")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics model or local weights file (default: yolo26n.pt).",
    )
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection threshold (0–1).")
    parser.add_argument("--image-size", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--duration", type=int, default=120, help="Hunt timer in seconds.")
    parser.add_argument(
        "--color-target",
        choices=TARGET_CLASSES,
        help="Optionally require a color for one target, e.g. --color-target book --required-color blue.",
    )
    parser.add_argument("--required-color", choices=COLOR_NAMES)
    return parser.parse_args()


def model_name_map(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a repeatable, reasonably bright BGR color for a class."""
    return (
        70 + (class_id * 67) % 170,
        70 + (class_id * 113) % 170,
        70 + (class_id * 157) % 170,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than 0")
    if bool(args.color_target) != bool(args.required_color):
        raise SystemExit("Use both --color-target and --required-color, or neither")

    print(f"Loading {args.model}; the first run may download model weights...")
    model = YOLO(args.model)
    names = model_name_map(model)
    class_id_by_name = {name: class_id for class_id, name in names.items()}
    class_ids = [class_id for class_id, name in names.items() if name in TARGET_CLASSES]
    available_targets = [names[class_id] for class_id in class_ids]
    missing_targets = sorted(set(TARGET_CLASSES) - set(available_targets))
    if missing_targets:
        raise SystemExit(
            "The selected model does not include these target classes: "
            + ", ".join(missing_targets)
        )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Check that it is connected, "
            "not in use by another app, and that camera permission is enabled."
        )

    print("Targets:", ", ".join(available_targets))
    print("Hunt started. Q quit | H hint | S skip target | R restart")
    fps_ema = 0.0
    stabilizer = PresenceStabilizer(TARGET_CLASSES)
    hunt_targets = [
        HuntTarget(name, args.required_color if name == args.color_target else None)
        for name in TARGET_CLASSES
    ]
    agent = HuntAgent(hunt_targets, duration_seconds=args.duration)
    window_name = "Webcam Scavenger Hunt - detector preview"

    try:
        while True:
            loop_started = time.perf_counter()
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            inference_started = time.perf_counter()
            result = model.predict(
                source=frame,
                classes=class_ids,
                conf=args.confidence,
                imgsz=args.image_size,
                verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - inference_started) * 1000

            detections: list[Detection] = []
            if result.boxes is not None:
                boxes = result.boxes
                detections = [
                    Detection(
                        name=names[int(class_id)],
                        confidence=float(confidence),
                        box=tuple(coords),
                    )
                    for coords, class_id, confidence in zip(
                        boxes.xyxy.tolist(),
                        boxes.cls.tolist(),
                        boxes.conf.tolist(),
                    )
                ]

            detected_targets = {detection.name for detection in detections}
            presence = stabilizer.update(detected_targets)
            agent.observe(detections, presence, frame)

            for detection in detections:
                x1, y1, x2, y2 = (int(value) for value in detection.box)
                name = detection.name
                confidence = detection.confidence
                confirmed, progress = presence[name]
                if confirmed:
                    label = f"{name} {confidence:.0%} - stable"
                    color = (60, 210, 90)
                else:
                    remaining = max(
                        0.0,
                        stabilizer.persistence_seconds * (1 - progress),
                    )
                    label = f"{name} {confidence:.0%} - hold {remaining:.1f}s"
                    color = color_for_class(class_id_by_name[name])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = max(y1 - 8, 22)
                cv2.putText(
                    frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            loop_seconds = time.perf_counter() - loop_started
            instant_fps = 1.0 / loop_seconds if loop_seconds > 0 else 0.0
            fps_ema = instant_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * instant_fps
            cv2.putText(
                frame,
                f"Inference: {inference_ms:.0f} ms   FPS: {fps_ema:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            target = agent.current_target
            remaining = agent.remaining_seconds()
            timer_text = f"{remaining // 60:02d}:{remaining % 60:02d}"
            hunt_text = f"HUNT: Find {target.label}" if target else (
                "HUNT COMPLETE" if not agent.is_time_up else "TIME'S UP"
            )
            cv2.putText(
                frame,
                hunt_text,
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Score: {agent.score}   Time: {timer_text}   Found: {len(agent.found)}/{len(agent.targets)}",
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            confirmed_names = [
                name for name, (confirmed, _) in presence.items() if confirmed
            ]
            confirming_names = [
                f"{name} {progress:.0%}"
                for name, (confirmed, progress) in presence.items()
                if not confirmed and progress > 0
            ]
            cv2.putText(
                frame,
                "Stable: " + (", ".join(confirmed_names) if confirmed_names else "none"),
                (12, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if confirming_names:
                cv2.putText(
                    frame,
                    "Confirming: " + ", ".join(confirming_names),
                    (12, 132),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            agent_text = agent.last_hint or agent.last_event
            if len(agent_text) > 76:
                agent_text = agent_text[:73] + "..."
            cv2.putText(
                frame,
                "Agent: " + agent_text,
                (12, 156),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key < 128:
                key = ord(chr(key).lower())
            if key == ord("q"):
                break
            if key == ord("h"):
                agent.request_hint()
            elif key == ord("s"):
                agent.skip_target()
            elif key == ord("r"):
                agent.reset()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
