# Webcam Scavenger Hunt

A local webcam game built with Python, OpenCV, and Ultralytics YOLO. Find **bottle, cup, carrot, potted plant, and book** before the timer runs out.

## Features

- Live webcam capture and YOLO26n object detection for five COCO classes
- One-second presence confirmation with a 350 ms grace period for brief detection flicker
- Two-minute hunt, 100 points per object, automatic hints, and a skip control
- Four small, logged agent tools: `detect_objects`, `check_color`, `give_hint`, and `mark_found`
- Optional color requirement for one target using a basic HSV color-area check
- Live inference latency and FPS display
- Local inference; this app does not save or upload webcam frames

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or 3.12
- A webcam and permission for Python to use it
- Internet access for installing packages and downloading `yolo26n.pt` on the first run
- CPU is enough to run the demo; a supported GPU can improve performance

## Install and run on Windows

Open PowerShell in the project directory. The commands below do not require virtual-environment activation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --confidence 0.15
```

The lower confidence threshold helps with objects such as books, but may allow more false detections. Aim the webcam at the current target shown on screen.

On macOS or Linux, use the platform’s Python 3.11/3.12 executable, create a virtual environment with `python3 -m venv .venv`, activate it, install `requirements.txt`, and run `python main.py --confidence 0.15`.

## Controls

| Key | Action |
|---|---|
| `H` | Ask the agent for a hint |
| `S` | Skip the current target |
| `R` | Restart the timer and score |
| `Q` | Quit |

The agent automatically gives a hint if the player has been stuck on a target for about 12 seconds. Tool actions are also printed in the terminal.

## Options

```powershell
# Change the webcam index or game duration
.\.venv\Scripts\python.exe main.py --camera 1 --duration 180

# Require a color for one target, e.g. a blue book
.\.venv\Scripts\python.exe main.py --color-target book --required-color blue
```

Supported color names are red, orange, yellow, green, blue, and purple. Color checks are approximate and can vary with lighting and the object’s background.

## Tests

Run the automated logic tests from the project directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover persistence and flicker handling, scoring, hints, and color checks. Webcam behavior still needs to be checked on the device running the app.

## How it works

`main.py` captures frames, runs YOLO, draws detections, and drives the game loop. `presence.py` stabilizes class detections over time. `game_agent.py` contains the rule-based agent and its tools. The agent runs locally and does not call an external LLM service.

## Current limitations

- The pretrained model recognizes generic COCO classes. It may detect a mug as a cup, but it does not identify handles or distinguish every kind of book.
- Persistence is tracked per class, not by the identity of an individual object.
- The optional color check uses a simple HSV area threshold and can be inaccurate for multicolored items or unusual lighting.
- Hints are rule-based; the app does not yet collect player feedback to measure whether a hint was correct.
- Printed-photo detection and Docker packaging are not included in this version. They remain stretch goals.

## Example performance

An initial webcam preview on the developer’s laptop showed about **45 ms inference latency** and **21 FPS**. This is a single machine-specific observation, not a benchmark; performance depends on the camera, lighting, model, and hardware.
