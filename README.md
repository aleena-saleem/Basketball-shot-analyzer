# AI Basketball Shot Detection & Tracker

A computer-vision system that detects a basketball and hoop in video, tracks ball trajectory across frames, and automatically classifies each shot attempt as a **make** or **miss** in real time.

## Core Technologies

| Component | Technology |
|---|---|
| Object Detection | **YOLOv8** (Ultralytics) — custom-trained on `Basketball` / `Basketball Hoop` classes |
| Video I/O & Rendering | **OpenCV** |
| Overlay Graphics | **cvzone** |
| Numerical Processing | **NumPy** (trajectory line-fitting via `polyfit`) |
| Deep Learning Backend | **PyTorch** (CUDA / MPS / CPU auto-selected) |
| Dashboard / Visualization | **Streamlit** |

## How It Works

1. **Detection** — Each frame is passed through a fine-tuned YOLOv8 model to localize the ball and hoop with bounding boxes and confidence scores.
2. **Position Cleaning** — `clean_ball_pos` / `clean_hoop_pos` filter out physically implausible detections (sudden jumps, wrong aspect ratios) to keep tracking stable.
3. **Shot Phase Detection** — `detect_up` identifies when the ball enters the backboard/rim region; `detect_down` identifies when it passes below the rim — together bounding a shot attempt.
4. **Scoring Logic** — `score()` fits a line through the ball's trajectory near the rim and checks whether the predicted crossing point falls within the rim's horizontal bounds (with a rebound buffer zone) to decide make vs. miss.
5. **Visualization** — Live overlay shows running score, make/miss text, ball trail, and a fading color flash for shot outcomes.

## Project Structure

```
├── shot_detector.py     # Main inference loop: detection, tracking, scoring, display
├── train.py             # YOLOv8 training script (Basketball / Basketball Hoop)
├── utils.py             # Device selection, position cleaning, shot/scoring geometry
├── config.yaml          # Dataset config for training (train/valid/test paths)
├── best.pt              # Trained YOLOv8 weights
└── video_test_5.mp4     # Sample input video
```

## Real-World Test Video

In addition to the bundled sample clip, the system has been validated against a real-world recorded shooting session:

```
[Watch the real-world test video](./Streamlitdemo/TEST%20VIDEO.mp4)
```

Update the `cv2.VideoCapture(...)` path in `shot_detector.py` to point to this file to reproduce the results shown below.

## Dashboard

A Streamlit dashboard provides an interactive interface for running the detector and reviewing shot statistics.

![Streamlit Dashboard](./Streamlitdemo/streamlit 1..png)
![Dashboard View 2](./Streamlitdemo/2.png)
![Dashboard View 3](./Streamlitdemo/3.png)
## Training

`train.py` fine-tunes a YOLOv8n model on a Roboflow-exported dataset:

1. Download and extract a labeled dataset (`train/`, `valid/`, `test/`).
2. Point `config.yaml` to the dataset paths.
3. Run:
   ```bash
   python train.py
   ```
4. Copy the resulting `best.pt` into the project root.

Training hyperparameters (epochs, augmentation, learning rate, etc.) are centralized in the `TRAIN_CONFIG` dictionary for easy experimentation.

## Usage

```bash
python shot_detector.py
```

Press `q` to exit the live view.

## Notes

- Device selection (`get_device`) automatically falls back CUDA → MPS → CPU, making the pipeline portable across GPU and CPU-only environments.
- Confidence thresholds are relaxed near the hoop region (`in_hoop_region`) to improve recall on the ball during critical scoring moments.
