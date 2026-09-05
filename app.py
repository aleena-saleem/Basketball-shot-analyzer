"""
Basketball Shot Detection & Tracker — Streamlit Dashboard
"""

import streamlit as st
import cv2
import cvzone
import math
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ultralytics import YOLO

from utils import (
    score, detect_down, detect_up,
    in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Basketball Shot Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --bg: #101614;
        --panel: #18201d;
        --panel-soft: #1d2723;
        --border: #2b3933;
        --border-strong: #3a5046;
        --text: #edf3ef;
        --muted: #8f9e96;
        --green: #67c58c;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 8%, rgba(103, 197, 140, 0.07), transparent 28%),
            linear-gradient(135deg, #0f1412 0%, #141b18 48%, #101614 100%);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #121916 !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.7rem 1.15rem;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text);
    }

    .sidebar-brand {
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.2px;
        margin-bottom: 0.2rem;
    }

    .sidebar-caption {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 1.35rem;
    }

    .sidebar-card {
        background: linear-gradient(145deg, #19221e, #151c19);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 13px 14px;
        margin: 10px 0;
    }

    .sidebar-card-title {
        color: var(--green);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 6px;
    }

    .sidebar-card-text {
        color: #c3cec8;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #b9c7bf;
        font-size: 0.82rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 0 4px rgba(103, 197, 140, 0.10);
    }

    /* Header */
    .main-header {
        color: var(--text);
        font-size: 2.65rem;
        font-weight: 750;
        line-height: 1.1;
        text-align: center;
        letter-spacing: -1.1px;
        padding: 4px 0 7px;
    }

    .main-header-accent {
        color: var(--green);
    }

    .sub-header {
        color: var(--muted);
        text-align: center;
        font-size: 0.94rem;
        margin: 0 auto 1.55rem;
        max-width: 760px;
    }

    .header-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
        margin: 0 0 1.55rem;
    }

    /* Section headings */
    .section-title {
        color: #dce8e1;
        font-size: 0.82rem;
        font-weight: 700;
        border-left: 3px solid var(--green);
        padding-left: 10px;
        margin: 1.25rem 0 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.15px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(24, 32, 29, 0.78);
        border: 1px solid var(--border-strong);
        border-radius: 12px;
        padding: 4px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #18201d;
        border: 1px dashed #3d5148;
        border-radius: 9px;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #496257;
        color: #dfe9e3;
        background: #202b26;
        border-radius: 7px;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1b2420, #171e1b);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 15px 16px;
        min-height: 108px;
        box-shadow: 0 7px 24px rgba(0, 0, 0, 0.16);
    }

    div[data-testid="metric-container"] label {
        color: #93a39a !important;
        font-weight: 600;
        font-size: 0.73rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: var(--green) !important;
    }

    /* Status badges */
    .status-make {
        background: rgba(103, 197, 140, 0.12);
        color: #78d69b;
        border: 1px solid rgba(103, 197, 140, 0.25);
        padding: 4px 11px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
    }

    .status-miss {
        background: rgba(220, 110, 110, 0.10);
        color: #df9292;
        border: 1px solid rgba(220, 110, 110, 0.22);
        padding: 4px 11px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
    }

    .status-waiting {
        background: rgba(143, 158, 150, 0.10);
        color: #aebbb4;
        border: 1px solid rgba(143, 158, 150, 0.20);
        padding: 4px 11px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #486154;
        background: #1d2923;
        color: #e5eee9;
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--green);
        color: #ffffff;
        background: #22332a;
    }

    /* Progress */
    .stProgress > div > div > div > div {
        background: var(--green);
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 9px;
        border: 1px solid var(--border);
        background: #18201d;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Landing feature cards */
    .feature-card {
        background: linear-gradient(145deg, #1b2420, #161d1a);
        border: 1px solid var(--border);
        border-radius: 12px;
        min-height: 150px;
        padding: 24px 22px;
        box-shadow: 0 7px 24px rgba(0, 0, 0, 0.14);
    }

    .feature-number {
        color: var(--green);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }

    .feature-title {
        color: #e5eee9;
        font-size: 1rem;
        font-weight: 700;
        margin: 12px 0 7px;
    }

    .feature-text {
        color: #899991;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "makes": 0,
        "attempts": 0,
        "shot_log": [],          # list of {"frame": int, "result": "Make"/"Miss", "conf": float}
        "conf_history": [],       # per-frame max ball confidence
        "processing_done": False,
        "output_video_path": None,
        "total_frames": 0,
        "fps": 30,
        "detection_counts": {"Basketball": 0, "Basketball Hoop": 0},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# Model loader (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    model_path = Path(__file__).parent / "best.pt"
    if not model_path.exists():
        st.error("best.pt not found in project directory!")
        st.stop()
    return YOLO(str(model_path))


# ─────────────────────────────────────────────
# Core processing function
# ─────────────────────────────────────────────
def process_video(video_path: str, conf_ball: float, conf_hoop: float,
                  progress_bar, status_text, frame_placeholder, stats_placeholder):
    """
    Run shot detection on the given video file.
    Returns (makes, attempts, shot_log, conf_history, output_path, fps, total_frames, detection_counts)
    """
    model   = load_model()
    device  = get_device()
    cap     = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    out_path = video_path.replace(".mp4", "_output.mp4").replace(".avi", "_output.avi")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    class_names = ["Basketball", "Basketball Hoop"]

    ball_pos  = []
    hoop_pos  = []
    makes     = 0
    attempts  = 0
    up        = False
    down      = False
    up_frame  = 0
    down_frame = 0
    fade_frames   = 20
    fade_counter  = 0
    overlay_color = (0, 0, 0)
    overlay_text  = "Waiting..."

    frame_count = 0
    shot_log    = []
    conf_history = []
    detection_counts = {"Basketball": 0, "Basketball Hoop": 0}

    PREVIEW_EVERY = max(1, total // 80)  # update preview ~80 times

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, device=device, verbose=False)

        frame_max_conf = 0.0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                current_class = class_names[cls]
                center = (int(x1 + w / 2), int(y1 + h / 2))

                if current_class == "Basketball":
                    frame_max_conf = max(frame_max_conf, conf)
                    if conf > conf_ball or (in_hoop_region(center, hoop_pos) and conf > 0.15):
                        ball_pos.append((center, frame_count, w, h, conf))
                        cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255, 165, 0))
                        detection_counts["Basketball"] += 1

                if current_class == "Basketball Hoop" and conf > conf_hoop:
                    hoop_pos.append((center, frame_count, w, h, conf))
                    cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(0, 200, 255))
                    detection_counts["Basketball Hoop"] += 1

        conf_history.append(round(frame_max_conf, 3))

        # ── Clean motion ──
        ball_pos = clean_ball_pos(ball_pos, frame_count)
        for bp in ball_pos:
            cv2.circle(frame, bp[0], 4, (0, 0, 255), -1)

        if len(hoop_pos) > 1:
            hoop_pos = clean_hoop_pos(hoop_pos)
            cv2.circle(frame, hoop_pos[-1][0], 5, (128, 128, 0), -1)

        # ── Shot detection ──
        if len(hoop_pos) > 0 and len(ball_pos) > 0:
            if not up:
                up = detect_up(ball_pos, hoop_pos)
                if up:
                    up_frame = ball_pos[-1][1]

            if up and not down:
                down = detect_down(ball_pos, hoop_pos)
                if down:
                    down_frame = ball_pos[-1][1]

            if frame_count % 10 == 0:
                if up and down and up_frame < down_frame:
                    attempts += 1
                    up = False
                    down = False

                    if score(ball_pos, hoop_pos):
                        makes += 1
                        overlay_color = (0, 255, 0)
                        overlay_text  = "MAKE"
                        shot_log.append({"frame": frame_count, "result": "Make",
                                         "conf": round(frame_max_conf, 3),
                                         "shot_num": attempts})
                    else:
                        overlay_color = (0, 0, 255)
                        overlay_text  = "MISS"
                        shot_log.append({"frame": frame_count, "result": "Miss",
                                         "conf": round(frame_max_conf, 3),
                                         "shot_num": attempts})
                    fade_counter = fade_frames

        # ── Draw HUD ──
        score_text = f"{makes} / {attempts}"

        # Background box for score
        cv2.rectangle(frame, (30, 60), (280, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (30, 60), (280, 150), (255, 107, 53), 2)
        cv2.putText(frame, "SCORE", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 107, 53), 2)
        cv2.putText(frame, score_text, (45, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
        cv2.putText(frame, score_text, (45, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 107, 53), 2)

        # Accuracy %
        acc = (makes / attempts * 100) if attempts > 0 else 0
        acc_text = f"ACC: {acc:.0f}%"
        cv2.putText(frame, acc_text, (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Overlay shot result
        if overlay_text not in ("Waiting...",):
            (tw, _th), _baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
            tx = frame.shape[1] - tw - 40
            ty = 110
            cv2.putText(frame, overlay_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, overlay_color, 6)
            cv2.putText(frame, overlay_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        # Frame fade
        if fade_counter > 0:
            alpha = 0.25 * (fade_counter / fade_frames)
            colored = np.full_like(frame, overlay_color)
            frame   = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
            fade_counter -= 1

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (30, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        writer.write(frame)

        # ── UI updates ──
        if frame_count % PREVIEW_EVERY == 0 and total > 0:
            pct = frame_count / total
            progress_bar.progress(min(pct, 1.0))
            status_text.markdown(
                f"**Processing frame {frame_count}/{total}** &nbsp;|&nbsp; "
                f"Makes: **{makes}** &nbsp;|&nbsp; Attempts: **{attempts}** &nbsp;|&nbsp; "
                f"Accuracy: **{acc:.1f}%**"
            )
            # Live preview (resize to save bandwidth)
            preview = cv2.resize(frame, (640, int(height * 640 / width)))
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(preview_rgb, channels="RGB", use_container_width=True)

            # Live mini-stats
            with stats_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Makes", makes)
                c2.metric("Attempts", attempts)
                c3.metric("Accuracy", f"{acc:.1f}%")

        frame_count += 1

    cap.release()
    writer.release()

    return makes, attempts, shot_log, conf_history, out_path, fps, total, detection_counts


# ─────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#1a1f2e",
    font=dict(color="#c9d1d9"),
    margin=dict(l=40, r=20, t=40, b=40),
)

def chart_shot_results(shot_log):
    if not shot_log:
        return None
    makes  = sum(1 for s in shot_log if s["result"] == "Make")
    misses = sum(1 for s in shot_log if s["result"] == "Miss")
    fig = go.Figure(go.Pie(
        labels=["Makes", "Misses"],
        values=[makes, misses],
        hole=0.55,
        marker=dict(colors=["#4ade80", "#f87171"],
                    line=dict(color="#1a1f2e", width=3)),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{makes}/{makes+misses}</b>", showarrow=False,
                       font=dict(size=18, color="white"), x=0.5, y=0.5)
    fig.update_layout(title="Shot Results", showlegend=True,
                      legend=dict(orientation="h", x=0.2, y=-0.1), **DARK_LAYOUT)
    return fig


def chart_shot_timeline(shot_log, total_frames, fps):
    if not shot_log:
        return None
    frames   = [s["frame"] for s in shot_log]
    results  = [s["result"] for s in shot_log]
    confs    = [s["conf"] for s in shot_log]
    colors   = ["#4ade80" if r == "Make" else "#f87171" for r in results]
    symbols  = ["circle" if r == "Make" else "x" for r in results]
    times    = [f / fps for f in frames]

    fig = go.Figure()
    for i, (t, r, c, col, sym) in enumerate(zip(times, results, confs, colors, symbols)):
        fig.add_trace(go.Scatter(
            x=[t], y=[i + 1],
            mode="markers+text",
            marker=dict(color=col, size=16, symbol=sym, line=dict(color="white", width=1)),
            text=[r], textposition="middle right",
            textfont=dict(color=col, size=11),
            name=r,
            showlegend=(i < 2),
            hovertemplate=f"Shot #{i+1}<br>Result: {r}<br>Time: {t:.1f}s<br>Conf: {c:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="Shot Timeline",
        xaxis=dict(title="Time (seconds)", gridcolor="#2d3748"),
        yaxis=dict(title="Shot #", gridcolor="#2d3748", dtick=1),
        **DARK_LAYOUT,
    )
    return fig


def chart_running_accuracy(shot_log):
    if not shot_log:
        return None
    makes_running = []
    acc_running   = []
    running_makes = 0
    for i, s in enumerate(shot_log):
        if s["result"] == "Make":
            running_makes += 1
        makes_running.append(running_makes)
        acc_running.append(round(running_makes / (i + 1) * 100, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(acc_running) + 1)),
        y=acc_running,
        mode="lines+markers",
        line=dict(color="#ff6b35", width=3),
        marker=dict(size=8, color="#ff6b35"),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.15)",
        name="Accuracy %",
        hovertemplate="Shot #%{x}<br>Running Accuracy: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#4b5563",
                  annotation_text="50%", annotation_font_color="#9ca3af")
    fig.update_layout(
        title="Running Accuracy",
        xaxis=dict(title="Shot Number", gridcolor="#2d3748"),
        yaxis=dict(title="Accuracy (%)", range=[0, 105], gridcolor="#2d3748"),
        **DARK_LAYOUT,
    )
    return fig


def chart_confidence(conf_history, fps):
    if not conf_history:
        return None
    times = [i / fps for i in range(len(conf_history))]
    # Smooth with rolling average
    window = max(1, len(conf_history) // 100)
    smoothed = np.convolve(conf_history, np.ones(window) / window, mode="same").tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=conf_history,
        mode="lines", line=dict(color="rgba(100,160,255,0.3)", width=1),
        name="Raw", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=smoothed,
        mode="lines", line=dict(color="#60a5fa", width=2),
        name="Smoothed", showlegend=True,
    ))
    fig.update_layout(
        title="Ball Detection Confidence Over Time",
        xaxis=dict(title="Time (seconds)", gridcolor="#2d3748"),
        yaxis=dict(title="Confidence", range=[0, 1.05], gridcolor="#2d3748"),
        **DARK_LAYOUT,
    )
    return fig


def chart_detection_bar(detection_counts):
    fig = go.Figure(go.Bar(
        x=list(detection_counts.keys()),
        y=list(detection_counts.values()),
        marker=dict(
            color=["#ff6b35", "#60a5fa"],
            line=dict(color="#1a1f2e", width=2),
        ),
        text=list(detection_counts.values()),
        textposition="outside",
        textfont=dict(color="white"),
    ))
    fig.update_layout(
        title="Total Object Detections",
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
        **DARK_LAYOUT,
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Basketball Shot Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Computer vision analysis system</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-text">
            An AI-powered computer vision system for detecting basketball shots,
            tracking ball movement, and automatically classifying attempts as
            makes or misses from video footage.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Object Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">YOLOv8</div>
        <div class="sidebar-card-text">
            A custom-trained YOLOv8 model detects two target classes:
            Basketball and Basketball Hoop.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Motion Tracking</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Trajectory Analysis</div>
        <div class="sidebar-card-text">
            Detected ball and hoop positions are cleaned and tracked across
            frames to estimate the ball trajectory and movement through the hoop region.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Shot Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Up / Down Region Logic</div>
        <div class="sidebar-card-text">
            A shot attempt is identified when the tracked basketball moves
            through the defined upper and lower regions in sequence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Shot Classification</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Make / Miss</div>
        <div class="sidebar-card-text">
            The scoring function evaluates the tracked ball trajectory relative
            to the detected hoop and classifies each completed attempt.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Inference</div>', unsafe_allow_html=True)
    model_path = Path("best.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_048_576
        st.markdown(
            f'<div class="sidebar-card"><div class="sidebar-status">'
            f'<span class="status-dot"></span><span>Model ready</span></div>'
            f'<div class="sidebar-card-text" style="margin-top:6px;">'
            f'best.pt &middot; {size_mb:.1f} MB</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.error("best.pt not found")

    device = get_device()
    st.markdown(
        f'<div class="sidebar-card"><div class="sidebar-card-title">Runtime</div>'
        f'<div class="sidebar-card-text">Inference device: <strong>{device.upper()}</strong></div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">Technical Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-text">
            Python &middot; Streamlit &middot; YOLOv8 &middot; OpenCV &middot; CVZone &middot; NumPy &middot; Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fixed inference thresholds matching the original detection implementation.
    conf_ball = 0.30
    conf_hoop = 0.50


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
st.markdown('<div class="main-header"><span class="main-header-accent">Basketball</span> Shot Detection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered video analysis for automated shot tracking and performance analytics</div>',
            unsafe_allow_html=True)
st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)

# ── Upload section ──
st.markdown('<div class="section-title">Video Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a basketball video",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported formats: MP4, AVI, MOV, MKV",
)

# Resolve video path
video_path_to_process = None
if uploaded_file:
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    video_path_to_process = tmp_path
    st.success(f"Uploaded: **{uploaded_file.name}**")


# ── Process ──
if video_path_to_process:
    st.markdown("---")
    st.markdown('<div class="section-title">Processing Video</div>', unsafe_allow_html=True)

    progress_bar     = st.progress(0.0)
    status_text      = st.empty()
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    with st.spinner("Running YOLO detection..."):
        t0 = time.time()
        (makes, attempts, shot_log, conf_history,
         out_path, fps, total_frames, detection_counts) = process_video(
            video_path_to_process, conf_ball, conf_hoop,
            progress_bar, status_text, frame_placeholder, stats_placeholder
        )
        elapsed = time.time() - t0

    progress_bar.progress(1.0)
    status_text.success(f"Processing complete in **{elapsed:.1f}s** ({total_frames} frames @ {fps:.0f} FPS)")
    frame_placeholder.empty()
    stats_placeholder.empty()

    # Store in session state
    st.session_state.makes           = makes
    st.session_state.attempts        = attempts
    st.session_state.shot_log        = shot_log
    st.session_state.conf_history    = conf_history
    st.session_state.output_video_path = out_path
    st.session_state.total_frames    = total_frames
    st.session_state.fps             = fps
    st.session_state.detection_counts = detection_counts
    st.session_state.processing_done = True

# ─────────────────────────────────────────────
# Results Dashboard
# ─────────────────────────────────────────────
if st.session_state.processing_done:
    makes    = st.session_state.makes
    attempts = st.session_state.attempts
    shot_log = st.session_state.shot_log
    fps      = st.session_state.fps
    total_f  = st.session_state.total_frames
    d_counts = st.session_state.detection_counts

    accuracy   = (makes / attempts * 100) if attempts > 0 else 0
    misses     = attempts - makes
    video_len  = total_f / fps if fps else 0

    st.markdown("---")
    st.markdown('<div class="section-title">Performance Dashboard</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Makes",    makes,    f"+{makes}")
    k2.metric("Misses",   misses,   delta=f"-{misses}", delta_color="inverse")
    k3.metric("Attempts", attempts)
    k4.metric("Accuracy", f"{accuracy:.1f}%",
              delta=f"{'Above' if accuracy >= 50 else 'Below'} 50%",
              delta_color="normal" if accuracy >= 50 else "inverse")
    k5.metric("Frames",  total_f)
    k6.metric("Duration", f"{video_len:.1f}s")

    st.markdown("---")

    # ── Last shot result ──
    if shot_log:
        last = shot_log[-1]
        badge_class = "status-make" if last["result"] == "Make" else "status-miss"
        st.markdown(
            f"**Last Shot:** <span class='{badge_class}'>{last['result']}</span> "
            f"&nbsp;·&nbsp; Shot #{last['shot_num']} &nbsp;·&nbsp; "
            f"Frame {last['frame']} &nbsp;·&nbsp; Conf: {last['conf']:.2f}",
            unsafe_allow_html=True
        )
    else:
        st.markdown("**Last Shot:** <span class='status-waiting'>No shots detected</span>",
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1 ──
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_shot_results(shot_log)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shots detected to chart.")

    with c2:
        fig = chart_running_accuracy(shot_log)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shots detected to chart.")

    # ── Charts Row 2 ──
    c3, c4 = st.columns(2)
    with c3:
        fig = chart_confidence(st.session_state.conf_history, fps)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = chart_detection_bar(d_counts)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── Shot timeline (full width) ──
    fig = chart_shot_timeline(shot_log, total_f, fps)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Shot log table ──
    st.markdown("---")
    st.markdown('<div class="section-title">Shot Log</div>', unsafe_allow_html=True)
    if shot_log:
        import pandas as pd
        df = pd.DataFrame(shot_log)
        df["time_sec"] = (df["frame"] / fps).round(2)
        df = df[["shot_num", "result", "time_sec", "frame", "conf"]]
        df.columns = ["Shot #", "Result", "Time (s)", "Frame", "Confidence"]
        df["Result"] = df["Result"].apply(
            lambda r: "Make" if r == "Make" else "Miss"
        )

        st.dataframe(
            df.style.apply(
                lambda col: [
                    "background-color: #14532d; color: #4ade80" if "Make" in v
                    else "background-color: #450a0a; color: #f87171"
                    for v in col
                ] if col.name == "Result" else [""] * len(col),
                axis=0
            ),
            use_container_width=True,
            height=min(400, 50 + len(df) * 35),
        )
    else:
        st.info("No shots were detected in this video.")

    # ── Output video download ──
    st.markdown("---")
    st.markdown('<div class="section-title">Download Output</div>', unsafe_allow_html=True)
    out_p = st.session_state.output_video_path
    if out_p and os.path.exists(out_p):
        with open(out_p, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Annotated Video",
            data=video_bytes,
            file_name="basketball_detection_output.mp4",
            mime="video/mp4",
            use_container_width=True,
        )

        # Also show inline preview
        st.markdown("**Preview of annotated output:**")
        st.video(video_bytes)

else:
    # Landing placeholder
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Workflow</div>', unsafe_allow_html=True)

    ph1, ph2, ph3 = st.columns(3)
    with ph1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">01 / INPUT</div>
            <div class="feature-title">Upload Video</div>
            <div class="feature-text">Upload basketball footage in MP4, AVI, MOV, or MKV format.</div>
        </div>
        """, unsafe_allow_html=True)

    with ph2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">02 / DETECTION</div>
            <div class="feature-title">Object Detection</div>
            <div class="feature-text">YOLOv8 identifies the basketball and hoop across the video frames.</div>
        </div>
        """, unsafe_allow_html=True)

    with ph3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">03 / ANALYTICS</div>
            <div class="feature-title">Shot Analytics</div>
            <div class="feature-text">Trajectory analysis produces shot outcomes, accuracy metrics, and visual reports.</div>
        </div>
        """, unsafe_allow_html=True)

