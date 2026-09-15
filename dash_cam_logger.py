"""
SIT225 - Data Capture Technologies
Distinction Task (6D): Camera Integration and Labeling

Student: JIANUO LIU  |  ID: 225160181

Features:
- Smooth update mechanism from 5C
- 10-second batched data collection
- OpenCV webcam capture
- Auto-saves matched .csv and .jpg files to `dataset/`
"""

import os
import time
import threading
import base64
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
BUFFER_SIZE     = 200    # visible rolling window (points)
UPDATE_MS       = 100    # interval for graph smooth update
CAPTURE_MS      = 10000  # 10 seconds for webcam capture & data saving
POINTS_PER_TICK = 1

# ── Arduino IoT Cloud Configuration ───────────────────────────────────────────
DEVICE_ID = "d7904e8f-a3e6-434e-aede-a643b89ea3a2"
SECRET_KEY = "!@SlazHSPnVbpaBj9oE7l#8Yg"
FORCE_SIMULATION = True  # Set to True to generate mock data automatically

# ── Setup Dataset Directory ───────────────────────────────────────────────────
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Shared buffers ────────────────────────────────────────────────────────────
buf_time = deque(maxlen=BUFFER_SIZE)
buf_x    = deque(maxlen=BUFFER_SIZE)
buf_y    = deque(maxlen=BUFFER_SIZE)
buf_z    = deque(maxlen=BUFFER_SIZE)

_total   = {"n": 0}
_sent    = {"n": 0}
_file_idx = {"n": 1}

# ── Data source: Arduino IoT Cloud (with simulation fallback) ─────────────────
def _start_cloud():
    if FORCE_SIMULATION:
        print("[Cloud] FORCE_SIMULATION is True. Using simulated data.")
        _simulate()
        return

    try:
        from arduino.iot.cloud import ArduinoCloudClient
        # Replace with your actual credentials
        client = ArduinoCloudClient(device_id=DEVICE_ID, secret_key=SECRET_KEY)
        def _cb_x(c, v): buf_x.append(v)
        def _cb_y(c, v): buf_y.append(v)
        def _cb_z(c, v):
            buf_z.append(v)
            buf_time.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
            _total["n"] += 1
        client.register("accelerometer_x", value=None, on_write=_cb_x)
        client.register("accelerometer_y", value=None, on_write=_cb_y)
        client.register("accelerometer_z", value=None, on_write=_cb_z)
        client.start()
    except Exception as e:
        print(f"[Cloud] {e} → using simulated data")
        _simulate()

def _simulate():
    start_t = time.time()
    last_state = -1
    while True:
        elapsed = time.time() - start_t
        # Change state every 180 seconds (3 minutes per activity for faster testing)
        state = int(elapsed // 180) % 3 
        
        if state != last_state:
            state_names = ["Idle (sit still)", "Walking (walk around)", "Shaking (shake/run)"]
            print(f"\n========================================================")
            print(f"  SIMULATION STATE: {state_names[state]} ")
            print(f"  (Please act this out in front of the camera!)")
            print(f"========================================================\n")
            last_state = state

        if state == 0: # Idle
            x = np.random.normal(0, 0.02)
            y = np.random.normal(0, 0.02)
            z = 1.0 + np.random.normal(0, 0.02)
        elif state == 1: # Walking
            t = elapsed
            x = np.sin(t * 2 * np.pi * 1.5) * 0.3 + np.random.normal(0, 0.05)
            y = np.cos(t * 2 * np.pi * 1.5) * 0.3 + np.random.normal(0, 0.05)
            z = 1.0 + np.sin(t * 2 * np.pi * 3.0) * 0.5 + np.random.normal(0, 0.05)
        else: # Shaking
            x = np.random.normal(0, 1.2)
            y = np.random.normal(0, 1.2)
            z = 1.0 + np.random.normal(0, 1.2)

        buf_x.append(round(float(x), 4))
        buf_y.append(round(float(y), 4))
        buf_z.append(round(float(z), 4))
        buf_time.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
        _total["n"] += 1
        time.sleep(0.1)

# ── OpenCV Webcam Handler ─────────────────────────────────────────────────────
# Initialize once globally to prevent delays during callback
cam = None
for _cam_idx in range(4):
    print(f"Trying camera index {_cam_idx}...")
    temp_cam = cv2.VideoCapture(_cam_idx)
    if temp_cam.isOpened():
        ret, frame = temp_cam.read()
        if ret:
            print(f"Success! Using camera index {_cam_idx}.")
            cam = temp_cam
            time.sleep(1) # warm up
            break
        temp_cam.release()
    else:
        temp_cam.release()

if cam is None:
    print("Warning: Could not open any webcam (tried indices 0-3). Please check macOS Camera Permissions in System Settings -> Privacy & Security.")

# ── Wrapper function from 5C ──────────────────────────────────────────────────
def smooth_live_dash_update(time_buffer, y_buffers, sent_ref, total_ref,
                            points_per_tick=POINTS_PER_TICK, max_points=BUFFER_SIZE):
    n_new = total_ref["n"] - sent_ref["n"]
    if n_new <= 0:
        return dash.no_update

    n_emit     = min(n_new, points_per_tick)
    times_snap = list(time_buffer)
    new_times  = times_snap[-n_emit:]

    xs, ys = [], []
    for buf in y_buffers.values():
        snap = list(buf)
        xs.append(new_times)
        ys.append(snap[-n_emit:])

    sent_ref["n"] += n_emit
    trace_indices  = list(range(len(y_buffers)))
    return ({"x": xs, "y": ys}, trace_indices, max_points)

# ── Dash layout ───────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Camera Sync Dashboard")

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif",
           "backgroundColor": "#1a1a2e", "minHeight": "100vh", "padding": "20px"},
    children=[
        html.H2("Data Collection Dashboard",
                style={"color": "#e0e0e0", "textAlign": "center"}),
        html.P("Synchronizing Accelerometer & Webcam every 10 seconds",
               style={"color": "#a0a0b0", "textAlign": "center"}),
        
        html.Div(style={"display": "flex", "flexDirection": "row", "justifyContent": "center", "gap": "20px"}, children=[
            # Left: Graph
            html.Div(style={"flex": "2", "minWidth": "600px"}, children=[
                dcc.Graph(
                    id="live-graph",
                    figure={
                        "data": [
                            go.Scatter(name="X", x=[], y=[], mode="lines", line=dict(color="#ff6b6b", width=1.5)),
                            go.Scatter(name="Y", x=[], y=[], mode="lines", line=dict(color="#4ecdc4", width=1.5)),
                            go.Scatter(name="Z", x=[], y=[], mode="lines", line=dict(color="#ffe66d", width=1.5)),
                        ],
                        "layout": go.Layout(
                            paper_bgcolor="#16213e", plot_bgcolor="#0f3460",
                            font=dict(color="#e0e0e0"),
                            xaxis=dict(title="Time", showgrid=True, gridcolor="#2a3f6f", tickangle=-45, nticks=15),
                            yaxis=dict(title="Acc (g)", showgrid=True, gridcolor="#2a3f6f", range=[-1.5, 1.5]),
                            margin=dict(l=70, r=20, t=30, b=80),
                            legend=dict(bgcolor="#16213e"),
                            transition={"duration": 0},
                            uirevision="constant",
                        ),
                    },
                    style={"height": "480px"},
                ),
            ]),
            # Right: Webcam Frame
            html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column", "alignItems": "center", "backgroundColor": "#16213e", "padding": "10px", "borderRadius": "8px"}, children=[
                html.H4("Latest 10s Capture", style={"color": "#e0e0e0", "marginTop": "0"}),
                html.Img(id="webcam-image", style={"width": "100%", "maxWidth": "400px", "borderRadius": "5px"}),
                html.Div(id="capture-log", style={"color": "#4ecdc4", "marginTop": "10px", "fontSize": "14px", "fontFamily": "monospace"})
            ])
        ]),

        dcc.Interval(id="interval-fast", interval=UPDATE_MS, n_intervals=0),
        dcc.Interval(id="interval-slow", interval=CAPTURE_MS, n_intervals=0),
        html.Div(id="status", style={"color": "#a0a0b0", "textAlign": "center", "marginTop": "20px", "fontSize": "13px"}),
    ],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("live-graph", "extendData"),
    Output("status", "children"),
    Input("interval-fast", "n_intervals")
)
def update_graph_fast(_n):
    result = smooth_live_dash_update(
        time_buffer=buf_time,
        y_buffers={"X": buf_x, "Y": buf_y, "Z": buf_z},
        sent_ref=_sent, total_ref=_total
    )
    total = _total["n"]
    visible = min(total, BUFFER_SIZE)
    status = (f"⏱ Graph: {UPDATE_MS}ms | Capture: {CAPTURE_MS//1000}s | "
              f"Buffer: {visible}/{BUFFER_SIZE} | Total: {total}")
    return result, status

@app.callback(
    Output("webcam-image", "src"),
    Output("capture-log", "children"),
    Input("interval-slow", "n_intervals"),
    prevent_initial_call=True
)
def capture_and_save(_n):
    if cam is None or not cam.isOpened():
        return dash.no_update, "Error: Webcam not available"

    # Grab snapshot of last 10 seconds of data (100 samples at 10Hz)
    samples_to_take = min(100, len(buf_time))
    if samples_to_take == 0:
        return dash.no_update, "Waiting for data..."

    df = pd.DataFrame({
        "time": list(buf_time)[-samples_to_take:],
        "x": list(buf_x)[-samples_to_take:],
        "y": list(buf_y)[-samples_to_take:],
        "z": list(buf_z)[-samples_to_take:]
    })

    # Capture frame from webcam
    ret, frame = cam.read()
    if not ret:
        return dash.no_update, "Failed to grab frame"

    # Generate matching filenames
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    idx = _file_idx["n"]
    _file_idx["n"] += 1
    
    csv_path = os.path.join(DATA_DIR, f"{idx}_{timestamp}.csv")
    jpg_path = os.path.join(DATA_DIR, f"{idx}_{timestamp}.jpg")

    # Save to disk
    df.to_csv(csv_path, index=False)
    cv2.imwrite(jpg_path, frame)

    # Encode for frontend display
    _, buffer = cv2.imencode('.jpg', frame)
    b64_image = base64.b64encode(buffer).decode("utf-8")
    src = f"data:image/jpeg;base64,{b64_image}"

    log_msg = f"Saved: {idx}_{timestamp}"
    return src, log_msg

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_start_cloud, daemon=True).start()
    time.sleep(1)
    print("=" * 60)
    print(f"  SIT225 6D – Camera Sync Dashboard started")
    print(f"  Saving to directory: ./{DATA_DIR}/")
    print("  Open: http://127.0.0.1:8050")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=8050)
