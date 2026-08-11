"""
Simple Gradio GUI for the SAM3 real-time server (server.py).

This file is JUST a client: it talks to the server over HTTP and shows the
results. Start the server first (python demo/server.py), then run this
(python demo/app.py).

Both tabs use the same single server endpoint, POST /predict, one frame at a
time. All the video work -- decoding, dropping frames, encoding the result --
lives here, which is why the server stays tiny.

Two tabs:
  1. Video file - upload a clip; prompt with text OR by clicking a point on the
                  first frame.
  2. Webcam     - live camera frames; type a prompt and click "Set prompt".
"""

import base64
import math
import os
import tempfile

import av  # PyAV: writes browser-playable H.264 (cv2's mp4v is not playable in browsers)
import cv2
import gradio as gr
import numpy as np
import requests

SERVER = os.environ.get("SAM3_SERVER", "http://localhost:8006")


# --- Tiny HTTP client for the server -----------------------------------------
def rgb_to_b64(arr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])  # JPEG: ~60x faster than PNG
    return base64.b64encode(buf).decode()


def b64_to_rgb(data: str) -> np.ndarray:
    """base64 image from the server -> (H, W, 3) uint8 rgb array."""
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def new_session() -> str:
    return requests.post(f"{SERVER}/sessions", timeout=60).json()["session_id"]


def close_session(session_id: str) -> None:
    requests.delete(f"{SERVER}/sessions/{session_id}", timeout=60)


def reset_session(session_id: str) -> None:
    """Forget the prompt and the tracked objects, but keep the session open."""
    requests.post(f"{SERVER}/sessions/{session_id}/reset", timeout=60).raise_for_status()


def predict(session_id, rgb, prompt=None, point=None):
    """Send one frame; get back the overlay the server rendered (masks painted red)."""
    payload = {"image": rgb_to_b64(rgb)}
    if prompt:
        payload["prompt"] = prompt
    if point:
        payload["point"] = point
    r = requests.post(f"{SERVER}/sessions/{session_id}/predict", json=payload, timeout=120)
    r.raise_for_status()
    return b64_to_rgb(r.json()["image"])


# --- Tab: real-time webcam ----------------------------------------------------
# Two pieces of state, each written by exactly one place:
#   requested - what the user last asked for   (only the "Set prompt" button writes it)
#   applied   - what the session is running    (only webcam_step writes it)
# They differ only right after a click, and that is what triggers the switch.
def request_prompt(prompt, requested):
    """The "Set prompt" button. It just records the wish; the actual switch happens
    on the next camera frame, in webcam_step. Doing the switch here instead would
    fight with the frame that is already being processed.

    The counter makes a click count even when the text did not change, so clicking
    again is a way to start the tracking over.
    """
    click = requested[0] + 1 if requested else 1
    return (click, prompt.strip())


def webcam_step(frame, session_id, requested, applied):
    """One camera frame -> the overlay. Runs continuously while the camera is on."""
    if frame is None or not requested or not requested[1]:
        return frame, session_id, applied  # no prompt set yet -> just show the camera

    prompt_to_send = None
    if requested != applied:
        if session_id is None:
            session_id = new_session()
        else:
            # drop the previous prompt and everything it was tracking
            reset_session(session_id)
        prompt_to_send = requested[1]
        applied = requested

    # after the first frame prompt_to_send is None, so the model just keeps tracking
    overlay = predict(session_id, frame, prompt=prompt_to_send)
    return overlay, session_id, applied


# --- Tab 2: whole video file --------------------------------------------------
def show_first_frame(video_path):
    """When a video is uploaded, show its first frame so a point can be clicked."""
    if not video_path:
        return None, None
    cap = cv2.VideoCapture(video_path)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None, None
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return frame, frame  # preview, stored-original


def sampled_frames(path: str, stride: int):
    """Yield RGB frames from a video file, keeping 1 out of every `stride`."""
    cap = cv2.VideoCapture(path)
    try:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


class H264Writer:
    """Write RGB frames to a browser-playable H.264 mp4 via PyAV/libx264.
    (cv2's built-in mp4v codec produces files browsers cannot play.)"""

    def __init__(self, path: str, fps: float):
        self.path, self.fps = path, fps
        self.container = self.stream = None  # opened lazily on the first frame

    def write(self, rgb: np.ndarray):
        # H.264 requires even width/height, so drop the last row/column if odd
        height, width = rgb.shape[:2]
        rgb = rgb[:height - height % 2, :width - width % 2]
        if self.container is None:
            self.container = av.open(self.path, "w", options={"movflags": "faststart"})
            self.stream = self.container.add_stream("libx264", rate=round(self.fps) or 1)
            self.stream.width, self.stream.height = rgb.shape[1], rgb.shape[0]
            self.stream.pix_fmt = "yuv420p"
        for pkt in self.stream.encode(av.VideoFrame.from_ndarray(rgb, "rgb24")):
            self.container.mux(pkt)

    def close(self):
        if self.container is not None:
            for pkt in self.stream.encode():  # flush frames still buffered in the encoder
                self.container.mux(pkt)
            self.container.close()


def pick_point(original, evt: gr.SelectData):
    """Click on the first frame -> store a normalized [x,y] point and mark it."""
    if original is None:
        raise gr.Error("Upload a video first.")
    x, y = int(evt.index[0]), int(evt.index[1])
    h, w = original.shape[:2]
    marked = original.copy()
    r = max(5, round(min(w, h) / 80))
    cv2.circle(marked, (x, y), r + 2, (0, 0, 0), -1)
    cv2.circle(marked, (x, y), r, (255, 235, 59), -1)
    cv2.circle(marked, (x, y), r, (255, 255, 255), 2)
    return marked, [x / w, y / h]


def run_video(video_path, mode, prompt, point, target_fps, progress=gr.Progress()):
    """Segment a whole video, one frame at a time, through the same /predict
    endpoint the webcam uses. A video is just frames, so there is nothing
    video-specific on the server -- decoding, thinning and encoding all happen
    here. The prompt (text, or a point clicked on the first frame) is sent with
    the first frame only; after that the model just keeps tracking.
    """
    if not video_path:
        return None, None
    if mode == "Point":
        if not point:
            raise gr.Error("Click the first frame to choose a point.")
        first_prompt, first_point = None, point
    else:
        if not prompt:
            raise gr.Error("Enter a text prompt (or switch to Point).")
        first_prompt, first_point = prompt, None

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # e.g. a 30 fps clip at target 6 fps -> segment every 5th frame
    if target_fps:
        stride = max(1, round(src_fps / target_fps))
    else:
        stride = 1
    # only for the progress bar; the count in the file's metadata is an estimate
    total = math.ceil(frame_count / stride) if frame_count > 0 else 0

    overlay_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    overlay_writer = H264Writer(overlay_path, src_fps / stride)

    session_id = new_session()
    processed = 0
    try:
        for frame in sampled_frames(video_path, stride):
            if processed == 0:
                overlay = predict(session_id, frame, prompt=first_prompt, point=first_point)
            else:
                overlay = predict(session_id, frame)
            overlay_writer.write(overlay)
            processed += 1
            if total:
                total = max(total, processed)  # never show 51/50
                progress(processed / total, desc=f"{processed}/{total}")
            else:
                progress(0.5, desc=str(processed))  # video didn't report a frame count
        overlay_writer.close()
    finally:
        # runs on Stop too, so an aborted job doesn't leak a session
        close_session(session_id)
    return overlay_path


# --- Build the UI -------------------------------------------------------------
CSS = """
/* keep the prompt row compact instead of letting it span the whole page */
#prompt_row { max-width: 560px; }
/* the textbox carries a label above it, so line the button up with its bottom edge */
#prompt_row button { align-self: flex-end; }
"""

with gr.Blocks(title="SAM3 real-time", css=CSS) as demo:
    with gr.Tab("Video file"):
        vid_point, vid_frame0 = gr.State(None), gr.State(None)
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="Input video", sources=["upload"])
                vid_mode = gr.Radio(["Text", "Point"], value="Text", label="Prompt type")
                vid_prompt = gr.Textbox(label="Text prompt")
                vid_fps = gr.Slider(1, 30, value=6, step=1, label="Target FPS")
                with gr.Row():
                    vid_btn = gr.Button("Run", variant="primary")
                    vid_stop = gr.Button("Stop")
            with gr.Column():
                vid_preview = gr.Image(label="First frame — click to set a point",
                                       type="numpy", interactive=False)
                vid_out = gr.Video(label="Overlay")
        vid_in.change(show_first_frame, vid_in, [vid_preview, vid_frame0])
        vid_preview.select(pick_point, vid_frame0, [vid_preview, vid_point])
        run_event = vid_btn.click(run_video, [vid_in, vid_mode, vid_prompt, vid_point, vid_fps],
                                  vid_out)
        vid_stop.click(None, None, None, cancels=[run_event])  # takes effect within one frame

    with gr.Tab("Webcam"):
        with gr.Row(elem_id="prompt_row"):
            wc_prompt = gr.Textbox(label="Prompt", scale=3)
            wc_set = gr.Button("Set prompt", variant="primary", scale=0)
        with gr.Row():
            wc_in = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
            wc_out = gr.Image(label="Overlay")
        wc_session_id = gr.State(None)
        wc_requested, wc_applied = gr.State(None), gr.State(None)
        wc_set.click(request_prompt, [wc_prompt, wc_requested], wc_requested)
        wc_in.stream(webcam_step, [wc_in, wc_session_id, wc_requested, wc_applied],
                     [wc_out, wc_session_id, wc_applied])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
