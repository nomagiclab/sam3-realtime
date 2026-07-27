import base64
import json
import os
import threading

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types

from sam3.model_builder import build_sam3_stream_predictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")
PREDICTOR = build_sam3_stream_predictor(device=DEVICE)

# Lock for multiple threads
LOCK = threading.Lock()

app = FastAPI(title="SAM3 server")

# Gemini (Vertex AI) client for /detect_with_model. Uses the credentials from
# `gcloud auth login` / `gcloud auth application-default login`, no API key needed.
# Built lazily so that people without Gemini access can still use every other
# endpoint; only hitting /detect_with_model requires it to succeed.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
_GEMINI_CLIENT = None


def get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        try:
            _GEMINI_CLIENT = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Gemini client unavailable: {e}")
    return _GEMINI_CLIENT

# Hardcoded task: point at the item sticking out of its box. Gemini's spatial
# convention is [y, x] normalized to 0..1000.
DETECT_PROMPT = (
    "Your primary goal is to identify the single, main crate located in "
    "the foreground of the image, directly underneath the robot's tool. "
    ""
    "Within *only* that specific crate, find the one item that sticks out / "
    "protrudes beyond the top edge of that crate. "
    ""
    "Strictly follow these rules: "
    "1. Focus *only* on the items within or on the edge of the one main foreground crate. "
    "2. Completely ignore all background containers, other crates, and background items. "
    "3. Ignore the robot arm, its tools, and items on the distant floor. "
    ""
    "Note: Be aware that the protruding item may have limited visibility. Pay special "
    "attention to the front wall of the crate (closest to the camera) where perspective "
    "makes it hard to see, and look carefully if the item is partially occluded by the robot's tool. "
    ""
)
DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "point": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["point"],
}


################## Helpers ##################
def b64_to_rgb(data: str) -> np.ndarray:
    """base64 image string -> (H, W, 3) uint8 rgb array"""
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mask_to_b64(mask: np.ndarray) -> str:
    """(H, W) bool mask -> base64 png string"""
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    return base64.b64encode(buf).decode()


def infer_frame(session_id: str, frame: np.ndarray, prompt=None, point=None):
    """Add one frame to a session, (optionally) set a prompt or point and run inference.

    Returns (frame_index, outputs).
    `outputs` describes every object found on this frame
    For N objects found we have these numpy arrays:
        out_obj_ids     (N,)      int   id of each tracked object
        out_probs        (N,)       float confidence 0..1
        out_boxes_xywh   (N, 4)     float box [x, y, w, h], normalized to 0..1
        out_binary_masks (N, H, W)  bool  segmentation mask, at frame resolution
    """
    with LOCK, torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        # 1. hand the frame to the session; it returns this frame's index
        add_response = PREDICTOR.handle_request(
            {"type": "add_frame", "session_id": session_id, "frame": frame}
        )
        idx = add_response["frame_index"]

        # 2. decide what to run on this frame
        if prompt:
            # a text prompt: set it once, the model reuses it on later frames
            request = {"type": "add_prompt", "session_id": session_id, "frame_index": idx, "text": prompt}
        elif point:
            request = {"type": "add_prompt", "session_id": session_id, "frame_index": idx,
                       "points": [list(point)], "point_labels": [1],
                       "obj_id": 1, "rel_coordinates": True}
        else:
            # no new prompt: just keep tracking whatever was asked earlier
            request = {"type": "run_inference", "session_id": session_id, "frame_index": idx}

        out = PREDICTOR.handle_request(request)["outputs"]
    return idx, out


################## Endpoints ##################
@app.post("/sessions")
def open_session():
    with LOCK:
        return PREDICTOR.handle_request({"type": "start_session"})  # -> {"session_id": ...}


@app.post("/sessions/{session_id}/predict")
def predict(session_id: str, body: dict):
    """
    Send the prompt with the first frame only, and after that just send frames and
    the model keeps tracking what it already found.

    Returns the raw masks.

    Body:   {"image": <b64 jpeg/png>, "prompt": "cat" | null, "point": [x, y] | null}
    Output: {"frame_index": int,
             "objects": [{"id": int, "box_xywh": [x, y, w, h], "prob": float, "mask": <b64 png, white = object>}, ...]}
    """
    frame = b64_to_rgb(body["image"])
    idx, out = infer_frame(session_id, frame, prompt=body.get("prompt"), point=body.get("point"))

    # turn the model's numpy arrays into a JSON list of objects
    objects = []
    for i in range(len(out["out_obj_ids"])):
        objects.append({
            "id": int(out["out_obj_ids"][i]),
            "box_xywh": [float(v) for v in out["out_boxes_xywh"][i]],
            "prob": float(out["out_probs"][i]),
            "mask": mask_to_b64(out["out_binary_masks"][i]),
        })

    return {"frame_index": idx, "objects": objects}


@app.post("/detect_with_model")
def detect_with_model(body: dict):
    """Ask Gemini to point at the item sticking out of the box (task is hardcoded).

    Body:   {"image": <b64 jpeg/png>}
    Output: {"point": [x, y]}  normalized 0..1, same convention as /predict's "point"
    """
    frame = b64_to_rgb(body["image"])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    image_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")

    response = get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=[DETECT_PROMPT, image_part],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DETECT_SCHEMA,
        ),
    )
    try:
        y1000, x1000 = json.loads(response.text)["point"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned no usable point: {e}")

    return {"point": [x1000 / 1000, y1000 / 1000]}


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str):
    """Forget the tracked objects but keep the session open (ready for a new prompt)."""
    with LOCK:
        PREDICTOR.handle_request({"type": "reset_session", "session_id": session_id})
    return {"ok": True}


@app.delete("/sessions/{session_id}")
def close_session(session_id: str):
    """Close the session and free the memory."""
    with LOCK:
        PREDICTOR.handle_request({"type": "close_session", "session_id": session_id})
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
