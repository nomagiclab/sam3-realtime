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


################## Helpers ##################
def b64_to_rgb(data: str) -> np.ndarray:
    """base64 image string -> (H, W, 3) uint8 rgb array"""
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def overlay_to_b64(frame: np.ndarray, masks: np.ndarray) -> str:
    """(H, W, 3) rgb frame + (N, H, W) bool masks -> base64 jpeg of the frame with
    every mask painted red at alpha 0.75."""
    out = frame.copy()
    if len(masks):
        any_mask = masks.any(axis=0)
        out[any_mask] = (0.25 * out[any_mask] + 0.75 * np.array([255, 0, 0])).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


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


def to_content(role: str, prompt: str, images: list) -> types.Content:
    """One chat turn: its text, then its images."""
    parts = [types.Part.from_text(text=prompt)] if prompt else []
    for image in images or []:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(b64_to_rgb(image), cv2.COLOR_RGB2BGR))
        parts.append(types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg"))
    return types.Content(role=role, parts=parts)


@app.post("/gemini")
def gemini(body: dict):
    """Generic Gemini call — the caller supplies the prompt, the JSON schema, and one or more images.

    Earlier turns can be replayed via "history", so a follow-up question can refer to
    what was asked and answered before ("the item you just pointed at").

    Body:   {"images": [<b64 jpeg/png>, ...], "prompt": str, "schema": <json schema dict>,
             "history": [{"role": "user" | "model", "prompt": str, "images": [...]}, ...]}
    Output: {"result": <parsed JSON matching schema>}
    """
    contents = [to_content(turn.get("role", "user"), turn.get("prompt", ""), turn.get("images"))
                for turn in body.get("history", [])]
    contents.append(to_content("user", body["prompt"], body.get("images")))

    response = get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=body["schema"],
        ),
    )
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid JSON: {e}")

    return {"result": result}


@app.post("/sessions")
def open_session():
    with LOCK:
        return PREDICTOR.handle_request({"type": "start_session"})  # -> {"session_id": ...}


@app.post("/sessions/{session_id}/predict")
def predict(session_id: str, body: dict):
    """
    Send the prompt with the first frame only, and after that just send frames and
    the model keeps tracking what it already found.

    Returns the frame with the masks painted on it, plus per-object metadata.

    Body:   {"image": <b64 jpeg/png>, "prompt": "cat" | null, "point": [x, y] | null}
    Output: {"frame_index": int,
             "image": <b64 jpeg, masks painted red at alpha 0.75>,
             "objects": [{"id": int, "box_xywh": [x, y, w, h], "prob": float,
                          "mask": <b64 png>}, ...]}
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

    return {
        "frame_index": idx,
        "image": overlay_to_b64(frame, out["out_binary_masks"]),
        "objects": objects,
    }


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
    uvicorn.run(app, host="0.0.0.0", port=8006)
