The server works in a real-time fashion.
It takes one image at a time and outputs a mask (and a few other things).
It is stateful, so it uses previous masks to make a prediciton.

## Run the server:
Set `HF_TOKEN` environment variable, then:
```bash
uv run demo/server.py
```

## Run the GUI:
```bash
uv run demo/app.py
```

## Server's API:
Prompt once (text or a point), with the first frame only. Every later frame is
just tracked:

  - "prompt": a text description.
  - "point":  an [x, y] click in 0..1 coords (0,0 = top-left).

```javascript
  POST /sessions
    Does:   opens a new streaming session.
    Input:  None.
    Output: {"session_id": "..."}.

  POST /sessions/{id}/predict
    Does:   segments one frame; if a prompt/point is given it is set first, then
            every later frame is tracked.
    Input:  {"image": <b64 jpeg/png>, "prompt": "cat" | null, "point": [x, y] | null}.
    Output: {"frame_index": int,
             "image": <b64 jpeg>,
             "objects": [{"id", "box_xywh", "prob"}, ...]}.
            "image" is the input frame with every mask painted red at alpha 0.75.

  POST /sessions/{id}/reset
    Does:   clears the session's tracking memory (keeps the session open).
    Input:  None.
    Output: {"ok": true}.

  DELETE /sessions/{id}
    Does:   closes the session and frees its memory.
    Input:  nothing.
    Output: {"ok": true}.

  POST /gemini
    Does:   generic Gemini call; the caller supplies the prompt, JSON schema, and
            image(s) -- the server has no hardcoded task. Needs Gemini/Vertex AI
            credentials (`gcloud auth application-default login`) and
            GOOGLE_CLOUD_PROJECT set.
    Input:  {"images": [<b64 jpeg/png>, ...], "prompt": "...", "schema": {<json schema>},
             "history": [{"role": "user" | "model", "prompt": "...", "images": [...]}, ...]}.
            "history" is optional: earlier turns replayed as context, so a follow-up
            question can refer to what was asked and answered before.
    Output: {"result": <parsed JSON matching schema>}.
```

## Masking a lerobot dataset
Three phases, so the cheap and unreliable step (Gemini) is not stuck behind the
expensive one (SAM: ~13 fps per camera, so ~2.5 h for a 40k-frame 3-camera dataset):

```bash
# 1. one point per camera per episode -> json + a preview jpeg per episode
python scripts/find_point_with_gemini.py data/clear/ind-iso-1 [--middle-frame]

# 2. browse the previews, click to fix whatever Gemini got wrong
python scripts/correct_point.py data/annotations/ind-iso-1.json

# 3. once the points look right, mask every camera with them
python scripts/mask.py data/annotations/ind-iso-1.json
```

Phase 1 is a short conversation per episode, one turn per camera. The overview
("side") camera goes first and gets the real question; each remaining camera is
then a follow-up turn -- with that first question and Gemini's own answer still in
the context -- asking where the same item is in this view. One turn holding every
view at once does not work: it answers the same coordinates for all of them, which
cannot be right for cameras in different places. A camera where the item is not
visible gets a `null` point and is copied through unmasked.

Without `--middle-frame` Gemini points at frame 0 (the item sticking out of the
crate) and the episode is tracked forward. With it, Gemini points at the middle
frame (the item the robot is holding -- a much easier question) and the episode is
tracked backwards to its start and forwards to its end from there.

`find_point_with_gemini.py` and `mask.py` take `--episode N` to try a single
episode first. Nothing but phase 1 talks to Gemini, so fixing points is free.

## Model weights
Download model weights and put them into `~/.cache/huggingface/hub/`

## Run the server in Docker
Needs a GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

1. Get the weights into your host's HF cache as described above (`~/.cache/huggingface/hub/`) — the
   container mounts that same directory, it doesn't ship the weights itself.
2. Build and run:
   ```bash
   docker compose up --build
   ```
3. The server is now reachable at `http://localhost:8006`.