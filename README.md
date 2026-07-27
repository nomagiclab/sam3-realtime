The server works in a real-time fashion.
It takes one image at a time and outputs a mask (and a few other things).
It is stateful, so it uses previous masks to make a prediciton.

## Run the server:
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
             "objects": [{"id", "box_xywh", "prob", "mask"}, ...]}.
            "mask" is a b64 png, white = object.

  POST /sessions/{id}/reset
    Does:   clears the session's tracking memory (keeps the session open).
    Input:  None.
    Output: {"ok": true}.

  DELETE /sessions/{id}
    Does:   closes the session and frees its memory.
    Input:  nothing.
    Output: {"ok": true}.
```

# Model weights: 
if you don't have access to the gated facebook/sam3 repo on Hugging Face paste the weights `~/.cache/huggingface/hub/`. (maybe setup HF_HUB_OFFLINE=1)

## Run the server in Docker
Needs a GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

1. Get the weights into your host's HF cache as described above (`~/.cache/huggingface/hub/`) — the
   container mounts that same directory, it doesn't ship the weights itself.
2. Build and run:
   ```bash
   docker compose up --build
   ```
3. The server is now reachable at `http://localhost:8000`.

The `/detect_with_model` endpoint (Gemini) is optional and needs `GOOGLE_CLOUD_PROJECT` set (and
`gcloud auth application-default login` credentials mounted/available) — skip it if you don't use that endpoint.