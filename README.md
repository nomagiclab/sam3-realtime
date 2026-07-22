## Run the server
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
