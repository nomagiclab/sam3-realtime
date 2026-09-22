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

## Masking a lerobot dataset, the short way

Pick-and-place recordings answer the "which item?" question themselves, so nobody has to
click 247 episodes:

```bash
python scripts/fetch_dataset.py nomagic/jr-pnp-tomek-1   # -> data/clear/jr-pnp-tomek-1
python scripts/auto_point.py data/clear/jr-pnp-tomek-1   # -> data/points/jr-pnp-tomek-1.json
python scripts/annotate.py                               # look at them, fix, run SAM3
```

`auto_point.py` reads each episode backwards to the last time the gripper opens, which is
the release, so the frames just before it are the robot carrying the item. In a wrist view
that item sits at a fixed spot between the fingers, the same in every episode, because the
camera is bolted to the hand. The overview camera is answered by what changed: before the
grasp the item lies on the table, after the release it is in the bin, so a pixel of the item
differs between EVERY frame before the grasp and EVERY frame after the release. The arm
differs too, but the arm moves, so over a dozen frames from before and a handful from after
the smallest difference per pixel keeps the item and drops the arm. The largest blob left
is where the item lay. The two cameras therefore get different seed frames, which is what
`frames` in the json holds. Each video file is decoded once and the files go to a process
pool, so a 300-episode recording takes about half a minute.

Two numbers are rig-specific and worth a look before trusting a new dataset: `--wrist-point`,
where a held item lands in the wrist view, and `--side-roi`, the part of the overview frame
that is table and never arm. Both are editable in the GUI, and the region is drawn over
every overview camera there, because a bad one is obvious on sight and invisible otherwise.

How far this gets on its own depends on the camera, and the two halves are not equally
reliable. The wrist answer comes from the gripper and is as good as the gripper signal:
right wherever a carry stretch is found, and unanswered where there is none (34 of the 298
`jr-pnp-tomek-3` episodes, whose gripper trace shows one closed stretch at the very start
and nothing after).

The overview answer was checked by eye on all five jr-pnp recordings, 745 episodes, by
independent readers: 100% on `jr-pnp-tomek-1`, 86% on `jr-pnp-tomek-3`, about 80% on the
others, 83% overall. What defeats it is not the arm being in the frame any more but the
arm standing over the item from frame 0 until the grasp, so the item is never seen on the
table at all and the point lands on the gripper; after that, items the operator added to
the table during an episode, which "arrive" exactly as the taken one "leaves"; and on
`jr-pnp-amanuel-1`, whose bin sits lower in the frame than the others', the item arriving
in the bin. Points the readers marked wrong were cleared, so the GUI offers those episodes
to be clicked rather than masking the wrong thing. Treat the overview column as a draft to
be looked at, which is what the GUI is for; the wrist column has earned more trust.

`annotate.py` is one window over the whole pass: choose a dataset, generate points for one
episode or all of them, scrub and click whatever came out wrong, watch what SAM3 makes of
each camera's points, then mask one episode into preview clips or the whole dataset. Clicks
save as they happen, into `data/points/<name>.json`, never into the Gemini files, and an
episode that was clicked is marked, so a later run over the whole dataset steps around it
rather than overwriting the answer somebody gave by hand.

A masked dataset comes out the same shape as the one it was made from, encoded the same
way: same resolution, same codec, same encoder settings, only the pixels under the mask
changed. The copy is re-encoded with what lerobot itself records with (libsvtav1, yuv420p,
a keyframe every second frame, crf 30, preset 12), because it is meant to be compared with
the original frame for frame, and any difference in compression would read as part of the
mask. Checked over 300 real frames: every ffprobe stream field and the keyframe pattern
match, bytes per frame within 3%. h264 would not have done: in yuv420p it refuses an odd
width, and the overview camera here is 355 wide, so it would have cropped the copy to 354.
The preview clips stay h264, because those are
watched in a browser and a column does not matter there.

Long work goes to a standing queue at the bottom of the window, which takes more at any
time, including while something is running. That is how the window is meant to be used:
queue a dataset for masking, carry on clicking the next one, queue that one behind it, and
so on -- nothing here ever refuses work because it is busy, and nothing waits for a whole
run to finish before the next can be asked for. Masking a dataset takes hours and the
window stays usable throughout, because the work runs behind it rather than in it.

The queue has two rails, because the two kinds of work are held up by different things.
Masking runs one dataset at a time: they share one SAM3 server on one GPU, and asking it
for four streams only makes all four slower, so a second dataset queued while the first is
being masked waits its turn. Generating points runs four at a time and never waits behind
masking: that job is decoding video, which is per core. So a dataset can have its points
generated while another is being masked.

Every job has its own bar with a time left on it, and one line above them says what the
queue as a whole is doing. The bars count the work rather than the datasets -- episodes
read for points, frames written for masking -- so they move smoothly and the estimate comes
from work really done. A job that fails says so in its row and the queue carries on with
the next; a dataset already sitting in `data/masked/<name>`, or already in the queue, is
skipped rather than queued twice, so "Mask every dataset" pressed a second time adds only
what is new. A masked dataset is written to `data/masked/<name>` and nowhere else.

Jobs are cancelled by ticking them in the picker under the bars. One that has not started
never starts; the one running stops between episodes and leaves that dataset half written,
to be removed from `data/masked` before it can be run again -- either way the rest of the
queue carries on. "Drop everything not yet started" empties the line without touching what
is running, and "stop everything" does both. The queue lives in the process: closing the
window's tab leaves it running, closing the process does not.

### If the clips play black

The clips under `data/clips` are only a browser-playable copy of what is in `data/masked`,
cut with ffmpeg on demand, and they are the only thing that can be wrong when a tile plays
black while the masked recording is whole. To tell the two apart, compare the masked file
with the clear one it was made from -- same duration and same frame count means the
masking is fine and only the clip is not:

```bash
python scripts/recut_clips.py --dataset jr-pnp-5 --list   # what is broken
python scripts/recut_clips.py --dataset jr-pnp-5          # cut those again
```

It runs beside an open window and beside a masking run, cutting only clips that are
already broken and touching neither the masked recording nor SAM3; refresh the tab when it
finishes. A broken clip is one two ffmpegs wrote at once, which used to be reachable by
refreshing the window, or opening a second one, while the first cut of a recording was
still going -- each window cuts every episode of the dataset it opens, four minutes for a
289-episode recording, and each had a pool of its own writing to one shared temporary name.
The result is an mp4 whose header comes from one write and whose body comes from the other:
whole enough that `ffprobe` reports a stream, empty enough that the player shows black. Cuts
now take a lock on the clip and a temporary name of their own, and a clip is checked box by
box before it is served rather than trusted because a file of that name exists, so two
windows on one dataset are fine.

## Reviewing a masked dataset and fixing the bad episodes

A masked dataset that is mostly right is not redone from scratch. The same window reviews it:

```bash
python scripts/fetch_pairs.py            # mim-chess-vlas/jr-pnp-N -> data/clear/jr-pnp-N,
                                         # mim-chess-vlas/jr-pnp-N-masked -> data/masked/jr-pnp-N
uv run demo/server.py                    # SAM3, in another terminal
python scripts/annotate.py --dataset data/clear/jr-pnp-1
```

Pick a dataset and the window is a 2x2 grid: the top row is the masked clip of the open
episode per camera, playing on a loop with the browser's own controls (seek bar, pause,
speed under the cog), the bottom row is the same cameras' seed frame masked by SAM3 from the
current points, i.e. what the re-mask would start from. A wrong mask is obvious in motion and
easy to miss in a frame. Clips are cut from `data/masked/<name>` by ffmpeg (about a second for
both cameras; the whole dataset is cut in the background the moment it is opened, so after
the first minute every episode opens at once) and cached under `data/clips/`.

**Clicking the item on the clip is the correction.** The player pauses, the frame it stands
on becomes that camera's seed frame and the spot its point, saved at once, and the tile below
redraws: that very frame, masked from the new point. Clicks accumulate on the same frame, and
the toggle decides whether a click grows the mask or carves out of it; a click on another
frame starts the camera over there. "undo last point" and "clear points" per camera sit under
the grid. The bottom strip of each player (the control bar) is not clickable for points; pause
and seek there, click higher up. Frames are decoded one at a time by seeking, so none of this
waits.

An episode that is wrong is flagged with the red button; the flag goes into the points
file (`"bad": true` on the episode) the moment it is pressed. "next flagged" walks the
flagged ones.

"Re-mask the flagged episodes of this dataset" (or "... of every dataset", which walks all
of `data/clear` in turn, one at a time since they share the GPU) then masks only those again, from the clear
recording with the corrected points, and splices them into the masked copy. Episodes share
video files, so a file that holds a flagged episode is re-encoded whole (its good episodes
decoded from the masked copy and written back with the same encoder settings, which costs
them one more compression pass and nothing else); a file with no flagged episode is copied
byte for byte. The result replaces `data/masked/<name>`, the copy it replaced moves to
`data/masked_prev/<name>-<time>`, the flags come off and the episodes are marked
`remasked`, so the summary line lists them as ones to look at again. Nothing is uploaded;
when the dataset looks right, push `data/masked/<name>` as before.

The five `jr-pnp-N` datasets on the Hub are the `*-target-visible-from-side-camera` cuts of
the original recordings (`jr-pnp-amanuel-1/2`, `jr-pnp-tomek-1/2/3`), with episodes dropped
and renumbered. `data/points/jr-pnp-N.map.json` records which original episode each Hub
episode is, and a `jr-pnp-N.json` points file that does not exist yet is seeded from the
original recording's points through that map, so the review starts from the points that
made the masks rather than from nothing.

## Masking a lerobot dataset, by hand
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