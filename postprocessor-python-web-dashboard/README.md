# Postprocessor Python Web Dashboard

A v5 post-processor that forwards real-time detection data to a companion web
app for visualization, while passing the objects message back to the AI Manager
**unmodified**. Two independent components:

| Component | File | Role |
|-----------|------|------|
| **Postprocessor** | `postprocessor-python-web-dashboard.py` | Compiled binary; receives inference results, POSTs per-frame counts and per-object size/position to the web app |
| **Web App** | `web_app.py` | Plain Python script; stores detection data in SQLite, serves a dashboard at `http://localhost:8111` |

## Architecture

```
NX AI Manager
    │  shared memory (MessagePack)
    ▼
Postprocessor  (Nuitka-compiled binary)
    │  HTTP POST /api/ingest  (JSON)
    ▼
Web App  (web_app.py)
    │  SQLite  plugin.web-dashboard.db
    ▼
Browser Dashboard  http://localhost:8111
```

## What it does

`handle_message` derives the web-app payload and POSTs it as a fire-and-forget
side effect, then returns the message untouched:

```python
counts = count_objects(msg)                 # {class_name: len(msg.objects[cls])}
sizes, positions = extract_bbox_data(msg)   # from each ObjectMetadata bbox
post_to_webapp(webapp_url, {"ts": msg.timestamp, "counts": counts,
                            "sizes": sizes, "positions": positions})
return msg                                  # forwarded unmodified
```

`extract_bbox_data` reads the normalized `ObjectMetadata` bbox:

```python
size = (round(obj.width, 4), round(obj.height, 4))
pos  = (round(obj.x + obj.width / 2, 4), round(obj.y + obj.height / 2, 4))
```

Adding the dashboard to a pipeline never changes downstream results: no
objects, attributes, or events are added.

## Behavior change from v4

- **Size/position come from the normalized bbox**: v5 bounding boxes are
  frame-relative `[0,1]` (`ObjectMetadata.x/y/width/height`), not pixel `xyxy`,
  so `extract_bbox_data` derives `size = (width, height)` and
  `pos = (x + width/2, y + height/2)`, rounded to 4 decimals (normalized
  coordinates need the precision).
- **Frame pixel dimensions are no longer sent**: the v5 objects message does not
  carry the frame `Width`/`Height`, so the `width`/`height` fields are dropped
  from the web-app payload (codec-forced).
- **The web-app URL is a launch-time `Config` arg** (`--webapp_url`), not an INI
  setting.

## Registration

Add the processor to `external_postprocessors.json` (see the colocated
`external_postprocessors.json` and `.claude/docs/external_postprocessors_json.md`):

```json
{
    "Name": "Web-Dashboard",
    "Command": "/path/to/postprocessor-python-web-dashboard",
    "Schedule": "OBJECTS",
    "LogFilepath": "/tmp/postprocess.web-dashboard.stdout.log",
    "Config": {
        "LogLevel": "DEBUG",
        "webapp_url": "http://localhost:8111"
    }
}
```

Key points:

- `Schedule: "OBJECTS"` — the processor runs on inference-results messages.
- `Config` delivers the launch-time `webapp_url` the processor POSTs frames to.
- No `Objects` or `Events`: the processor forwards the message unmodified and
  declares nothing.

## Build, test, install

Only the postprocessor binary is compiled; `web_app.py` runs directly with
Python.

```shell
mkdir -p build && cd build
cmake ..
cmake --build . --target postprocessor-python-web-dashboard
cmake --install . --component postprocessor-python-web-dashboard
```

The install step copies both the postprocessor binary and `web_app.py` to the
postprocessors directory.

Run the colocated unit tests — `test_postprocessor_python_web_dashboard` for the
post-processor and `test_web_app` for the web app's shutdown path:

```shell
cd postprocessor-python-web-dashboard
PYTHONPATH=../nxai-utilities python -m unittest discover -p 'test_*.py' -v
```

## Running the web app

```shell
# With defaults
python3 web_app.py

# Override port or database path
python3 web_app.py --port 9000 --db /path/to/data.db

# Tune retention
python3 web_app.py --timeline-capacity 100000 --scatter-capacity 10000
```

All settings are command-line flags; the web app reads no configuration file.

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | 8111 | HTTP port |
| `--db` | `plugin.web-dashboard.db` in `../etc` relative to the script if that directory exists, otherwise next to the script | SQLite database path |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `--timeline-capacity` | 50000 | Per-second timeline buckets kept (1 to 1000000) |
| `--scatter-capacity` | 5000 | Reservoir samples per scatter plot (1 to 1000000) |

The web app restores all previously collected data from SQLite on startup, so it
can be restarted independently of the postprocessor without data loss. The
dashboard is then available at `http://localhost:8111`.

## Troubleshooting

**Dashboard not loading** — confirm `web_app.py` is running and port 8111 is free.

**No data appearing** — run `curl http://localhost:8111/api/stats` to verify the
web app is reachable, and check the postprocessor log for `Could not reach web
app` warnings.

## Security

The web app binds to `0.0.0.0` with no authentication. For production
deployments, put it behind a reverse proxy with authentication and HTTPS.

## Licence

Copyright 2026, Network Optix, All rights reserved.
