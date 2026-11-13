#!/usr/bin/env python3
"""
Web Dashboard — standalone monitoring web application.

Receives detection data from the postprocessor via POST /api/ingest,
persists everything in SQLite, and serves an interactive dashboard at
http://localhost:<port>/.
"""
import os, sys, logging, logging.handlers, json, signal, random, sqlite3, time, argparse
from datetime import datetime
from threading import Thread, Lock, Event
from collections import deque, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

script_location = os.path.dirname(os.path.realpath(sys.argv[0]))

# ── Paths ─────────────────────────────────────────────────────────────────────
_etc        = os.path.join(script_location, "..", "etc")
_log_dir    = _etc if os.path.exists(_etc) else script_location
LOG_FILE    = os.path.join(_log_dir, "plugin.web-dashboard-app.log")
DEFAULT_DB  = os.path.join(_log_dir, "plugin.web-dashboard.db")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - web-dashboard-app - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=3),
    ]
)

DEFAULT_PORT          = 8111
DEFAULT_TIMELINE_CAP  = 50_000   # per-second buckets (~13.9 h at 1 fps)
DEFAULT_SCATTER_CAP   = 5_000    # reservoir samples per scatter plot
DEFAULT_FLUSH_SECS    = 10       # how often to flush meta/sampler state to DB
FLUSH_JOIN_SECS       = 5        # how long shutdown waits for the final flush

shutdown_event = Event()
web_server     = None
store          = None
logger         = None


# ── Reservoir sampler ─────────────────────────────────────────────────────────

class ReservoirSampler:
    """
    Fixed-size unbiased random sample (Vitter's Algorithm R).
    add() returns the pool index that was written, or None if the point was
    not sampled — callers use this to keep the DB in sync.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples  = []   # list[dict]
        self.n        = 0    # total points seen

    def restore(self, samples, n):
        self.samples = list(samples)
        self.n       = n

    def add(self, point):
        self.n += 1
        if len(self.samples) < self.capacity:
            idx = len(self.samples)
            self.samples.append(point)
            return idx
        idx = random.randint(0, self.n - 1)
        if idx < self.capacity:
            self.samples[idx] = point
            return idx
        return None

    def get(self):
        return list(self.samples)

    def clear(self):
        self.samples.clear()
        self.n = 0


# ── Detection store ───────────────────────────────────────────────────────────

class DetectionStore:
    """
    Thread-safe detection store backed by SQLite for persistence.

    Write strategy:
      - Timeline buckets are written to DB as soon as they are finalised (≤1/s).
      - Scatter pool updates are written incrementally per frame.
      - Counters and sampler n-values are flushed by a background thread every
        FLUSH_SECS seconds and on shutdown, to avoid a DB write on every frame.
    """

    def __init__(self, db_path, timeline_cap, scatter_cap):
        self._lock         = Lock()
        self._db_path      = db_path
        self._timeline_cap = timeline_cap
        self._scatter_cap  = scatter_cap
        self._dirty        = False

        self._init_db()
        self._load()

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _connect(self):
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS timeline (
                    ts    INTEGER PRIMARY KEY,
                    total REAL    NOT NULL,
                    pc    TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS size_samples (
                    idx INTEGER PRIMARY KEY,
                    x   REAL NOT NULL,
                    y   REAL NOT NULL,
                    c   TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS position_samples (
                    idx INTEGER PRIMARY KEY,
                    x   REAL NOT NULL,
                    y   REAL NOT NULL,
                    c   TEXT NOT NULL
                );
            """)

    def _load(self):
        """Restore in-memory state from DB on startup."""
        with self._connect() as conn:
            def meta(key, default=None):
                row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                return row[0] if row else default

            self.frame_count          = int(meta('frame_count', 0))
            self.total_objects        = int(meta('total_objects', 0))
            self.cumulative_per_class = defaultdict(int, json.loads(meta('cumulative_per_class', '{}')))
            self.current_per_class    = {}

            raw_start = meta('start_time')
            self.start_time = datetime.fromisoformat(raw_start) if raw_start else datetime.now()
            if not raw_start:
                conn.execute("INSERT OR REPLACE INTO meta VALUES ('start_time', ?)",
                             (self.start_time.isoformat(),))

            # Timeline
            rows = conn.execute(
                "SELECT ts, total, pc FROM timeline ORDER BY ts"
            ).fetchall()
            self._timeline = deque(
                [{'ts': r[0], 'total': r[1], 'pc': json.loads(r[2])} for r in rows],
                maxlen=self._timeline_cap,
            )
            self._cur_sec   = None
            self._cur_total = 0
            self._cur_pc    = defaultdict(int)

            # Scatter samplers
            def load_sampler(table, n_key):
                rows = conn.execute(
                    f"SELECT idx, x, y, c FROM {table} ORDER BY idx"
                ).fetchall()
                samples = [{'x': r[1], 'y': r[2], 'c': r[3]} for r in rows]
                n       = int(meta(n_key, len(samples)))
                s       = ReservoirSampler(self._scatter_cap)
                s.restore(samples, n)
                return s

            self._sizes     = load_sampler('size_samples',     'size_sampler_n')
            self._positions = load_sampler('position_samples', 'position_sampler_n')

        logger.info("Loaded from DB: %d timeline buckets, %d size samples, %d position samples",
                    len(self._timeline), len(self._sizes.samples), len(self._positions.samples))

    # ── Write ──────────────────────────────────────────────────────────────────

    def update(self, ts, counts, sizes, positions):
        """
        Process one detection frame.
        ts       — Unix float seconds (from the inference message Timestamp).
        counts   — {class: count}
        sizes    — flat list of {x, y, c} dicts (bbox width × height)
        positions— flat list of {x, y, c} dicts (bbox centre)
        """
        bucket_to_write = None
        sz_updates      = []   # (idx, point) pairs for DB sync
        ps_updates      = []

        with self._lock:
            self.frame_count  += 1
            frame_total        = sum(counts.values())
            self.total_objects += frame_total
            self.current_per_class = dict(counts)

            for cls, cnt in counts.items():
                self.cumulative_per_class[cls] += cnt

            # Accumulate into per-second buckets keyed by the inference timestamp
            sec = int(ts)
            if self._cur_sec != sec:
                if self._cur_sec is not None:
                    bucket_to_write = {
                        'ts':    self._cur_sec,
                        'total': self._cur_total,
                        'pc':    dict(self._cur_pc),
                    }
                    self._timeline.append(bucket_to_write)
                self._cur_sec   = sec
                self._cur_total = frame_total
                self._cur_pc    = defaultdict(int, counts)
            else:
                self._cur_total += frame_total
                for cls, cnt in counts.items():
                    self._cur_pc[cls] += cnt

            # Reservoir-sample bounding boxes
            for p in sizes:
                idx = self._sizes.add(p)
                if idx is not None:
                    sz_updates.append((idx, p))
            for p in positions:
                idx = self._positions.add(p)
                if idx is not None:
                    ps_updates.append((idx, p))

            self._dirty = True

        # DB writes happen outside the in-memory lock
        if bucket_to_write or sz_updates or ps_updates:
            self._write_incremental(bucket_to_write, sz_updates, ps_updates)

    def _write_incremental(self, bucket, sz_updates, ps_updates):
        with self._connect() as conn:
            if bucket:
                conn.execute(
                    "INSERT OR REPLACE INTO timeline (ts, total, pc) VALUES (?,?,?)",
                    (bucket['ts'], bucket['total'], json.dumps(bucket['pc'])),
                )
                # Keep only the most recent timeline_cap rows
                conn.execute(
                    "DELETE FROM timeline WHERE ts NOT IN "
                    "(SELECT ts FROM timeline ORDER BY ts DESC LIMIT ?)",
                    (self._timeline_cap,),
                )
            for idx, pt in sz_updates:
                conn.execute(
                    "INSERT OR REPLACE INTO size_samples (idx,x,y,c) VALUES (?,?,?,?)",
                    (idx, pt['x'], pt['y'], pt['c']),
                )
            for idx, pt in ps_updates:
                conn.execute(
                    "INSERT OR REPLACE INTO position_samples (idx,x,y,c) VALUES (?,?,?,?)",
                    (idx, pt['x'], pt['y'], pt['c']),
                )

    def flush_meta(self):
        """Persist counters and sampler n-values. Called by the background flush thread."""
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            pairs = [
                ('frame_count',          str(self.frame_count)),
                ('total_objects',        str(self.total_objects)),
                ('cumulative_per_class', json.dumps(dict(self.cumulative_per_class))),
                ('size_sampler_n',       str(self._sizes.n)),
                ('position_sampler_n',   str(self._positions.n)),
            ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", pairs
            )

    def clear(self):
        with self._lock:
            self.frame_count          = 0
            self.total_objects        = 0
            self.current_per_class    = {}
            self.cumulative_per_class = defaultdict(int)
            self._timeline.clear()
            self._cur_sec   = None
            self._cur_total = 0
            self._cur_pc    = defaultdict(int)
            self._sizes.clear()
            self._positions.clear()
            self.start_time = datetime.now()
            self._dirty     = False

        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM timeline;
                DELETE FROM size_samples;
                DELETE FROM position_samples;
                DELETE FROM meta;
            """)
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('start_time', ?)",
                         (self.start_time.isoformat(),))

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_stats(self):
        with self._lock:
            return {
                'frame_count':   self.frame_count,
                'total_objects': self.total_objects,
                'current_total': sum(self.current_per_class.values()),
                'uptime':        (datetime.now() - self.start_time).total_seconds(),
            }

    def get_timeline(self, max_pts=500):
        with self._lock:
            data = list(self._timeline)
            if self._cur_sec is not None:
                data.append({'ts': self._cur_sec, 'total': self._cur_total,
                             'pc': dict(self._cur_pc)})
        return _downsample(data, max_pts)

    def get_scatter(self):
        with self._lock:
            return {
                'sizes':     self._sizes.get(),
                'positions': self._positions.get(),
            }

    def get_distribution(self):
        with self._lock:
            return dict(self.cumulative_per_class)


def _downsample(data, max_pts):
    """Average-bucket downsampling for timeline data."""
    if len(data) <= max_pts:
        return data
    bsz = len(data) / max_pts
    out = []
    for i in range(max_pts):
        bucket = data[int(i * bsz): int((i + 1) * bsz)]
        if not bucket:
            continue
        classes = {c for b in bucket for c in b['pc']}
        out.append({
            'ts':    bucket[len(bucket) // 2]['ts'],
            'total': round(sum(b['total'] for b in bucket) / len(bucket), 1),
            'pc':    {c: round(sum(b['pc'].get(c, 0) for b in bucket) / len(bucket), 1)
                      for c in classes},
        })
    return out


# ── HTTP server ────────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    def do_GET(self):
        p  = urlparse(self.path)
        qs = parse_qs(p.query)
        routes = {
            '/':               lambda: self._html(get_dashboard_html()),
            '/api/stats':      lambda: self._json(store.get_stats()),
            '/api/timeline':   lambda: self._json(
                store.get_timeline(max(1, min(int(qs.get('points', [500])[0]), 5000)))),
            '/api/scatter':    lambda: self._json(store.get_scatter()),
            '/api/distribution': lambda: self._json(store.get_distribution()),
        }
        h = routes.get(p.path)
        if h:
            h()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)

        if self.path == '/api/ingest':
            try:
                data = json.loads(body)
                store.update(
                    ts        = float(data['ts']),
                    counts    = data.get('counts', {}),
                    sizes     = data.get('sizes', []),
                    positions = data.get('positions', []),
                )
                self._json({'ok': True})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.warning("Bad ingest payload: %s", e)
                self.send_error(400, str(e))

        elif self.path == '/api/clear':
            store.clear()
            self._json({'ok': True})

        else:
            self.send_error(404)

    def _html(self, content):
        b = content.encode()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)

    def _json(self, data):
        b = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(b))
        self.end_headers()
        self.wfile.write(b)


class _ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # BrokenPipeError is normal — client disconnected before we finished writing
        if issubclass(sys.exc_info()[0], BrokenPipeError):
            return
        logger.error("Unhandled error for request from %s", client_address, exc_info=True)


# ── Web server ─────────────────────────────────────────────────────────────────

def start_web_server(port):
    global web_server
    try:
        web_server = _ReusableHTTPServer(('0.0.0.0', port), DashboardHandler)
    except Exception as e:
        logger.error("Could not bind to port %d: %s", port, e)
        raise
    def run():
        try:
            web_server.serve_forever()
        except Exception as e:
            logger.error("Web server error: %s", e, exc_info=True)
    Thread(target=run, daemon=True, name="http").start()
    logger.info("Dashboard running at http://localhost:%d", port)


# ── Background flush thread ────────────────────────────────────────────────────

def start_flush_thread(interval_secs):
    def run():
        while not shutdown_event.wait(timeout=interval_secs):
            try:
                store.flush_meta()
            except Exception as e:
                logger.error("Flush error: %s", e, exc_info=True)
        # Final flush on shutdown
        try:
            store.flush_meta()
        except Exception as e:
            logger.error("Final flush error: %s", e, exc_info=True)
    thread = Thread(target=run, daemon=True, name="flush")
    thread.start()
    return thread


# ── Server lifecycle ───────────────────────────────────────────────────────────

def signal_handler(signum, _):
    logger.info("Signal %s received, shutting down.", signal.Signals(signum).name)
    shutdown_event.set()
    if web_server:
        web_server.shutdown()


def wait_for_shutdown(flush_thread, join_timeout=FLUSH_JOIN_SECS):
    """
    Block until a signal sets the shutdown event, then give the flush thread
    time to complete its final flush. Without this the thread is a daemon and
    the interpreter can exit while that flush is still in progress, losing the
    meta/sampler state it was about to write.
    """
    shutdown_event.wait()
    flush_thread.join(timeout=join_timeout)
    if flush_thread.is_alive():
        logger.warning("Flush thread still running after %ss; final flush may be incomplete",
                       join_timeout)


def set_log_level(level):
    try:
        logger.setLevel(getattr(logging, level.upper()))
    except Exception as e:
        logger.error("Log level error: %s", e, exc_info=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Web Dashboard — NX AI Manager monitoring app")
    parser.add_argument('--port',      type=int, default=DEFAULT_PORT, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument('--db',        metavar='PATH', default=DEFAULT_DB, help="SQLite database path")
    parser.add_argument('--log-level', metavar='LEVEL', default='INFO', help="Log level: DEBUG|INFO|WARNING|ERROR")
    parser.add_argument('--timeline-capacity', type=int, default=DEFAULT_TIMELINE_CAP,
                        help=f"Per-second timeline buckets kept (default: {DEFAULT_TIMELINE_CAP})")
    parser.add_argument('--scatter-capacity', type=int, default=DEFAULT_SCATTER_CAP,
                        help=f"Reservoir samples per scatter plot (default: {DEFAULT_SCATTER_CAP})")
    return parser.parse_args(argv)


def config(args):
    port, tc, sc = args.port, args.timeline_capacity, args.scatter_capacity
    if not (1 <= port <= 65535):
        logger.warning("Invalid port %d, using default %d", port, DEFAULT_PORT)
        port = DEFAULT_PORT
    if not (1 <= tc <= 1_000_000):
        logger.warning("Invalid timeline_capacity %d, using default %d", tc, DEFAULT_TIMELINE_CAP)
        tc = DEFAULT_TIMELINE_CAP
    if not (1 <= sc <= 1_000_000):
        logger.warning("Invalid scatter_capacity %d, using default %d", sc, DEFAULT_SCATTER_CAP)
        sc = DEFAULT_SCATTER_CAP
    return port, tc, sc, args.db


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

def get_dashboard_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detection Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#222;min-height:100vh;padding:16px}
.page{max-width:1400px;margin:0 auto}

.header{background:#1a1a2e;color:#fff;border-radius:10px;padding:18px 24px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.header h1{font-size:20px;font-weight:600}
.header p{font-size:13px;color:#aaa;margin-top:3px}
.badge{display:inline-flex;align-items:center;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600}
.live{background:#064e3b;color:#34d399}
.paused{background:#78350f;color:#fbbf24}
.offline{background:#374151;color:#9ca3af}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}
.card{background:#fff;border-radius:10px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.card-value{font-size:28px;font-weight:700;color:#1a1a2e;line-height:1.1}

.controls{background:#fff;border-radius:10px;padding:10px 16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08);display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.btn{padding:7px 14px;border:1px solid #ddd;border-radius:6px;background:#fff;color:#333;cursor:pointer;font-size:13px;transition:background .15s}
.btn:hover{background:#f5f5f5}
.btn.primary{background:#1a1a2e;color:#fff;border-color:#1a1a2e}
.btn.primary:hover{background:#2d2d4e}

.panel{background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:16px}
.panel-title{font-size:14px;font-weight:600;color:#444;margin-bottom:14px;text-transform:uppercase;letter-spacing:.4px}
.chart-wrap{position:relative}

.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}
.two-col .panel{margin-bottom:0}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div>
      <h1>Object Detection Dashboard</h1>
      <p>Real-time monitoring · NX AI Manager</p>
    </div>
    <span class="badge live" id="status">● Live</span>
  </div>

  <div class="stats">
    <div class="card">
      <div class="card-label">Frames Processed</div>
      <div class="card-value" id="v-frames">0</div>
    </div>
    <div class="card">
      <div class="card-label">Total Objects Detected</div>
      <div class="card-value" id="v-objects">0</div>
    </div>
    <div class="card">
      <div class="card-label">Current Frame</div>
      <div class="card-value" id="v-current">0</div>
    </div>
    <div class="card">
      <div class="card-label">Uptime</div>
      <div class="card-value" id="v-uptime">0s</div>
    </div>
  </div>

  <div class="controls">
    <button class="btn primary" onclick="resetZoom()">Reset Zoom</button>
    <button class="btn" id="btn-pause" onclick="togglePause()">Pause</button>
    <button class="btn" onclick="clearData()">Clear Data</button>
  </div>

  <div class="panel">
    <div class="panel-title">Objects Detected Over Time</div>
    <div class="chart-wrap" style="height:300px">
      <canvas id="timelineChart"></canvas>
    </div>
  </div>

  <div class="panel">
    <div class="panel-title">Cumulative Distribution per Class</div>
    <div class="chart-wrap" id="distWrap" style="height:200px">
      <canvas id="distChart"></canvas>
    </div>
  </div>

  <div class="two-col">
    <div class="panel">
      <div class="panel-title">Bounding Box Sizes (Width × Height)</div>
      <div class="chart-wrap" style="height:300px">
        <canvas id="sizeChart"></canvas>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Bounding Box Positions (Center X, Y)</div>
      <div class="chart-wrap" style="height:300px">
        <canvas id="posChart"></canvas>
      </div>
    </div>
  </div>

</div>
<script>
// ── Color management ──────────────────────────────────────────────────────────
const PALETTE = [
  '#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f',
  '#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac',
  '#54a0ff','#5f27cd','#00d2d3','#ff9f43','#2ecc71'
];
const clsColors = {};

function getColor(cls) {
  if (!clsColors[cls])
    clsColors[cls] = PALETTE[Object.keys(clsColors).length % PALETTE.length];
  return clsColors[cls];
}

function hex2rgba(hex, a) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getHidden(chart) {
  const hidden = new Set();
  chart.data.datasets.forEach((ds, i) => {
    if (!chart.isDatasetVisible(i)) hidden.add(ds.label);
  });
  return hidden;
}

function restoreHidden(chart, hidden) {
  if (!hidden.size) return;
  chart.data.datasets.forEach((ds, i) => {
    if (hidden.has(ds.label)) chart.setDatasetVisibility(i, false);
  });
  chart.update('none');
}

function fmtTime(ts) { return new Date(ts * 1000).toLocaleTimeString(); }

function fmtDur(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h > 0 ? `${h}h ${m}m ${sec}s` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function groupByClass(pts) {
  const g = {};
  for (const p of pts) {
    if (!g[p.c]) g[p.c] = [];
    g[p.c].push({ x: p.x, y: p.y });
  }
  return g;
}

// ── Chart initialisation ──────────────────────────────────────────────────────
let timelineChart, distChart, sizeChart, posChart;

function initCharts() {
  timelineChart = new Chart(document.getElementById('timelineChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 12 } } },
        tooltip: { backgroundColor: 'rgba(0,0,0,.85)', padding: 10 },
        zoom: {
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
          pan:  { enabled: true, mode: 'x' }
        }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 11 } },
             title: { display: true, text: 'Time' } },
        y: { beginAtZero: true, title: { display: true, text: 'Count / second' } }
      }
    }
  });

  distChart = new Chart(document.getElementById('distChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3 }] },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(0,0,0,.85)',
          callbacks: { label: ctx => '  ' + ctx.parsed.x.toLocaleString() + ' total' }
        }
      },
      scales: {
        x: { type: 'logarithmic', title: { display: true, text: 'Cumulative count (log scale)' } },
        y: { ticks: { font: { size: 11 } } }
      }
    }
  });

  function scatterOpts(xLabel, yLabel, reverseY) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } },
                  onClick: Chart.defaults.plugins.legend.onClick },
        tooltip: { backgroundColor: 'rgba(0,0,0,.85)' },
        zoom: {
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' },
          pan:  { enabled: true, mode: 'xy' }
        }
      },
      scales: {
        x: { title: { display: true, text: xLabel } },
        y: { title: { display: true, text: yLabel }, reverse: !!reverseY }
      }
    };
  }

  sizeChart = new Chart(document.getElementById('sizeChart'), {
    type: 'scatter', data: { datasets: [] },
    options: scatterOpts('Width (normalized)', 'Height (normalized)', false)
  });

  posChart = new Chart(document.getElementById('posChart'), {
    type: 'scatter', data: { datasets: [] },
    options: scatterOpts('Center X (normalized)', 'Center Y (normalized)', true)
  });
}

// ── Update functions ──────────────────────────────────────────────────────────
function updateStats(s) {
  document.getElementById('v-frames').textContent  = s.frame_count.toLocaleString();
  document.getElementById('v-objects').textContent = s.total_objects.toLocaleString();
  document.getElementById('v-current').textContent = s.current_total.toLocaleString();
  document.getElementById('v-uptime').textContent  = fmtDur(s.uptime);
}

function updateTimeline(data) {
  if (!data.length) return;
  const classes = new Set();
  data.forEach(pt => Object.keys(pt.pc).forEach(c => classes.add(c)));

  const hidden = getHidden(timelineChart);
  timelineChart.data.labels = data.map(pt => fmtTime(pt.ts));
  timelineChart.data.datasets = Array.from(classes).map(cls => ({
    label: cls,
    data:  data.map(pt => pt.pc[cls] || 0),
    borderColor:     getColor(cls),
    backgroundColor: hex2rgba(getColor(cls), 0.08),
    fill: false,
    tension: 0.3,
    pointRadius: 0,
    pointHoverRadius: 4,
    borderWidth: 1.5,
  }));
  timelineChart.update('none');
  restoreHidden(timelineChart, hidden);
}

function updateDistribution(data) {
  const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
  if (!sorted.length) return;

  const h = Math.min(500, Math.max(120, sorted.length * 24));
  document.getElementById('distWrap').style.height = h + 'px';

  distChart.data.labels = sorted.map(([cls]) => cls);
  distChart.data.datasets[0].data            = sorted.map(([, cnt]) => cnt);
  distChart.data.datasets[0].backgroundColor = sorted.map(([cls]) => getColor(cls));
  distChart.update('none');
}

function updateScatter(data) {
  const bySz = groupByClass(data.sizes);
  const byPs = groupByClass(data.positions);

  const szHidden = getHidden(sizeChart);
  sizeChart.data.datasets = Object.entries(bySz).map(([cls, pts]) => ({
    label: cls,
    data:  pts,
    backgroundColor: hex2rgba(getColor(cls), 0.55),
    pointRadius: 3,
    pointHoverRadius: 5,
  }));
  sizeChart.update('none');
  restoreHidden(sizeChart, szHidden);

  const psHidden = getHidden(posChart);
  posChart.data.datasets = Object.entries(byPs).map(([cls, pts]) => ({
    label: cls,
    data:  pts,
    backgroundColor: hex2rgba(getColor(cls), 0.55),
    pointRadius: 3,
    pointHoverRadius: 5,
  }));
  posChart.update('none');
  restoreHidden(posChart, psHidden);
}

// ── Fetch loop ────────────────────────────────────────────────────────────────
let isPaused = false;

async function fetchFast() {
  if (isPaused) return;
  try {
    const [sRes, tRes] = await Promise.all([fetch('/api/stats'), fetch('/api/timeline')]);
    const [stats, timeline] = await Promise.all([sRes.json(), tRes.json()]);
    updateStats(stats);
    updateTimeline(timeline);
    setStatus('live');
  } catch { setStatus('offline'); }
}

async function fetchSlow() {
  if (isPaused) return;
  try {
    const [scRes, dRes] = await Promise.all([fetch('/api/scatter'), fetch('/api/distribution')]);
    const [scatter, dist] = await Promise.all([scRes.json(), dRes.json()]);
    updateScatter(scatter);
    updateDistribution(dist);
  } catch { /* silently skip */ }
}

function setStatus(s) {
  const el = document.getElementById('status');
  el.className = 'badge ' + s;
  el.textContent = s === 'live' ? '● Live' : s === 'paused' ? '⏸ Paused' : '● Offline';
}

function resetZoom() {
  [timelineChart, sizeChart, posChart].forEach(c => c.resetZoom());
}

function togglePause() {
  isPaused = !isPaused;
  document.getElementById('btn-pause').textContent = isPaused ? 'Resume' : 'Pause';
  setStatus(isPaused ? 'paused' : 'live');
}

function clearData() {
  if (!confirm('Clear all data?')) return;
  fetch('/api/clear', { method: 'POST' })
    .then(res => {
      if (!res.ok) { alert('Clear failed'); return; }
      timelineChart.data.labels = [];
      timelineChart.data.datasets = [];
      timelineChart.update();

      distChart.data.labels = [];
      distChart.data.datasets = [{ data: [], backgroundColor: [], borderRadius: 3 }];
      distChart.update();
      document.getElementById('distWrap').style.height = '200px';

      sizeChart.data.datasets = [];
      sizeChart.update();
      posChart.data.datasets = [];
      posChart.update();

      Object.keys(clsColors).forEach(k => delete clsColors[k]);
      updateStats({ frame_count: 0, total_objects: 0, current_total: 0, uptime: 0 });
    })
    .catch(() => alert('Clear failed'));
}

// ── Boot ──────────────────────────────────────────────────────────────────────
initCharts();
fetchFast();
fetchSlow();
setInterval(fetchFast, 1000);
setInterval(fetchSlow, 5000);
</script>
</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger(__name__)
    set_log_level(args.log_level)
    port, tc, sc, db_path = config(args)

    store = DetectionStore(db_path, tc, sc)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)

    flush_thread = start_flush_thread(DEFAULT_FLUSH_SECS)

    try:
        start_web_server(port)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Web Dashboard App running — http://localhost:%d  db=%s", port, db_path)
    wait_for_shutdown(flush_thread)
    logger.info("Web Dashboard App stopped.")
