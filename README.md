# Event Clustering

Rust tools for parsing Prophesee EVT2 `.raw` event-camera captures and tracking the largest plausible moving cluster, intended for estimating the position of a thrown ball.

## Features

- Streams EVT2 data from Prophesee RAW files.
- Reconstructs event timestamps from `EVT_TIME_HIGH` words.
- Tracks clusters over a rolling time window.
- Reports centroid, bounding box, event count, and confidence.
- Provides an interactive viewer with sliders, toggles, and tooltips.
- Runs parsing, tracking, and frame rendering on a worker thread so the UI stays responsive.

## Requirements

- Rust 2024 edition toolchain.
- A Prophesee EVT2 RAW file, for example `spinner.raw`.

## Quick Start

Open the interactive viewer:

```bash
cargo run -- spinner.raw view
```

Print detections in the terminal:

```bash
cargo run -- spinner.raw track-ball
```

Show CLI help:

```bash
cargo run -- --help
```

## Viewer

The viewer shows recent events and overlays the current cluster detection:

- Positive events: white
- Negative events: blue
- Bounding box: red
- Centroid: yellow cross

The left panel contains grouped controls. Hover any control for a short explanation.

### Playback

- `Play/Pause`: pause or resume file playback.
- `Restart`: rewind the RAW file and clear tracker state.
- `speed`: playback speed relative to event timestamps. `1.0` is real time, `0.5` is half speed, `2.0` is double speed, and `0.0` processes as fast as possible.

### Cluster Window

- `window us`: amount of recent event history used for each detection. Smaller is more responsive; larger is more stable but creates longer motion trails.
- `step us`: how often detections are emitted in event time. Smaller gives smoother updates and more CPU work.
- `cell size`: spatial bin size before clustering. Larger values bridge gaps and reduce noise, but reduce spatial precision.

### Cluster Filters

- `min events`: reject clusters with fewer events. Increase to suppress noise; decrease if the ball is faint or far away.
- `min cells`: reject clusters occupying too few spatial cells. Useful for ignoring hot pixels.
- `max bbox w`: reject clusters wider than this.
- `max bbox h`: reject clusters taller than this.

### Polarity

- `use events`: cluster using all events, positive-only events, or negative-only events.
- `invert labels`: swaps positive/negative interpretation for clustering labels and render colors. This is enabled by default because the sample capture appeared swapped.

### Performance

- `events/tick`: worker-thread event budget per UI update. Increase for faster catch-up/playback; decrease if CPU usage is too high.

`speed` and `events/tick` are different: `speed` is the target playback rate, while `events/tick` is how much work the worker is allowed to do to keep up.

## CLI Examples

View positive events only for clustering:

```bash
cargo run -- spinner.raw view --polarity positive
```

Use raw EVT2 ON/OFF polarity labels instead of the default inverted labels:

```bash
cargo run -- spinner.raw view --raw-polarity
```

Run slower for tuning:

```bash
cargo run -- spinner.raw view --speed 0.25
```

Process as fast as possible with a higher worker budget:

```bash
cargo run -- spinner.raw view --speed 0 --events-per-tick 20000
```

Print the first 10 detections:

```bash
cargo run -- spinner.raw track-ball --max-detections 10
```

Tune text-mode tracking:

```bash
cargo run -- spinner.raw track-ball \
  --window-us 20000 \
  --step-us 5000 \
  --cell-size 2 \
  --min-events 20 \
  --min-cells 3 \
  --max-bbox-width 200 \
  --max-bbox-height 200
```

## Options

```text
--window-us <us>          rolling window size, default 20000
--step-us <us>            output interval, default 5000
--cell-size <pixels>      spatial bin size, default 2
--min-events <count>      minimum cluster size, default 20
--min-cells <count>       minimum occupied cells, default 3
--max-bbox-width <px>     reject wider clusters, default 200
--max-bbox-height <px>    reject taller clusters, default 200
--polarity <mode>         all, positive/on/+, or negative/off/-; default all
--invert-polarity         invert positive/negative labels
--raw-polarity            use raw EVT2 ON/OFF labels
--width <px>              viewer width, default 640
--height <px>             viewer height, default 480
--speed <factor>          playback speed, default 1.0; use 0 for fastest
--events-per-tick <n>     worker event budget per UI update, default 5000
--max-detections <count>  stop text output after this many detections
--endian <little|big>     EVT2 word endianness, default little
```

## EVT2 Parser Notes

The parser handles Prophesee EVT2 RAW files with a text header followed by 32-bit EVT2 words. It currently decodes:

- `CD_OFF`
- `CD_ON`
- `EVT_TIME_HIGH`
- `EXT_TRIGGER`

Vendor-specific or reserved words are surfaced as `Other` values and ignored by the tracker.

## Project Layout

```text
src/main.rs                    CLI and interactive viewer
src/event.rs                   Event and bounding-box types
src/evt2.rs                    EVT2 RAW parser
src/algorithms/rolling_cluster.rs  Rolling-window clustering tracker
```

## Testing

Run:

```bash
cargo test
```
