# Event Clustering

Rust tools for parsing Prophesee EVT2 `.raw` event-camera captures and tracking the largest plausible moving cluster, intended for estimating the position of a thrown ball.

## Features

- Streams event data through a modular parser interface.
- Implements EVT2.0, EVT2.1, and EVT3 decoding.
- Reconstructs event timestamps from `EVT_TIME_HIGH` words.
- Tracks clusters over a rolling time window.
- Supports modular event filters before clustering.
- Can remove isolated noise with a background activity filter.
- Uses polarity selection as a filter rather than coupling it to clustering.
- Can suppress static flicker sources such as background LEDs.
- Reports centroid, bounding box, event count, confidence, and optional RANSAC circle fit.
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
- Filtered events: hidden by default, or dim gray/dim blue when `show filtered events` is enabled
- Bounding box: red
- Centroid: yellow cross
- 3D ball view: OpenGL XYZ position and recent trajectory when Ball Projection is enabled; drag to orbit and scroll to zoom

The left panel contains grouped controls. Hover any control for a short explanation.

### Playback

- `Play/Pause`: pause or resume file playback.
- `Restart`: rewind the RAW file and clear tracker state.
- `timeline us`: scrub through the recording by event timestamp. Moving the slider pauses playback and seeks to that point.
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
- `circle fit`: fit a circle to events in the selected cluster.
- `circle tol px`: radial inlier tolerance for RANSAC circle fitting.

### Polarity

- `use events`: cluster using all events, positive-only events, or negative-only events.
- `invert labels`: swaps positive/negative interpretation for clustering labels and render colors. This is enabled by default because the sample capture appeared swapped.

Polarity is implemented as a modular event filter. Events that do not match the selected polarity are dropped before clustering.

### Background Activity Filter

- `enabled`: suppress isolated events that have no recent neighbor nearby.
- `radius px`: spatial neighbor radius.
- `time window us`: how recently a neighboring event must have occurred.

This is intended for random sensor noise. The first event in a local burst is suppressed, then nearby follow-up events pass through.

### Static Filter

- `enabled`: suppress event cells that keep firing in the same location over time.
- `cell size`: spatial bin size used by the static filter.
- `static after us`: how long a cell must remain active before it can be suppressed.
- `min events`: how many events are required before a cell can be considered static.

This is intended for background flicker sources, such as stationary LEDs. Moving objects should pass through cells before they become static.

### Performance

- `show filtered events`: display rejected events dimmed instead of hiding them.
- `events/tick`: worker-thread event budget per UI update. Increase for faster catch-up/playback; decrease if CPU usage is too high.

`speed` and `events/tick` are different: `speed` is the target playback rate, while `events/tick` is how much work the worker is allowed to do to keep up.

## CLI Examples

View positive events only for clustering:

```bash
cargo run -- spinner.raw view --polarity positive
```

Enable static-event filtering for flickering LEDs:

```bash
cargo run -- spinner.raw view --filter-static
```

Tune the static filter:

```bash
cargo run -- spinner.raw view \
  --filter-static \
  --static-cell-size 4 \
  --static-after-us 250000 \
  --static-min-events 200
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
--no-circle-fit           disable RANSAC circle fitting, enabled by default
--circle-inlier-tolerance-px <px>
                          circle-fit radial inlier tolerance, default 2.5
--polarity <mode>         all, positive/on/+, or negative/off/-; default all
--invert-polarity         invert positive/negative labels
--raw-polarity            use raw EVT2 ON/OFF labels
--filter-background-activity
                          suppress isolated events without recent spatial neighbors
--background-radius-px <px>
                          background activity neighbor radius, default 2
--background-time-window-us <us>
                          background activity neighbor time window, default 5000
--filter-static           suppress cells that stay active in one place
--static-cell-size <px>   static filter cell size, default 4
--static-after-us <us>    mark cells static after this duration, default 250000
--static-min-events <n>   minimum events before a cell is static, default 200
--static-inactive-us <us> forget inactive static cells, default 1000000
--format <format>         auto, evt2, evt21, or evt3; default auto
--width <px>              viewer width, default 640
--height <px>             viewer height, default 480
--speed <factor>          playback speed, default 1.0; use 0 for fastest
--events-per-tick <n>     worker event budget per UI update, default 5000
--max-detections <count>  stop text output after this many detections
--endian <mode>           little, little32, or big; default little
```

## Parser Notes

The event input layer is built around a modular parser interface:

```rust
trait EventStream {
    fn next_record(&mut self) -> io::Result<Option<EventRecord>>;
}
```

The CLI accepts:

```bash
--format auto
--format evt2
--format evt21
--format evt3
```

`auto` reads the RAW header and selects the parser based on `% evt ...`.

Use `--endian little32` for EVT2.1 streams that transmit each 32-bit half of the 64-bit word in little-endian order, as documented for IMX636.

Current implementation status:

- `evt2`: implemented.
- `evt21`: implemented.
- `evt3`: implemented.

This keeps the downstream tracker and filters independent of the underlying event encoding.

## Event Parser Notes

The parser handles Prophesee RAW files with a text header followed by EVT words. EVT2.0 uses 32-bit words, EVT2.1 uses 64-bit vector words, and EVT3 uses 16-bit stateful words. It currently decodes:

- `CD_OFF`
- `CD_ON`
- `EVT_TIME_HIGH`
- `EXT_TRIGGER`
- EVT2.1 vector CD events
- EVT3 address and vector CD events

Vendor-specific or reserved words are surfaced as `Other` values and ignored by the tracker.

## Project Layout

```text
src/main.rs                    CLI and text-mode tracking
src/viewer.rs                  Interactive viewer and worker thread
src/render.rs                  Software RGB rendering helpers
src/event.rs                   Event and bounding-box types
src/evt2.rs                    EVT2 RAW parser
src/parser.rs                  Modular parser interface and format selection
src/parser/evt21.rs            EVT2.1 vector parser
src/parser/evt3.rs             EVT3 stateful parser
src/filters.rs                 Modular event filters
src/pipeline.rs                Shared parser/filter/algorithm event pipeline
src/algorithms/rolling_cluster.rs  Rolling-window clustering tracker
```

## Testing

Run:

```bash
cargo test
```
