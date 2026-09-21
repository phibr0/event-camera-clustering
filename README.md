# achtung: vibe coding deluxe

# event-camera-clustering

## IMU motion comparison

Open a synchronized four-panel comparison of a RAW recording and its iPhone IMU CSV:

```bash
cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv --start-s 4
```

The top row shows all events with camera rotation compensation **off / on**.
The bottom row applies the **same foreground filter** to those two streams.
Both columns use the same timestamps, lens correction, and brightness scale.
Candidate moving regions are displayed; this mode does not select only a ball.

The viewer starts paused. Play, step, scrub, change playback speed, and adjust the
IMU time offset, integration window, motion threshold, minimum region size, and
mounting rotation. Uncovered IMU intervals are explicitly marked unavailable.
Backward seeks reread the RAW file. Use a release build for interactive playback.

By default, startup estimates the time offset and the fixed phone-to-camera rotation
by maximizing event alignment over short windows distributed through the recording.
It searches +/-4 seconds, tries the 24 proper axis mappings, then refines rotation
and timing. A weak fit fails with an explanation instead of silently claiming success.
This assumes the static background dominates and there is sufficiently varied rotation.
The focus score measures event-image sharpness, **not segmentation accuracy**.

Save and reuse the alignment to avoid repeating the search:

```bash
cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv --save-alignment /tmp/recording-3.alignment

cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv --alignment /tmp/recording-3.alignment --start-s 4
```

Export a labeled, synchronized MP4 without opening a window (requires `ffmpeg`):

```bash
cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv --alignment /tmp/recording-3.alignment \
  --start-s 4 --end-s 10 --export-mp4 /tmp/motion-comparison.mp4
```

Outputs are never overwritten. The video contains all four panels and camera time.
The alignment file contains the offset in seconds on its first line and a row-major
3x3 IMU-to-camera rotation on the second. The convention is
`IMU time since first sample = camera time since first event + offset`.
Each recording can need its own offset. `--motion-offset` and `--imu-to-camera`
provide manual values; `--auto-align` forces recalibration, and `--alignment-range`
changes the search radius. The CSV's monotonic `timestamp` and `gx,gy,gz` in rad/s
are used; packet arrival times and absolute phone orientation are not used.

Default segmentation uses a 30 ms window, 3 px cells, a normalized mean-timestamp
threshold of 0.12, at least three events per cell, and eight connected cells.
Tune with `--motion-window-ms`, `--motion-cell-px`, `--motion-threshold`, and
`--motion-min-cells`. The camera calibration must match the recording resolution.

`src/motion.rs` handles IMU interpolation, gyro integration, undistortion, rotation
warping, automatic alignment, and foreground masks. Mask indices refer to the input
events, so consumers can retain their original camera coordinates.
`src/comparison.rs` supplies recording playback, the comparison UI, and video export.

This compensates **rotation**, not depth-dependent translation/parallax. Static
edges may remain when the camera translates, and slowly moving objects can be missed.
Lighting changes are also not distinguished from motion. A 100 Hz IMU cannot resolve
arbitrarily fast vibration. No extrapolation is used outside IMU coverage or across
sample gaps longer than 50 ms. Validate foreground retention visually; fewer events
alone does not mean better detection.

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

<<<<<<< HEAD
Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!
||||||| parent of 24bf818 (feat: add static filtering)

- Streams EVT2 data from Prophesee RAW files.
- Reconstructs event timestamps from `EVT_TIME_HIGH` words.
- Tracks clusters over a rolling time window.
- Reports centroid, bounding box, event count, and confidence.
- Provides an interactive viewer with sliders, toggles, and tooltips.
- # Runs parsing, tracking, and frame rendering on a worker thread so the UI stays responsive.
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
  > > > > > > > 24bf818 (feat: add static filtering)

## Add your files

- [Create](https://docs.gitlab.com/user/project/repository/web_editor/#create-a-file) or [upload](https://docs.gitlab.com/user/project/repository/web_editor/#upload-a-file) files
- [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/event-camera-clustering.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [Set up project integrations](https://gitlab.cc-asp.fraunhofer.de/iml/oe130/projektgruppe-h-react/event-camera-clustering/-/settings/integrations)

## Collaborate with your team

- [Invite team members and collaborators](https://docs.gitlab.com/user/project/members/)
- [Create a new merge request](https://docs.gitlab.com/user/project/merge_requests/creating_merge_requests/)
- [Automatically close issues from merge requests](https://docs.gitlab.com/user/project/issues/managing_issues/#closing-issues-automatically)
- [Enable merge request approvals](https://docs.gitlab.com/user/project/merge_requests/approvals/)
- [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- Positive events: white
- Negative events: blue
- Filtered events: hidden by default, or dim gray/dim blue when `show filtered events` is enabled
- Bounding box: red
- Centroid: yellow cross
- 3D ball view: OpenGL XYZ position, recent trajectory, and optional two-shot parabola fit; drag to orbit and scroll to zoom

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

### Two-Shot Parabola

- `enabled`: fit a gravity-constrained 3D parabola directly from cluster bounding-box area.
- `diameter m`: known real-world ball diameter used for depth from apparent bbox area.
- `fx px`, `fy px`, `cx px`, `cy px`: camera intrinsics used for raw detection-to-3D reconstruction.
- `gravity axis`: direction of gravity in camera-space coordinates. Default is `y positive`, matching the referenced UZH RPG project.
- `inlier m`: 3D residual threshold used to score two-point parabola hypotheses.
- `buffer pts` and `min pts`: rolling point buffer size and minimum inlier count before reporting a fit.

This path is independent from Ball Projection and does not use the Kalman-smoothed 3D reconstruction. It follows the referenced project's minimal two-point parabola model, then refines the best inliers with least squares.

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

Print detections with independent two-shot parabola fitting:

```bash
cargo run -- spinner.raw track-ball \
  --parabola-fit \
  --parabola-ball-diameter-m 0.067 \
  --parabola-gravity-axis y-positive
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
--parabola-fit            enable independent two-shot parabola fitting
--no-parabola-fit         disable independent two-shot parabola fitting
--parabola-ball-diameter-m <m>
                          real-world ball diameter for bbox-area depth, default 0.067
--parabola-fx-px <px>     parabola reconstruction focal length x
--parabola-fy-px <px>     parabola reconstruction focal length y
--parabola-cx-px <px>     parabola reconstruction principal point x
--parabola-cy-px <px>     parabola reconstruction principal point y
--parabola-gravity-axis <axis>
                          x-positive/x+, x-negative/x-, y-positive/y+, y-negative/y-, z-positive/z+, or z-negative/z-
--parabola-inlier-threshold-m <m>
                          3D residual threshold for RANSAC inliers, default 0.25
--parabola-buffer-len <n> rolling 3D point buffer length, default 50
--parabola-min-points <n> minimum inliers required for a fit, default 4
--parabola-max-depth-m <m>
                          reject reconstructed points beyond this depth, default 30
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
