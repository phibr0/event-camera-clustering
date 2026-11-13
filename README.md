# Event camera clustering

Rust tools for EVT2, EVT2.1, and EVT3 recordings: event filtering, cluster tracking,
IMU rotation compensation, foreground detection, and 3D ball tracking.

## Build and run

Use a current stable Rust toolchain. Run commands from the repository root.
MP4 export requires `ffmpeg` on `PATH`. Install [Git LFS](https://git-lfs.com/)
to download the example recordings after cloning:

```sh
git lfs install
git lfs pull
cargo build --release --locked
cargo run --release -- examples/ball_klein_3.raw view
cargo run --release -- examples/ball_klein_3.raw track-ball --max-detections 10
cargo run --release -- --help
```

`examples/` contains a 25 cm ball recording (`ball_gross_1`) and a 7 cm ball
recording (`ball_klein_3`), about 30 MB combined. RAW files use LFS; matching IMU
CSVs, camera calibration, and estimated alignment profiles use regular Git.

`view` provides playback, a timeline, filter controls, and optional 3D ball
projection. Hover a control for its description. `track-ball` prints detections.
These commands use the cluster tracker; IMU processing uses `motion-compare` and
`ball-3d`.

| Setting | Use |
| --- | --- |
| `--window-us`, `--step-us` | Set event history and detection interval; defaults: 20000 and 5000 µs. |
| `--cell-size`, `--min-events`, `--min-cells` | Set clustering resolution and minimum support. |
| `--max-bbox-width`, `--max-bbox-height` | Reject large clusters. |
| `--no-circle-fit` | Disable the optional cluster circle fit. |
| `--polarity positive`, `--polarity negative` | Select events for clustering. Default: both. |
| `--raw-polarity` | Disable the default polarity inversion. |
| `--filter-background-activity` | Reject events without recent spatial neighbors. |
| `--filter-static` | Suppress cells with persistent activity, such as flickering LEDs. |
| `--speed 0.25`, `--speed 0` | Use quarter-speed or unrestricted playback. |
| `--events-per-tick` | Set the worker event budget per UI update. |
| `--parabola-fit --parabola-ball-diameter-m 0.07` | Enable the independent two-point ballistic fit using the specified ball diameter. |

The parser reads the RAW header by default. Use `--format evt2`, `evt21`, or `evt3`
to override it. Use `--endian little32` for EVT2.1 input with little-endian 32-bit
halves. Unknown event words are ignored by the tracker.

## IMU motion comparison

Provide a RAW recording, its IMU CSV, and `calibration.json`. The CSV must contain
`timestamp` in seconds and `gx,gy,gz` in rad/s. The IMU must be rigidly attached to
the camera. Calibration must match the recording resolution and contain
`image_width`, `image_height`, a 3×3 `camera_matrix` in pixels, and
`distortion_coefficients` in `[k1,k2,p1,p2,k3]` order. Use `--calibration` to select
another file.

Run an example with its saved time offset and IMU-to-camera rotation:

```sh
cargo run --release -- examples/ball_klein_3.raw motion-compare \
  --imu examples/ball_klein_3.csv --calibration examples/calibration.json \
  --alignment examples/ball_klein_3.alignment --start-s 4
```

The viewer starts paused. The top row shows events with rotation compensation
off and on. The bottom row applies foreground detection to each result. All
panels use the same timestamps, lens correction, and brightness scale.

Reuse the alignment to export a four-panel video:

```sh
mkdir -p outputs/motion-comparison
cargo run --release -- examples/ball_klein_3.raw motion-compare \
  --imu examples/ball_klein_3.csv --calibration examples/calibration.json \
  --alignment examples/ball_klein_3.alignment \
  --start-s 4 --end-s 6 --export-mp4 outputs/motion-comparison/ball_klein_3.mp4
```

Omit `--export-mp4` to open the viewer. Replace it and its path with `--benchmark`
to measure contiguous processing windows. Existing MP4 and alignment files are
not overwritten.

For a new recording, omit `--alignment` and add
`--save-alignment outputs/motion-comparison/<name>.alignment` to estimate and save
its alignment. Use calibration for that camera and resolution.

Automatic alignment searches ±4 s and estimates a fixed mounting rotation. It
requires background structure and varied camera rotation. Weak fits fail. Use
`--alignment-range` to change the search range, `--auto-align` to recalibrate, or
`--motion-offset` and `--imu-to-camera` to supply manual values. The mounting
matrix has nine comma-separated values in row-major order.

An alignment file has two lines: offset in seconds, then the mounting matrix.
The time convention is `relative IMU time = relative camera time + offset`, with
each stream measured from its first sample or event. Each recording needs its
own time alignment. The CSV's arrival times and absolute orientation are unused.

The gyroscope measures angular velocity. The code integrates it into relative
rotations, converts these to camera axes, and projects each undistorted event
into the camera orientation at the end of the window. Foreground detection then
selects regions with recent activity that extends beyond earlier background
activity. It detects moving regions; it does not identify balls.

Defaults are a 30 ms window, 3 px cells, at least 3 events per cell, a normalized
mean-time residual above 0.12, and at least 8 connected cells. At least 10% of each
region must lie outside the neighborhood of activity in the first quarter of the
window. Tune `--motion-window-ms`, `--motion-cell-px`, `--motion-threshold`,
`--motion-min-cells`, and `--motion-min-new-fraction`. Set the last option to `0`
to disable the early-activity check.

Compensation covers rotation. Translation, parallax, lighting changes, and slow
targets can cause errors. IMU gaps above 50 ms or missing coverage disable the
compensated foreground. The benchmark includes decoding, compensation, both
foreground paths, and CPU panel rendering. It excludes setup, sensor transport,
GPU display, and video encoding. Live IMU input is not implemented; it requires
a receiver for the IMU stream and a time alignment that is estimated while running.

## 3D ball position

`ball-3d` tracks a ball of known diameter and reports its position relative to
the camera. It takes the same recording, IMU CSV, calibration, and alignment as
`motion-compare`, and the real ball diameter in meters:

```sh
cargo run --release -- examples/ball_klein_3.raw ball-3d \
  --imu examples/ball_klein_3.csv --calibration examples/calibration.json \
  --alignment examples/ball_klein_3.alignment --ball-diameter-m 0.07 --start-s 4
```

Use `--ball-diameter-m 0.25` with `ball_gross_1`. The viewer starts paused. The
left panel shows the rotation-compensated events, with foreground events bright.
A green circle marks a ball whose outline was measured in this update. A green
box marks a match without an outline, and orange marks a prediction. The pink
line is the filtered flight path for the next 0.3 s. The `candidates` option
shows every cluster proposal. The right panel shows the ball path in meters, in
the camera orientation at the start of playback: dots are single measurements,
the curve is the filtered flight, and its pink end is the prediction. A seek or
a changed setting restarts the tracker 0.3 s earlier.

The tracker runs every 10 ms on the preceding `--motion-window-ms` of events and
uses no later data:

1. Rotation compensation and foreground detection, as in `motion-compare`.
2. Up to six connected clusters of the foreground events of the last 10 ms.
3. A circle search around each cluster in all events of the last 4 ms. The
   foreground often keeps only part of a ball, so background events may support
   the circle. A circle needs events in 9 of its 16 sectors.
4. Association with a constant-velocity model in a rotation-stabilized image. A
   track starts after six observations, three of them with a circle, that move
   at least 120 px/s. It is predicted for up to 80 ms without a match.
5. Depth `Z = fx × ball diameter / circle diameter`, then `X` and `Y` from the
   pixel position. This single measurement uses the median of the last five
   circles and keeps the last diameter in an update without a circle.
6. A Kalman filter of position and velocity whose prediction step adds gravity.
   It is corrected with the circle center and the depth of each circle. A
   circle whose size contradicts the flight so far counts less.

Positions use the camera frame at the end of each window: X right, Y down, Z
forward. The reported position and velocity are the filtered ones.

The filter needs the direction of gravity. It reads the optional
`gravity_x,gravity_y,gravity_z` columns of the IMU CSV, a vector toward the
ground in IMU axes, such as Core Motion's `gravity`. Without these columns the
positions are the single measurements.

Replace the viewer with a batch run using any of these options:

| Option | Result |
| --- | --- |
| `--benchmark` | Print track and timing summaries only. |
| `--tracks-csv <path>` | Write one row per update: pixel position, single measurement, filtered position and velocity. |
| `--export-mp4 <path>` | Write a video of the left panel with the position as text. |

Batch runs always print the summaries. Existing output files are not overwritten.
The summary includes a check of the filter: the flight path predicted 100 ms
ahead, compared in pixels with the circle measured at that time.

### Live camera

`ball-3d` reads a connected camera instead of a recording when the file name is
`camera`. Build with the `camera` feature, which adds the
[`neuromorphic-drivers`](https://github.com/neuromorphicsystems/neuromorphic-drivers)
crate and compiles libusb:

```sh
cargo run --release --features camera -- camera ball-3d \
  --calibration examples/calibration.json --ball-diameter-m 0.07 \
  --mask-pixels 87:24,121:306,291:328,237:359,711:391,755:433
```

The driver supports IMX636 cameras such as the IDS uEye XCP-E and the Prophesee
EVK4; it was tested with an IDS UE-39B0XCP-E on macOS. The viewer starts running
and shows the event rate and how long events wait between USB and the tracker.
`--mask-pixels` switches defective pixels off inside the sensor. A pixel that
fires without pause shows as a dot that never fades and can dominate the event
rate; the list above belongs to one camera. With `--benchmark`, `--tracks-csv`
or `--export-mp4` the run lasts `--end-s` seconds, 10 by default.

There is no live IMU input yet. The camera must stand still, and the flight
filter takes the ground to be down in the image. When processing falls more
than 150 ms behind, the tracker skips to the newest events. Batch runs keep one
CPU core busy while waiting, because a thread that sleeps between updates runs
several times slower on an Apple Silicon Mac. The viewer does not, and its
updates take a few milliseconds instead of a fraction of one.

Camera translation is not measured, so the path is relative to the camera, and
the filter treats an accelerating camera as noise. The depth depends on the
fitted circle and is not validated against measured distances. The filter makes
the path smooth; it did not bring the depth closer to hand-labelled circle
sizes. The depth rate is weakly observed, so the first 0.15 s of a track and
any prediction far ahead are uncertain. The circle search samples candidate circles with a fixed seed: repeated
runs give identical results, but small input changes can change a track. Fast
blur, a ball clipped by the image border, and round background objects cause
errors. Tracking thresholds assume the 10 ms step.

## Development

| Location | Responsibility |
| --- | --- |
| `src/main.rs`, `src/viewer.rs` | CLI, interactive playback, and worker thread. |
| `src/parser.rs`, `src/parser/` | Common event stream and EVT decoders. |
| `src/filter/`, `src/pipeline.rs` | Event filtering and algorithm dispatch. |
| `src/algorithms/rolling_cluster.rs` | Connected clusters and optional circle fit. |
| `src/ball.rs`, `src/parabola.rs` | Ball projection and ballistic fitting. |
| `src/ball_tracker.rs` | Cluster proposals, circle search, association, and 3D position. |
| `src/ball_view.rs` | `ball-3d` viewer, track export, and benchmark. |
| `src/camera.rs` | Live events from a USB camera, behind the `camera` feature. |
| `src/motion.rs` | IMU integration, alignment, rotation warping, and foreground masks. |
| `src/comparison.rs`, `src/render.rs` | Comparison UI, export, benchmark, and rendering. |

`motion::compensate` accepts one event window. Output masks preserve input event
indices. `ball_tracker::BallTracker::update` accepts one event window and keeps
the track between calls. Camera rays are cached; rotations use 0.5 ms bins. Processing uses no
future event windows, but IMU interpolation requires samples bracketing the window.

```sh
cargo test --locked
```

## Local files

Git excludes recordings (`*.raw`, root CSVs/ZIPs, `motion-compensation/`),
`calibration.json`, generated output (`outputs/`, MP4s, alignment profiles), viewer
settings, logs, OS metadata, and `target/`.
The recordings, calibration, and alignment profiles in `examples/` are exceptions.
Keep exports and local diagnostics under `outputs/`. Keep reusable source outside
that directory. Commit source, `.gitattributes`, `Cargo.toml`, and `Cargo.lock`.
