# Event camera clustering

Rust tools for EVT2, EVT2.1, and EVT3 recordings: event filtering, cluster tracking,
IMU rotation compensation, and foreground detection.

## Build and run

Use a current stable Rust toolchain. Run commands from the repository root.
MP4 export requires `ffmpeg` on `PATH`. Recordings and camera calibration are local
inputs and are not included in Git.

```sh
cargo build --release --locked
cargo run --release -- spinner.raw view
cargo run --release -- spinner.raw track-ball --max-detections 10
cargo run --release -- --help
```

`view` provides playback, a timeline, filter controls, and optional 3D ball
projection. Hover a control for its description. `track-ball` prints detections.
These commands use the cluster tracker; IMU processing uses `motion-compare`.

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

Estimate and save the time offset and IMU-to-camera rotation:

```sh
mkdir -p outputs/motion-comparison
cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv --start-s 4 \
  --save-alignment outputs/motion-comparison/recording-3.alignment
```

The viewer starts paused. The top row shows events with rotation compensation
off and on. The bottom row applies foreground detection to each result. All
panels use the same timestamps, lens correction, and brightness scale.

Reuse the alignment to export a four-panel video:

```sh
cargo run --release -- motion-compensation/aufnahme_tracking_3.raw motion-compare \
  --imu motion-compensation/3.csv \
  --alignment outputs/motion-comparison/recording-3.alignment \
  --start-s 4 --end-s 10 --export-mp4 outputs/motion-comparison/recording-3.mp4
```

Omit `--export-mp4` to open the viewer. Replace it and its path with `--benchmark`
to measure contiguous processing windows. Existing MP4 and alignment files are
not overwritten.

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
GPU display, and video encoding. Live ingestion is not implemented; it requires
timestamped camera/IMU input, alignment buffering, and an end-to-end latency test.

## Development

| Location | Responsibility |
| --- | --- |
| `src/main.rs`, `src/viewer.rs` | CLI, interactive playback, and worker thread. |
| `src/parser.rs`, `src/parser/` | Common event stream and EVT decoders. |
| `src/filter/`, `src/pipeline.rs` | Event filtering and algorithm dispatch. |
| `src/algorithms/rolling_cluster.rs` | Connected clusters and optional circle fit. |
| `src/ball.rs`, `src/parabola.rs` | Ball projection and ballistic fitting. |
| `src/motion.rs` | IMU integration, alignment, rotation warping, and foreground masks. |
| `src/comparison.rs`, `src/render.rs` | Comparison UI, export, benchmark, and rendering. |

`motion::compensate` accepts one event window. Output masks preserve input event
indices. Camera rays are cached; rotations use 0.5 ms bins. Processing uses no
future event windows, but IMU interpolation requires samples bracketing the window.

```sh
cargo test --locked
```

## Local files

Git excludes recordings (`*.raw`, root CSVs/ZIPs, `motion-compensation/`),
`calibration.json`, generated output (`outputs/`, MP4s, alignment profiles), viewer
settings, logs, OS metadata, and `target/`.
Keep exports and local diagnostics under `outputs/`. Keep reusable source outside
that directory. Commit source, `Cargo.toml`, and `Cargo.lock`.
