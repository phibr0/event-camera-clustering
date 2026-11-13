//! `ball-3d`: tracks a ball of known diameter in a recording and reports its
//! position relative to the camera, in a viewer or as a batch run.
use crate::camera::{self, LiveCamera};
use crate::comparison::{self, Engine, LiveSetup, MotionArgs, Result, Session};
use clap::Args;
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use event_clustering::Event;
use event_clustering::ball_tracker::{BallTracker, BallUpdate, Flight, TrackState};
use event_clustering::motion::{self, ImuTrack, MotionConfig};
use event_clustering::parser::{Endian, EventFormat};
use std::{
    collections::VecDeque,
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::{Duration, Instant},
};

/// The tracker runs on this grid of recording time; its thresholds assume it.
const STEP_US: u64 = 10_000;
const TRAIL_S: f64 = 0.35;
const PATH_POINTS: usize = 6000;
/// Replayed before a seek target, so that a ball in flight is already tracked there.
const PREROLL_US: u64 = 300_000;
/// Playback slows down instead of freezing the window when it cannot keep up.
const MAX_STEPS_PER_FRAME: usize = 20;
const TIMING_SAMPLES: usize = 500;
/// Live input skips ahead when this much sensor time waits to be processed.
const MAX_LIVE_LAG_US: u64 = 150_000;
/// Batch runs on live input last this long unless `--end-s` says otherwise.
const LIVE_BATCH_S: f64 = 10.;
/// The path that selects the first connected camera instead of a recording.
const CAMERA_PATH: &str = "camera";
const FLIGHTS: usize = 32;
/// The flight path is drawn this far ahead of the ball.
const AHEAD_S: f64 = 0.3;
/// Batch runs compare the path predicted this far ahead with the later outline.
const CHECK_US: u64 = 100_000;

const MEASURED: [u8; 3] = [40, 240, 100];
const PREDICTED: [u8; 3] = [255, 175, 45];
const TRAIL: [u8; 3] = [40, 170, 220];
const CANDIDATE: [u8; 3] = [150, 120, 40];
const AHEAD: [u8; 3] = [255, 110, 230];

#[derive(Debug, Args)]
pub(crate) struct BallArgs {
    /// Real diameter of the tracked ball in meters; required by ball-3d.
    #[arg(long)]
    ball_diameter_m: Option<f64>,
    /// Write one row per 10 ms update instead of opening the viewer.
    #[arg(long)]
    tracks_csv: Option<PathBuf>,
    /// Live input: defective pixels to switch off in the sensor, as x:y,x:y,...
    #[arg(long, value_name = "X:Y,...")]
    mask_pixels: Option<String>,
}

pub(crate) fn run(
    path: PathBuf,
    motion: MotionArgs,
    ball: BallArgs,
    format: EventFormat,
    endian: Endian,
) -> Result<()> {
    let diameter = ball.ball_diameter_m.ok_or(
        "ball-3d requires --ball-diameter-m, the real ball diameter in meters \
         (0.07 for examples/ball_klein_3, 0.25 for examples/ball_gross_1)",
    )?;
    if !diameter.is_finite() || !(0.005..=2.).contains(&diameter) {
        return Err("--ball-diameter-m must be between 0.005 and 2 meters".into());
    }
    if ball.tracks_csv.as_ref().is_some_and(|p| p.exists()) {
        return Err("track output already exists; choose a new path".into());
    }
    let batch_run = motion.benchmark || motion.export_mp4.is_some() || ball.tracks_csv.is_some();
    let player = if path.as_os_str() == CAMERA_PATH {
        let masked = match &ball.mask_pixels {
            Some(text) => camera::parse_pixels(text)?,
            None => Vec::new(),
        };
        let setup = comparison::live_setup(&motion, camera::WIDTH, camera::HEIGHT)?;
        let stream = LiveCamera::open(&masked)?;
        eprintln!(
            "Camera {} opened, {} pixel(s) masked. No IMU input yet: the camera must stand still and level.",
            stream.serial,
            masked.len()
        );
        Player::from_camera(stream, setup, diameter, batch_run)
    } else {
        if ball.mask_pixels.is_some() {
            return Err("--mask-pixels applies to live input only".into());
        }
        Player::new(
            comparison::prepare(&path, &motion, format, endian)?,
            diameter,
        )?
    };
    if batch_run {
        return batch(player, motion, ball);
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default().with_inner_size([1500., 760.]),
        ..Default::default()
    };
    let title = match player.live() {
        Some(live) => format!("Ball position — live camera {}", live.stream.serial),
        None => format!(
            "Ball position — {}",
            path.file_name().unwrap().to_string_lossy()
        ),
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |_| Ok(Box::new(BallApp::new(player)))),
    )?;
    Ok(())
}

/// One unfiltered position, in the stabilized orientation.
struct PathPoint {
    track_id: u32,
    position: [f32; 3],
    supported: bool,
}

/// The newest flight estimate of a track and the time it covers.
struct FlightPath {
    track_id: u32,
    start_s: f64,
    time_s: f64,
    flight: Flight,
}

impl FlightPath {
    /// The path from the start of the track to `ahead_s` after its newest update.
    fn points(&self, ahead_s: f64) -> Vec<[f64; 3]> {
        let span = self.time_s - self.start_s + ahead_s;
        let steps = (span / 0.02).ceil().max(1.) as usize;
        (0..=steps)
            .map(|i| {
                self.flight
                    .stabilized_after(self.start_s - self.time_s + span * i as f64 / steps as f64)
            })
            .collect()
    }
}

/// Events arriving from a camera that stands still.
struct Live {
    stream: LiveCamera,
    camera: motion::Camera,
    /// Zero rotation and gravity along the image's vertical, until there is a live IMU.
    imu: ImuTrack,
    imu_until_s: f64,
    config: MotionConfig,
    window_us: u64,
    /// Events of the current window and newer, in sensor time.
    events: VecDeque<Event>,
    /// How long a batch run lasts.
    duration_us: Option<u64>,
    /// Events per second over the last second, and what it is counted from.
    rate: f64,
    counted: (Instant, u64),
    /// Time the newest events waited between USB arrival and processing.
    wait_ms: f64,
    /// Updates dropped to stay live.
    skipped: u64,
}

impl Live {
    /// Takes over the events that have arrived.
    fn poll(&mut self) -> Result<()> {
        while let Some(batch) = self.stream.try_next()? {
            self.wait_ms = batch.received.elapsed().as_secs_f64() * 1000.;
            self.counted.1 += batch.events.len() as u64;
            self.events.extend(batch.events);
        }
        let elapsed = self.counted.0.elapsed().as_secs_f64();
        if elapsed >= 1. {
            self.rate = self.counted.1 as f64 / elapsed;
            self.counted = (Instant::now(), 0);
        }
        Ok(())
    }
}

enum Source {
    File(Engine),
    Live(Box<Live>),
}

enum Step {
    /// `Player::update` is new.
    Updated,
    /// Live input has not delivered the next window yet.
    Waiting,
    /// The end of the recording or of a timed live run.
    Finished,
}

/// Steps the tracker through a recording or a live stream and keeps what the views draw.
struct Player {
    source: Source,
    tracker: BallTracker,
    /// Times of the first, last and next update: recording time, or sensor time
    /// for live input, where they are zero until the first events arrive.
    first_us: u64,
    last_us: u64,
    next_us: u64,
    update: Option<BallUpdate>,
    /// Recent measured centers of the current track in its reference orientation.
    trail: VecDeque<(f64, [f64; 2])>,
    trail_track: u32,
    path: VecDeque<PathPoint>,
    flights: VecDeque<FlightPath>,
    path_frame: u32,
    compute_ms: VecDeque<f64>,
}

impl Player {
    fn new(session: Session, ball_diameter_m: f64) -> Result<Self> {
        let Session {
            engine,
            start_s,
            end_s,
        } = session;
        // One step after the first complete window, like the experiment this replaces.
        let first_us = ((start_s * 1e6).round() as u64 + STEP_US).div_ceil(STEP_US) * STEP_US;
        let last_us = ((end_s * 1e6).round() as u64).saturating_sub(STEP_US) / STEP_US * STEP_US;
        if last_us < first_us {
            return Err("the interval is shorter than one tracker update".into());
        }
        Ok(Self::with_source(
            Source::File(engine),
            ball_diameter_m,
            first_us,
            last_us,
        ))
    }

    /// `timed` runs stop after the duration of the setup; a viewer runs until closed.
    fn from_camera(
        stream: LiveCamera,
        setup: LiveSetup,
        ball_diameter_m: f64,
        timed: bool,
    ) -> Self {
        let duration_s = setup.duration_s.unwrap_or(LIVE_BATCH_S);
        let live = Live {
            stream,
            camera: setup.camera,
            imu: ImuTrack::live(),
            imu_until_s: 0.,
            config: setup.config,
            window_us: setup.window_us,
            events: VecDeque::new(),
            duration_us: timed.then_some((duration_s * 1e6) as u64),
            rate: 0.,
            counted: (Instant::now(), 0),
            wait_ms: 0.,
            skipped: 0,
        };
        Self::with_source(Source::Live(Box::new(live)), ball_diameter_m, 0, u64::MAX)
    }

    fn with_source(source: Source, ball_diameter_m: f64, first_us: u64, last_us: u64) -> Self {
        Self {
            source,
            tracker: BallTracker::new(ball_diameter_m),
            first_us,
            last_us,
            next_us: first_us,
            update: None,
            trail: VecDeque::new(),
            trail_track: 0,
            path: VecDeque::new(),
            flights: VecDeque::new(),
            path_frame: 0,
            compute_ms: VecDeque::new(),
        }
    }

    fn live(&self) -> Option<&Live> {
        match &self.source {
            Source::Live(live) => Some(live),
            Source::File(_) => None,
        }
    }

    fn camera(&self) -> &motion::Camera {
        match &self.source {
            Source::File(engine) => &engine.camera,
            Source::Live(live) => &live.camera,
        }
    }

    fn config_mut(&mut self) -> &mut MotionConfig {
        match &mut self.source {
            Source::File(engine) => &mut engine.config,
            Source::Live(live) => &mut live.config,
        }
    }

    /// Recording time of the newest update.
    fn time_us(&self) -> u64 {
        self.next_us.saturating_sub(STEP_US).max(self.first_us)
    }

    /// Processes the next update, if it is available.
    fn advance(&mut self) -> Result<Step> {
        let update = match &mut self.source {
            Source::File(engine) => {
                if self.next_us > self.last_us {
                    return Ok(Step::Finished);
                }
                let end = engine.origin_us + self.next_us;
                let (start, events) = engine.window(end)?;
                self.tracker.update(
                    &events,
                    engine.origin_us,
                    start,
                    end,
                    &engine.camera,
                    &engine.imu,
                    engine.config,
                )
            }
            Source::Live(live) => {
                live.poll()?;
                let Some(newest) = live.events.back().map(|event| event.timestamp_us) else {
                    return Ok(Step::Waiting);
                };
                if self.next_us == 0 {
                    let first = live.events[0].timestamp_us;
                    self.first_us = (first + live.window_us).div_ceil(STEP_US) * STEP_US;
                    self.next_us = self.first_us;
                    if let Some(duration) = live.duration_us {
                        self.last_us = self.first_us + duration;
                    }
                }
                if self.next_us > self.last_us {
                    return Ok(Step::Finished);
                }
                if newest < self.next_us {
                    return Ok(Step::Waiting);
                }
                if newest - self.next_us > MAX_LIVE_LAG_US {
                    // Too far behind to be live: continue at the newest complete step.
                    let target = newest / STEP_US * STEP_US;
                    live.skipped += (target - self.next_us) / STEP_US;
                    self.next_us = target;
                    self.tracker.reset();
                }
                let end = self.next_us;
                let start = end.saturating_sub(live.window_us);
                while live.events.front().is_some_and(|e| e.timestamp_us < start) {
                    live.events.pop_front();
                }
                let events: Vec<Event> = live
                    .events
                    .iter()
                    .take_while(|event| event.timestamp_us <= end)
                    .copied()
                    .collect();
                // A still camera: no rotation, and the ground is down in the image.
                while live.imu_until_s < end as f64 * 1e-6 + 0.05 {
                    live.imu.push(live.imu_until_s, [0.; 3], Some([0., 1., 0.]));
                    live.imu_until_s += 0.02;
                }
                self.tracker
                    .update(&events, 0, start, end, &live.camera, &live.imu, live.config)
            }
        };
        self.next_us += STEP_US;
        if update.track_id != self.trail_track || update.state == TrackState::Lost {
            self.trail.clear();
        }
        self.trail_track = update.track_id;
        if update.stabilized_frame != self.path_frame {
            self.path.clear();
            self.flights.clear();
            self.path_frame = update.stabilized_frame;
        }
        if let Some(ball) = update.ball {
            if update.state == TrackState::Measured {
                self.trail.push_back((update.time_s, ball.reference_px));
            }
            if self.path.len() == PATH_POINTS {
                self.path.pop_front();
            }
            self.path.push_back(PathPoint {
                track_id: update.track_id,
                position: ball.measured_stabilized_m.map(|v| v as f32),
                supported: update.outline_supported,
            });
            if let Some(flight) = ball.flight {
                match self.flights.back_mut() {
                    Some(path) if path.track_id == update.track_id => {
                        path.time_s = update.time_s;
                        path.flight = flight;
                    }
                    _ => {
                        if self.flights.len() == FLIGHTS {
                            self.flights.pop_front();
                        }
                        self.flights.push_back(FlightPath {
                            track_id: update.track_id,
                            start_s: update.time_s,
                            time_s: update.time_s,
                            flight,
                        });
                    }
                }
            }
        }
        while self
            .trail
            .front()
            .is_some_and(|&(t, _)| update.time_s - t > TRAIL_S)
        {
            self.trail.pop_front();
        }
        if self.compute_ms.len() == TIMING_SAMPLES {
            self.compute_ms.pop_front();
        }
        self.compute_ms.push_back(update.compute_ms());
        self.update = Some(update);
        Ok(Step::Updated)
    }

    /// Forgets all tracks and what was drawn of them.
    fn clear(&mut self) {
        self.tracker.reset();
        self.trail.clear();
        self.path.clear();
        self.flights.clear();
        self.update = None;
    }

    /// Restarts the tracker shortly before `target_us` and replays up to it.
    /// Live input cannot go back, so it only restarts the tracker.
    fn seek(&mut self, target_us: u64) -> Result<()> {
        self.clear();
        if self.live().is_some() {
            return Ok(());
        }
        let target = target_us.clamp(self.first_us, self.last_us) / STEP_US * STEP_US;
        self.next_us = target.saturating_sub(PREROLL_US).max(self.first_us);
        while self.next_us <= target && matches!(self.advance()?, Step::Updated) {}
        Ok(())
    }

    /// Drops the events that arrived while a live view was paused.
    fn discard_live(&mut self) -> Result<()> {
        if let Source::Live(live) = &mut self.source {
            live.poll()?;
            live.events.clear();
            // Start over with the next events.
            self.next_us = 0;
            self.first_us = 0;
        }
        Ok(())
    }
}

fn state_label(update: &BallUpdate) -> String {
    match (update.state, update.outline_supported) {
        (TrackState::Lost, _) if !update.imu_covered => "No IMU coverage".into(),
        (TrackState::Lost, _) => "No ball track".into(),
        (TrackState::Measured, true) => format!("Track {}: outline measured", update.track_id),
        (TrackState::Measured, false) => {
            format!("Track {}: cluster only, depth held", update.track_id)
        }
        (TrackState::Predicted, _) => format!("Track {}: predicted", update.track_id),
    }
}

/// Software image of the compensated events with the tracking overlay. The
/// viewer and the video export show the same pixels.
struct Canvas {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
    background: Vec<u8>,
    foreground: Vec<u8>,
    levels: [[u8; 256]; 2],
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        // Brightness saturates with event density; foreground stays brighter than background.
        let level = |base: f32, gain: f32, scale: f32| {
            std::array::from_fn(|n| {
                if n == 0 {
                    0
                } else {
                    (base + gain * (1. - (-((n - 1) as f32) / scale).exp())) as u8
                }
            })
        };
        Self {
            width,
            height,
            rgb: vec![0; width * height * 3],
            background: vec![0; width * height],
            foreground: vec![0; width * height],
            levels: [level(65., 65., 2.), level(165., 90., 1.5)],
        }
    }

    fn draw(&mut self, player: &Player, show_candidates: bool) {
        self.background.fill(0);
        self.foreground.fill(0);
        let frame = player.tracker.frame();
        for (p, &foreground) in frame.points.iter().zip(&frame.foreground) {
            if p[0] < 0. || p[1] < 0. || p[0] >= self.width as f32 || p[1] >= self.height as f32 {
                continue;
            }
            let i = p[1] as usize * self.width + p[0] as usize;
            let count = if foreground {
                &mut self.foreground[i]
            } else {
                &mut self.background[i]
            };
            *count = count.saturating_add(1);
        }
        for (i, rgb) in self.rgb.chunks_exact_mut(3).enumerate() {
            let level = if self.foreground[i] > 0 {
                self.levels[1][self.foreground[i] as usize]
            } else {
                self.levels[0][self.background[i] as usize]
            };
            rgb.fill(level);
        }
        if show_candidates {
            for candidate in player.tracker.candidates() {
                let center = [candidate.x as f32, candidate.y as f32];
                match candidate.outline {
                    Some(outline) => self.circle(center, outline.radius as f32, CANDIDATE, 1),
                    None => self.rectangle(
                        center,
                        [candidate.width as f32 / 2., candidate.height as f32 / 2.],
                        CANDIDATE,
                        1,
                    ),
                }
            }
        }
        let Some(update) = &player.update else {
            return;
        };
        let Some(ball) = update.ball else {
            return;
        };
        let trail: Vec<_> = player
            .trail
            .iter()
            .filter_map(|&(_, p)| player.tracker.camera_px(p))
            .map(|p| p.map(|v| v as f32))
            .collect();
        for segment in trail.windows(2) {
            self.line(segment[0], segment[1], TRAIL, 2);
        }
        if let Some(flight) = ball.flight {
            // Where the flight path leads, seen from the current camera orientation.
            let ahead: Vec<_> = (0..=15)
                .filter_map(|i| {
                    player
                        .tracker
                        .stabilized_px(flight.stabilized_after(AHEAD_S * i as f64 / 15.))
                })
                .map(|p| p.map(|v| v as f32))
                .collect();
            for segment in ahead.windows(2) {
                self.line(segment[0], segment[1], AHEAD, 2);
            }
        }
        let color = if update.state == TrackState::Measured {
            MEASURED
        } else {
            PREDICTED
        };
        let center = ball.pixel.map(|v| v as f32);
        let half = ball.size_px.map(|v| v as f32 / 2.);
        if update.outline_supported {
            self.circle(center, half[0], color, 2);
        } else {
            self.rectangle(center, [half[0] + 6., half[1] + 6.], color, 2);
        }
        self.line(
            [center[0] - 8., center[1]],
            [center[0] + 8., center[1]],
            color,
            2,
        );
        self.line(
            [center[0], center[1] - 8.],
            [center[0], center[1] + 8.],
            color,
            2,
        );
    }

    fn line(&mut self, a: [f32; 2], b: [f32; 2], color: [u8; 3], thickness: i32) {
        if !a.iter().chain(&b).all(|v| v.is_finite()) {
            return;
        }
        let steps = (a[0] - b[0])
            .abs()
            .max((a[1] - b[1]).abs())
            .ceil()
            .min(4000.) as usize;
        for i in 0..=steps {
            let f = i as f32 / steps.max(1) as f32;
            let x = (a[0] * (1. - f) + b[0] * f).round() as i32;
            let y = (a[1] * (1. - f) + b[1] * f).round() as i32;
            for (x, y) in
                (0..thickness).flat_map(|dy| (0..thickness).map(move |dx| (x + dx, y + dy)))
            {
                if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                    let i = (y as usize * self.width + x as usize) * 3;
                    self.rgb[i..i + 3].copy_from_slice(&color);
                }
            }
        }
    }

    fn circle(&mut self, center: [f32; 2], radius: f32, color: [u8; 3], thickness: i32) {
        let point = |i: usize| {
            let angle = i as f32 * std::f32::consts::TAU / 64.;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]
        };
        for i in 0..64 {
            self.line(point(i), point(i + 1), color, thickness);
        }
    }

    fn rectangle(&mut self, center: [f32; 2], half: [f32; 2], color: [u8; 3], thickness: i32) {
        let corner = |sx: f32, sy: f32| [center[0] + sx * half[0], center[1] + sy * half[1]];
        for (a, b) in [
            (corner(-1., -1.), corner(1., -1.)),
            (corner(1., -1.), corner(1., 1.)),
            (corner(1., 1.), corner(-1., 1.)),
            (corner(-1., 1.), corner(-1., -1.)),
        ] {
            self.line(a, b, color, thickness);
        }
    }
}

fn speed(velocity_mps: [f64; 3]) -> f64 {
    velocity_mps.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() as f64 * fraction) as usize).min(sorted.len() - 1)]
}

struct TrackSummary {
    track_id: u32,
    start_s: f64,
    end_s: f64,
    measured: usize,
    outlines: usize,
    depth_m: [f64; 2],
}

/// Runs the whole interval without a window: timing, and optionally a track table and a video.
fn batch(mut player: Player, motion: MotionArgs, ball: BallArgs) -> Result<()> {
    let mut csv = match &ball.tracks_csv {
        Some(path) => {
            let mut file = BufWriter::new(
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)?,
            );
            writeln!(
                file,
                "time_s,state,track_id,outline_supported,x_px,y_px,width_px,height_px,diameter_px,\
                 measured_x_m,measured_y_m,measured_z_m,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,\
                 depth_sigma_m,stabilized_x_m,stabilized_y_m,stabilized_z_m,\
                 stabilized_vx_mps,stabilized_vy_mps,stabilized_vz_mps"
            )?;
            Some(file)
        }
        None => None,
    };
    let (width, height) = (player.camera().width, player.camera().height);
    let mut canvas = Canvas::new(width, height);
    // Label bars above and below the image; the encoder needs even dimensions.
    let video_width = width.next_multiple_of(2);
    let video_height = (height + 64).next_multiple_of(2);
    let mut video = vec![0u8; video_width * video_height * 3];
    let text_context = egui::Context::default();
    let mut encoder = match &motion.export_mp4 {
        Some(path) => Some(comparison::spawn_ffmpeg(
            video_width,
            video_height,
            motion.motion_fps,
            path,
        )?),
        None => None,
    };
    let mut encoder_input = encoder.as_mut().map(|child| child.stdin.take().unwrap());
    let frame_us = 1e6 / motion.motion_fps as f64;
    let mut next_frame_us: Option<f64> = None;
    let mut waits_ms = Vec::new();
    if let Some(live) = player.live() {
        eprintln!(
            "Running on live input for {:.0} s...",
            live.duration_us.unwrap_or(0) as f64 * 1e-6
        );
    }
    let mut frames = 0;

    let mut read = Vec::new();
    let mut stages: [Vec<f64>; 3] = Default::default();
    let mut total = Vec::new();
    let mut states = [0usize; 3];
    let mut tracks: Vec<TrackSummary> = Vec::new();
    // Predicted positions waiting for the outline measured at their time.
    let mut predictions: VecDeque<(u64, u32, [f64; 3])> = VecDeque::new();
    let mut prediction_px = Vec::new();
    let diameter = player.tracker.ball_diameter_m();
    let result = (|| -> Result<()> {
        loop {
            match player.advance()? {
                Step::Updated => {}
                Step::Waiting => {
                    // Wait without sleeping. A thread that sleeps between updates is
                    // slowed down by the system; on an Apple Silicon Mac an update
                    // then takes ten times as long. This keeps one core busy.
                    std::hint::spin_loop();
                    continue;
                }
                Step::Finished => break,
            }
            let update = player.update.unwrap();
            match &player.source {
                Source::File(engine) => read.push(engine.read_ms),
                Source::Live(live) => waits_ms.push(live.wait_ms),
            }
            for (samples, ms) in stages.iter_mut().zip(update.stage_ms) {
                samples.push(ms);
            }
            total.push(update.compute_ms());
            states[match update.state {
                TrackState::Measured => 0,
                TrackState::Predicted => 1,
                TrackState::Lost => 2,
            }] += 1;
            if let (TrackState::Measured, Some(ball)) = (update.state, update.ball) {
                if tracks.last().is_none_or(|t| t.track_id != update.track_id) {
                    tracks.push(TrackSummary {
                        track_id: update.track_id,
                        start_s: update.time_s,
                        end_s: update.time_s,
                        measured: 0,
                        outlines: 0,
                        depth_m: [f64::INFINITY, 0.],
                    });
                }
                let track = tracks.last_mut().unwrap();
                track.end_s = update.time_s;
                track.measured += 1;
                track.outlines += usize::from(update.outline_supported);
                let depth = ball.position_m()[2];
                track.depth_m = [track.depth_m[0].min(depth), track.depth_m[1].max(depth)];
            }
            let now_us = player.time_us();
            while predictions.front().is_some_and(|&(due, ..)| due < now_us) {
                predictions.pop_front();
            }
            if let Some(ball) = update.ball {
                if predictions.front().is_some_and(|&(due, ..)| due == now_us) {
                    let (_, track_id, point) = predictions.pop_front().unwrap();
                    let predicted = player.tracker.stabilized_px(point);
                    if let (true, true, Some(measured), Some(predicted)) = (
                        track_id == update.track_id,
                        update.outline_supported,
                        ball.measured_px,
                        predicted,
                    ) {
                        prediction_px
                            .push((predicted[0] - measured[0]).hypot(predicted[1] - measured[1]));
                    }
                }
                if let Some(flight) = ball.flight {
                    predictions.push_back((
                        now_us + CHECK_US,
                        update.track_id,
                        flight.stabilized_after(CHECK_US as f64 * 1e-6),
                    ));
                }
            }
            if let Some(file) = &mut csv {
                let state = match update.state {
                    TrackState::Lost => "lost",
                    TrackState::Measured => "measured",
                    TrackState::Predicted => "predicted",
                };
                write!(
                    file,
                    "{:.2},{state},{},{}",
                    update.time_s, update.track_id, update.outline_supported
                )?;
                match update.ball {
                    Some(b) => {
                        write!(
                            file,
                            ",{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4}",
                            b.pixel[0],
                            b.pixel[1],
                            b.size_px[0],
                            b.size_px[1],
                            b.diameter_px,
                            b.measured_m[0],
                            b.measured_m[1],
                            b.measured_m[2],
                        )?;
                        match b.flight {
                            Some(f) => {
                                for v in f.position_m.iter().chain(&f.velocity_mps) {
                                    write!(file, ",{v:.4}")?;
                                }
                                write!(file, ",{:.4}", f.depth_sigma_m)?;
                                for v in f.stabilized_m.iter().chain(&f.stabilized_velocity_mps) {
                                    write!(file, ",{v:.4}")?;
                                }
                                writeln!(file)?;
                            }
                            // Without a flight estimate the position is the measurement.
                            None => writeln!(
                                file,
                                ",{:.4},{:.4},{:.4},,,,,{:.4},{:.4},{:.4},,,",
                                b.measured_m[0],
                                b.measured_m[1],
                                b.measured_m[2],
                                b.measured_stabilized_m[0],
                                b.measured_stabilized_m[1],
                                b.measured_stabilized_m[2],
                            )?,
                        }
                    }
                    None => writeln!(file, ",,,,,,,,,,,,,,,,,,,,,")?,
                }
            }
            if let Some(input) = &mut encoder_input {
                let due = next_frame_us.get_or_insert(player.first_us as f64);
                if player.time_us() as f64 >= *due {
                    *due += frame_us;
                    canvas.draw(&player, false);
                    video.fill(0);
                    for row in 0..height {
                        let dst = (row + 32) * video_width * 3;
                        video[dst..dst + width * 3]
                            .copy_from_slice(&canvas.rgb[row * width * 3..(row + 1) * width * 3]);
                    }
                    let mut status = state_label(&update);
                    if let Some(b) = update.ball {
                        let [x, y, z] = b.position_m();
                        status += &format!(" | X {x:+.2} m  Y {y:+.2} m  Z {z:.2} m");
                        if let Some(f) = b.flight {
                            status += &format!(" | {:.1} m/s", speed(f.velocity_mps));
                        }
                    }
                    comparison::draw_labels(
                        &text_context,
                        &mut video,
                        video_width,
                        video_height,
                        &[
                            (
                                12,
                                5,
                                format!(
                                    "Rotation-compensated events, foreground bright | ball diameter {:.0} cm",
                                    diameter * 100.
                                ),
                            ),
                            (
                                12,
                                height + 37,
                                format!("Time {:.2}s | {status}", update.time_s),
                            ),
                        ],
                    );
                    input.write_all(&video)?;
                    frames += 1;
                }
            }
        }
        Ok(())
    })();
    drop(encoder_input);
    if let Some(mut child) = encoder {
        let status = child.wait()?;
        result?;
        if !status.success() {
            return Err("ffmpeg video export failed".into());
        }
        eprintln!(
            "Saved {} ({frames} frames)",
            motion.export_mp4.as_ref().unwrap().display()
        );
    } else {
        result?;
    }
    if let Some(mut file) = csv {
        file.flush()?;
        eprintln!("Saved {}", ball.tracks_csv.as_ref().unwrap().display());
    }

    if total.is_empty() {
        return Err("no events arrived".into());
    }
    println!(
        "{} updates every {} ms from {:.2}s to {:.2}s, ball diameter {diameter} m",
        total.len(),
        STEP_US / 1000,
        player.first_us as f64 * 1e-6,
        player.time_us() as f64 * 1e-6
    );
    if let Some(live) = player.live() {
        waits_ms.sort_by(f64::total_cmp);
        println!(
            "live input: {:.2} million events/s at the end, {} updates skipped to stay live, events waited p50={:.2} ms p95={:.2} ms between USB and tracker",
            live.rate / 1e6,
            live.skipped,
            percentile(&waits_ms, 0.5),
            percentile(&waits_ms, 0.95)
        );
    }
    println!(
        "measured {}, predicted {}, without track {}",
        states[0], states[1], states[2]
    );
    for track in &tracks {
        println!(
            "track {}: {:.2}s to {:.2}s, {} measured updates ({} with outline), depth {:.2} to {:.2} m",
            track.track_id,
            track.start_s,
            track.end_s,
            track.measured,
            track.outlines,
            track.depth_m[0],
            track.depth_m[1]
        );
    }
    if !prediction_px.is_empty() {
        prediction_px.sort_by(f64::total_cmp);
        println!(
            "flight path predicted {} ms ahead against the outline measured then: median {:.1} px, 90% within {:.1} px, {} outlines",
            CHECK_US / 1000,
            percentile(&prediction_px, 0.5),
            percentile(&prediction_px, 0.9),
            prediction_px.len()
        );
    }
    let budget_ms = STEP_US as f64 / 1000.;
    let misses = total.iter().filter(|&&ms| ms > budget_ms).count();
    let [compensation, outlines, association] = stages;
    for (label, mut samples) in [
        ("file read", read),
        ("compensation + foreground", compensation),
        ("clusters + outlines", outlines),
        ("association", association),
        ("tracking total", total.clone()),
    ] {
        if samples.is_empty() {
            continue;
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{label}: p50={:.3} ms p95={:.3} ms max={:.3} ms",
            percentile(&samples, 0.5),
            percentile(&samples, 0.95),
            samples.last().unwrap()
        );
    }
    println!(
        "Tracking updates over the {budget_ms} ms step: {misses}/{}; excludes file read, sensor transport, display and video encoding",
        total.len()
    );
    Ok(())
}

struct Orbit {
    yaw: f32,
    pitch: f32,
    pixels_per_m: f32,
}

struct BallApp {
    player: Player,
    canvas: Canvas,
    texture: Option<TextureHandle>,
    paused: bool,
    speed: f64,
    last_tick: Instant,
    /// Recording time that playback has reached, in microseconds.
    clock_us: f64,
    dirty: bool,
    show_candidates: bool,
    error: Option<String>,
    orbit: Orbit,
}

impl BallApp {
    fn new(player: Player) -> Self {
        let canvas = Canvas::new(player.camera().width, player.camera().height);
        let live = player.live().is_some();
        let mut app = Self {
            canvas,
            texture: None,
            // A recording waits for Play; a camera shows what it sees.
            paused: !live,
            speed: 0.5,
            last_tick: Instant::now(),
            clock_us: 0.,
            dirty: true,
            show_candidates: false,
            error: None,
            orbit: Orbit {
                yaw: 0.6,
                pitch: 0.5,
                pixels_per_m: 70.,
            },
            player,
        };
        if !live {
            let first = app.player.first_us;
            app.seek(first);
        }
        app
    }

    fn report(&mut self, result: Result<()>) {
        if let Err(error) = result {
            self.error = Some(error.to_string());
            self.paused = true;
        }
    }

    fn seek(&mut self, target_us: u64) {
        let result = self.player.seek(target_us);
        self.report(result);
        self.clock_us = self.player.time_us() as f64;
        self.last_tick = Instant::now();
        self.dirty = true;
    }

    fn play(&mut self) {
        let elapsed = self.last_tick.elapsed().as_secs_f64();
        self.last_tick = Instant::now();
        let live = self.player.live().is_some();
        if self.paused {
            if live {
                let result = self.player.discard_live();
                self.report(result);
            }
            return;
        }
        self.clock_us += elapsed * self.speed * 1e6;
        let mut steps = 0;
        // A recording follows the playback clock; live input takes what has arrived.
        while (live || (self.player.next_us as f64) <= self.clock_us) && steps < MAX_STEPS_PER_FRAME
        {
            match self.player.advance() {
                Ok(Step::Updated) => steps += 1,
                Ok(Step::Waiting) => break,
                Ok(Step::Finished) => {
                    self.paused = true;
                    break;
                }
                Err(error) => {
                    self.report(Err(error));
                    break;
                }
            }
        }
        if steps == MAX_STEPS_PER_FRAME || self.paused {
            self.clock_us = self.player.time_us() as f64;
        }
        self.dirty |= steps > 0;
    }

    fn ui_live_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused { "▶ Resume" } else { "⏸ Pause" })
                .clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Space))
            {
                self.paused = !self.paused;
                if !self.paused {
                    self.player.clear();
                }
            }
            if let Some(live) = self.player.live() {
                ui.strong(format!("Live · camera {}", live.stream.serial));
                ui.label(format!(
                    "{:.2} million events/s · sensor time {:.1} s · events wait {:.1} ms · {} updates skipped",
                    live.rate / 1e6,
                    self.player.time_us() as f64 * 1e-6,
                    live.wait_ms,
                    live.skipped
                ))
                .on_hover_text("The wait is the time between the arrival of events over USB and their processing. Updates are skipped when processing falls more than 150 ms behind.");
            }
        });
    }

    fn ui_controls(&mut self, ui: &mut egui::Ui) {
        let live = self.player.live().is_some();
        if live {
            self.ui_live_controls(ui);
        } else {
            self.ui_playback_controls(ui);
        }
        self.ui_settings(ui, live);
    }

    fn ui_playback_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                .clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Space))
            {
                self.paused = !self.paused;
                if !self.paused && self.player.next_us > self.player.last_us {
                    let first = self.player.first_us;
                    self.seek(first);
                }
                self.clock_us = self.player.time_us() as f64;
                self.last_tick = Instant::now();
            }
            if ui
                .button("Step")
                .on_hover_text("Process one 10 ms update.")
                .clicked()
            {
                self.paused = true;
                let result = self.player.advance().map(|_| ());
                self.report(result);
                self.dirty = true;
            }
            let mut time_s = self.player.time_us() as f64 * 1e-6;
            let range = self.player.first_us as f64 * 1e-6..=self.player.last_us as f64 * 1e-6;
            let slider_width = ui.spacing().slider_width;
            ui.spacing_mut().slider_width = (ui.available_width() - 480.).clamp(slider_width, 900.);
            let time = ui.add(egui::Slider::new(&mut time_s, range).text("recording seconds"));
            ui.spacing_mut().slider_width = slider_width;
            if time.changed() {
                self.paused = true;
                self.seek((time_s * 1e6).round() as u64);
            }
            ui.add(egui::Slider::new(&mut self.speed, 0.05..=2.0).text("speed"));
        });
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui, live: bool) {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Ball diameter");
            let mut diameter = self.player.tracker.ball_diameter_m();
            if ui
                .add(
                    egui::DragValue::new(&mut diameter)
                        .speed(0.001)
                        .range(0.005..=2.0)
                        .max_decimals(3)
                        .suffix(" m"),
                )
                .on_hover_text("Real diameter of the ball. The distance scales with it.")
                .changed()
            {
                self.player.tracker.set_ball_diameter_m(diameter);
                changed = true;
            }
            let config = self.player.config_mut();
            if !live {
                ui.label("IMU offset");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.offset_s)
                            .speed(0.001)
                            .max_decimals(4)
                            .suffix(" s"),
                    )
                    .changed();
            }
            changed |= ui
                .add(egui::Slider::new(&mut config.threshold, 0.0..=0.4).text("motion threshold"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut config.min_component_cells, 1..=100).text("min cells"))
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut config.min_new_fraction, 0.0..=1.0)
                        .text("min new fraction"),
                )
                .on_hover_text("Part of each foreground region that must lie beyond early activity.")
                .changed();
            self.dirty |= ui
                .checkbox(&mut self.show_candidates, "candidates")
                .on_hover_text("Show every cluster proposal: a circle where an outline was found, otherwise its box.")
                .changed();
        });
        if changed {
            // Settings change the track history, so replay the moments before this time.
            self.seek(self.player.time_us());
        }
    }

    fn ui_status(&self, ui: &mut egui::Ui) {
        let Some(update) = &self.player.update else {
            ui.label("No update yet");
            return;
        };
        ui.horizontal(|ui| {
            let color = match update.state {
                TrackState::Lost => ui.visuals().weak_text_color(),
                TrackState::Measured => egui::Color32::from_rgb(MEASURED[0], MEASURED[1], MEASURED[2]),
                TrackState::Predicted => {
                    egui::Color32::from_rgb(PREDICTED[0], PREDICTED[1], PREDICTED[2])
                }
            };
            ui.colored_label(color, egui::RichText::new(state_label(update)).strong());
            if let Some(ball) = update.ball {
                let [x, y, z] = ball.position_m();
                let mut text = format!("X {x:+.2} m   Y {y:+.2} m   Z {z:.2} m");
                match ball.flight {
                    Some(flight) => {
                        text += &format!(
                            " ± {:.2}   speed {:.1} m/s   unfiltered Z {:.2} m",
                            flight.depth_sigma_m,
                            speed(flight.velocity_mps),
                            ball.measured_m[2]
                        );
                    }
                    None => text += "   unfiltered: the IMU file has no gravity columns",
                }
                ui.monospace(text)
                .on_hover_text("Camera frame at the time shown: X right, Y down, Z forward. Filtered with a Kalman filter that assumes free flight under gravity. Camera translation is not measured.");
            }
        });
        let mut sorted: Vec<_> = self.player.compute_ms.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        if !sorted.is_empty() {
            ui.label(format!(
                "Tracking {:.2} ms per update · last {}: p95 {:.2} ms, max {:.2} ms · step {} ms · {} events in window",
                update.compute_ms(),
                sorted.len(),
                percentile(&sorted, 0.95),
                sorted.last().unwrap(),
                STEP_US / 1000,
                self.player.tracker.frame().points.len(),
            ));
        }
    }

    fn ui_image(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        if self.dirty {
            self.dirty = false;
            self.canvas.draw(&self.player, self.show_candidates);
            let image =
                ColorImage::from_rgb([self.canvas.width, self.canvas.height], &self.canvas.rgb);
            match &mut self.texture {
                Some(texture) => texture.set(image, TextureOptions::LINEAR),
                None => {
                    self.texture =
                        Some(ui.ctx().load_texture("ball", image, TextureOptions::LINEAR));
                }
            }
        }
        let rect = match &self.texture {
            Some(texture) => ui.image((texture.id(), size)).rect,
            None => ui.allocate_space(size).1,
        };
        let Some(ball) = self.player.update.and_then(|update| update.ball) else {
            return;
        };
        let scale = rect.width() / self.canvas.width as f32;
        let anchor = rect.min
            + egui::vec2(
                (ball.pixel[0] + ball.size_px[0] / 2.) as f32 * scale + 10.,
                ball.pixel[1] as f32 * scale,
            );
        ui.painter_at(rect).text(
            anchor,
            egui::Align2::LEFT_CENTER,
            format!("{:.2} m", ball.position_m()[2]),
            egui::FontId::proportional(16.),
            egui::Color32::WHITE,
        );
    }

    fn ui_space(&mut self, ui: &mut egui::Ui, size: egui::Vec2) {
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(8, 10, 14));
        if response.dragged() {
            let delta = response.drag_delta();
            self.orbit.yaw += delta.x * 0.01;
            self.orbit.pitch = (self.orbit.pitch + delta.y * 0.01).clamp(-1.5, 1.5);
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            self.orbit.pixels_per_m =
                (self.orbit.pixels_per_m * (1. + scroll * 0.002)).clamp(5., 2000.);
        }
        // Orthographic view of the stabilized camera frame: x right, y down, z forward.
        let (sin_yaw, cos_yaw) = self.orbit.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.orbit.pitch.sin_cos();
        let origin = egui::pos2(
            rect.left() + rect.width() * 0.4,
            rect.bottom() - rect.height() * 0.22,
        );
        let scale = self.orbit.pixels_per_m;
        let project = |p: [f32; 3]| {
            let x = p[0] * cos_yaw + p[2] * sin_yaw;
            let z = -p[0] * sin_yaw + p[2] * cos_yaw;
            origin + egui::vec2(x, p[1] * cos_pitch - z * sin_pitch) * scale
        };
        let line = |a: [f32; 3], b: [f32; 3], width: f32, color: egui::Color32| {
            painter.line_segment([project(a), project(b)], egui::Stroke::new(width, color));
        };
        let text =
            |position: egui::Pos2, align: egui::Align2, text: String, color: egui::Color32| {
                painter.text(
                    position,
                    align,
                    text,
                    egui::FontId::proportional(12.),
                    color,
                );
            };
        let grid = egui::Color32::from_gray(45);
        let label = egui::Color32::from_gray(150);
        for i in 0..=8 {
            let z = i as f32;
            line([-4., 0., z], [4., 0., z], 1., grid);
            let x = z - 4.;
            line([x, 0., 0.], [x, 0., 8.], 1., grid);
            if i > 0 && i % 2 == 0 {
                text(
                    project([-4., 0., z]),
                    egui::Align2::RIGHT_CENTER,
                    format!("{i} m "),
                    label,
                );
            }
        }
        for (axis, name, color) in [
            ([0.5, 0., 0.], "X", egui::Color32::from_rgb(230, 80, 80)),
            ([0., 0.5, 0.], "Y", egui::Color32::from_rgb(90, 200, 90)),
            ([0., 0., 0.5], "Z", egui::Color32::from_rgb(90, 140, 255)),
        ] {
            line([0.; 3], axis, 2., color);
            text(project(axis), egui::Align2::LEFT_BOTTOM, name.into(), color);
        }
        // The camera's field of view at the current orientation is not drawn: the
        // frame is the orientation at the start of playback.
        let palette = [
            egui::Color32::from_rgb(40, 240, 100),
            egui::Color32::from_rgb(90, 180, 255),
            egui::Color32::from_rgb(255, 200, 60),
            egui::Color32::from_rgb(240, 110, 220),
        ];
        // Single measurements as dots; the filtered flight as a curve through them.
        for point in &self.player.path {
            let color = palette[point.track_id as usize % palette.len()];
            let color = color.gamma_multiply(if point.supported { 0.55 } else { 0.25 });
            painter.circle_filled(project(point.position), 1.5, color);
        }
        let current = self
            .player
            .update
            .filter(|update| update.ball.is_some())
            .map(|update| update.track_id);
        for path in &self.player.flights {
            let color = palette[path.track_id as usize % palette.len()];
            let flying = current == Some(path.track_id);
            let points = path.points(if flying { AHEAD_S } else { 0. });
            let covered = path.time_s - path.start_s;
            let span = covered + if flying { AHEAD_S } else { 0. };
            for (i, segment) in points.windows(2).enumerate() {
                // The part ahead of the ball is a prediction.
                let ahead = span * (i + 1) as f64 / (points.len() - 1) as f64 > covered + 1e-9;
                line(
                    segment[0].map(|v| v as f32),
                    segment[1].map(|v| v as f32),
                    if ahead { 1. } else { 2.5 },
                    if ahead {
                        egui::Color32::from_rgb(AHEAD[0], AHEAD[1], AHEAD[2])
                    } else {
                        color
                    },
                );
            }
        }
        if let Some(ball) = self.player.update.and_then(|update| update.ball) {
            let p = ball.stabilized_m().map(|v| v as f32);
            // Foot point on the grid plane, to read depth and height apart.
            line(p, [p[0], 0., p[2]], 1., egui::Color32::from_gray(110));
            painter.circle_filled(project([p[0], 0., p[2]]), 2., egui::Color32::from_gray(110));
            painter.circle_filled(project(p), 5., egui::Color32::WHITE);
            text(
                project(p) + egui::vec2(9., -9.),
                egui::Align2::LEFT_BOTTOM,
                format!("{:+.2}, {:+.2}, {:.2} m", p[0], p[1], p[2]),
                egui::Color32::WHITE,
            );
        }
        text(
            rect.left_top() + egui::vec2(8., 6.),
            egui::Align2::LEFT_TOP,
            "Ball path in the camera orientation at the start of playback. Grid: 1 m, at camera height.\nDrag to rotate, scroll to zoom. Dots: single measurements. Curve: filtered flight. Pink: predicted.".into(),
            label,
        );
    }
}

impl eframe::App for BallApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Frame::new().inner_margin(8.0).show(ui, |ui| {
            self.ui_controls(ui);
            self.play();
            self.ui_status(ui);
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::RED, error);
            }
            let available = ui.available_size();
            let aspect = self.canvas.width as f32 / self.canvas.height as f32;
            let width = (available.x * 0.62).min(available.y * aspect).max(160.);
            let image_size = egui::vec2(width, width / aspect);
            ui.horizontal_top(|ui| {
                self.ui_image(ui, image_size);
                let space = egui::vec2(
                    (available.x - width - ui.spacing().item_spacing.x).max(160.),
                    available.y.max(image_size.y),
                );
                self.ui_space(ui, space);
            });
        });
        ui.ctx()
            .request_repaint_after(Duration::from_millis(if self.paused { 100 } else { 8 }));
    }
}
