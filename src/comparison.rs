use clap::Args;
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use event_clustering::{
    Event,
    motion::{self, Camera, ImuTrack, MotionConfig, MotionFrame, Rotation},
    parabola::parse_calibration_json,
    parser::{Endian, EventFormat, EventRecord, EventStream, open_event_stream},
};
use std::{
    collections::VecDeque,
    error::Error,
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Debug, Args)]
pub(crate) struct MotionArgs {
    /// iPhone CSV with timestamp,gx,gy,gz (seconds and rad/s).
    #[arg(long)]
    imu: Option<PathBuf>,
    #[arg(long, default_value = "calibration.json")]
    calibration: PathBuf,
    /// Relative IMU time = relative camera time + this offset, in seconds.
    #[arg(long, allow_hyphen_values = true)]
    motion_offset: Option<f64>,
    /// Nine comma-separated values, row-major rotation from IMU to camera axes.
    #[arg(long, allow_hyphen_values = true)]
    imu_to_camera: Option<String>,
    /// Reuse an alignment written by --save-alignment.
    #[arg(long)]
    alignment: Option<PathBuf>,
    #[arg(long)]
    save_alignment: Option<PathBuf>,
    /// Search +/- this many seconds around zero (or --motion-offset).
    #[arg(long, default_value_t = 4.0)]
    alignment_range: f64,
    /// Force joint time/mount calibration, including when an offset was supplied.
    #[arg(long)]
    auto_align: bool,
    #[arg(long, default_value_t = 30.0)]
    motion_window_ms: f64,
    #[arg(long, default_value_t = 0.12)]
    motion_threshold: f32,
    #[arg(long, default_value_t = 3)]
    motion_cell_px: usize,
    #[arg(long, default_value_t = 8)]
    motion_min_cells: usize,
    #[arg(long, default_value_t = 0.0)]
    start_s: f64,
    #[arg(long)]
    end_s: Option<f64>,
    /// Export a labeled four-panel MP4 instead of opening the viewer; requires ffmpeg.
    #[arg(long)]
    export_mp4: Option<PathBuf>,
    #[arg(long, default_value_t = 25)]
    motion_fps: u32,
}

fn mount_from_text(text: &str) -> Result<Rotation> {
    let values: Vec<f64> = text
        .split(',')
        .map(|v| v.trim().parse())
        .collect::<std::result::Result<_, _>>()?;
    if values.len() != 9 || !values.iter().all(|v| v.is_finite()) {
        return Err("mount needs nine finite numbers".into());
    }
    let m: Rotation = std::array::from_fn(|i| std::array::from_fn(|j| values[i * 3 + j]));
    let orthogonal = motion::multiply(m, motion::transpose(m));
    if (0..3).any(|i| (0..3).any(|j| (orthogonal[i][j] - motion::IDENTITY[i][j]).abs() > 0.002)) {
        return Err("mount must be an orthonormal rotation".into());
    }
    let cross = [
        m[0][1] * m[1][2] - m[0][2] * m[1][1],
        m[0][2] * m[1][0] - m[0][0] * m[1][2],
        m[0][0] * m[1][1] - m[0][1] * m[1][0],
    ];
    if cross.iter().zip(m[2]).map(|(a, b)| a * b).sum::<f64>() < 0.99 {
        return Err("mount must be right-handed (determinant +1)".into());
    }
    Ok(m)
}

pub(crate) fn run(
    path: PathBuf,
    args: MotionArgs,
    format: EventFormat,
    endian: Endian,
) -> Result<()> {
    let imu_path = args
        .imu
        .as_ref()
        .ok_or("motion-compare requires --imu path/to/recording.csv")?;
    if !args.motion_window_ms.is_finite()
        || !(1.0..=200.0).contains(&args.motion_window_ms)
        || !args.motion_threshold.is_finite()
        || !(0.0..=0.5).contains(&args.motion_threshold)
        || !(1..=16).contains(&args.motion_cell_px)
        || args.motion_min_cells == 0
        || !(1..=120).contains(&args.motion_fps)
        || !args.start_s.is_finite()
        || args.start_s < 0.
        || !args.alignment_range.is_finite()
        || !(0.1..=30.).contains(&args.alignment_range)
        || args
            .end_s
            .is_some_and(|v| !v.is_finite() || v <= args.start_s)
        || args.motion_offset.is_some_and(|v| !v.is_finite())
    {
        return Err("invalid motion settings: window 1..200 ms, threshold 0..0.5, cell 1..16 px, positive FPS/interval".into());
    }
    if args.export_mp4.as_ref().is_some_and(|p| p.exists()) {
        return Err("output video already exists; choose a new path".into());
    }
    if let Some(p) = &args.save_alignment {
        if p.exists() {
            return Err("alignment output already exists; choose a new path".into());
        }
    }
    let imu = ImuTrack::load(imu_path)?;
    let calibration = parse_calibration_json(&fs::read_to_string(&args.calibration)?)
        .ok_or("invalid calibration.json")?;
    let header = open_event_stream(&path, format, endian)?.header;
    let width = header
        .width
        .or(calibration.image_width)
        .ok_or("recording width unavailable")?;
    let height = header
        .height
        .or(calibration.image_height)
        .ok_or("recording height unavailable")?;
    let camera = Camera::new(calibration, width, height)?;
    eprintln!("Scanning recording and collecting short calibration windows...");
    let (origin_us, duration_s, windows) = scan(&path, format, endian)?;
    let mut config = MotionConfig {
        cell_px: args.motion_cell_px,
        threshold: args.motion_threshold,
        min_component_cells: args.motion_min_cells,
        ..Default::default()
    };
    if let Some(p) = &args.alignment {
        let text = fs::read_to_string(p)?;
        let mut lines = text.lines();
        config.offset_s = lines.next().ok_or("missing alignment offset")?.parse()?;
        config.mount = mount_from_text(lines.next().ok_or("missing mounting rotation")?)?;
        if !config.offset_s.is_finite() {
            return Err("nonfinite alignment offset".into());
        }
    }
    if let Some(offset) = args.motion_offset {
        config.offset_s = offset;
    }
    if let Some(m) = &args.imu_to_camera {
        config.mount = mount_from_text(m)?;
    }
    if args.auto_align
        || (args.alignment.is_none()
            && args.motion_offset.is_none()
            && args.imu_to_camera.is_none())
    {
        eprintln!(
            "Fitting IMU time offset and mounting rotation from {} event windows...",
            windows.len()
        );
        let fit = motion::auto_align(
            &windows,
            origin_us,
            &camera,
            &imu,
            config.offset_s,
            args.alignment_range,
        )?;
        config.offset_s = fit.offset_s;
        config.mount = fit.mount;
        eprintln!(
            "Alignment: offset={:.4}s, focus gain={:.3}x over {} windows",
            fit.offset_s, fit.focus_gain, fit.windows
        );
        if fit.focus_gain < 1.05 {
            return Err("automatic alignment is inconclusive (<5% focus gain); use a manual alignment or a recording with more camera rotation".into());
        }
    }
    let matrix = config
        .mount
        .iter()
        .flatten()
        .map(|v| format!("{v:.9}"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "IMU -> camera: {matrix}\nIMU time = camera time {:+.4}s",
        config.offset_s
    );
    if let Some(p) = &args.save_alignment {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(p)?;
        writeln!(file, "{:.9}\n{matrix}", config.offset_s)?;
    }
    let window_us = (args.motion_window_ms * 1000.).round() as u64;
    let start_s = args
        .start_s
        .max(-config.offset_s + args.motion_window_ms * 0.001)
        .max(args.motion_window_ms * 0.001);
    let end_s = args
        .end_s
        .unwrap_or(duration_s)
        .min(duration_s)
        .min(imu.duration_s() - config.offset_s);
    if end_s <= start_s {
        return Err("camera and IMU have no overlap at this offset/interval".into());
    }
    let reader = WindowReader::new(path.clone(), format, endian, origin_us)?;
    let engine = Engine {
        reader,
        camera,
        imu,
        config,
        window_us,
        origin_us,
        duration_s,
    };
    if let Some(output) = args.export_mp4 {
        return export(engine, start_s, end_s, args.motion_fps, output);
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default().with_inner_size([1360., 1000.]),
        ..Default::default()
    };
    let title = format!(
        "Motion comparison — {}",
        path.file_name().unwrap().to_string_lossy()
    );
    eframe::run_native(
        &title,
        options,
        Box::new(move |_| {
            Ok(Box::new(ComparisonApp {
                engine,
                time_s: start_s,
                end_s,
                paused: true,
                speed: 0.5,
                last_tick: Instant::now(),
                dirty: true,
                textures: Vec::new(),
                status: String::new(),
                error: None,
                base_mount: config.mount,
                angles: [0.; 3],
            }))
        }),
    )?;
    Ok(())
}

fn scan(
    path: &PathBuf,
    format: EventFormat,
    endian: Endian,
) -> Result<(u64, f64, Vec<Vec<Event>>)> {
    let mut stream = open_event_stream(path, format, endian)?.stream;
    let mut origin = None;
    let mut last = 0;
    let mut slot = 0;
    let mut window = Vec::new();
    let mut windows = Vec::new();
    while let Some(record) = stream.next_record()? {
        if let EventRecord::Event(event) = record {
            let first = *origin.get_or_insert(event.timestamp_us);
            if event.timestamp_us < last {
                return Err("recording timestamps go backwards; unwrap timestamps before motion compensation".into());
            }
            last = event.timestamp_us;
            let t = event.timestamp_us - first;
            if t / 350_000 != slot {
                take_window(&mut window, &mut windows);
                slot = t / 350_000;
            }
            if t % 350_000 < 30_000 {
                window.push(event);
            }
        }
    }
    take_window(&mut window, &mut windows);
    let origin = origin.ok_or("recording contains no events")?;
    // Use windows distributed over the whole capture, not just its beginning.
    let stride = windows.len().div_ceil(40).max(1);
    Ok((
        origin,
        (last - origin) as f64 * 1e-6,
        windows.into_iter().step_by(stride).collect(),
    ))
}

fn take_window(window: &mut Vec<Event>, windows: &mut Vec<Vec<Event>>) {
    if window.len() >= 1500 {
        windows.push(
            window
                .iter()
                .step_by((window.len() / 5000).max(1))
                .take(5000)
                .copied()
                .collect(),
        );
    }
    window.clear();
}

struct WindowReader {
    path: PathBuf,
    format: EventFormat,
    endian: Endian,
    stream: Box<dyn EventStream>,
    pending: Option<Event>,
    events: VecDeque<Event>,
    last_start: u64,
    origin: u64,
}
impl WindowReader {
    fn new(path: PathBuf, format: EventFormat, endian: Endian, origin: u64) -> Result<Self> {
        let stream = open_event_stream(&path, format, endian)?.stream;
        Ok(Self {
            path,
            format,
            endian,
            stream,
            pending: None,
            events: VecDeque::new(),
            last_start: origin,
            origin,
        })
    }
    fn window(&mut self, start: u64, end: u64) -> Result<Vec<Event>> {
        if start < self.last_start {
            *self = Self::new(self.path.clone(), self.format, self.endian, self.origin)?;
        }
        self.last_start = start;
        loop {
            if let Some(e) = self.pending.take() {
                if e.timestamp_us > end {
                    self.pending = Some(e);
                    break;
                }
                if e.timestamp_us >= start {
                    self.events.push_back(e);
                }
            }
            match self.stream.next_record()? {
                Some(EventRecord::Event(e)) => self.pending = Some(e),
                Some(_) => {}
                None => break,
            }
        }
        while self.events.front().is_some_and(|e| e.timestamp_us < start) {
            self.events.pop_front();
        }
        Ok(self.events.iter().copied().collect())
    }
}

struct Engine {
    reader: WindowReader,
    camera: Camera,
    imu: ImuTrack,
    config: MotionConfig,
    window_us: u64,
    origin_us: u64,
    duration_s: f64,
}
impl Engine {
    fn frame(&mut self, time_s: f64) -> Result<(Vec<Event>, MotionFrame)> {
        let end = self.origin_us + (time_s * 1e6).round() as u64;
        let start = end.saturating_sub(self.window_us).max(self.origin_us);
        let events = self.reader.window(start, end)?;
        let frame = motion::compensate(
            &events,
            self.origin_us,
            start,
            end,
            &self.camera,
            &self.imu,
            self.config,
        );
        Ok((events, frame))
    }
}

const LABELS: [&str; 4] = [
    "All events | compensation OFF",
    "All events | compensation ON",
    "Foreground | compensation OFF",
    "Foreground | compensation ON",
];

fn panels(frame: &MotionFrame, camera: &Camera) -> (usize, usize, Vec<Vec<u8>>) {
    let width = camera.width.div_ceil(2);
    let height = camera.height.div_ceil(2);
    let mut output = Vec::new();
    for panel in 0..4 {
        let points = if panel % 2 == 0 {
            &frame.original
        } else {
            &frame.compensated
        };
        let mask = if panel == 2 {
            Some(&frame.foreground_off)
        } else if panel == 3 {
            Some(&frame.foreground_on)
        } else {
            None
        };
        let mut counts = vec![0u32; width * height];
        for (i, &p) in points.iter().enumerate() {
            if mask.is_some_and(|m| !m[i])
                || p[0] < 0.
                || p[1] < 0.
                || p[0] >= camera.width as f32
                || p[1] >= camera.height as f32
            {
                continue;
            }
            counts[p[1] as usize / 2 * width + p[0] as usize / 2] += 1;
        }
        let mut pixels = vec![0u8; width * height * 3];
        for (n, rgb) in counts.into_iter().zip(pixels.chunks_exact_mut(3)) {
            let level = (255. * (1. - (-(n as f32) / 2.).exp())) as u8;
            rgb.copy_from_slice(&[level, level, level]);
        }
        output.push(pixels);
    }
    (width, height, output)
}

struct ComparisonApp {
    engine: Engine,
    time_s: f64,
    end_s: f64,
    paused: bool,
    speed: f64,
    last_tick: Instant,
    dirty: bool,
    textures: Vec<TextureHandle>,
    status: String,
    error: Option<String>,
    base_mount: Rotation,
    angles: [f64; 3],
}

impl eframe::App for ComparisonApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("Camera motion compensation");
        ui.label("Same event window and filter settings on both sides. Both views use lens correction. Rotation only; translation/parallax can remain.");
        ui.horizontal(|ui| {
            if ui
                .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                .clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Space))
            {
                self.paused = !self.paused;
                self.last_tick = Instant::now();
            }
            if ui.button("Step").clicked() {
                self.time_s = (self.time_s + 0.04).min(self.end_s);
                self.paused = true;
                self.dirty = true;
            }
            let response = ui.add(
                egui::Slider::new(&mut self.time_s, 0.0..=self.engine.duration_s)
                    .text("recording seconds"),
            );
            if response.changed() {
                self.paused = true;
                self.dirty = true;
            }
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=2.0).text("speed"));
        });
        ui.horizontal(|ui| {
            ui.label("IMU offset (s)");
            self.dirty |= ui
                .add(
                    egui::DragValue::new(&mut self.engine.config.offset_s)
                        .speed(0.001)
                        .max_decimals(4),
                )
                .changed();
            let mut ms = self.engine.window_us as f64 / 1000.;
            if ui
                .add(egui::Slider::new(&mut ms, 5.0..=100.0).text("window ms"))
                .changed()
            {
                self.engine.window_us = (ms * 1000.) as u64;
                self.dirty = true;
            }
            self.dirty |= ui
                .add(
                    egui::Slider::new(&mut self.engine.config.threshold, 0.0..=0.4)
                        .text("motion threshold"),
                )
                .changed();
            self.dirty |= ui
                .add(
                    egui::Slider::new(&mut self.engine.config.min_component_cells, 1..=100)
                        .text("min cells"),
                )
                .changed();
        });
        ui.collapsing("Mounting adjustment", |ui| {
            ui.horizontal(|ui| {
                let mut changed = false;
                for axis in 0..3 {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.angles[axis], -30.0..=30.0)
                                .text(["X degrees", "Y degrees", "Z degrees"][axis]),
                        )
                        .changed();
                }
                if changed {
                    self.engine.config.mount = motion::multiply(
                        motion::rotation_vector(self.angles.map(f64::to_radians)),
                        self.base_mount,
                    );
                    self.dirty = true;
                }
            });
        });
        if !self.paused && self.last_tick.elapsed() >= Duration::from_millis(40) {
            self.time_s =
                (self.time_s + self.last_tick.elapsed().as_secs_f64() * self.speed).min(self.end_s);
            self.last_tick = Instant::now();
            self.dirty = true;
            if self.time_s >= self.end_s {
                self.paused = true;
            }
        }
        if self.dirty {
            self.dirty = false;
            match self.engine.frame(self.time_s) {
                Ok((events, frame)) => {
                    let (w, h, images) = panels(&frame, &self.engine.camera);
                    for (i, pixels) in images.iter().enumerate() {
                        let image = ColorImage::from_rgb([w, h], pixels);
                        if let Some(texture) = self.textures.get_mut(i) {
                            texture.set(image, TextureOptions::LINEAR);
                        } else {
                            self.textures.push(ui.ctx().load_texture(
                                LABELS[i],
                                image,
                                TextureOptions::LINEAR,
                            ));
                        }
                    }
                    let before = frame.foreground_off.iter().filter(|v| **v).count();
                    let after = frame.foreground_on.iter().filter(|v| **v).count();
                    self.status = if frame.imu_covered {
                        format!(
                            "{} input events · foreground off/on: {} / {} · alignment focus: {:.2}× (not a detection-accuracy score)",
                            events.len(),
                            before,
                            after,
                            frame.focus_on / frame.focus_off.max(1.)
                        )
                    } else {
                        "NO IMU COVERAGE — compensated foreground unavailable; top-right shows the unwarped input".into()
                    };
                    self.error = None;
                }
                Err(e) => {
                    self.error = Some(e.to_string());
                    self.paused = true;
                }
            }
        }
        ui.label(&self.status);
        if let Some(error) = &self.error {
            ui.colored_label(egui::Color32::RED, error);
        }
        let available = ui.available_size();
        let w = (available.x - 16.) / 2.;
        let h = ((available.y - 60.) / 2.).max(40.);
        let aspect = self.engine.camera.width as f32 / self.engine.camera.height as f32;
        let size = egui::vec2(w.min(h * aspect), h.min(w / aspect));
        for row in 0..2 {
            ui.columns(2, |cols| {
                for col in 0..2 {
                    let index = row * 2 + col;
                    cols[col].strong(LABELS[index]);
                    if let Some(texture) = self.textures.get(index) {
                        cols[col].image((texture.id(), size));
                    }
                }
            });
        }
        ui.ctx().request_repaint_after(Duration::from_millis(30));
    }
}

fn export(mut engine: Engine, start_s: f64, end_s: f64, fps: u32, path: PathBuf) -> Result<()> {
    let w = engine.camera.width.div_ceil(2);
    let h = engine.camera.height.div_ceil(2);
    let width = (w * 2).next_multiple_of(2);
    let height = (h * 2 + 96).next_multiple_of(2);
    let text_context = egui::Context::default();
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start ffmpeg for video export: {e}"))?;
    let mut input = child.stdin.take().unwrap();
    let mut rgb = vec![0u8; width * height * 3];
    let frames = ((end_s - start_s) * fps as f64).floor() as usize;
    let mut off_total = 0u64;
    let mut on_total = 0u64;
    let mut input_total = 0u64;
    let mut gains = Vec::new();
    let result = (|| -> Result<()> {
        for i in 0..frames {
            let time = start_s + i as f64 / fps as f64;
            let (events, frame) = engine.frame(time)?;
            if !frame.imu_covered {
                return Err(format!(
                    "IMU data unavailable at {time:.3}s; export stopped rather than extrapolating"
                )
                .into());
            }
            off_total += frame.foreground_off.iter().filter(|v| **v).count() as u64;
            on_total += frame.foreground_on.iter().filter(|v| **v).count() as u64;
            input_total += events.len() as u64;
            if frame.focus_off > 0. {
                gains.push(frame.focus_on / frame.focus_off);
            }
            let (_, _, images) = panels(&frame, &engine.camera);
            rgb.fill(0);
            for panel in 0..4 {
                let x = panel % 2 * w;
                let y = panel / 2 * (h + 32) + 32;
                for row in 0..h {
                    let dst = ((y + row) * width + x) * 3;
                    rgb[dst..dst + w * 3]
                        .copy_from_slice(&images[panel][row * w * 3..(row + 1) * w * 3]);
                }
            }
            let mut labels: Vec<_> = LABELS
                .iter()
                .enumerate()
                .map(|(i, s)| (12 + i % 2 * w, 5 + i / 2 * (h + 32), s.to_string()))
                .collect();
            labels.push((12,height-27,format!("Time {time:.2}s | rotation only | IMU offset {:+.4}s | foreground OFF / ON: {} / {}",
                engine.config.offset_s,frame.foreground_off.iter().filter(|v|**v).count(),frame.foreground_on.iter().filter(|v|**v).count())));
            draw_labels(&text_context, &mut rgb, width, height, &labels);
            input.write_all(&rgb)?;
            if i % fps as usize == 0 {
                eprintln!("Export {:.2}s / {:.2}s", time, end_s);
            }
        }
        Ok(())
    })();
    drop(input);
    let status = child.wait()?;
    result?;
    if !status.success() {
        return Err("ffmpeg video export failed".into());
    }
    gains.sort_by(f64::total_cmp);
    eprintln!(
        "Saved {} ({frames} frames). Input window events={input_total}, foreground off={off_total}, on={on_total}, median focus gain={:.3}x",
        path.display(),
        gains.get(gains.len() / 2).copied().unwrap_or(0.)
    );
    Ok(())
}

/// Reuse egui's existing font rasterizer for headless video labels. No dependency
/// on a particular system font or ffmpeg's optional drawtext build feature.
fn draw_labels(
    ctx: &egui::Context,
    rgb: &mut [u8],
    width: usize,
    height: usize,
    labels: &[(usize, usize, String)],
) {
    ctx.begin_pass(Default::default());
    let (galleys, atlas) = ctx.fonts_mut(|fonts| {
        let galleys: Vec<_> = labels
            .iter()
            .map(|(_, _, text)| {
                fonts.layout_no_wrap(
                    text.clone(),
                    egui::FontId::proportional(19.),
                    egui::Color32::WHITE,
                )
            })
            .collect();
        (galleys, fonts.image())
    });
    for ((ox, oy, _), galley) in labels.iter().zip(galleys) {
        for row in &galley.rows {
            for quad in
                row.visuals.mesh.vertices[row.visuals.glyph_vertex_range.clone()].chunks_exact(4)
            {
                let min = quad
                    .iter()
                    .fold(egui::pos2(f32::INFINITY, f32::INFINITY), |p, v| {
                        p.min(v.pos)
                    });
                let max = quad
                    .iter()
                    .fold(egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY), |p, v| {
                        p.max(v.pos)
                    });
                let uvmin = quad
                    .iter()
                    .fold(egui::pos2(f32::INFINITY, f32::INFINITY), |p, v| p.min(v.uv));
                let uvmax = quad
                    .iter()
                    .fold(egui::pos2(f32::NEG_INFINITY, f32::NEG_INFINITY), |p, v| {
                        p.max(v.uv)
                    });
                let x0 = (min.x + row.pos.x + *ox as f32).round() as i32;
                let y0 = (min.y + row.pos.y + *oy as f32).round() as i32;
                let qw = (max.x - min.x).round().max(1.) as usize;
                let qh = (max.y - min.y).round().max(1.) as usize;
                for y in 0..qh {
                    for x in 0..qw {
                        let px = x0 + x as i32;
                        let py = y0 + y as i32;
                        if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
                            continue;
                        }
                        let tx =
                            (uvmin.x + (x as f32 + 0.5) / qw as f32 * (uvmax.x - uvmin.x)) as usize;
                        let ty =
                            (uvmin.y + (y as f32 + 0.5) / qh as f32 * (uvmax.y - uvmin.y)) as usize;
                        if tx >= atlas.size[0] || ty >= atlas.size[1] {
                            continue;
                        }
                        let a = atlas.pixels[ty * atlas.size[0] + tx].a();
                        let i = (py as usize * width + px as usize) * 3;
                        rgb[i..i + 3].fill(a);
                    }
                }
            }
        }
    }
    let _ = ctx.end_pass();
}
