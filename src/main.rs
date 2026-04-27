use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use event_clustering::Event;
use event_clustering::algorithms::{
    ClusterDetection, PolarityFilter, RollingClusterTracker, RollingClusterTrackerConfig,
};
use event_clustering::evt2::{DecodedEvt2, Endian, Evt2Reader};
use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse(env::args().skip(1))?;

    match cli.command {
        Command::TrackBall {
            path,
            config,
            endian,
            max_detections,
        } => track_ball(path, config, endian, max_detections),
        Command::View {
            path,
            tracker_config,
            view_config,
            endian,
        } => view(path, tracker_config, view_config, endian),
        Command::Help => {
            print_usage();
            Ok(())
        }
    }
}

fn view(
    path: PathBuf,
    tracker_config: RollingClusterTrackerConfig,
    view_config: ViewConfig,
    endian: Endian,
) -> Result<(), Box<dyn Error>> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([view_config.width as f32 + 280.0, view_config.height as f32]),
        ..Default::default()
    };

    eframe::run_native(
        "event-clustering viewer",
        native_options,
        Box::new(move |_cc| {
            Ok(Box::new(ViewerApp::new(
                path,
                tracker_config,
                view_config,
                endian,
            )))
        }),
    )?;

    Ok(())
}

fn drop_old_render_events(events: &mut VecDeque<Event>, now_us: u64, window_us: u64) {
    let min_timestamp_us = now_us.saturating_sub(window_us);
    while events
        .front()
        .is_some_and(|event| event.timestamp_us < min_timestamp_us)
    {
        events.pop_front();
    }
}

#[derive(Clone, Copy)]
struct ViewerControls {
    tracker_config: RollingClusterTrackerConfig,
    view_config: ViewConfig,
    paused: bool,
    restart_generation: u64,
}

struct ViewerFrame {
    rgb_buffer: Vec<u8>,
    generation: u64,
    frame_timestamp_us: u64,
    processed_events: u64,
    latest_detection: Option<ClusterDetection>,
    finished: bool,
    error: Option<String>,
}

struct ViewerApp {
    controls: Arc<Mutex<ViewerControls>>,
    frame: Arc<Mutex<ViewerFrame>>,
    stop_worker: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    tracker_config: RollingClusterTrackerConfig,
    view_config: ViewConfig,
    paused: bool,
    texture: Option<TextureHandle>,
    rgb_buffer: Vec<u8>,
    seen_frame_generation: u64,
    restart_generation: u64,
    frame_timestamp_us: u64,
    processed_events: u64,
    latest_detection: Option<ClusterDetection>,
    finished: bool,
    error: Option<String>,
}

impl ViewerApp {
    fn new(
        path: PathBuf,
        mut tracker_config: RollingClusterTrackerConfig,
        view_config: ViewConfig,
        endian: Endian,
    ) -> Self {
        tracker_config.invert_polarity = true;
        let controls = Arc::new(Mutex::new(ViewerControls {
            tracker_config,
            view_config,
            paused: false,
            restart_generation: 0,
        }));
        let frame = Arc::new(Mutex::new(ViewerFrame {
            rgb_buffer: vec![0; view_config.width * view_config.height * 3],
            generation: 0,
            frame_timestamp_us: 0,
            processed_events: 0,
            latest_detection: None,
            finished: false,
            error: None,
        }));
        let stop_worker = Arc::new(AtomicBool::new(false));
        let worker = Some(spawn_viewer_worker(
            path,
            endian,
            controls.clone(),
            frame.clone(),
            stop_worker.clone(),
        ));

        Self {
            controls,
            frame,
            stop_worker,
            worker,
            tracker_config,
            view_config,
            paused: false,
            texture: None,
            rgb_buffer: vec![0; view_config.width * view_config.height * 3],
            seen_frame_generation: 0,
            restart_generation: 0,
            frame_timestamp_us: 0,
            processed_events: 0,
            latest_detection: None,
            finished: false,
            error: None,
        }
    }

    fn pull_frame(&mut self, ctx: &egui::Context) {
        let frame = self.frame.lock().expect("viewer frame lock poisoned");
        self.frame_timestamp_us = frame.frame_timestamp_us;
        self.processed_events = frame.processed_events;
        self.latest_detection = frame.latest_detection.clone();
        self.finished = frame.finished;
        self.error = frame.error.clone();

        if frame.generation == self.seen_frame_generation {
            return;
        }

        self.seen_frame_generation = frame.generation;
        self.rgb_buffer.copy_from_slice(&frame.rgb_buffer);
        let image = ColorImage::from_rgb(
            [self.view_config.width, self.view_config.height],
            &self.rgb_buffer,
        );

        if let Some(texture) = self.texture.as_mut() {
            texture.set(image, TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture("events", image, TextureOptions::NEAREST));
        }
    }

    fn push_controls(&self) {
        let mut controls = self.controls.lock().expect("viewer controls lock poisoned");
        controls.tracker_config = self.tracker_config;
        controls.view_config = self.view_config;
        controls.paused = self.paused;
        controls.restart_generation = self.restart_generation;
    }

    fn ui_controls(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing.y = 5.0;
        ui.heading("Ball Tracker");
        ui.label(
            egui::RichText::new("Hover controls for tuning hints.")
                .small()
                .weak(),
        );

        ui.group(|ui| {
            ui.strong("Playback");
            ui.horizontal(|ui| {
                if ui
                    .button(if self.paused { "Play" } else { "Pause" })
                    .on_hover_text("Pause or resume file playback. Tracking state is preserved while paused.")
                    .clicked()
                {
                    self.paused = !self.paused;
                }
                if ui
                    .button("Restart")
                    .on_hover_text("Rewind the RAW file and clear tracker history.")
                    .clicked()
                {
                    self.restart_generation += 1;
                }
            });
            slider_f64(
                ui,
                &mut self.view_config.speed,
                0.0..=5.0,
                "speed",
                "Playback speed relative to timestamps. 1.0 is real time; 0 runs as fast as the worker can process.",
            );
        });

        ui.group(|ui| {
            ui.strong("Cluster Window");
            slider_u64(
                ui,
                &mut self.tracker_config.window_us,
                1_000..=100_000,
                "window us",
                "Amount of recent event history used per detection. Smaller is more responsive; larger is more stable but creates longer motion trails.",
            );
            slider_u64(
                ui,
                &mut self.tracker_config.step_us,
                500..=30_000,
                "step us",
                "How often the tracker emits detections in event time. Smaller gives smoother updates but more CPU work.",
            );
            slider_u16(
                ui,
                &mut self.tracker_config.cell_size,
                1..=16,
                "cell size",
                "Spatial bin size before connected-component clustering. Larger values bridge gaps and reduce noise, but reduce precision.",
            );
        });

        ui.group(|ui| {
            ui.strong("Cluster Filters");
            slider_usize(
                ui,
                &mut self.tracker_config.min_events,
                1..=10_000,
                "min events",
                "Reject clusters with fewer events. Increase to suppress noise; decrease if the ball is faint or far away.",
            );
            slider_usize(
                ui,
                &mut self.tracker_config.min_cells,
                1..=200,
                "min cells",
                "Reject clusters occupying too few spatial cells. Useful for ignoring hot pixels and single-point noise.",
            );
            slider_u16(
                ui,
                &mut self.tracker_config.max_bbox_width,
                1..=640,
                "max bbox w",
                "Reject clusters wider than this. Lower it to ignore large background motion or object trails.",
            );
            slider_u16(
                ui,
                &mut self.tracker_config.max_bbox_height,
                1..=480,
                "max bbox h",
                "Reject clusters taller than this. Lower it to ignore large vertical motion or full-scene flicker.",
            );
        });

        ui.group(|ui| {
            ui.strong("Polarity");
            egui::ComboBox::from_label("use events")
                .selected_text(match self.tracker_config.polarity_filter {
                    PolarityFilter::All => "all",
                    PolarityFilter::Positive => "positive",
                    PolarityFilter::Negative => "negative",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.tracker_config.polarity_filter,
                        PolarityFilter::All,
                        "all",
                    );
                    ui.selectable_value(
                        &mut self.tracker_config.polarity_filter,
                        PolarityFilter::Positive,
                        "positive only",
                    );
                    ui.selectable_value(
                        &mut self.tracker_config.polarity_filter,
                        PolarityFilter::Negative,
                        "negative only",
                    );
                })
                .response
                .on_hover_text("Choose which event polarity contributes to clustering. The viewer still displays both polarities.");
            ui.checkbox(
                &mut self.tracker_config.invert_polarity,
                "invert labels",
            )
            .on_hover_text("Swap positive/negative interpretation for both clustering labels and render colors. Enabled by default because this capture appeared swapped.");
        });

        ui.group(|ui| {
            ui.strong("Performance");
            slider_usize(
                ui,
                &mut self.view_config.max_events_per_ui_update,
                500..=50_000,
                "events/tick",
                "Worker-thread event budget per UI update. Increase for faster playback; decrease if CPU usage is too high.",
            );
        });

        ui.group(|ui| {
            ui.strong("Status");
            ui.label(format!("time: {} us", self.frame_timestamp_us));
            ui.label(format!("events: {}", self.processed_events));
            if self.finished {
                ui.label(egui::RichText::new("finished").weak());
            }
            if let Some(detection) = &self.latest_detection {
                ui.label(format!(
                    "centroid: {:.1}, {:.1}",
                    detection.centroid_x, detection.centroid_y
                ));
                ui.label(format!(
                    "bbox: ({}, {})-({}, {})",
                    detection.bbox.min_x,
                    detection.bbox.min_y,
                    detection.bbox.max_x,
                    detection.bbox.max_y
                ));
                ui.label(format!("cluster events: {}", detection.event_count));
            } else {
                ui.label(egui::RichText::new("no cluster yet").weak());
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::YELLOW, error);
            }
        });
    }
}

fn slider_u64(
    ui: &mut egui::Ui,
    value: &mut u64,
    range: std::ops::RangeInclusive<u64>,
    label: &str,
    tooltip: &str,
) {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tooltip);
}

fn slider_u16(
    ui: &mut egui::Ui,
    value: &mut u16,
    range: std::ops::RangeInclusive<u16>,
    label: &str,
    tooltip: &str,
) {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tooltip);
}

fn slider_usize(
    ui: &mut egui::Ui,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
    label: &str,
    tooltip: &str,
) {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tooltip);
}

fn slider_f64(
    ui: &mut egui::Ui,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    label: &str,
    tooltip: &str,
) {
    ui.add(egui::Slider::new(value, range).text(label))
        .on_hover_text(tooltip);
}

impl Drop for ViewerApp {
    fn drop(&mut self) {
        self.stop_worker.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.push_controls();
        self.pull_frame(&ctx);

        ui.horizontal(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(260.0, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(260.0);
                    self.ui_controls(ui);
                },
            );
            ui.separator();
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                if let Some(texture) = &self.texture {
                    ui.image((
                        texture.id(),
                        egui::vec2(
                            self.view_config.width as f32,
                            self.view_config.height as f32,
                        ),
                    ));
                } else {
                    ui.allocate_space(egui::vec2(
                        self.view_config.width as f32,
                        self.view_config.height as f32,
                    ));
                }
            });
        });

        ctx.request_repaint_after(Duration::from_millis(1));
    }
}

fn spawn_viewer_worker(
    path: PathBuf,
    endian: Endian,
    controls: Arc<Mutex<ViewerControls>>,
    frame: Arc<Mutex<ViewerFrame>>,
    stop: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut state = WorkerState::new(path, endian, controls, frame);
        while !stop.load(Ordering::Relaxed) {
            state.tick();
        }
    })
}

struct WorkerState {
    path: PathBuf,
    endian: Endian,
    controls: Arc<Mutex<ViewerControls>>,
    frame: Arc<Mutex<ViewerFrame>>,
    reader: Option<Evt2Reader<BufReader<File>>>,
    tracker: RollingClusterTracker,
    render_events: VecDeque<Event>,
    latest_detection: Option<ClusterDetection>,
    buffer: Vec<u32>,
    rgb_buffer: Vec<u8>,
    next_frame_us: Option<u64>,
    playback_start_us: Option<u64>,
    playback_start: Instant,
    frame_timestamp_us: u64,
    processed_events: u64,
    finished: bool,
    error: Option<String>,
    restart_generation: u64,
    frame_generation: u64,
}

impl WorkerState {
    fn new(
        path: PathBuf,
        endian: Endian,
        controls: Arc<Mutex<ViewerControls>>,
        frame: Arc<Mutex<ViewerFrame>>,
    ) -> Self {
        let controls_snapshot = *controls.lock().expect("viewer controls lock poisoned");
        let mut state = Self {
            path,
            endian,
            controls,
            frame,
            reader: None,
            tracker: RollingClusterTracker::new(controls_snapshot.tracker_config),
            render_events: VecDeque::new(),
            latest_detection: None,
            buffer: vec![
                0;
                controls_snapshot.view_config.width * controls_snapshot.view_config.height
            ],
            rgb_buffer: vec![
                0;
                controls_snapshot.view_config.width
                    * controls_snapshot.view_config.height
                    * 3
            ],
            next_frame_us: None,
            playback_start_us: None,
            playback_start: Instant::now(),
            frame_timestamp_us: 0,
            processed_events: 0,
            finished: false,
            error: None,
            restart_generation: controls_snapshot.restart_generation,
            frame_generation: 0,
        };
        state.reset_stream(controls_snapshot);
        state
    }

    fn tick(&mut self) {
        let controls = *self.controls.lock().expect("viewer controls lock poisoned");
        if controls.restart_generation != self.restart_generation {
            self.restart_generation = controls.restart_generation;
            self.reset_stream(controls);
        }

        if controls.paused || self.finished || !self.frame_is_due(controls) {
            thread::sleep(Duration::from_millis(1));
            return;
        }

        self.process_events(controls);
    }

    fn reset_stream(&mut self, controls: ViewerControls) {
        self.reader = None;
        self.tracker = RollingClusterTracker::new(controls.tracker_config);
        self.render_events.clear();
        self.latest_detection = None;
        self.next_frame_us = None;
        self.playback_start_us = None;
        self.playback_start = Instant::now();
        self.frame_timestamp_us = 0;
        self.processed_events = 0;
        self.finished = false;
        self.error = None;
        self.buffer
            .resize(controls.view_config.width * controls.view_config.height, 0);
        self.rgb_buffer.resize(
            controls.view_config.width * controls.view_config.height * 3,
            0,
        );

        match Evt2Reader::from_path(&self.path, self.endian) {
            Ok((header, reader)) => {
                if header.evt_version.as_deref() != Some("2.0") {
                    self.error = Some(format!(
                        "RAW header evt version is {:?}; decoding as EVT 2.0",
                        header.evt_version
                    ));
                }
                self.reader = Some(reader);
            }
            Err(error) => {
                self.error = Some(error.to_string());
                self.finished = true;
            }
        }
        self.publish_frame(controls);
    }

    fn process_events(&mut self, controls: ViewerControls) {
        let Some(reader) = self.reader.as_mut() else {
            return;
        };

        for _ in 0..controls.view_config.max_events_per_ui_update {
            let decoded = match reader.next() {
                Some(Ok(decoded)) => decoded,
                Some(Err(error)) => {
                    self.error = Some(error.to_string());
                    self.finished = true;
                    self.publish_frame(controls);
                    return;
                }
                None => {
                    self.finished = true;
                    self.publish_frame(controls);
                    return;
                }
            };

            let DecodedEvt2::Event(event) = decoded else {
                continue;
            };

            self.processed_events += 1;
            self.render_events.push_back(event);
            drop_old_render_events(
                &mut self.render_events,
                event.timestamp_us,
                controls.tracker_config.window_us,
            );

            self.tracker.set_config(controls.tracker_config);
            for detection in self.tracker.process_event(event) {
                self.latest_detection = Some(detection);
            }

            let next_frame = self.next_frame_us.get_or_insert(event.timestamp_us);
            self.playback_start_us.get_or_insert(*next_frame);
            if event.timestamp_us >= *next_frame {
                self.frame_timestamp_us = *next_frame;
                self.next_frame_us = Some(*next_frame + controls.tracker_config.step_us);
                self.publish_frame(controls);
                return;
            }
        }
    }

    fn frame_is_due(&self, controls: ViewerControls) -> bool {
        if controls.view_config.speed <= 0.0 {
            return true;
        }

        let Some(playback_start_us) = self.playback_start_us else {
            return true;
        };

        let elapsed_us = self.frame_timestamp_us.saturating_sub(playback_start_us) as f64
            / controls.view_config.speed;
        self.playback_start.elapsed() >= Duration::from_micros(elapsed_us as u64)
    }

    fn publish_frame(&mut self, controls: ViewerControls) {
        render_frame(
            &mut self.buffer,
            controls.view_config.width,
            controls.view_config.height,
            &self.render_events,
            self.latest_detection.as_ref(),
            controls.tracker_config.invert_polarity,
        );
        fill_rgb_bytes(&self.buffer, &mut self.rgb_buffer);
        self.frame_generation += 1;

        let mut frame = self.frame.lock().expect("viewer frame lock poisoned");
        if frame.rgb_buffer.len() != self.rgb_buffer.len() {
            frame.rgb_buffer.resize(self.rgb_buffer.len(), 0);
        }
        frame.rgb_buffer.copy_from_slice(&self.rgb_buffer);
        frame.generation = self.frame_generation;
        frame.frame_timestamp_us = self.frame_timestamp_us;
        frame.processed_events = self.processed_events;
        frame.latest_detection = self.latest_detection.clone();
        frame.finished = self.finished;
        frame.error = self.error.clone();
    }
}

fn fill_rgb_bytes(buffer: &[u32], bytes: &mut [u8]) {
    for (color, rgb) in buffer.iter().zip(bytes.chunks_exact_mut(3)) {
        rgb[0] = ((color >> 16) & 0xff) as u8;
        rgb[1] = ((color >> 8) & 0xff) as u8;
        rgb[2] = (color & 0xff) as u8;
    }
}

fn render_frame(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    events: &VecDeque<Event>,
    detection: Option<&ClusterDetection>,
    invert_polarity: bool,
) {
    buffer.fill(0x000000);

    for event in events {
        let x = usize::from(event.x);
        let y = usize::from(event.y);
        if x >= width || y >= height {
            continue;
        }

        let index = y * width + x;
        let polarity = if invert_polarity {
            !event.polarity
        } else {
            event.polarity
        };
        buffer[index] = if polarity { 0xffffff } else { 0x3060ff };
    }

    if let Some(detection) = detection {
        draw_bbox(buffer, width, height, detection, 0xff2020);
        draw_cross(
            buffer,
            width,
            height,
            detection.centroid_x.round() as i32,
            detection.centroid_y.round() as i32,
            6,
            0xffff00,
        );
    }
}

fn draw_bbox(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    detection: &ClusterDetection,
    color: u32,
) {
    let min_x = usize::from(detection.bbox.min_x).min(width.saturating_sub(1));
    let min_y = usize::from(detection.bbox.min_y).min(height.saturating_sub(1));
    let max_x = usize::from(detection.bbox.max_x).min(width.saturating_sub(1));
    let max_y = usize::from(detection.bbox.max_y).min(height.saturating_sub(1));

    for x in min_x..=max_x {
        set_pixel(buffer, width, height, x as i32, min_y as i32, color);
        set_pixel(buffer, width, height, x as i32, max_y as i32, color);
    }
    for y in min_y..=max_y {
        set_pixel(buffer, width, height, min_x as i32, y as i32, color);
        set_pixel(buffer, width, height, max_x as i32, y as i32, color);
    }
}

fn draw_cross(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    radius: i32,
    color: u32,
) {
    for offset in -radius..=radius {
        set_pixel(buffer, width, height, x + offset, y, color);
        set_pixel(buffer, width, height, x, y + offset, color);
    }
}

fn set_pixel(buffer: &mut [u32], width: usize, height: usize, x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x < width && y < height {
        buffer[y * width + x] = color;
    }
}

fn track_ball(
    path: PathBuf,
    config: RollingClusterTrackerConfig,
    endian: Endian,
    max_detections: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let (header, reader) = Evt2Reader::from_path(path, endian)?;
    if header.evt_version.as_deref() != Some("2.0") {
        eprintln!(
            "warning: RAW header evt version is {:?}, decoding as EVT 2.0",
            header.evt_version
        );
    }

    let mut tracker = RollingClusterTracker::new(config);
    let mut event_count = 0_u64;
    let mut detection_count = 0_u64;

    for decoded in reader {
        let decoded = decoded?;
        let DecodedEvt2::Event(event) = decoded else {
            continue;
        };

        event_count += 1;
        for detection in tracker.process_event(event) {
            detection_count += 1;
            print_detection(&detection);
            if max_detections.is_some_and(|max| detection_count >= max) {
                eprintln!("processed_events={event_count} detections={detection_count}");
                return Ok(());
            }
        }
    }

    if let Some(detection) = tracker.finish() {
        detection_count += 1;
        print_detection(&detection);
    }

    eprintln!("processed_events={event_count} detections={detection_count}");
    Ok(())
}

fn print_detection(detection: &ClusterDetection) {
    println!(
        "t={}us window={}..{}us centroid=({:.2},{:.2}) bbox=({},{})->({},{}) events={} confidence={:.3}",
        detection.timestamp_us,
        detection.window_start_us,
        detection.window_end_us,
        detection.centroid_x,
        detection.centroid_y,
        detection.bbox.min_x,
        detection.bbox.min_y,
        detection.bbox.max_x,
        detection.bbox.max_y,
        detection.event_count,
        detection.confidence,
    );
}

#[derive(Debug)]
struct Cli {
    command: Command,
}

#[derive(Debug)]
enum Command {
    TrackBall {
        path: PathBuf,
        config: RollingClusterTrackerConfig,
        endian: Endian,
        max_detections: Option<u64>,
    },
    View {
        path: PathBuf,
        tracker_config: RollingClusterTrackerConfig,
        view_config: ViewConfig,
        endian: Endian,
    },
    Help,
}

#[derive(Debug, Clone, Copy)]
struct ViewConfig {
    width: usize,
    height: usize,
    speed: f64,
    max_events_per_ui_update: usize,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            speed: 1.0,
            max_events_per_ui_update: 5_000,
        }
    }
}

impl Cli {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut args = args.into_iter();
        let Some(first) = args.next() else {
            return Ok(Self {
                command: Command::Help,
            });
        };

        if first == "--help" || first == "-h" || first == "help" {
            return Ok(Self {
                command: Command::Help,
            });
        }

        let known_command = first == "track-ball" || first == "view";
        let (path, command) = if known_command {
            let path = args
                .next()
                .ok_or_else(|| format!("missing RAW file path after {first}"))?;
            (PathBuf::from(path), first)
        } else {
            let command = args.next().unwrap_or_else(|| "track-ball".to_owned());
            (PathBuf::from(first), command)
        };

        if command != "track-ball" && command != "view" {
            return Err(format!("unknown command: {command}"));
        }

        let mut config = RollingClusterTrackerConfig::default();
        let mut view_config = ViewConfig::default();
        let mut endian = Endian::Little;
        let mut max_detections = None;

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--window-us" => config.window_us = parse_next(&mut args, &flag)?,
                "--step-us" => config.step_us = parse_next(&mut args, &flag)?,
                "--cell-size" => config.cell_size = parse_next(&mut args, &flag)?,
                "--min-events" => config.min_events = parse_next(&mut args, &flag)?,
                "--min-cells" => config.min_cells = parse_next(&mut args, &flag)?,
                "--max-bbox-width" => config.max_bbox_width = parse_next(&mut args, &flag)?,
                "--max-bbox-height" => config.max_bbox_height = parse_next(&mut args, &flag)?,
                "--polarity" => {
                    let value: String = parse_next(&mut args, &flag)?;
                    config.polarity_filter = parse_polarity_filter(&value)?;
                }
                "--invert-polarity" => config.invert_polarity = true,
                "--raw-polarity" | "--no-invert-polarity" => config.invert_polarity = false,
                "--max-detections" => max_detections = Some(parse_next(&mut args, &flag)?),
                "--width" => view_config.width = parse_next(&mut args, &flag)?,
                "--height" => view_config.height = parse_next(&mut args, &flag)?,
                "--speed" => view_config.speed = parse_next(&mut args, &flag)?,
                "--events-per-tick" => {
                    view_config.max_events_per_ui_update = parse_next(&mut args, &flag)?
                }
                "--endian" => {
                    let value: String = parse_next(&mut args, &flag)?;
                    endian = match value.as_str() {
                        "little" => Endian::Little,
                        "big" => Endian::Big,
                        _ => return Err("--endian must be little or big".to_owned()),
                    };
                }
                "--help" | "-h" => {
                    return Ok(Self {
                        command: Command::Help,
                    });
                }
                _ => return Err(format!("unknown flag: {flag}")),
            }
        }

        let command = if command == "view" {
            Command::View {
                path,
                tracker_config: config,
                view_config,
                endian,
            }
        } else {
            Command::TrackBall {
                path,
                config,
                endian,
                max_detections,
            }
        };

        Ok(Self { command })
    }
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))?
        .parse::<T>()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}

fn parse_polarity_filter(value: &str) -> Result<PolarityFilter, String> {
    match value {
        "all" => Ok(PolarityFilter::All),
        "positive" | "on" | "+" => Ok(PolarityFilter::Positive),
        "negative" | "off" | "-" => Ok(PolarityFilter::Negative),
        _ => Err("--polarity must be all, positive/on/+, or negative/off/-".to_owned()),
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n  cargo run -- <raw-file> track-ball [options]\n  cargo run -- <raw-file> view [options]\n  cargo run -- track-ball <raw-file> [options]\n  cargo run -- view <raw-file> [options]\n\ntracking options:\n  --window-us <us>          rolling window size, default 20000\n  --step-us <us>            output interval, default 5000\n  --cell-size <pixels>      spatial bin size, default 2\n  --min-events <count>      minimum cluster size, default 20\n  --min-cells <count>       minimum occupied cells, default 3\n  --max-bbox-width <px>     reject wider clusters, default 200\n  --max-bbox-height <px>    reject taller clusters, default 200\n  --polarity <mode>         cluster all, positive/on/+, or negative/off/- events; default all\n  --invert-polarity         invert positive/negative labels\n  --raw-polarity            use raw EVT2 ON/OFF polarity labels\n\nview options:\n  --width <px>              viewer width, default 640\n  --height <px>             viewer height, default 480\n  --speed <factor>          playback speed, default 1.0; use 0 for fastest\n  --events-per-tick <n>     parser work budget per UI tick, default 5000\n\nother options:\n  --max-detections <count>  stop text output after this many detections\n  --endian <little|big>     EVT2 word endianness, default little"
    );
}
