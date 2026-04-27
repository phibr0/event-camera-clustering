use crate::render::{fill_rgb_bytes, render_frame};
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use event_clustering::Event;
use event_clustering::algorithms::{
    ClusterDetection, RollingClusterTracker, RollingClusterTrackerConfig,
};
use event_clustering::filters::{
    EventFilterChain, PolarityFilterConfig, PolarityMode, StaticEventFilterConfig,
};
use event_clustering::parser::{Endian, EventFormat, open_event_stream};
use event_clustering::pipeline::EventPipeline;
use std::collections::VecDeque;
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewConfig {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) speed: f64,
    pub(crate) max_events_per_ui_update: usize,
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

pub(crate) fn view(
    path: PathBuf,
    tracker_config: RollingClusterTrackerConfig,
    polarity_filter_config: PolarityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
    view_config: ViewConfig,
    format: EventFormat,
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
                polarity_filter_config,
                static_filter_config,
                view_config,
                format,
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
    polarity_filter_config: PolarityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
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
    polarity_filter_config: PolarityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
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
        tracker_config: RollingClusterTrackerConfig,
        polarity_filter_config: PolarityFilterConfig,
        static_filter_config: StaticEventFilterConfig,
        view_config: ViewConfig,
        format: EventFormat,
        endian: Endian,
    ) -> Self {
        let controls = Arc::new(Mutex::new(ViewerControls {
            tracker_config,
            polarity_filter_config,
            static_filter_config,
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
            format,
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
            polarity_filter_config,
            static_filter_config,
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
        controls.polarity_filter_config = self.polarity_filter_config;
        controls.static_filter_config = self.static_filter_config;
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
                    .on_hover_text(
                        "Pause or resume file playback. Tracking state is preserved while paused.",
                    )
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
                .selected_text(match self.polarity_filter_config.mode {
                    PolarityMode::All => "all",
                    PolarityMode::Positive => "positive",
                    PolarityMode::Negative => "negative",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.polarity_filter_config.mode,
                        PolarityMode::All,
                        "all",
                    );
                    ui.selectable_value(
                        &mut self.polarity_filter_config.mode,
                        PolarityMode::Positive,
                        "positive only",
                    );
                    ui.selectable_value(
                        &mut self.polarity_filter_config.mode,
                        PolarityMode::Negative,
                        "negative only",
                    );
                })
                .response
                .on_hover_text("Choose which event polarity contributes to clustering. The viewer still displays both polarities.");
            ui.checkbox(&mut self.polarity_filter_config.invert_polarity, "invert labels")
                .on_hover_text("Swap positive/negative interpretation for both clustering labels and render colors. Enabled by default because this capture appeared swapped.");
        });

        ui.group(|ui| {
            ui.strong("Static Filter");
            ui.checkbox(&mut self.static_filter_config.enabled, "enabled")
                .on_hover_text("Suppress cells that keep firing in the same location over time. Useful for flickering LEDs in the background.");
            slider_u16(
                ui,
                &mut self.static_filter_config.cell_size,
                1..=32,
                "cell size",
                "Spatial bin size used to decide whether activity is static. Larger values suppress a wider area around flickering points.",
            );
            slider_u64(
                ui,
                &mut self.static_filter_config.stable_after_us,
                10_000..=2_000_000,
                "static after us",
                "A cell must stay active for at least this long before it is suppressed as static.",
            );
            slider_u32(
                ui,
                &mut self.static_filter_config.min_events,
                1..=10_000,
                "min events",
                "Minimum number of events in a cell before it can be considered static.",
            );
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

fn slider_u32(
    ui: &mut egui::Ui,
    value: &mut u32,
    range: std::ops::RangeInclusive<u32>,
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
    format: EventFormat,
    endian: Endian,
    controls: Arc<Mutex<ViewerControls>>,
    frame: Arc<Mutex<ViewerFrame>>,
    stop: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut state = WorkerState::new(path, format, endian, controls, frame);
        while !stop.load(Ordering::Relaxed) {
            state.tick();
        }
    })
}

struct WorkerState {
    path: PathBuf,
    format: EventFormat,
    endian: Endian,
    controls: Arc<Mutex<ViewerControls>>,
    frame: Arc<Mutex<ViewerFrame>>,
    pipeline: Option<EventPipeline<EventFilterChain, RollingClusterTracker>>,
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
        format: EventFormat,
        endian: Endian,
        controls: Arc<Mutex<ViewerControls>>,
        frame: Arc<Mutex<ViewerFrame>>,
    ) -> Self {
        let controls_snapshot = *controls.lock().expect("viewer controls lock poisoned");
        let mut state = Self {
            path,
            format,
            endian,
            controls,
            frame,
            pipeline: None,
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
        self.pipeline = None;
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

        match open_event_stream(&self.path, self.format, self.endian) {
            Ok(opened) => {
                if opened.header.evt_version.as_deref() != Some("2.0")
                    && opened.format == EventFormat::Evt2
                {
                    self.error = Some(format!(
                        "RAW header evt version is {:?}; decoding as EVT 2.0",
                        opened.header.evt_version
                    ));
                }
                match (
                    EventFilterChain::new(
                        controls.polarity_filter_config,
                        controls.static_filter_config,
                    ),
                    RollingClusterTracker::new(controls.tracker_config),
                ) {
                    (Ok(filters), Ok(tracker)) => {
                        self.pipeline = Some(EventPipeline::new(opened.stream, filters, tracker));
                    }
                    (Err(error), _) | (_, Err(error)) => {
                        self.error = Some(error.to_string());
                        self.finished = true;
                    }
                }
            }
            Err(error) => {
                self.error = Some(error.to_string());
                self.finished = true;
            }
        }
        self.publish_frame(controls);
    }

    fn process_events(&mut self, controls: ViewerControls) {
        let Some(pipeline) = self.pipeline.as_mut() else {
            return;
        };

        for _ in 0..controls.view_config.max_events_per_ui_update {
            if let Err(error) = pipeline.algorithm_mut().set_config(controls.tracker_config) {
                self.error = Some(error.to_string());
                self.finished = true;
                self.publish_frame(controls);
                return;
            }
            if let Err(error) = pipeline.filters_mut().set_config(
                controls.polarity_filter_config,
                controls.static_filter_config,
            ) {
                self.error = Some(error.to_string());
                self.finished = true;
                self.publish_frame(controls);
                return;
            }

            let processed = match pipeline.next_event() {
                Ok(Some(processed)) => processed,
                Ok(None) => {
                    self.finished = true;
                    self.publish_frame(controls);
                    return;
                }
                Err(error) => {
                    self.error = Some(error.to_string());
                    self.finished = true;
                    self.publish_frame(controls);
                    return;
                }
            };

            self.processed_events += 1;
            let event = processed.event;
            self.render_events.push_back(event);
            drop_old_render_events(
                &mut self.render_events,
                event.timestamp_us,
                controls.tracker_config.window_us,
            );

            for detection in processed.outputs {
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
            controls.polarity_filter_config.invert_polarity,
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
