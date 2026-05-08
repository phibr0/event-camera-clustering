use crate::render::{RenderEvent, fill_rgb_bytes, render_frame};
use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use eframe::{egui_glow, glow};
use event_clustering::Event;
use event_clustering::algorithms::{
    ClusterDetection, RollingClusterTracker, RollingClusterTrackerConfig,
};
use event_clustering::ball::{
    BallDiameterSource, BallPathEstimator, BallProjectionConfig, BallTrackEstimate,
};
use event_clustering::filter::{
    BackgroundActivityFilterConfig, ConfiguredEventFilters, PolarityFilterConfig, PolarityMode,
    StaticEventFilterConfig,
};
use event_clustering::parser::{
    Endian, EventFormat, EventRecord, open_event_stream, read_event_header,
};
use event_clustering::pipeline::EventPipeline;
use std::collections::VecDeque;
use std::error::Error;
use std::path::PathBuf;
use std::slice;
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
    background_activity_filter_config: BackgroundActivityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
    mut view_config: ViewConfig,
    format: EventFormat,
    endian: Endian,
) -> Result<(), Box<dyn Error>> {
    let time_range = recording_time_range(&path, format, endian).ok().flatten();
    if view_config.uses_default_size() {
        if let Ok(header) = read_event_header(&path) {
            if let (Some(width), Some(height)) = (header.width, header.height) {
                view_config.width = width;
                view_config.height = height;
            }
        }
    }

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        multisampling: 4,
        viewport: egui::ViewportBuilder::default().with_inner_size([
            view_config.width as f32 + 280.0,
            view_config.height as f32 + 340.0,
        ]),
        ..Default::default()
    };

    eframe::run_native(
        "event-clustering viewer",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(ViewerApp::new(
                cc,
                path,
                tracker_config,
                polarity_filter_config,
                background_activity_filter_config,
                static_filter_config,
                view_config,
                format,
                endian,
                time_range,
            )))
        }),
    )?;

    Ok(())
}

fn recording_time_range(
    path: impl AsRef<std::path::Path>,
    format: EventFormat,
    endian: Endian,
) -> Result<Option<(u64, u64)>, Box<dyn Error>> {
    let mut stream = open_event_stream(path, format, endian)?.stream;
    let mut first = None;
    let mut last = None;

    while let Some(record) = stream.next_record()? {
        if let EventRecord::Event(event) = record {
            first.get_or_insert(event.timestamp_us);
            last = Some(event.timestamp_us);
        }
    }

    Ok(first.zip(last))
}

impl ViewConfig {
    fn uses_default_size(&self) -> bool {
        self.width == Self::default().width && self.height == Self::default().height
    }
}

fn drop_old_render_events(events: &mut VecDeque<RenderEvent>, now_us: u64, window_us: u64) {
    let min_timestamp_us = now_us.saturating_sub(window_us);
    while events
        .front()
        .is_some_and(|event| event.event.timestamp_us < min_timestamp_us)
    {
        events.pop_front();
    }
}

fn format_duration_us(duration_us: u64) -> String {
    let total_ms = duration_us / 1_000;
    let minutes = total_ms / 60_000;
    let seconds = (total_ms / 1_000) % 60;
    let millis = total_ms % 1_000;
    format!("{minutes:02}:{seconds:02}.{millis:03}")
}

#[derive(Clone, Copy)]
struct ViewerControls {
    tracker_config: RollingClusterTrackerConfig,
    polarity_filter_config: PolarityFilterConfig,
    background_activity_filter_config: BackgroundActivityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
    ball_projection_config: BallProjectionConfig,
    view_config: ViewConfig,
    show_filtered_events: bool,
    paused: bool,
    restart_generation: u64,
    seek_generation: u64,
    seek_target_us: u64,
}

struct ViewerFrame {
    rgb_buffer: Vec<u8>,
    generation: u64,
    frame_timestamp_us: u64,
    processed_events: u64,
    latest_detection: Option<ClusterDetection>,
    latest_ball_estimate: Option<BallTrackEstimate>,
    ball_path: Vec<BallTrackEstimate>,
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
    background_activity_filter_config: BackgroundActivityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
    ball_projection_config: BallProjectionConfig,
    view_config: ViewConfig,
    show_filtered_events: bool,
    paused: bool,
    texture: Option<TextureHandle>,
    rgb_buffer: Vec<u8>,
    seen_frame_generation: u64,
    restart_generation: u64,
    seek_generation: u64,
    scrubber_timestamp_us: u64,
    timeline_start_us: u64,
    timeline_end_us: u64,
    frame_timestamp_us: u64,
    processed_events: u64,
    latest_detection: Option<ClusterDetection>,
    latest_ball_estimate: Option<BallTrackEstimate>,
    ball_path: Vec<BallTrackEstimate>,
    ball_3d_view: Arc<Mutex<Ball3dView>>,
    ball_3d_camera: Ball3dCamera,
    finished: bool,
    error: Option<String>,
}

impl ViewerApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        path: PathBuf,
        tracker_config: RollingClusterTrackerConfig,
        polarity_filter_config: PolarityFilterConfig,
        background_activity_filter_config: BackgroundActivityFilterConfig,
        static_filter_config: StaticEventFilterConfig,
        view_config: ViewConfig,
        format: EventFormat,
        endian: Endian,
        time_range: Option<(u64, u64)>,
    ) -> Self {
        let gl = cc
            .gl
            .as_ref()
            .expect("viewer must run with the glow renderer");
        let ball_projection_config =
            BallProjectionConfig::for_view(view_config.width, view_config.height);
        let (timeline_start_us, timeline_end_us) = time_range.unwrap_or((0, 0));
        let controls = Arc::new(Mutex::new(ViewerControls {
            tracker_config,
            polarity_filter_config,
            background_activity_filter_config,
            static_filter_config,
            ball_projection_config,
            view_config,
            show_filtered_events: false,
            paused: false,
            restart_generation: 0,
            seek_generation: 0,
            seek_target_us: timeline_start_us,
        }));
        let frame = Arc::new(Mutex::new(ViewerFrame {
            rgb_buffer: vec![0; view_config.width * view_config.height * 3],
            generation: 0,
            frame_timestamp_us: 0,
            processed_events: 0,
            latest_detection: None,
            latest_ball_estimate: None,
            ball_path: Vec::new(),
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
            background_activity_filter_config,
            static_filter_config,
            ball_projection_config,
            view_config,
            show_filtered_events: false,
            paused: false,
            texture: None,
            rgb_buffer: vec![0; view_config.width * view_config.height * 3],
            seen_frame_generation: 0,
            restart_generation: 0,
            seek_generation: 0,
            scrubber_timestamp_us: timeline_start_us,
            timeline_start_us,
            timeline_end_us,
            frame_timestamp_us: 0,
            processed_events: 0,
            latest_detection: None,
            latest_ball_estimate: None,
            ball_path: Vec::new(),
            ball_3d_view: Arc::new(Mutex::new(Ball3dView::new(gl))),
            ball_3d_camera: Ball3dCamera::default(),
            finished: false,
            error: None,
        }
    }

    fn pull_frame(&mut self, ctx: &egui::Context) {
        let frame = self.frame.lock().expect("viewer frame lock poisoned");
        self.frame_timestamp_us = frame.frame_timestamp_us;
        if !self.paused && frame.frame_timestamp_us >= self.timeline_start_us {
            self.scrubber_timestamp_us = frame.frame_timestamp_us;
        }
        self.processed_events = frame.processed_events;
        self.latest_detection = frame.latest_detection.clone();
        self.latest_ball_estimate = frame.latest_ball_estimate.clone();
        self.ball_path.clone_from(&frame.ball_path);
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
        controls.background_activity_filter_config = self.background_activity_filter_config;
        controls.static_filter_config = self.static_filter_config;
        controls.ball_projection_config = self.ball_projection_config;
        controls.view_config = self.view_config;
        controls.show_filtered_events = self.show_filtered_events;
        controls.paused = self.paused;
        controls.restart_generation = self.restart_generation;
        controls.seek_generation = self.seek_generation;
        controls.seek_target_us = self.scrubber_timestamp_us;
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
                    self.seek_generation = 0;
                    self.scrubber_timestamp_us = self.timeline_start_us;
                }
            });
            if self.timeline_end_us > self.timeline_start_us {
                let mut timestamp_us = self.scrubber_timestamp_us.clamp(
                    self.timeline_start_us,
                    self.timeline_end_us,
                );
                let response = ui
                    .add(
                        egui::Slider::new(
                            &mut timestamp_us,
                            self.timeline_start_us..=self.timeline_end_us,
                        )
                        .text("timeline us"),
                    )
                    .on_hover_text("Scrub through the recording by event timestamp, like a video timeline.");
                if response.changed() {
                    self.scrubber_timestamp_us = timestamp_us;
                    self.seek_generation += 1;
                    self.paused = true;
                }
                ui.label(format!(
                    "{} / {}",
                    format_duration_us(timestamp_us.saturating_sub(self.timeline_start_us)),
                    format_duration_us(self.timeline_end_us.saturating_sub(self.timeline_start_us))
                ));
            } else {
                ui.label(egui::RichText::new("timeline unavailable").weak());
            }
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
            ui.checkbox(&mut self.tracker_config.circle_fit, "circle fit")
                .on_hover_text("Fit a circle to events in the selected cluster for a more stable apparent ball diameter.");
            slider_f32(
                ui,
                &mut self.tracker_config.circle_inlier_tolerance_px,
                0.5..=10.0,
                "circle tol px",
                "Maximum radial error for RANSAC circle inliers.",
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
            ui.strong("Background Activity Filter");
            ui.checkbox(&mut self.background_activity_filter_config.enabled, "enabled")
                .on_hover_text("Suppress isolated events that have no recent spatial neighbor. Useful for random sensor noise.");
            slider_u16(
                ui,
                &mut self.background_activity_filter_config.radius_px,
                1..=8,
                "radius px",
                "Spatial neighbor radius in pixels.",
            );
            slider_u64(
                ui,
                &mut self.background_activity_filter_config.time_window_us,
                100..=20_000,
                "time window us",
                "Neighbor must have occurred within this recent time window.",
            );
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
            ui.strong("Ball Projection");
            ui.checkbox(&mut self.ball_projection_config.enabled, "enabled")
                .on_hover_text("Estimate 3D ball position from the cluster bbox size and smooth it with a Kalman filter.");
            slider_f32(
                ui,
                &mut self.ball_projection_config.diameter_m,
                0.01..=0.30,
                "diameter m",
                "Known real-world ball diameter in meters. Tennis ball default is about 0.067 m.",
            );
            slider_f32(
                ui,
                &mut self.ball_projection_config.focal_length_x_px,
                50.0..=5_000.0,
                "fx px",
                "Camera focal length in horizontal pixels. Use calibrated value when available.",
            );
            slider_f32(
                ui,
                &mut self.ball_projection_config.focal_length_y_px,
                50.0..=5_000.0,
                "fy px",
                "Camera focal length in vertical pixels. Use calibrated value when available.",
            );
            slider_f32(
                ui,
                &mut self.ball_projection_config.principal_x_px,
                0.0..=self.view_config.width as f32,
                "cx px",
                "Optical center x coordinate in pixels. Usually near image width / 2.",
            );
            slider_f32(
                ui,
                &mut self.ball_projection_config.principal_y_px,
                0.0..=self.view_config.height as f32,
                "cy px",
                "Optical center y coordinate in pixels. Usually near image height / 2.",
            );
            egui::ComboBox::from_label("diameter from")
                .selected_text(match self.ball_projection_config.diameter_source {
                    BallDiameterSource::CircleFit => "circle fit",
                    BallDiameterSource::BboxWidth => "bbox width",
                    BallDiameterSource::BboxHeight => "bbox height",
                    BallDiameterSource::BboxAverage => "bbox average",
                    BallDiameterSource::BboxMax => "bbox max",
                    BallDiameterSource::BboxMin => "bbox min",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::CircleFit,
                        "circle fit",
                    );
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::BboxAverage,
                        "bbox average",
                    );
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::BboxWidth,
                        "bbox width",
                    );
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::BboxHeight,
                        "bbox height",
                    );
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::BboxMax,
                        "bbox max",
                    );
                    ui.selectable_value(
                        &mut self.ball_projection_config.diameter_source,
                        BallDiameterSource::BboxMin,
                        "bbox min",
                    );
                })
                .response
                .on_hover_text("Which fitted or bbox dimension to treat as the apparent ball diameter.");
        });

        ui.group(|ui| {
            ui.strong("Performance");
            ui.checkbox(&mut self.show_filtered_events, "show filtered events")
                .on_hover_text("Display rejected events as dim pixels. When disabled, only events accepted by the filters are rendered.");
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
                if let Some(circle) = detection.circle_fit {
                    ui.label(format!(
                        "circle: ({:.1}, {:.1}) r={:.1}px inliers={:.0}%",
                        circle.center_x,
                        circle.center_y,
                        circle.radius_px,
                        circle.inlier_ratio * 100.0
                    ));
                }
            } else {
                ui.label(egui::RichText::new("no cluster yet").weak());
            }
            if let Some(estimate) = &self.latest_ball_estimate {
                ui.separator();
                ui.label(format!(
                    "ball xyz: {:.2}, {:.2}, {:.2} m",
                    estimate.position_m[0], estimate.position_m[1], estimate.position_m[2]
                ));
                ui.label(format!(
                    "velocity: {:.2}, {:.2}, {:.2} m/s",
                    estimate.velocity_mps[0], estimate.velocity_mps[1], estimate.velocity_mps[2]
                ));
                ui.label(format!("speed: {:.2} m/s", estimate.speed_mps));
                ui.label(format!(
                    "bbox diameter: {:.1} px",
                    estimate.measurement.pixel_diameter
                ));
            } else if self.ball_projection_config.enabled {
                ui.separator();
                ui.label(egui::RichText::new("no ball estimate yet").weak());
            }
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::YELLOW, error);
            }
        });
    }

    fn ui_3d_view(&mut self, ui: &mut egui::Ui) {
        ui.strong("3D Ball Position");
        ui.label(
            egui::RichText::new("Drag to orbit, scroll to zoom. X/Y/Z are camera-space meters.")
                .small()
                .weak(),
        );

        let desired_size = egui::vec2(self.view_config.width as f32, 260.0);
        let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(8, 10, 14));

        if response.dragged() {
            let delta = response.drag_delta();
            self.ball_3d_camera.yaw += delta.x * 0.01;
            self.ball_3d_camera.pitch =
                (self.ball_3d_camera.pitch + delta.y * 0.01).clamp(-1.35, 1.35);
        }
        if response.hovered() {
            let scroll_y = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll_y != 0.0 {
                self.ball_3d_camera.distance =
                    (self.ball_3d_camera.distance * (1.0 - scroll_y * 0.001)).clamp(0.4, 50.0);
            }
        }

        let status_text = if let Some(estimate) = &self.latest_ball_estimate {
            format!(
                "x={:.2}m  y={:.2}m  z={:.2}m  speed={:.2}m/s",
                estimate.position_m[0],
                estimate.position_m[1],
                estimate.position_m[2],
                estimate.speed_mps
            )
        } else if self.ball_projection_config.enabled {
            "No 3D estimate yet".to_owned()
        } else {
            "Enable Ball Projection to show 3D position".to_owned()
        };
        painter.text(
            rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            status_text,
            egui::FontId::monospace(13.0),
            egui::Color32::WHITE,
        );

        if !self.ball_projection_config.enabled {
            return;
        }

        let ball_path = self.ball_path.clone();
        let camera = self.ball_3d_camera;
        let ball_3d_view = Arc::clone(&self.ball_3d_view);
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                ball_3d_view
                    .lock()
                    .expect("ball 3d view lock poisoned")
                    .paint(
                        painter.gl(),
                        rect.width() / rect.height().max(1.0),
                        &ball_path,
                        camera,
                    );
            })),
        };
        ui.painter().add(callback);
    }
}

#[derive(Debug, Clone, Copy)]
struct Ball3dCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Default for Ball3dCamera {
    fn default() -> Self {
        Self {
            yaw: -0.75,
            pitch: 0.45,
            distance: 4.0,
        }
    }
}

struct Ball3dView {
    program: glow::Program,
    vertex_array: glow::VertexArray,
    vertex_buffer: glow::Buffer,
}

impl Ball3dView {
    fn new(gl: &glow::Context) -> Self {
        use glow::HasContext as _;
        unsafe {
            let program = gl
                .create_program()
                .expect("cannot create 3D shader program");
            let shader_version = if cfg!(target_arch = "wasm32") {
                "#version 300 es"
            } else {
                "#version 330"
            };
            let vertex_shader = compile_shader(
                gl,
                program,
                glow::VERTEX_SHADER,
                shader_version,
                r#"
                    uniform mat4 u_mvp;
                    in vec3 a_position;
                    in vec3 a_color;
                    out vec3 v_color;

                    void main() {
                        v_color = a_color;
                        gl_Position = u_mvp * vec4(a_position, 1.0);
                    }
                "#,
            );
            let fragment_shader = compile_shader(
                gl,
                program,
                glow::FRAGMENT_SHADER,
                shader_version,
                r#"
                    precision mediump float;
                    in vec3 v_color;
                    out vec4 out_color;

                    void main() {
                        out_color = vec4(v_color, 1.0);
                    }
                "#,
            );

            gl.link_program(program);
            assert!(
                gl.get_program_link_status(program),
                "{}",
                gl.get_program_info_log(program)
            );
            gl.detach_shader(program, vertex_shader);
            gl.detach_shader(program, fragment_shader);
            gl.delete_shader(vertex_shader);
            gl.delete_shader(fragment_shader);

            let vertex_array = gl.create_vertex_array().expect("cannot create 3D VAO");
            let vertex_buffer = gl.create_buffer().expect("cannot create 3D VBO");
            gl.bind_vertex_array(Some(vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vertex_buffer));

            let stride = 6 * std::mem::size_of::<f32>() as i32;
            let position_location = gl
                .get_attrib_location(program, "a_position")
                .expect("missing a_position attribute");
            let color_location = gl
                .get_attrib_location(program, "a_color")
                .expect("missing a_color attribute");
            gl.enable_vertex_attrib_array(position_location);
            gl.vertex_attrib_pointer_f32(position_location, 3, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(color_location);
            gl.vertex_attrib_pointer_f32(
                color_location,
                3,
                glow::FLOAT,
                false,
                stride,
                3 * std::mem::size_of::<f32>() as i32,
            );

            Self {
                program,
                vertex_array,
                vertex_buffer,
            }
        }
    }

    fn paint(
        &mut self,
        gl: &glow::Context,
        aspect_ratio: f32,
        ball_path: &[BallTrackEstimate],
        camera: Ball3dCamera,
    ) {
        use glow::HasContext as _;
        let vertices = ball_scene_vertices(ball_path);
        let mvp = ball_scene_mvp(ball_path, aspect_ratio, camera);

        unsafe {
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(
                gl.get_uniform_location(self.program, "u_mvp").as_ref(),
                false,
                &mvp,
            );
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vertex_buffer));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                f32s_as_u8s(&vertices),
                glow::STREAM_DRAW,
            );
            gl.line_width(2.0);
            gl.draw_arrays(glow::LINES, 0, (vertices.len() / 6) as i32);
        }
    }

    fn destroy(&self, gl: &glow::Context) {
        use glow::HasContext as _;
        unsafe {
            gl.delete_buffer(self.vertex_buffer);
            gl.delete_vertex_array(self.vertex_array);
            gl.delete_program(self.program);
        }
    }
}

fn compile_shader(
    gl: &glow::Context,
    program: glow::Program,
    shader_type: u32,
    shader_version: &str,
    source: &str,
) -> glow::Shader {
    use glow::HasContext as _;
    unsafe {
        let shader = gl.create_shader(shader_type).expect("cannot create shader");
        gl.shader_source(shader, &format!("{shader_version}\n{source}"));
        gl.compile_shader(shader);
        assert!(
            gl.get_shader_compile_status(shader),
            "failed to compile 3D shader: {}",
            gl.get_shader_info_log(shader)
        );
        gl.attach_shader(program, shader);
        shader
    }
}

fn f32s_as_u8s(values: &[f32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

fn ball_scene_vertices(ball_path: &[BallTrackEstimate]) -> Vec<f32> {
    let extent = ball_scene_extent(ball_path);
    let mut vertices = Vec::new();
    let grid_color = [0.18, 0.20, 0.24];
    let axis_x = [1.0, 0.25, 0.25];
    let axis_y = [0.25, 1.0, 0.35];
    let axis_z = [0.25, 0.55, 1.0];
    let path_color = [0.25, 0.75, 1.0];
    let ball_color = [1.0, 0.88, 0.15];
    let grid_step = (extent / 4.0).max(0.25);

    for index in -4..=4 {
        let v = index as f32 * grid_step;
        push_line(
            &mut vertices,
            [-extent, 0.0, v],
            [extent, 0.0, v],
            grid_color,
        );
        push_line(&mut vertices, [v, 0.0, 0.0], [v, 0.0, extent], grid_color);
    }

    push_line(&mut vertices, [0.0, 0.0, 0.0], [extent, 0.0, 0.0], axis_x);
    push_line(&mut vertices, [0.0, 0.0, 0.0], [0.0, extent, 0.0], axis_y);
    push_line(&mut vertices, [0.0, 0.0, 0.0], [0.0, 0.0, extent], axis_z);

    for segment in ball_path.windows(2) {
        push_line(
            &mut vertices,
            segment[0].position_m,
            segment[1].position_m,
            path_color,
        );
    }

    if let Some(estimate) = ball_path.last() {
        let position = estimate.position_m;
        let radius = (estimate.measurement.pixel_diameter * 0.002).clamp(0.04, 0.16);
        push_line(
            &mut vertices,
            [position[0] - radius, position[1], position[2]],
            [position[0] + radius, position[1], position[2]],
            ball_color,
        );
        push_line(
            &mut vertices,
            [position[0], position[1] - radius, position[2]],
            [position[0], position[1] + radius, position[2]],
            ball_color,
        );
        push_line(
            &mut vertices,
            [position[0], position[1], position[2] - radius],
            [position[0], position[1], position[2] + radius],
            ball_color,
        );
    }

    vertices
}

fn push_line(vertices: &mut Vec<f32>, a: [f32; 3], b: [f32; 3], color: [f32; 3]) {
    vertices.extend_from_slice(&[a[0], a[1], a[2], color[0], color[1], color[2]]);
    vertices.extend_from_slice(&[b[0], b[1], b[2], color[0], color[1], color[2]]);
}

fn ball_scene_mvp(
    ball_path: &[BallTrackEstimate],
    aspect_ratio: f32,
    camera: Ball3dCamera,
) -> [f32; 16] {
    let target = ball_scene_center(ball_path);
    let distance = camera.distance.max(ball_scene_extent(ball_path) * 1.25);
    let eye = [
        target[0] + distance * camera.yaw.sin() * camera.pitch.cos(),
        target[1] + distance * camera.pitch.sin(),
        target[2] - distance * camera.yaw.cos() * camera.pitch.cos(),
    ];
    let view = look_at(eye, target, [0.0, 1.0, 0.0]);
    let projection = perspective(55.0_f32.to_radians(), aspect_ratio.max(0.1), 0.01, 200.0);
    mat4_mul(projection, view)
}

fn ball_scene_center(ball_path: &[BallTrackEstimate]) -> [f32; 3] {
    if ball_path.is_empty() {
        return [0.0, 0.0, 1.0];
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for estimate in ball_path {
        for axis in 0..3 {
            min[axis] = min[axis].min(estimate.position_m[axis]);
            max[axis] = max[axis].max(estimate.position_m[axis]);
        }
    }

    [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ]
}

fn ball_scene_extent(ball_path: &[BallTrackEstimate]) -> f32 {
    ball_path
        .iter()
        .flat_map(|estimate| estimate.position_m)
        .map(f32::abs)
        .fold(1.0_f32, f32::max)
        .max(1.0)
}

fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y_rad / 2.0).tan();
    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        f,
        0.0,
        0.0,
        0.0,
        0.0,
        (far + near) / (near - far),
        -1.0,
        0.0,
        0.0,
        (2.0 * far * near) / (near - far),
        0.0,
    ]
}

fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = vec3_normalize(vec3_sub(target, eye));
    let s = vec3_normalize(vec3_cross(f, up));
    let u = vec3_cross(s, f);

    [
        s[0],
        u[0],
        -f[0],
        0.0,
        s[1],
        u[1],
        -f[1],
        0.0,
        s[2],
        u[2],
        -f[2],
        0.0,
        -vec3_dot(s, eye),
        -vec3_dot(u, eye),
        vec3_dot(f, eye),
        1.0,
    ]
}

fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let length = vec3_dot(v, v).sqrt().max(0.000_001);
    [v[0] / length, v[1] / length, v[2] / length]
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

fn slider_f32(
    ui: &mut egui::Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
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
            ui.vertical(|ui| {
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
                ui.add_space(8.0);
                egui::Frame::canvas(ui.style()).show(ui, |ui| {
                    self.ui_3d_view(ui);
                });
            });
        });

        ctx.request_repaint_after(Duration::from_millis(1));
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.ball_3d_view
                .lock()
                .expect("ball 3d view lock poisoned")
                .destroy(gl);
        }
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
    pipeline: Option<EventPipeline<ConfiguredEventFilters, RollingClusterTracker>>,
    ball_estimator: Option<BallPathEstimator>,
    render_events: VecDeque<RenderEvent>,
    latest_detection: Option<ClusterDetection>,
    latest_ball_estimate: Option<BallTrackEstimate>,
    ball_path: VecDeque<BallTrackEstimate>,
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
    seek_generation: u64,
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
            ball_estimator: None,
            render_events: VecDeque::new(),
            latest_detection: None,
            latest_ball_estimate: None,
            ball_path: VecDeque::new(),
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
            seek_generation: controls_snapshot.seek_generation,
            frame_generation: 0,
        };
        state.reset_stream(controls_snapshot);
        state
    }

    fn tick(&mut self) {
        let controls = *self.controls.lock().expect("viewer controls lock poisoned");
        if controls.restart_generation != self.restart_generation {
            self.restart_generation = controls.restart_generation;
            self.seek_generation = controls.seek_generation;
            self.reset_stream(controls);
        }

        if controls.seek_generation != self.seek_generation {
            self.seek_generation = controls.seek_generation;
            self.seek_to(controls, controls.seek_target_us);
            return;
        }

        if controls.paused || self.finished || !self.frame_is_due(controls) {
            thread::sleep(Duration::from_millis(1));
            return;
        }

        self.process_events(controls);
    }

    fn reset_stream(&mut self, controls: ViewerControls) {
        self.pipeline = None;
        self.ball_estimator = None;
        self.render_events.clear();
        self.latest_detection = None;
        self.latest_ball_estimate = None;
        self.ball_path.clear();
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
                    ConfiguredEventFilters::new(
                        controls.polarity_filter_config,
                        controls.background_activity_filter_config,
                        controls.static_filter_config,
                    ),
                    RollingClusterTracker::new(controls.tracker_config),
                    BallPathEstimator::new(controls.ball_projection_config),
                ) {
                    (Ok(filters), Ok(tracker), Ok(ball_estimator)) => {
                        self.pipeline = Some(EventPipeline::new(opened.stream, filters, tracker));
                        self.ball_estimator = Some(ball_estimator);
                    }
                    (Err(error), _, _) | (_, Err(error), _) | (_, _, Err(error)) => {
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
        for _ in 0..controls.view_config.max_events_per_ui_update {
            let Some(event) = self.process_one_event(controls) else {
                self.publish_frame(controls);
                return;
            };

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

    fn seek_to(&mut self, controls: ViewerControls, target_us: u64) {
        self.reset_stream(controls);
        while !self.finished {
            let Some(event) = self.process_one_event(controls) else {
                break;
            };
            if event.timestamp_us >= target_us {
                self.frame_timestamp_us = event.timestamp_us;
                self.next_frame_us = Some(event.timestamp_us + controls.tracker_config.step_us);
                self.playback_start_us = Some(event.timestamp_us);
                self.playback_start = Instant::now();
                break;
            }
        }
        self.publish_frame(controls);
    }

    fn process_one_event(&mut self, controls: ViewerControls) -> Option<Event> {
        let Some(pipeline) = self.pipeline.as_mut() else {
            return None;
        };

        if let Err(error) = pipeline.algorithm_mut().set_config(controls.tracker_config) {
            self.error = Some(error.to_string());
            self.finished = true;
            return None;
        }
        if let Err(error) = pipeline.filters_mut().set_config(
            controls.polarity_filter_config,
            controls.background_activity_filter_config,
            controls.static_filter_config,
        ) {
            self.error = Some(error.to_string());
            self.finished = true;
            return None;
        }
        if let Some(ball_estimator) = self.ball_estimator.as_mut() {
            if let Err(error) = ball_estimator.set_config(controls.ball_projection_config) {
                self.error = Some(error.to_string());
                self.finished = true;
                return None;
            }
            if !controls.ball_projection_config.enabled {
                self.latest_ball_estimate = None;
                self.ball_path.clear();
            }
        }

        let processed = match pipeline.next_event() {
            Ok(Some(processed)) => processed,
            Ok(None) => {
                self.finished = true;
                return None;
            }
            Err(error) => {
                self.error = Some(error.to_string());
                self.finished = true;
                return None;
            }
        };

        self.processed_events += 1;
        let event = processed.event;
        self.render_events.push_back(RenderEvent {
            event,
            accepted: processed.accepted,
        });
        drop_old_render_events(
            &mut self.render_events,
            event.timestamp_us,
            controls.tracker_config.window_us,
        );

        for detection in processed.outputs {
            if let Some(ball_estimator) = self.ball_estimator.as_mut() {
                self.latest_ball_estimate = ball_estimator.estimate(&detection);
                if let Some(estimate) = &self.latest_ball_estimate {
                    self.ball_path.push_back(estimate.clone());
                    while self.ball_path.len() > 180 {
                        self.ball_path.pop_front();
                    }
                }
            }
            self.latest_detection = Some(detection);
        }

        Some(event)
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
            controls.show_filtered_events,
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
        frame.latest_ball_estimate = self.latest_ball_estimate.clone();
        frame.ball_path = self.ball_path.iter().cloned().collect();
        frame.finished = self.finished;
        frame.error = self.error.clone();
    }
}
