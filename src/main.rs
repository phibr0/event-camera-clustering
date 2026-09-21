mod render;
mod viewer;
mod comparison;

use clap::{Args, Parser, ValueEnum};
use event_clustering::algorithms::{
    ClusterDetection, RollingClusterTracker, RollingClusterTrackerConfig,
};
use event_clustering::filter::{
    BackgroundActivityFilterConfig, ConfiguredEventFilters, PolarityFilterConfig, PolarityMode,
    StaticEventFilterConfig,
};
use event_clustering::parabola::{
    GravityAxis, ParabolaFitConfig, TwoShotParabolaFitter, parse_calibration_json,
};
use event_clustering::parser::{Endian, EventFormat, open_event_stream};
use event_clustering::pipeline::EventPipeline;
use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use viewer::{ViewConfig, view};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse_from(normalize_args(env::args().collect()));
    let tracker_config = cli.tracking.tracker_config();
    let view_config = cli.view.view_config();
    let mut parabola_config = cli
        .parabola
        .parabola_config(view_config.width, view_config.height);
    apply_calibration_from_project_root(&mut parabola_config);
    let polarity_filter_config = cli.filters.polarity_config();
    let background_activity_filter_config = cli.filters.background_activity_config();
    let static_filter_config = cli.filters.static_config();
    let format = cli.parser.format.unwrap_or_default();
    let endian = cli.parser.endian.unwrap_or(Endian::Little);

    if matches!(cli.command, CommandName::MotionCompare) {
        return comparison::run(cli.path, cli.motion, format, endian);
    }

    match cli.command {
        CommandName::TrackBall => track_ball(
            cli.path,
            tracker_config,
            polarity_filter_config,
            background_activity_filter_config,
            static_filter_config,
            parabola_config,
            format,
            endian,
            cli.max_detections,
        ),
        CommandName::View => view(
            cli.path,
            tracker_config,
            polarity_filter_config,
            background_activity_filter_config,
            static_filter_config,
            parabola_config,
            view_config,
            format,
            endian,
        ),
        CommandName::MotionCompare => unreachable!(),
    }
}

fn apply_calibration_from_project_root(config: &mut ParabolaFitConfig) {
    let path = Path::new("calibration.json");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };
    if let Some(calibration) = parse_calibration_json(&contents) {
        config.apply_calibration(calibration);
    }
}

fn normalize_args(mut args: Vec<String>) -> Vec<String> {
    if args.len() == 1 {
        args.push("--help".to_owned());
    }
    if args.len() >= 3 && matches!(args[1].as_str(), "track-ball" | "view" | "motion-compare") {
        args.swap(1, 2);
    }
    args
}

fn track_ball(
    path: PathBuf,
    config: RollingClusterTrackerConfig,
    polarity_filter_config: PolarityFilterConfig,
    background_activity_filter_config: BackgroundActivityFilterConfig,
    static_filter_config: StaticEventFilterConfig,
    mut parabola_config: ParabolaFitConfig,
    format: EventFormat,
    endian: Endian,
    max_detections: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let opened = open_event_stream(path, format, endian)?;
    if let (Some(width), Some(height)) = (opened.header.width, opened.header.height) {
        parabola_config.adjust_default_intrinsics_for_view(640, 480, width, height);
    }
    if opened.header.evt_version.as_deref() != Some("2.0") && opened.format == EventFormat::Evt2 {
        eprintln!(
            "warning: RAW header evt version is {:?}, decoding as EVT 2.0",
            opened.header.evt_version
        );
    }

    let tracker = RollingClusterTracker::new(config)?;
    let filters = ConfiguredEventFilters::new(
        polarity_filter_config,
        background_activity_filter_config,
        static_filter_config,
    )?;
    let mut pipeline = EventPipeline::new(opened.stream, filters, tracker);
    let mut parabola_fitter = TwoShotParabolaFitter::new(parabola_config)?;
    let mut event_count = 0_u64;
    let mut detection_count = 0_u64;

    while let Some(processed) = pipeline.next_event()? {
        event_count += 1;
        for detection in processed.outputs {
            detection_count += 1;
            let parabola_fit = parabola_fitter.push_detection(&detection);
            print_detection(&detection, parabola_fit.as_ref());
            if max_detections.is_some_and(|max| detection_count >= max) {
                eprintln!("processed_events={event_count} detections={detection_count}");
                return Ok(());
            }
        }
    }

    for detection in pipeline.finish() {
        detection_count += 1;
        let parabola_fit = parabola_fitter.push_detection(&detection);
        print_detection(&detection, parabola_fit.as_ref());
    }

    eprintln!("processed_events={event_count} detections={detection_count}");
    Ok(())
}

fn print_detection(
    detection: &ClusterDetection,
    parabola_fit: Option<&event_clustering::parabola::ParabolaFitEstimate>,
) {
    let circle = detection.circle_fit.map_or_else(
        || "circle=none".to_owned(),
        |circle| {
            format!(
                "circle=({:.2},{:.2}) r={:.2}px inliers={} ratio={:.3} err={:.2}px",
                circle.center_x,
                circle.center_y,
                circle.radius_px,
                circle.inlier_count,
                circle.inlier_ratio,
                circle.mean_error_px
            )
        },
    );
    let parabola = parabola_fit.map_or_else(
        || "parabola=none".to_owned(),
        |fit| {
            format!(
                "parabola=p({:.2},{:.2},{:.2})m v({:.2},{:.2},{:.2})m/s inliers={}/{} err={:.3}m",
                fit.initial_position_m[0],
                fit.initial_position_m[1],
                fit.initial_position_m[2],
                fit.initial_velocity_mps[0],
                fit.initial_velocity_mps[1],
                fit.initial_velocity_mps[2],
                fit.inlier_count,
                fit.total_count,
                fit.mean_error_m,
            )
        },
    );
    println!(
        "t={}us window={}..{}us centroid=({:.2},{:.2}) bbox=({},{})->({},{}) events={} confidence={:.3} {} {}",
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
        circle,
        parabola,
    );
}

#[derive(Debug, Parser)]
#[command(about = "Parse event-camera RAW files and track plausible moving clusters")]
struct Cli {
    #[arg(value_name = "RAW_FILE")]
    path: PathBuf,

    #[arg(value_enum, default_value = "track-ball")]
    command: CommandName,

    #[command(flatten)]
    tracking: TrackingArgs,

    #[command(flatten)]
    filters: FilterArgs,

    #[command(flatten)]
    parser: ParserArgs,

    #[command(flatten)]
    view: ViewArgs,

    #[command(flatten)]
    parabola: ParabolaArgs,

    #[command(flatten)]
    motion: comparison::MotionArgs,

    #[arg(long)]
    max_detections: Option<u64>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CommandName {
    TrackBall,
    View,
    MotionCompare,
}

#[derive(Debug, Args)]
struct TrackingArgs {
    #[arg(long)]
    window_us: Option<u64>,
    #[arg(long)]
    step_us: Option<u64>,
    #[arg(long)]
    cell_size: Option<u16>,
    #[arg(long)]
    min_events: Option<usize>,
    #[arg(long)]
    min_cells: Option<usize>,
    #[arg(long)]
    max_bbox_width: Option<u16>,
    #[arg(long)]
    max_bbox_height: Option<u16>,
    #[arg(long)]
    no_circle_fit: bool,
    #[arg(long)]
    circle_inlier_tolerance_px: Option<f32>,
}

impl TrackingArgs {
    fn tracker_config(&self) -> RollingClusterTrackerConfig {
        let mut config = RollingClusterTrackerConfig::default();
        if let Some(value) = self.window_us {
            config.window_us = value;
        }
        if let Some(value) = self.step_us {
            config.step_us = value;
        }
        if let Some(value) = self.cell_size {
            config.cell_size = value;
        }
        if let Some(value) = self.min_events {
            config.min_events = value;
        }
        if let Some(value) = self.min_cells {
            config.min_cells = value;
        }
        if let Some(value) = self.max_bbox_width {
            config.max_bbox_width = value;
        }
        if let Some(value) = self.max_bbox_height {
            config.max_bbox_height = value;
        }
        if self.no_circle_fit {
            config.circle_fit = false;
        }
        if let Some(value) = self.circle_inlier_tolerance_px {
            config.circle_inlier_tolerance_px = value;
        }
        config
    }
}

#[derive(Debug, Args)]
struct FilterArgs {
    #[arg(long, value_parser = parse_polarity_mode, default_value = "all")]
    polarity: PolarityMode,
    #[arg(long)]
    invert_polarity: bool,
    #[arg(long, alias = "no-invert-polarity")]
    raw_polarity: bool,
    #[arg(long)]
    filter_background_activity: bool,
    #[arg(long)]
    no_filter_background_activity: bool,
    #[arg(long)]
    background_radius_px: Option<u16>,
    #[arg(long)]
    background_time_window_us: Option<u64>,
    #[arg(long)]
    background_cleanup_after_us: Option<u64>,
    #[arg(long)]
    filter_static: bool,
    #[arg(long)]
    no_filter_static: bool,
    #[arg(long)]
    static_cell_size: Option<u16>,
    #[arg(long)]
    static_after_us: Option<u64>,
    #[arg(long)]
    static_min_events: Option<u32>,
    #[arg(long)]
    static_inactive_us: Option<u64>,
}

impl FilterArgs {
    fn polarity_config(&self) -> PolarityFilterConfig {
        let mut config = PolarityFilterConfig::default();
        config.mode = self.polarity;
        if self.invert_polarity {
            config.invert_polarity = true;
        }
        if self.raw_polarity {
            config.invert_polarity = false;
        }
        config
    }

    fn background_activity_config(&self) -> BackgroundActivityFilterConfig {
        let mut config = BackgroundActivityFilterConfig::default();
        if self.filter_background_activity {
            config.enabled = true;
        }
        if self.no_filter_background_activity {
            config.enabled = false;
        }
        if let Some(value) = self.background_radius_px {
            config.radius_px = value;
        }
        if let Some(value) = self.background_time_window_us {
            config.time_window_us = value;
        }
        if let Some(value) = self.background_cleanup_after_us {
            config.cleanup_after_us = value;
        }
        config
    }

    fn static_config(&self) -> StaticEventFilterConfig {
        let mut config = StaticEventFilterConfig::default();
        if self.filter_static {
            config.enabled = true;
        }
        if self.no_filter_static {
            config.enabled = false;
        }
        if let Some(value) = self.static_cell_size {
            config.cell_size = value;
        }
        if let Some(value) = self.static_after_us {
            config.stable_after_us = value;
        }
        if let Some(value) = self.static_min_events {
            config.min_events = value;
        }
        if let Some(value) = self.static_inactive_us {
            config.inactive_after_us = value;
        }
        config
    }
}

#[derive(Debug, Args)]
struct ParserArgs {
    #[arg(long)]
    format: Option<EventFormat>,
    #[arg(long)]
    endian: Option<Endian>,
}

#[derive(Debug, Args)]
struct ViewArgs {
    #[arg(long)]
    width: Option<usize>,
    #[arg(long)]
    height: Option<usize>,
    #[arg(long)]
    speed: Option<f64>,
    #[arg(long)]
    events_per_tick: Option<usize>,
}

#[derive(Debug, Args)]
struct ParabolaArgs {
    #[arg(long)]
    parabola_fit: bool,
    #[arg(long)]
    no_parabola_fit: bool,
    #[arg(long)]
    parabola_ball_diameter_m: Option<f32>,
    #[arg(long)]
    parabola_fx_px: Option<f32>,
    #[arg(long)]
    parabola_fy_px: Option<f32>,
    #[arg(long)]
    parabola_cx_px: Option<f32>,
    #[arg(long)]
    parabola_cy_px: Option<f32>,
    #[arg(long)]
    parabola_gravity_mps2: Option<f32>,
    #[arg(long, value_parser = parse_gravity_axis)]
    parabola_gravity_axis: Option<GravityAxis>,
    #[arg(long)]
    parabola_buffer_len: Option<usize>,
    #[arg(long)]
    parabola_min_points: Option<usize>,
    #[arg(long)]
    parabola_max_pair_samples: Option<usize>,
    #[arg(long)]
    parabola_inlier_threshold_m: Option<f32>,
    #[arg(long)]
    parabola_min_inlier_ratio: Option<f32>,
    #[arg(long)]
    parabola_min_sample_dt_s: Option<f32>,
    #[arg(long)]
    parabola_min_depth_m: Option<f32>,
    #[arg(long)]
    parabola_max_depth_m: Option<f32>,
}

impl ParabolaArgs {
    fn parabola_config(&self, width: usize, height: usize) -> ParabolaFitConfig {
        let mut config = ParabolaFitConfig::for_view(width, height);
        if self.parabola_fit {
            config.enabled = true;
        }
        if self.no_parabola_fit {
            config.enabled = false;
        }
        if let Some(value) = self.parabola_ball_diameter_m {
            config.ball_diameter_m = value;
        }
        if let Some(value) = self.parabola_fx_px {
            config.focal_length_x_px = value;
        }
        if let Some(value) = self.parabola_fy_px {
            config.focal_length_y_px = value;
        }
        if let Some(value) = self.parabola_cx_px {
            config.principal_x_px = value;
        }
        if let Some(value) = self.parabola_cy_px {
            config.principal_y_px = value;
        }
        if let Some(value) = self.parabola_gravity_mps2 {
            config.gravity_mps2 = value;
        }
        if let Some(value) = self.parabola_gravity_axis {
            config.gravity_axis = value;
        }
        if let Some(value) = self.parabola_buffer_len {
            config.buffer_len = value;
        }
        if let Some(value) = self.parabola_min_points {
            config.min_points = value;
        }
        if let Some(value) = self.parabola_max_pair_samples {
            config.max_pair_samples = value;
        }
        if let Some(value) = self.parabola_inlier_threshold_m {
            config.inlier_threshold_m = value;
        }
        if let Some(value) = self.parabola_min_inlier_ratio {
            config.min_inlier_ratio = value;
        }
        if let Some(value) = self.parabola_min_sample_dt_s {
            config.min_sample_dt_s = value;
        }
        if let Some(value) = self.parabola_min_depth_m {
            config.min_depth_m = value;
        }
        if let Some(value) = self.parabola_max_depth_m {
            config.max_depth_m = value;
        }
        config
    }
}

impl ViewArgs {
    fn view_config(&self) -> ViewConfig {
        let mut config = ViewConfig::default();
        if let Some(value) = self.width {
            config.width = value;
        }
        if let Some(value) = self.height {
            config.height = value;
        }
        if let Some(value) = self.speed {
            config.speed = value;
        }
        if let Some(value) = self.events_per_tick {
            config.max_events_per_ui_update = value;
        }
        config
    }
}

fn parse_polarity_mode(value: &str) -> Result<PolarityMode, String> {
    match value {
        "all" => Ok(PolarityMode::All),
        "positive" | "on" | "+" => Ok(PolarityMode::Positive),
        "negative" | "off" | "-" => Ok(PolarityMode::Negative),
        _ => Err("--polarity must be all, positive/on/+, or negative/off/-".to_owned()),
    }
}

fn parse_gravity_axis(value: &str) -> Result<GravityAxis, String> {
    match value {
        "x-positive" | "x+" => Ok(GravityAxis::XPositive),
        "x-negative" | "x-" => Ok(GravityAxis::XNegative),
        "y-positive" | "y+" => Ok(GravityAxis::YPositive),
        "y-negative" | "y-" => Ok(GravityAxis::YNegative),
        "z-positive" | "z+" => Ok(GravityAxis::ZPositive),
        "z-negative" | "z-" => Ok(GravityAxis::ZNegative),
        _ => Err("--parabola-gravity-axis must be x-positive/x+, x-negative/x-, y-positive/y+, y-negative/y-, z-positive/z+, or z-negative/z-".to_owned()),
    }
}
