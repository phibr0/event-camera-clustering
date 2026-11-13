#[allow(dead_code)]
mod comparison {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/comparison.rs"));
    pub fn probe() -> Result<()> {
        fs::create_dir_all("outputs/ball-tracking")?;
        use event_clustering::algorithms::{RollingClusterTracker, RollingClusterTrackerConfig};
        for name in [
            "ball_gross_1",
            "ball_gross_2",
            "ball_gross_3",
            "ball_klein_2",
            "ball_klein_3",
            "ball_klein_4",
            "ball_klein_5",
        ] {
            let path = PathBuf::from(format!("motion-compensation/{name}.raw"));
            let (origin_us, duration_s, _) = scan(&path, EventFormat::Auto, Endian::Little)?;
            let calibration =
                parse_calibration_json(&fs::read_to_string("calibration.json")?).unwrap();
            let camera = Camera::new(calibration, 1280, 720)?;
            let imu = ImuTrack::load(&PathBuf::from(format!("motion-compensation/{name}.csv")))?;
            let alignment = fs::read_to_string(format!(
                "outputs/motion-comparison/ball-recordings/alignments/{name}.alignment"
            ))?;
            let mut lines = alignment.lines();
            let config = MotionConfig {
                offset_s: lines.next().unwrap().parse()?,
                mount: mount_from_text(lines.next().unwrap())?,
                ..Default::default()
            };
            let start = (((-config.offset_s + 0.04).max(0.04)) * 100.).ceil() / 100.;
            let end = duration_s.min(imu.duration_s() - config.offset_s);
            let reader = WindowReader::new(path, EventFormat::Auto, Endian::Little, origin_us)?;
            let mut engine = Engine {
                reader,
                camera,
                imu,
                config,
                origin_us,
                duration_s,
                window_us: 30_000,
                read_ms: 0.,
                process_ms: 0.,
            };
            let mut output = std::io::BufWriter::new(fs::File::create(format!(
                "outputs/ball-tracking/{name}-components.csv"
            ))?);
            let mut latency = Vec::new();
            writeln!(
                output,
                "time_s,foreground_events,recent_events,rank,cx,cy,width,height,events,ref_x,ref_y,r00,r01,r02,r10,r11,r12,r20,r21,r22"
            )?;
            for i in 0..((end - start) * 100.) as usize {
                let time = start + i as f64 / 100.;
                let (events, frame) = engine.frame(time)?;
                let tick = Instant::now();
                let end_us = origin_us + (time * 1e6).round() as u64;
                let recent: Vec<_> = events
                    .iter()
                    .zip(&frame.compensated)
                    .zip(&frame.foreground_on)
                    .filter_map(|((&e, &p), &keep)| {
                        (keep && e.timestamp_us >= end_us.saturating_sub(10_000)).then_some(Event {
                            x: p[0].round().clamp(0., 1279.) as u16,
                            y: p[1].round().clamp(0., 719.) as u16,
                            ..e
                        })
                    })
                    .collect();
                let rotation = engine
                    .imu
                    .relative_rotation(
                        time + config.offset_s,
                        start + config.offset_s,
                        config.mount,
                    )
                    .unwrap();
                let matrix = rotation
                    .iter()
                    .flatten()
                    .map(|v| format!("{v:.9}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let foreground = frame.foreground_on.iter().filter(|&&v| v).count();
                // ponytail: six bounded reruns reuse the existing tracker; overlapping
                // boxes can hide candidates. Expose all components if that becomes limiting.
                let mut remaining = recent.clone();
                let mut rank = 0;
                while rank < 6 && remaining.len() >= 15 {
                    let mut tracker = RollingClusterTracker::new(RollingClusterTrackerConfig {
                        window_us: 10_000,
                        step_us: 30_000,
                        cell_size: 3,
                        min_events: 15,
                        min_cells: 3,
                        max_bbox_width: 800,
                        max_bbox_height: 800,
                        circle_fit: false,
                        ..Default::default()
                    })?;
                    for &e in &remaining {
                        tracker.process_event(e);
                    }
                    let Some(d) = tracker.finish() else { break };
                    let ray = [
                        (d.centroid_x as f64 - calibration.principal_x_px as f64)
                            / calibration.focal_length_x_px as f64,
                        (d.centroid_y as f64 - calibration.principal_y_px as f64)
                            / calibration.focal_length_y_px as f64,
                        1.,
                    ];
                    let point = engine
                        .camera
                        .project(motion::transform(rotation, ray))
                        .unwrap_or([-1., -1.]);
                    writeln!(
                        output,
                        "{time:.6},{foreground},{},{rank},{},{},{},{},{},{},{},{matrix}",
                        recent.len(),
                        d.centroid_x,
                        d.centroid_y,
                        d.bbox.width(),
                        d.bbox.height(),
                        d.event_count,
                        point[0],
                        point[1]
                    )?;
                    remaining.retain(|e| {
                        e.x < d.bbox.min_x
                            || e.x > d.bbox.max_x
                            || e.y < d.bbox.min_y
                            || e.y > d.bbox.max_y
                    });
                    rank += 1;
                }
                if rank == 0 {
                    writeln!(
                        output,
                        "{time:.6},{foreground},{},-1,,,,,,,,{matrix}",
                        recent.len()
                    )?;
                }
                if i > 0 {
                    latency.push(
                        engine.read_ms + engine.process_ms + tick.elapsed().as_secs_f64() * 1000.,
                    );
                }
            }
            latency.sort_by(f64::total_cmp);
            let report = format!(
                "Read + rotation/foreground OFF/ON + up to 6 cluster proposals: p50={:.3} ms p95={:.3} ms max={:.3} ms; over 10 ms: {}/{}. Excludes setup, association, rendering and encoding.\n",
                latency[latency.len() / 2],
                latency[latency.len() * 95 / 100],
                latency.last().unwrap(),
                latency.iter().filter(|&&v| v > 10.).count(),
                latency.len()
            );
            fs::write(format!("outputs/ball-tracking/{name}-timing.txt"), &report)?;
            eprintln!("Finished {name}: {report}");
        }
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    comparison::probe()
}
