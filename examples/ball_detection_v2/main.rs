#[allow(dead_code)]
mod comparison {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/comparison.rs"));
    pub fn probe() -> Result<()> {
        fs::create_dir_all("outputs/ball-detection-v2")?;
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
                "outputs/ball-detection-v2/{name}-events.bin"
            ))?);
            for i in 0..((end - start) * 100.) as usize {
                let time = start + i as f64 / 100.;
                let (events, frame) = engine.frame(time)?;
                let end_us = origin_us + (time * 1e6).round() as u64;
                let points: Vec<_> = events
                    .iter()
                    .zip(&frame.compensated)
                    .zip(&frame.foreground_on)
                    .filter(|((e, p), _)| {
                        e.timestamp_us >= end_us.saturating_sub(10_000)
                            && p[0] >= 0.
                            && p[1] >= 0.
                            && p[0] < 1280.
                            && p[1] < 720.
                    })
                    .map(|((e, p), fg)| {
                        [
                            p[0],
                            p[1],
                            end_us.saturating_sub(e.timestamp_us) as f32 * 0.001,
                            if *fg { 1. } else { 0. },
                        ]
                    })
                    .collect();
                output.write_all(&time.to_le_bytes())?;
                output.write_all(&(points.len() as u32).to_le_bytes())?;
                for p in points {
                    for v in p {
                        output.write_all(&v.to_le_bytes())?;
                    }
                }
            }
            eprintln!("Cached {name}");
        }
        Ok(())
    }
    fn line(rgb: &mut [u8], width: usize, height: usize, a: [f32; 2], b: [f32; 2], color: [u8; 3]) {
        if !a.iter().chain(&b).all(|v| v.is_finite()) {
            return;
        }
        let steps = (a[0] - b[0])
            .abs()
            .max((a[1] - b[1]).abs())
            .ceil()
            .min(3000.) as usize;
        for i in 0..=steps {
            let f = i as f32 / steps.max(1) as f32;
            let x = (a[0] * (1. - f) + b[0] * f).round() as i32;
            let y = (a[1] * (1. - f) + b[1] * f).round() as i32;
            if x >= 640 && x < width as i32 && y >= 32 && y < 392 && y < height as i32 {
                let j = (y as usize * width + x as usize) * 3;
                rgb[j..j + 3].copy_from_slice(&color);
            }
        }
    }
    pub fn render_tracks() -> Result<()> {
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
            let text = fs::read_to_string(format!("outputs/ball-detection-v2/{name}-tracks.csv"))?;
            let rows: Vec<Vec<&str>> = text
                .lines()
                .skip(1)
                .map(|s| s.trim().split(',').collect())
                .collect();
            let reference: f64 = rows[0][0].parse()?;
            let diameter = if name.contains("gross") { 25 } else { 7 };
            let output = PathBuf::from(format!("outputs/ball-detection-v2/{name}-tracked.mp4"));
            if output.exists() {
                return Err(format!("output already exists: {}", output.display()).into());
            }
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
                    "1280x424",
                    "-framerate",
                    "25",
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
                .arg(&output)
                .stdin(Stdio::piped())
                .spawn()?;
            let mut input = child.stdin.take().unwrap();
            let ctx = egui::Context::default();
            let mut rgb = vec![0u8; 1280 * 424 * 3];
            let mut trail = VecDeque::new();
            let mut previous_id = String::new();
            let mut frames = 0;
            let result = (|| -> Result<()> {
                for (i, row) in rows.iter().enumerate() {
                    let time: f64 = row[0].parse()?;
                    let state = row[1];
                    if previous_id != row[2] || state == "lost" {
                        trail.clear();
                    }
                    previous_id = row[2].to_string();
                    if state == "measured" {
                        trail.push_back((time, [row[3].parse::<f32>()?, row[4].parse::<f32>()?]));
                    }
                    while trail.front().is_some_and(|(t, _)| time - t > 0.35) {
                        trail.pop_front();
                    }
                    if i % 4 != 0 {
                        continue;
                    }
                    let (_, frame) = engine.frame(time)?;
                    if !frame.imu_covered {
                        return Err("IMU coverage missing during track export".into());
                    }
                    let (_, _, images) = panels(&frame, &engine.camera);
                    rgb.fill(0);
                    for (panel, src) in [0, 3].into_iter().enumerate() {
                        for y in 0..360 {
                            let j = ((y + 32) * 1280 + panel * 640) * 3;
                            rgb[j..j + 640 * 3]
                                .copy_from_slice(&images[src][y * 640 * 3..(y + 1) * 640 * 3]);
                        }
                    }
                    let mut status = "NO BALL TRACK".to_string();
                    if state != "lost" {
                        let x: f32 = row[7].parse()?;
                        let y: f32 = row[8].parse()?;
                        let w: f32 = row[5].parse()?;
                        let h: f32 = row[6].parse()?;
                        let color = if state == "measured" {
                            [40, 240, 100]
                        } else {
                            [255, 175, 45]
                        };
                        let cx = 640. + x * 0.5;
                        let cy = 32. + y * 0.5;
                        let rx = w * 0.25 + 3.;
                        let ry = h * 0.25 + 3.;
                        if row[12] == "True" {
                            for i in 0..48 {
                                let a = i as f32 * std::f32::consts::TAU / 48.;
                                let b = (i + 1) as f32 * std::f32::consts::TAU / 48.;
                                line(
                                    &mut rgb,
                                    1280,
                                    424,
                                    [cx + (rx - 3.) * a.cos(), cy + (ry - 3.) * a.sin()],
                                    [cx + (rx - 3.) * b.cos(), cy + (ry - 3.) * b.sin()],
                                    color,
                                );
                            }
                        } else {
                            for (a, b) in [
                                ([cx - rx, cy - ry], [cx + rx, cy - ry]),
                                ([cx + rx, cy - ry], [cx + rx, cy + ry]),
                                ([cx + rx, cy + ry], [cx - rx, cy + ry]),
                                ([cx - rx, cy + ry], [cx - rx, cy - ry]),
                            ] {
                                line(&mut rgb, 1280, 424, a, b, color);
                            }
                        }
                        line(&mut rgb, 1280, 424, [cx - 5., cy], [cx + 5., cy], color);
                        line(&mut rgb, 1280, 424, [cx, cy - 5.], [cx, cy + 5.], color);
                        let rotation = engine
                            .imu
                            .relative_rotation(
                                reference + config.offset_s,
                                time + config.offset_s,
                                config.mount,
                            )
                            .unwrap();
                        let points: Vec<_> = trail
                            .iter()
                            .filter_map(|(_, p)| {
                                let ray = [
                                    (p[0] as f64 - calibration.principal_x_px as f64)
                                        / calibration.focal_length_x_px as f64,
                                    (p[1] as f64 - calibration.principal_y_px as f64)
                                        / calibration.focal_length_y_px as f64,
                                    1.,
                                ];
                                engine
                                    .camera
                                    .project(motion::transform(rotation, ray))
                                    .map(|p| [640. + p[0] * 0.5, 32. + p[1] * 0.5])
                            })
                            .collect();
                        for segment in points.windows(2) {
                            line(&mut rgb, 1280, 424, segment[0], segment[1], [40, 170, 220]);
                        }
                        let geometry = if row[12] == "True" {
                            "outline checked"
                        } else {
                            "no depth update"
                        };
                        status = format!("Track {}: {} | {geometry}", row[2], state);
                    }
                    draw_labels(
                        &ctx,
                        &mut rgb,
                        1280,
                        424,
                        &[
                            (12, 5, format!("All events | ball diameter {diameter} cm")),
                            (
                                652,
                                5,
                                "Improved ball tracking | compensated foreground".to_string(),
                            ),
                            (
                                12,
                                397,
                                format!(
                                    "Time {time:.2}s | {status} | green: measured, orange: predicted"
                                ),
                            ),
                        ],
                    );
                    input.write_all(&rgb)?;
                    frames += 1;
                }
                Ok(())
            })();
            drop(input);
            let status = child.wait()?;
            result?;
            if !status.success() {
                return Err("ffmpeg failed".into());
            }
            eprintln!("Saved {} ({frames} frames)", output.display());
        }
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().nth(1).as_deref() == Some("render") {
        comparison::render_tracks()
    } else {
        comparison::probe()
    }
}
