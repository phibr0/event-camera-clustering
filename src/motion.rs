//! Rotation compensation and motion segmentation in short event windows.
use crate::parabola::CameraCalibration;
use crate::{Event, EventError, Result};
use std::{fs, path::Path};

pub type Rotation = [[f64; 3]; 3];
pub const IDENTITY: Rotation = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

fn invalid(message: &'static str) -> EventError {
    EventError::InvalidConfig {
        field: "motion compensation",
        message,
    }
}

#[derive(Clone, Debug)]
pub struct ImuSample {
    pub time_s: f64,
    pub gyro: [f64; 3],
    orientation: [f64; 4],
}

pub struct ImuTrack {
    pub samples: Vec<ImuSample>,
}

impl ImuTrack {
    /// Reads the numeric CSV produced by ../imu/receive.py. Uses sensor time,
    /// never UDP arrival time. Integration is local in effect: only relative
    /// orientations within one event window are used downstream.
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_csv(&fs::read_to_string(path)?)
    }

    pub fn from_csv(csv: &str) -> Result<Self> {
        let mut lines = csv.lines();
        let columns: Vec<_> = lines
            .next()
            .ok_or(invalid("empty IMU CSV"))?
            .trim_start_matches('\u{feff}')
            .split(',')
            .map(str::trim)
            .collect();
        let indices = ["timestamp", "gx", "gy", "gz"]
            .map(|name| columns.iter().position(|&column| column == name));
        let [Some(t), Some(x), Some(y), Some(z)] = indices else {
            return Err(invalid(
                "IMU CSV needs timestamp,gx,gy,gz columns (seconds and rad/s)",
            ));
        };
        let mut samples = Vec::new();
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let values: Vec<_> = line.split(',').collect();
            let mut parsed = [0.; 4];
            for (dest, index) in parsed.iter_mut().zip([t, x, y, z]) {
                *dest = values
                    .get(index)
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .filter(|v| v.is_finite())
                    .ok_or(invalid("invalid IMU sample"))?;
            }
            samples.push(ImuSample {
                time_s: parsed[0],
                gyro: [parsed[1], parsed[2], parsed[3]],
                orientation: [1., 0., 0., 0.],
            });
        }
        if samples.len() < 2 {
            return Err(invalid("at least two IMU samples are required"));
        }
        // UDP may reorder packets. Duplicate sensor times are not extra samples.
        samples.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
        samples.dedup_by(|a, b| a.time_s == b.time_s);
        if samples.len() < 2 {
            return Err(invalid("IMU timestamps must span a nonzero interval"));
        }
        let origin = samples[0].time_s;
        for sample in &mut samples {
            sample.time_s -= origin;
        }
        for i in 1..samples.len() {
            let dt = samples[i].time_s - samples[i - 1].time_s;
            let angle = std::array::from_fn(|axis| {
                (samples[i - 1].gyro[axis] + samples[i].gyro[axis]) * dt * 0.5
            });
            samples[i].orientation =
                quaternion_product(samples[i - 1].orientation, angle_quaternion(angle));
        }
        Ok(Self { samples })
    }

    pub fn duration_s(&self) -> f64 {
        self.samples.last().unwrap().time_s
    }

    fn bracket(&self, time_s: f64) -> Option<(usize, f64)> {
        if !time_s.is_finite() || time_s < 0. || time_s > self.duration_s() {
            return None;
        }
        let i = self
            .samples
            .partition_point(|s| s.time_s <= time_s)
            .clamp(1, self.samples.len() - 1);
        let dt = self.samples[i].time_s - self.samples[i - 1].time_s;
        if dt > 0.05 {
            return None;
        } // Do not invent motion across dropped packets.
        Some((i, (time_s - self.samples[i - 1].time_s) / dt))
    }

    pub fn gyro_at(&self, time_s: f64) -> Option<[f64; 3]> {
        let (i, a) = self.bracket(time_s)?;
        Some(std::array::from_fn(|k| {
            self.samples[i - 1].gyro[k] * (1. - a) + self.samples[i].gyro[k] * a
        }))
    }

    fn orientation(&self, time_s: f64) -> Option<Rotation> {
        let (i, a) = self.bracket(time_s)?;
        let p = self.samples[i - 1].orientation;
        let mut q = self.samples[i].orientation;
        if p.iter().zip(q).map(|(p, q)| p * q).sum::<f64>() < 0. {
            q.iter_mut().for_each(|v| *v = -*v);
        }
        // Normalized interpolation is sufficient between adjacent 100 Hz samples.
        let mut q: [f64; 4] = std::array::from_fn(|k| p[k] * (1. - a) + q[k] * a);
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        q.iter_mut().for_each(|v| *v /= norm);
        Some(quaternion_matrix(q))
    }

    pub fn relative_rotation(&self, from_s: f64, to_s: f64, mount: Rotation) -> Option<Rotation> {
        // Refuse a window that crosses an IMU gap, even if both endpoints exist.
        let first = self
            .samples
            .partition_point(|s| s.time_s < from_s.min(to_s));
        let last = self
            .samples
            .partition_point(|s| s.time_s <= from_s.max(to_s));
        if self.samples[first.saturating_sub(1)..last.min(self.samples.len())]
            .windows(2)
            .any(|s| s[1].time_s - s[0].time_s > 0.05)
        {
            return None;
        }
        let relative = multiply(
            transpose(self.orientation(to_s)?),
            self.orientation(from_s)?,
        );
        Some(multiply(multiply(mount, relative), transpose(mount)))
    }
}

pub struct Camera {
    pub width: usize,
    pub height: usize,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    rays: Vec<[f32; 3]>,
}

impl Camera {
    pub fn new(calibration: CameraCalibration, width: usize, height: usize) -> Result<Self> {
        let c = calibration;
        if width == 0
            || height == 0
            || width > 8192
            || height > 8192
            || ![
                c.focal_length_x_px,
                c.focal_length_y_px,
                c.principal_x_px,
                c.principal_y_px,
            ]
            .iter()
            .all(|v| v.is_finite())
            || c.focal_length_x_px <= 0.
            || c.focal_length_y_px <= 0.
            || !c.distortion_coefficients.iter().all(|v| v.is_finite())
        {
            return Err(invalid("invalid camera calibration or image dimensions"));
        }
        if c.image_width.is_some_and(|w| w != width) || c.image_height.is_some_and(|h| h != height)
        {
            return Err(invalid(
                "camera calibration dimensions do not match recording",
            ));
        }
        let mut camera = Self {
            width,
            height,
            fx: c.focal_length_x_px as f64,
            fy: c.focal_length_y_px as f64,
            cx: c.principal_x_px as f64,
            cy: c.principal_y_px as f64,
            rays: Vec::with_capacity(width * height),
        };
        for y in 0..height {
            for x in 0..width {
                let (x, y) = crate::parabola::undistort_normalized(
                    (x as f32 - c.principal_x_px) / c.focal_length_x_px,
                    (y as f32 - c.principal_y_px) / c.focal_length_y_px,
                    c.distortion_coefficients,
                );
                camera.rays.push([x, y, 1.]);
            }
        }
        Ok(camera)
    }

    pub fn ray(&self, event: Event) -> Option<[f64; 3]> {
        if event.x as usize >= self.width || event.y as usize >= self.height {
            return None;
        }
        Some(self.rays[event.y as usize * self.width + event.x as usize].map(f64::from))
    }

    pub fn project(&self, ray: [f64; 3]) -> Option<[f32; 2]> {
        if ray[2] <= 0. {
            return None;
        }
        let p = [
            (self.fx * ray[0] / ray[2] + self.cx) as f32,
            (self.fy * ray[1] / ray[2] + self.cy) as f32,
        ];
        (p[0].is_finite() && p[1].is_finite()).then_some(p)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MotionConfig {
    /// IMU time relative to its first sample = camera relative time + offset.
    pub offset_s: f64,
    /// IMU axes -> camera axes, a proper rotation (not a reflection).
    pub mount: Rotation,
    pub cell_px: usize,
    pub threshold: f32,
    pub min_events: u32,
    pub min_component_cells: usize,
    /// Fraction of a region that must be outside the early-event neighborhood; 0 disables.
    pub min_new_fraction: f32,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            offset_s: 0.,
            mount: IDENTITY,
            cell_px: 3,
            threshold: 0.12,
            min_events: 3,
            min_component_cells: 8,
            min_new_fraction: 0.1,
        }
    }
}

pub struct MotionFrame {
    pub original: Vec<[f32; 2]>,
    pub compensated: Vec<[f32; 2]>,
    pub foreground_off: Vec<bool>,
    pub foreground_on: Vec<bool>,
    pub imu_covered: bool,
    pub focus_off: f64,
    pub focus_on: f64,
}

pub fn compensate(
    events: &[Event],
    origin_us: u64,
    start_us: u64,
    end_us: u64,
    camera: &Camera,
    imu: &ImuTrack,
    config: MotionConfig,
) -> MotionFrame {
    let reference = end_us.saturating_sub(origin_us) as f64 * 1e-6 + config.offset_s;
    let start = start_us.saturating_sub(origin_us) as f64 * 1e-6 + config.offset_s;
    let covered = imu
        .relative_rotation(start, reference, config.mount)
        .is_some();
    // Cache rotations every 0.5 ms; interpolate rays between bins to retain subpixel precision.
    let bins = ((end_us - start_us).div_ceil(500) as usize).max(1);
    let rotations: Vec<_> = (0..=bins)
        .map(|i| {
            let t = start + (reference - start) * i as f64 / bins as f64;
            imu.relative_rotation(t, reference, config.mount)
                .unwrap_or(IDENTITY)
        })
        .collect();
    let mut original = Vec::with_capacity(events.len());
    let mut compensated = Vec::with_capacity(events.len());
    for &event in events {
        let ray = camera.ray(event);
        original.push(ray.and_then(|r| camera.project(r)).unwrap_or([-1., -1.]));
        let p = ray.and_then(|r| {
            if !covered {
                return camera.project(r);
            }
            let a = ((event.timestamp_us.saturating_sub(start_us)) as f64
                / (end_us - start_us).max(1) as f64
                * bins as f64)
                .clamp(0., bins as f64);
            let i = (a as usize).min(bins - 1);
            let w = a - i as f64;
            let p = transform(rotations[i], r);
            let q = transform(rotations[i + 1], r);
            camera.project(std::array::from_fn(|k| p[k] * (1. - w) + q[k] * w))
        });
        compensated.push(p.unwrap_or([-1., -1.]));
    }
    let foreground_off = segment(events, &original, start_us, end_us, camera, config);
    let foreground_on = if covered {
        segment(events, &compensated, start_us, end_us, camera, config)
    } else {
        vec![false; events.len()]
    };
    let focus_off = focus(&original, camera.width, camera.height, 3);
    let focus_on = focus(&compensated, camera.width, camera.height, 3);
    MotionFrame {
        original,
        compensated,
        foreground_off,
        foreground_on,
        imu_covered: covered,
        focus_off,
        focus_on,
    }
}

fn cell_index(p: [f32; 2], width: usize, height: usize, cell: usize) -> Option<usize> {
    if p[0] < 0. || p[1] < 0. || p[0] >= width as f32 || p[1] >= height as f32 {
        return None;
    }
    Some(p[1] as usize / cell * width.div_ceil(cell) + p[0] as usize / cell)
}

fn segment(
    events: &[Event],
    points: &[[f32; 2]],
    start_us: u64,
    end_us: u64,
    camera: &Camera,
    config: MotionConfig,
) -> Vec<bool> {
    let cell = config.cell_px.max(1);
    let width = camera.width.div_ceil(cell);
    let height = camera.height.div_ceil(cell);
    let mut count = vec![0u32; width * height];
    let mut times = vec![0f32; width * height];
    let mut early = vec![0u32; width * height];
    let duration = (end_us - start_us).max(1) as f32;
    for (&event, &p) in events.iter().zip(points) {
        if let Some(i) = cell_index(p, camera.width, camera.height, cell) {
            count[i] += 1;
            let t = event.timestamp_us.saturating_sub(start_us) as f32 / duration;
            times[i] += t;
            if config.min_new_fraction > 0. && t < 0.25 {
                early[i] += 1;
            }
        }
    }
    let mut mean = 0.;
    let mut occupied = 0;
    for i in 0..count.len() {
        if count[i] >= config.min_events {
            times[i] /= count[i] as f32;
            mean += times[i];
            occupied += 1;
        }
    }
    if occupied == 0 {
        return vec![false; events.len()];
    }
    mean /= occupied as f32;
    // ponytail: mean-time residual assumes a dominant static background. Parallax
    // and abrupt illumination can pass; use depth-aware motion if rotation is insufficient.
    let mut candidate: Vec<_> = count
        .iter()
        .zip(&times)
        .map(|(&n, &t)| n >= config.min_events && t - mean > config.threshold)
        .collect();
    let mut keep = vec![false; count.len()];
    let mut component = Vec::new();
    for seed in 0..count.len() {
        if !candidate[seed] {
            continue;
        }
        component.clear();
        component.push(seed);
        candidate[seed] = false;
        let mut head = 0;
        let mut new_cells = 0;
        while head < component.len() {
            let i = component[head];
            head += 1;
            let x = i % width;
            let y = i / width;
            let mut early_support = 0u64;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let j = ny * width + nx;
                    early_support += u64::from(early[j]);
                    if candidate[j] {
                        candidate[j] = false;
                        component.push(j);
                    }
                }
            }
            // Activity split across cell boundaries is still earlier background evidence.
            new_cells += usize::from(early_support < u64::from(config.min_events));
        }
        // A late burst on an already occupied edge is not enough evidence of motion.
        // Test the whole region so its slower edges survive alongside a moving part.
        // ponytail: one-cell tolerance suppresses small residual drift but can also
        // reject slow targets; lower min_new_fraction (0 disables) for that tradeoff.
        if component.len() >= config.min_component_cells
            && new_cells as f32 >= component.len() as f32 * config.min_new_fraction
        {
            for &i in &component {
                keep[i] = true;
            }
        }
    }
    points
        .iter()
        .map(|&p| cell_index(p, camera.width, camera.height, cell).is_some_and(|i| keep[i]))
        .collect()
}

/// Bilinear event-image energy, used only to measure alignment, not foreground recall.
pub fn focus(points: &[[f32; 2]], width: usize, height: usize, scale: usize) -> f64 {
    let w = width.div_ceil(scale);
    let h = height.div_ceil(scale);
    let mut counts = vec![0f32; w * h];
    for &p in points {
        let x = p[0] / scale as f32;
        let y = p[1] / scale as f32;
        if x < 0. || y < 0. || x >= w as f32 - 1. || y >= h as f32 - 1. {
            continue;
        }
        let ix = x as usize;
        let iy = y as usize;
        let dx = x - ix as f32;
        let dy = y - iy as f32;
        counts[iy * w + ix] += (1. - dx) * (1. - dy);
        counts[iy * w + ix + 1] += dx * (1. - dy);
        counts[(iy + 1) * w + ix] += (1. - dx) * dy;
        counts[(iy + 1) * w + ix + 1] += dx * dy;
    }
    counts.iter().map(|&n| (n * n) as f64).sum()
}

pub fn transform(m: Rotation, v: [f64; 3]) -> [f64; 3] {
    m.map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
}
pub fn transpose(m: Rotation) -> Rotation {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}
pub fn multiply(a: Rotation, b: Rotation) -> Rotation {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
pub fn rotation_vector(v: [f64; 3]) -> Rotation {
    quaternion_matrix(angle_quaternion(v))
}

pub struct Alignment {
    pub offset_s: f64,
    pub mount: Rotation,
    pub focus_gain: f64,
    pub windows: usize,
}

/// Jointly search time offset and the 24 right-handed axis mappings, then refine
/// the mounting rotation. Diverse background motion is needed to identify these.
pub fn auto_align(
    windows: &[Vec<Event>],
    origin_us: u64,
    camera: &Camera,
    imu: &ImuTrack,
    center_s: f64,
    radius_s: f64,
) -> Result<Alignment> {
    let samples: Vec<_> = windows
        .iter()
        .filter_map(|events| {
            let first = events.first()?;
            let last = events.last()?;
            if events.len() < 1500 {
                return None;
            }
            let center = (first.timestamp_us as f64 + last.timestamp_us as f64) * 0.5;
            let points: Vec<_> = events
                .iter()
                .step_by((events.len() / 2500).max(1))
                .take(2500)
                .filter(|e| {
                    e.x > 50
                        && e.y > 50
                        && (e.x as usize) + 50 < camera.width
                        && (e.y as usize) + 50 < camera.height
                })
                .filter_map(|&e| Some((camera.ray(e)?, (e.timestamp_us as f64 - center) * 1e-6)))
                .collect();
            let original: Vec<_> = points
                .iter()
                .filter_map(|&(r, _)| camera.project(r))
                .collect();
            let baseline = focus(&original, camera.width, camera.height, 3);
            (baseline > 0.).then_some(((center - origin_us as f64) * 1e-6, points, baseline))
        })
        .collect();
    let score = |offset: f64, mount: Rotation| {
        let mut total = 0.;
        let mut used = 0;
        for (t, points, baseline) in &samples {
            let Some(gyro) = imu.gyro_at(t + offset) else {
                continue;
            };
            if gyro.iter().map(|v| v * v).sum::<f64>() < 0.08f64.powi(2) {
                continue;
            }
            let omega = transform(mount, gyro);
            let projected: Vec<_> = points
                .iter()
                .filter_map(|&(ray, dt)| {
                    let cross = [
                        omega[1] * ray[2] - omega[2] * ray[1],
                        omega[2] * ray[0] - omega[0] * ray[2],
                        omega[0] * ray[1] - omega[1] * ray[0],
                    ];
                    camera.project(std::array::from_fn(|k| ray[k] + cross[k] * dt))
                })
                .collect();
            total += (focus(&projected, camera.width, camera.height, 3) / baseline)
                .max(1e-12)
                .ln();
            used += 1;
        }
        (
            if used >= 6 {
                total / used as f64
            } else {
                f64::NEG_INFINITY
            },
            used,
        )
    };
    let mut best = (f64::NEG_INFINITY, 0);
    let mut offset = center_s;
    let mut mount = IDENTITY;
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for tick in 0..=((radius_s * 20.).ceil() as usize) {
        let candidate_offset = center_s - radius_s + tick as f64 * 0.1;
        for axes in permutations {
            for mask in 0..8 {
                let mut m = [[0.; 3]; 3];
                for i in 0..3 {
                    m[i][axes[i]] = if mask & (1 << i) == 0 { -1. } else { 1. };
                }
                let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                if det < 0. {
                    continue;
                }
                let s = score(candidate_offset, m);
                if s.0 > best.0 {
                    best = s;
                    offset = candidate_offset;
                    mount = m;
                }
            }
        }
    }
    if !best.0.is_finite() {
        return Err(invalid(
            "not enough overlapping camera/IMU motion to auto-align; supply a manual offset and mounting rotation",
        ));
    }
    for step in [0.06, 0.025, 0.01, 0.004] {
        for _ in 0..10 {
            let mut changed = false;
            for axis in 0..4 {
                for sign in [-1., 1.] {
                    let mut m = mount;
                    let mut t = offset;
                    if axis == 3 {
                        t += step * sign;
                    } else {
                        let mut v = [0.; 3];
                        v[axis] = step * sign;
                        m = multiply(rotation_vector(v), mount);
                    }
                    let s = score(t, m);
                    if s.0 > best.0 {
                        best = s;
                        mount = m;
                        offset = t;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
    Ok(Alignment {
        offset_s: offset,
        mount,
        focus_gain: best.0.exp(),
        windows: best.1,
    })
}

fn angle_quaternion(v: [f64; 3]) -> [f64; 4] {
    let angle = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = if angle < 1e-12 {
        0.5
    } else {
        (angle * 0.5).sin() / angle
    };
    [
        (angle * 0.5).cos(),
        v[0] * scale,
        v[1] * scale,
        v[2] * scale,
    ]
}
fn quaternion_product(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn quaternion_matrix(q: [f64; 4]) -> Rotation {
    let [w, x, y, z] = q;
    [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_motion_check_rejects_repeated_edges_without_eroding_moving_regions() {
        let camera = Camera::new(
            CameraCalibration {
                image_width: Some(96),
                image_height: Some(72),
                focal_length_x_px: 80.,
                focal_length_y_px: 80.,
                principal_x_px: 48.,
                principal_y_px: 36.,
                distortion_coefficients: [0.; 5],
            },
            96,
            72,
        )
        .unwrap();
        let mut events = Vec::new();
        for y in (48..66).step_by(3) {
            for x in (3..90).step_by(3) {
                for timestamp_us in [4000, 5000, 6000] {
                    events.push(Event {
                        x,
                        y,
                        timestamp_us,
                        polarity: true,
                    });
                }
            }
        }
        for y in (0..24).step_by(3) {
            for repeat in 0..3 {
                // Sparse early activity straddles the late background cell: no
                // single early cell reaches min_events. The moving region
                // has both overlapping and new cells, and should survive intact.
                for (x, timestamp_us) in [
                    (if repeat < 2 { 30 } else { 36 }, 1000),
                    (33, 9000),
                    (60, 1000),
                    (if y < 12 { 63 } else { 66 }, 9000),
                ] {
                    events.push(Event {
                        x,
                        y,
                        timestamp_us,
                        polarity: true,
                    });
                }
            }
        }
        let points: Vec<_> = events.iter().map(|e| [e.x as f32, e.y as f32]).collect();
        let baseline = segment(
            &events,
            &points,
            0,
            10_000,
            &camera,
            MotionConfig {
                min_new_fraction: 0.,
                ..Default::default()
            },
        );
        let filtered = segment(
            &events,
            &points,
            0,
            10_000,
            &camera,
            MotionConfig::default(),
        );
        assert_eq!(baseline.iter().filter(|&&v| v).count(), 48);
        assert_eq!(filtered.iter().filter(|&&v| v).count(), 24);
        for (event, &keep) in events.iter().zip(&filtered) {
            assert_eq!(keep, event.timestamp_us == 9000 && event.x >= 63);
        }
    }

    #[test]
    fn rotation_compensation_preserves_independent_motion_and_rejects_invalid_imu() {
        assert!(ImuTrack::from_csv("timestamp,gx,gy,gz\n0,NaN,0,0\n1,0,0,0").is_err());
        let mut csv = String::from("timestamp,gx,gy,gz\n");
        for i in 0..=20 {
            csv.push_str(&format!("{},0,1,0\n", i as f64 * 0.01));
        }
        let imu = ImuTrack::from_csv(&csv).unwrap();
        assert!(imu.gyro_at(-0.01).is_none());
        assert!(imu.gyro_at(0.21).is_none());
        let camera = Camera::new(
            CameraCalibration {
                image_width: Some(320),
                image_height: Some(240),
                focal_length_x_px: 300.,
                focal_length_y_px: 300.,
                principal_x_px: 160.,
                principal_y_px: 120.,
                distortion_coefficients: [0.; 5],
            },
            320,
            240,
        )
        .unwrap();
        let mut events = Vec::new();
        let mut moving = Vec::new();
        // Static vertical edges and an object moving opposite to the camera flow.
        for i in 0..=40 {
            let t = i as f64 * 0.001;
            for x in [60., 100., 140., 180., 220.] {
                for y in (30..110).step_by(3) {
                    let ray = transform(
                        rotation_vector([0., -t, 0.]),
                        [(x - 160.) / 300., (y as f64 - 120.) / 300., 1.],
                    );
                    let p = camera.project(ray).unwrap();
                    events.push(Event {
                        timestamp_us: i * 1000,
                        x: p[0].round() as u16,
                        y: p[1].round() as u16,
                        polarity: true,
                    });
                    moving.push(false);
                }
            }
            for y in (150..200).step_by(2) {
                let ray = transform(
                    rotation_vector([0., -t, 0.]),
                    [(50. + t * 900. - 160.) / 300., (y as f64 - 120.) / 300., 1.],
                );
                let p = camera.project(ray).unwrap();
                events.push(Event {
                    timestamp_us: i * 1000,
                    x: p[0].round() as u16,
                    y: p[1].round() as u16,
                    polarity: true,
                });
                moving.push(true);
            }
        }
        let frame = compensate(
            &events,
            0,
            0,
            40_000,
            &camera,
            &imu,
            MotionConfig::default(),
        );
        assert!(frame.imu_covered);
        assert!(frame.focus_on > frame.focus_off * 1.5);
        let kept_background = frame
            .foreground_on
            .iter()
            .zip(&moving)
            .filter(|(keep, obj)| **keep && !**obj)
            .count();
        let kept_object = frame
            .foreground_on
            .iter()
            .zip(&moving)
            .filter(|(keep, obj)| **keep && **obj)
            .count();
        assert!(
            kept_background < 100,
            "background events retained: {kept_background}"
        );
        assert!(kept_object > 60, "object events retained: {kept_object}");
        let missing = compensate(
            &events,
            0,
            0,
            40_000,
            &camera,
            &imu,
            MotionConfig {
                offset_s: -1.,
                ..Default::default()
            },
        );
        assert!(!missing.imu_covered);
        assert!(!missing.foreground_on.iter().any(|v| *v));
    }
}
