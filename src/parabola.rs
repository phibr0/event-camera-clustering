use crate::algorithms::ClusterDetection;
use crate::{EventError, Result};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityAxis {
    XPositive,
    XNegative,
    YPositive,
    YNegative,
    ZPositive,
    ZNegative,
}

impl GravityAxis {
    fn vector(self, magnitude: f32) -> [f32; 3] {
        match self {
            GravityAxis::XPositive => [magnitude, 0.0, 0.0],
            GravityAxis::XNegative => [-magnitude, 0.0, 0.0],
            GravityAxis::YPositive => [0.0, magnitude, 0.0],
            GravityAxis::YNegative => [0.0, -magnitude, 0.0],
            GravityAxis::ZPositive => [0.0, 0.0, magnitude],
            GravityAxis::ZNegative => [0.0, 0.0, -magnitude],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParabolaFitConfig {
    pub enabled: bool,
    pub ball_diameter_m: f32,
    pub focal_length_x_px: f32,
    pub focal_length_y_px: f32,
    pub principal_x_px: f32,
    pub principal_y_px: f32,
    pub distortion_coefficients: [f32; 5],
    pub gravity_mps2: f32,
    pub gravity_axis: GravityAxis,
    pub buffer_len: usize,
    pub min_points: usize,
    pub max_pair_samples: usize,
    pub inlier_threshold_m: f32,
    pub min_inlier_ratio: f32,
    pub min_sample_dt_s: f32,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraCalibration {
    pub image_width: Option<usize>,
    pub image_height: Option<usize>,
    pub focal_length_x_px: f32,
    pub focal_length_y_px: f32,
    pub principal_x_px: f32,
    pub principal_y_px: f32,
    pub distortion_coefficients: [f32; 5],
}

impl ParabolaFitConfig {
    pub fn for_view(width: usize, height: usize) -> Self {
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        Self {
            enabled: false,
            ball_diameter_m: 0.067,
            focal_length_x_px: 410.0 * width / 640.0,
            focal_length_y_px: 409.0 * height / 480.0,
            principal_x_px: width / 2.0,
            principal_y_px: height / 2.0,
            distortion_coefficients: [0.0; 5],
            gravity_mps2: 9.81,
            gravity_axis: GravityAxis::YPositive,
            buffer_len: 50,
            min_points: 4,
            max_pair_samples: 200,
            inlier_threshold_m: 0.25,
            min_inlier_ratio: 0.5,
            min_sample_dt_s: 0.01,
            min_depth_m: 0.05,
            max_depth_m: 30.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.ball_diameter_m <= 0.0 {
            return Err(invalid_config(
                "parabola_ball_diameter_m",
                "must be positive",
            ));
        }
        if self.focal_length_x_px <= 0.0 {
            return Err(invalid_config(
                "parabola_focal_length_x_px",
                "must be positive",
            ));
        }
        if self.focal_length_y_px <= 0.0 {
            return Err(invalid_config(
                "parabola_focal_length_y_px",
                "must be positive",
            ));
        }
        if self.gravity_mps2 <= 0.0 {
            return Err(invalid_config("parabola_gravity_mps2", "must be positive"));
        }
        if self.buffer_len < 2 {
            return Err(invalid_config("parabola_buffer_len", "must be at least 2"));
        }
        if self.min_points < 2 {
            return Err(invalid_config("parabola_min_points", "must be at least 2"));
        }
        if self.min_points > self.buffer_len {
            return Err(invalid_config(
                "parabola_min_points",
                "must not exceed buffer_len",
            ));
        }
        if self.max_pair_samples == 0 {
            return Err(invalid_config(
                "parabola_max_pair_samples",
                "must be positive",
            ));
        }
        if self.inlier_threshold_m <= 0.0 {
            return Err(invalid_config(
                "parabola_inlier_threshold_m",
                "must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&self.min_inlier_ratio) {
            return Err(invalid_config(
                "parabola_min_inlier_ratio",
                "must be between 0 and 1",
            ));
        }
        if self.min_sample_dt_s <= 0.0 {
            return Err(invalid_config(
                "parabola_min_sample_dt_s",
                "must be positive",
            ));
        }
        if self.min_depth_m <= 0.0 {
            return Err(invalid_config("parabola_min_depth_m", "must be positive"));
        }
        if self.max_depth_m <= self.min_depth_m {
            return Err(invalid_config(
                "parabola_max_depth_m",
                "must be greater than min_depth_m",
            ));
        }
        Ok(())
    }

    pub fn adjust_default_intrinsics_for_view(
        &mut self,
        previous_width: usize,
        previous_height: usize,
        width: usize,
        height: usize,
    ) {
        let previous = Self::for_view(previous_width, previous_height);
        let updated = Self::for_view(width, height);
        if self.focal_length_x_px == previous.focal_length_x_px {
            self.focal_length_x_px = updated.focal_length_x_px;
        }
        if self.focal_length_y_px == previous.focal_length_y_px {
            self.focal_length_y_px = updated.focal_length_y_px;
        }
        if self.principal_x_px == previous.principal_x_px {
            self.principal_x_px = updated.principal_x_px;
        }
        if self.principal_y_px == previous.principal_y_px {
            self.principal_y_px = updated.principal_y_px;
        }
    }

    pub fn apply_calibration(&mut self, calibration: CameraCalibration) {
        self.focal_length_x_px = calibration.focal_length_x_px;
        self.focal_length_y_px = calibration.focal_length_y_px;
        self.principal_x_px = calibration.principal_x_px;
        self.principal_y_px = calibration.principal_y_px;
        self.distortion_coefficients = calibration.distortion_coefficients;
    }
}

pub fn parse_calibration_json(contents: &str) -> Option<CameraCalibration> {
    let matrix_start = contents.find("\"camera_matrix\"")?;
    let matrix_end = contents[matrix_start..]
        .find("\"distortion")
        .map(|offset| matrix_start + offset)
        .unwrap_or(contents.len());
    let matrix_values = extract_numbers(&contents[matrix_start..matrix_end]);
    if matrix_values.len() < 6 {
        return None;
    }
    let distortion_coefficients = extract_key_array_numbers(contents, "distortion_coefficients")
        .and_then(|values| {
            if values.len() >= 5 {
                Some([
                    values[0] as f32,
                    values[1] as f32,
                    values[2] as f32,
                    values[3] as f32,
                    values[4] as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or([0.0; 5]);

    Some(CameraCalibration {
        image_width: extract_key_number(contents, "image_width").map(|value| value as usize),
        image_height: extract_key_number(contents, "image_height").map(|value| value as usize),
        focal_length_x_px: matrix_values[0] as f32,
        focal_length_y_px: matrix_values[4] as f32,
        principal_x_px: matrix_values[2] as f32,
        principal_y_px: matrix_values[5] as f32,
        distortion_coefficients,
    })
}

fn extract_key_array_numbers(contents: &str, key: &str) -> Option<Vec<f64>> {
    let key_index = contents.find(&format!("\"{key}\""))?;
    let array_start = contents[key_index..].find('[')? + key_index;
    let array_end = contents[array_start..].find(']')? + array_start;
    Some(extract_numbers(&contents[array_start..=array_end]))
}

fn extract_key_number(contents: &str, key: &str) -> Option<f64> {
    let key_index = contents.find(&format!("\"{key}\""))?;
    extract_numbers(&contents[key_index..]).into_iter().next()
}

fn extract_numbers(contents: &str) -> Vec<f64> {
    let mut values = Vec::new();
    let mut token = String::new();
    for ch in contents.chars() {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            token.push(ch);
        } else if !token.is_empty() {
            if let Ok(value) = token.parse() {
                values.push(value);
            }
            token.clear();
        }
    }
    if !token.is_empty() {
        if let Ok(value) = token.parse() {
            values.push(value);
        }
    }
    values
}

impl Default for ParabolaFitConfig {
    fn default() -> Self {
        Self::for_view(640, 480)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParabolaPoint3d {
    pub timestamp_us: u64,
    pub position_m: [f32; 3],
    pub pixel_area: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParabolaFitEstimate {
    pub timestamp_us: u64,
    pub origin_timestamp_us: u64,
    pub initial_position_m: [f32; 3],
    pub initial_velocity_mps: [f32; 3],
    pub gravity_mps2: [f32; 3],
    pub inlier_count: usize,
    pub total_count: usize,
    pub mean_error_m: f32,
    pub max_error_m: f32,
    pub inliers: Vec<bool>,
}

impl ParabolaFitEstimate {
    pub fn sample_at_s(&self, t_s: f32) -> [f32; 3] {
        sample_parabola(
            self.initial_position_m,
            self.initial_velocity_mps,
            self.gravity_mps2,
            t_s,
        )
    }

    pub fn sample_at_timestamp_us(&self, timestamp_us: u64) -> [f32; 3] {
        let t_s = timestamp_us.saturating_sub(self.origin_timestamp_us) as f32 / 1_000_000.0;
        self.sample_at_s(t_s)
    }
}

pub struct TwoShotParabolaFitter {
    config: ParabolaFitConfig,
    points: VecDeque<ParabolaPoint3d>,
}

impl TwoShotParabolaFitter {
    pub fn new(config: ParabolaFitConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            points: VecDeque::new(),
        })
    }

    pub fn set_config(&mut self, config: ParabolaFitConfig) -> Result<()> {
        config.validate()?;
        if self.config != config {
            self.config = config;
            while self.points.len() > self.config.buffer_len {
                self.points.pop_front();
            }
        }
        Ok(())
    }

    pub fn config(&self) -> ParabolaFitConfig {
        self.config
    }

    pub fn points(&self) -> &VecDeque<ParabolaPoint3d> {
        &self.points
    }

    pub fn reset(&mut self) {
        self.points.clear();
    }

    pub fn push_detection(&mut self, detection: &ClusterDetection) -> Option<ParabolaFitEstimate> {
        if !self.config.enabled {
            self.points.clear();
            return None;
        }

        let point = point_from_detection(detection, self.config)?;
        self.push_point(point)
    }

    pub fn push_point(&mut self, point: ParabolaPoint3d) -> Option<ParabolaFitEstimate> {
        self.points.push_back(point);
        while self.points.len() > self.config.buffer_len {
            self.points.pop_front();
        }
        self.fit()
    }

    pub fn fit(&self) -> Option<ParabolaFitEstimate> {
        if !self.config.enabled || self.points.len() < self.config.min_points {
            return None;
        }

        let points: Vec<_> = self.points.iter().copied().collect();
        fit_points(&points, self.config, self.points.front()?.timestamp_us)
    }
}

pub fn point_from_detection(
    detection: &ClusterDetection,
    config: ParabolaFitConfig,
) -> Option<ParabolaPoint3d> {
    let width = f32::from(detection.bbox.width());
    let height = f32::from(detection.bbox.height());
    let ellipse_area_px = width * height * std::f32::consts::PI / 4.0;
    if !ellipse_area_px.is_finite() || ellipse_area_px <= 0.0 {
        return None;
    }

    let ball_radius_m = config.ball_diameter_m * 0.5;
    let ball_area_m2 = std::f32::consts::PI * ball_radius_m * ball_radius_m;
    let depth_m = config.focal_length_x_px * (ball_area_m2 / ellipse_area_px).sqrt();
    if !depth_m.is_finite() || depth_m < config.min_depth_m || depth_m > config.max_depth_m {
        return None;
    }

    let x_distorted = (detection.centroid_x - config.principal_x_px) / config.focal_length_x_px;
    let y_distorted = (detection.centroid_y - config.principal_y_px) / config.focal_length_y_px;
    let (x_norm, y_norm) =
        undistort_normalized(x_distorted, y_distorted, config.distortion_coefficients);
    let x_m = x_norm * depth_m;
    let y_m = y_norm * depth_m;

    Some(ParabolaPoint3d {
        timestamp_us: detection.timestamp_us,
        position_m: [x_m, y_m, depth_m],
        pixel_area: ellipse_area_px,
    })
}

pub(crate) fn undistort_normalized(x_distorted: f32, y_distorted: f32, coeffs: [f32; 5]) -> (f32, f32) {
    if coeffs.iter().all(|value| value.abs() <= f32::EPSILON) {
        return (x_distorted, y_distorted);
    }

    let [k1, k2, p1, p2, k3] = coeffs;
    let mut x = x_distorted;
    let mut y = y_distorted;
    for _ in 0..8 {
        let r2 = x * x + y * y;
        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        let delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (x_distorted - delta_x) / radial;
        y = (y_distorted - delta_y) / radial;
    }
    (x, y)
}

fn fit_points(
    points: &[ParabolaPoint3d],
    config: ParabolaFitConfig,
    origin_timestamp_us: u64,
) -> Option<ParabolaFitEstimate> {
    let gravity = config.gravity_axis.vector(config.gravity_mps2);
    let times = relative_times(points, origin_timestamp_us);
    let mut best: Option<CandidateScore> = None;
    let mut sampled_pairs = 0_usize;
    let pair_stride = pair_stride(points.len(), config.max_pair_samples);

    for i in (0..points.len()).step_by(pair_stride) {
        for j in ((i + 1)..points.len()).step_by(pair_stride) {
            if sampled_pairs >= config.max_pair_samples {
                break;
            }
            sampled_pairs += 1;
            if (times[j] - times[i]).abs() < config.min_sample_dt_s {
                continue;
            }
            let (position, velocity) = minimal_solution(
                points[i].position_m,
                times[i],
                points[j].position_m,
                times[j],
                gravity,
            )?;
            let score = score_model(points, &times, position, velocity, gravity, config);
            if score.inlier_count < config.min_points {
                continue;
            }
            if best.as_ref().is_none_or(|best| score.better_than(best)) {
                best = Some(score);
            }
        }
    }

    let best = best?;
    let required_inliers = (config.min_inlier_ratio * points.len() as f32).ceil() as usize;
    if best.inlier_count < required_inliers.max(config.min_points) {
        return None;
    }

    let (position, velocity) = least_squares_solution(points, &times, gravity, &best.inliers)?;
    let refined = score_model(points, &times, position, velocity, gravity, config);
    if refined.inlier_count < required_inliers.max(config.min_points) {
        return None;
    }

    Some(ParabolaFitEstimate {
        timestamp_us: points.last()?.timestamp_us,
        origin_timestamp_us,
        initial_position_m: position,
        initial_velocity_mps: velocity,
        gravity_mps2: gravity,
        inlier_count: refined.inlier_count,
        total_count: points.len(),
        mean_error_m: refined.mean_error_m,
        max_error_m: refined.max_error_m,
        inliers: refined.inliers,
    })
}

#[derive(Debug, Clone)]
struct CandidateScore {
    inlier_count: usize,
    mean_error_m: f32,
    max_error_m: f32,
    inliers: Vec<bool>,
}

impl CandidateScore {
    fn better_than(&self, other: &Self) -> bool {
        self.inlier_count > other.inlier_count
            || (self.inlier_count == other.inlier_count && self.mean_error_m < other.mean_error_m)
    }
}

fn relative_times(points: &[ParabolaPoint3d], origin_timestamp_us: u64) -> Vec<f32> {
    points
        .iter()
        .map(|point| point.timestamp_us.saturating_sub(origin_timestamp_us) as f32 / 1_000_000.0)
        .collect()
}

fn pair_stride(point_count: usize, max_pair_samples: usize) -> usize {
    let pair_count = point_count.saturating_mul(point_count.saturating_sub(1)) / 2;
    if pair_count <= max_pair_samples {
        1
    } else {
        ((pair_count as f32 / max_pair_samples as f32).sqrt().floor() as usize).max(1)
    }
}

fn minimal_solution(
    p0: [f32; 3],
    t0: f32,
    p1: [f32; 3],
    t1: f32,
    gravity: [f32; 3],
) -> Option<([f32; 3], [f32; 3])> {
    let dt = t1 - t0;
    if dt.abs() <= f32::EPSILON {
        return None;
    }

    let g0 = vec3_scale(gravity, 0.5 * t0 * t0);
    let g1 = vec3_scale(gravity, 0.5 * t1 * t1);
    let p0_comp = vec3_sub(p0, g0);
    let p1_comp = vec3_sub(p1, g1);
    let velocity = vec3_scale(vec3_sub(p1_comp, p0_comp), 1.0 / dt);
    let position = vec3_sub(p0_comp, vec3_scale(velocity, t0));

    if position
        .iter()
        .chain(velocity.iter())
        .all(|value| value.is_finite())
    {
        Some((position, velocity))
    } else {
        None
    }
}

fn least_squares_solution(
    points: &[ParabolaPoint3d],
    times: &[f32],
    gravity: [f32; 3],
    inliers: &[bool],
) -> Option<([f32; 3], [f32; 3])> {
    let mut n = 0.0_f32;
    let mut t_sum = 0.0_f32;
    let mut t2_sum = 0.0_f32;
    let mut p_sum = [0.0; 3];
    let mut tp_sum = [0.0; 3];

    for ((point, &time), &inlier) in points.iter().zip(times).zip(inliers) {
        if !inlier {
            continue;
        }
        let gravity_comp = vec3_sub(point.position_m, vec3_scale(gravity, 0.5 * time * time));
        n += 1.0;
        t_sum += time;
        t2_sum += time * time;
        p_sum = vec3_add(p_sum, gravity_comp);
        tp_sum = vec3_add(tp_sum, vec3_scale(gravity_comp, time));
    }

    let denom = t_sum * t_sum - n * t2_sum;
    if n < 2.0 || denom.abs() <= f32::EPSILON {
        return None;
    }

    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for axis in 0..3 {
        position[axis] = (t_sum * tp_sum[axis] - p_sum[axis] * t2_sum) / denom;
        velocity[axis] = (t_sum * p_sum[axis] - n * tp_sum[axis]) / denom;
    }

    Some((position, velocity))
}

fn score_model(
    points: &[ParabolaPoint3d],
    times: &[f32],
    position: [f32; 3],
    velocity: [f32; 3],
    gravity: [f32; 3],
    config: ParabolaFitConfig,
) -> CandidateScore {
    let mut inliers = Vec::with_capacity(points.len());
    let mut inlier_count = 0_usize;
    let mut error_sum = 0.0_f32;
    let mut max_error_m = 0.0_f32;

    for (point, &time) in points.iter().zip(times) {
        let predicted = sample_parabola(position, velocity, gravity, time);
        let error = vec3_length(vec3_sub(point.position_m, predicted));
        let inlier = error <= config.inlier_threshold_m;
        inliers.push(inlier);
        if inlier {
            inlier_count += 1;
            error_sum += error;
            max_error_m = max_error_m.max(error);
        }
    }

    CandidateScore {
        inlier_count,
        mean_error_m: if inlier_count == 0 {
            f32::INFINITY
        } else {
            error_sum / inlier_count as f32
        },
        max_error_m,
        inliers,
    }
}

pub fn sample_parabola(
    position: [f32; 3],
    velocity: [f32; 3],
    gravity: [f32; 3],
    t_s: f32,
) -> [f32; 3] {
    vec3_add(
        vec3_add(position, vec3_scale(velocity, t_s)),
        vec3_scale(gravity, 0.5 * t_s * t_s),
    )
}

fn invalid_config(field: &'static str, message: &'static str) -> EventError {
    EventError::InvalidConfig { field, message }
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_scale(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundingBox;

    fn config() -> ParabolaFitConfig {
        ParabolaFitConfig {
            enabled: true,
            ball_diameter_m: 0.1,
            focal_length_x_px: 1_000.0,
            focal_length_y_px: 1_000.0,
            principal_x_px: 320.0,
            principal_y_px: 240.0,
            distortion_coefficients: [0.0; 5],
            gravity_mps2: 9.81,
            gravity_axis: GravityAxis::YPositive,
            buffer_len: 50,
            min_points: 4,
            max_pair_samples: 200,
            inlier_threshold_m: 0.08,
            min_inlier_ratio: 0.5,
            min_sample_dt_s: 0.01,
            min_depth_m: 0.05,
            max_depth_m: 30.0,
        }
    }

    fn point(timestamp_us: u64, position_m: [f32; 3]) -> ParabolaPoint3d {
        ParabolaPoint3d {
            timestamp_us,
            position_m,
            pixel_area: 1.0,
        }
    }

    #[test]
    fn fits_noiseless_y_positive_parabola() {
        let config = config();
        let gravity = config.gravity_axis.vector(config.gravity_mps2);
        let position = [0.2, 0.3, 4.0];
        let velocity = [1.0, -2.0, -3.0];
        let mut fitter = TwoShotParabolaFitter::new(config).unwrap();
        let mut latest = None;

        for index in 0..8 {
            let t = index as f32 * 0.05;
            latest = fitter.push_point(point(
                (t * 1_000_000.0) as u64,
                sample_parabola(position, velocity, gravity, t),
            ));
        }

        let fit = latest.unwrap();
        for axis in 0..3 {
            assert!((fit.initial_position_m[axis] - position[axis]).abs() < 0.001);
            assert!((fit.initial_velocity_mps[axis] - velocity[axis]).abs() < 0.001);
        }
        assert_eq!(fit.inlier_count, 8);
    }

    #[test]
    fn rejects_outliers_with_two_shot_scoring() {
        let config = config();
        let gravity = config.gravity_axis.vector(config.gravity_mps2);
        let position = [0.0, 0.0, 3.0];
        let velocity = [0.5, 1.0, -1.0];
        let mut fitter = TwoShotParabolaFitter::new(config).unwrap();

        for index in 0..8 {
            let t = index as f32 * 0.05;
            fitter.push_point(point(
                (t * 1_000_000.0) as u64,
                sample_parabola(position, velocity, gravity, t),
            ));
        }
        fitter.push_point(point(450_000, [6.0, -4.0, 1.0]));
        let fit = fitter.push_point(point(500_000, [-5.0, 8.0, 4.0])).unwrap();

        assert_eq!(fit.inlier_count, 8);
        assert_eq!(fit.total_count, 10);
        assert!(fit.mean_error_m < 0.001);
    }

    #[test]
    fn skips_degenerate_timestamps() {
        let mut config = config();
        config.min_points = 2;
        let mut fitter = TwoShotParabolaFitter::new(config).unwrap();
        fitter.push_point(point(0, [0.0, 0.0, 1.0]));
        assert!(fitter.push_point(point(0, [1.0, 1.0, 1.0])).is_none());
    }

    #[test]
    fn projects_detection_depth_from_bbox_area() {
        let detection = ClusterDetection {
            timestamp_us: 0,
            window_start_us: 0,
            window_end_us: 0,
            centroid_x: 320.0,
            centroid_y: 240.0,
            bbox: BoundingBox {
                min_x: 310,
                min_y: 230,
                max_x: 329,
                max_y: 249,
            },
            event_count: 100,
            confidence: 1.0,
            circle_fit: None,
        };

        let projected = point_from_detection(&detection, config()).unwrap();
        assert!((projected.position_m[2] - 5.0).abs() < 0.001);
        assert_eq!(projected.position_m[0], 0.0);
        assert_eq!(projected.position_m[1], 0.0);
    }
}
