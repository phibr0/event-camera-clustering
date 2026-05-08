use crate::algorithms::ClusterDetection;
use crate::{EventError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BallDiameterSource {
    CircleFit,
    BboxWidth,
    BboxHeight,
    BboxAverage,
    BboxMax,
    BboxMin,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BallProjectionConfig {
    pub enabled: bool,
    pub diameter_m: f32,
    pub focal_length_x_px: f32,
    pub focal_length_y_px: f32,
    pub principal_x_px: f32,
    pub principal_y_px: f32,
    pub diameter_source: BallDiameterSource,
}

impl BallProjectionConfig {
    pub fn for_view(_width: usize, _height: usize) -> Self {
        Self {
            enabled: false,
            diameter_m: 0.067,
            focal_length_x_px: 410.0,
            focal_length_y_px: 409.0,
            principal_x_px: 320.0,
            principal_y_px: 240.0,
            diameter_source: BallDiameterSource::CircleFit,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.diameter_m <= 0.0 {
            return Err(EventError::InvalidConfig {
                field: "ball_diameter_m",
                message: "must be positive",
            });
        }
        if self.focal_length_x_px <= 0.0 {
            return Err(EventError::InvalidConfig {
                field: "focal_length_x_px",
                message: "must be positive",
            });
        }
        if self.focal_length_y_px <= 0.0 {
            return Err(EventError::InvalidConfig {
                field: "focal_length_y_px",
                message: "must be positive",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BallMeasurement3d {
    pub timestamp_us: u64,
    pub position_m: [f32; 3],
    pub pixel_diameter: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BallTrackEstimate {
    pub timestamp_us: u64,
    pub measurement: BallMeasurement3d,
    pub position_m: [f32; 3],
    pub velocity_mps: [f32; 3],
    pub speed_mps: f32,
    pub confidence: f32,
}

pub struct BallPathEstimator {
    config: BallProjectionConfig,
    x: Kalman1d,
    y: Kalman1d,
    z: Kalman1d,
    last_timestamp_us: Option<u64>,
    initialized: bool,
}

impl BallPathEstimator {
    pub fn new(config: BallProjectionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            x: Kalman1d::default(),
            y: Kalman1d::default(),
            z: Kalman1d::default(),
            last_timestamp_us: None,
            initialized: false,
        })
    }

    pub fn set_config(&mut self, config: BallProjectionConfig) -> Result<()> {
        config.validate()?;
        if self.config != config {
            self.reset();
            self.config = config;
        }
        Ok(())
    }

    pub fn estimate(&mut self, detection: &ClusterDetection) -> Option<BallTrackEstimate> {
        if !self.config.enabled {
            return None;
        }

        let measurement = measurement_from_detection(detection, self.config)?;

        if !self.initialized {
            self.x = Kalman1d::new(measurement.position_m[0]);
            self.y = Kalman1d::new(measurement.position_m[1]);
            self.z = Kalman1d::new(measurement.position_m[2]);
            self.last_timestamp_us = Some(measurement.timestamp_us);
            self.initialized = true;
        } else {
            let previous_us = self.last_timestamp_us.unwrap_or(measurement.timestamp_us);
            let dt = measurement.timestamp_us.saturating_sub(previous_us) as f32 / 1_000_000.0;
            self.x.step(dt, measurement.position_m[0]);
            self.y.step(dt, measurement.position_m[1]);
            self.z.step(dt, measurement.position_m[2]);
            self.last_timestamp_us = Some(measurement.timestamp_us);
        }

        let velocity_mps = [self.x.velocity, self.y.velocity, self.z.velocity];
        let speed_mps =
            (velocity_mps[0].powi(2) + velocity_mps[1].powi(2) + velocity_mps[2].powi(2)).sqrt();

        Some(BallTrackEstimate {
            timestamp_us: measurement.timestamp_us,
            measurement,
            position_m: [self.x.position, self.y.position, self.z.position],
            velocity_mps,
            speed_mps,
            confidence: detection.confidence,
        })
    }

    pub fn reset(&mut self) {
        self.x = Kalman1d::default();
        self.y = Kalman1d::default();
        self.z = Kalman1d::default();
        self.last_timestamp_us = None;
        self.initialized = false;
    }
}

fn measurement_from_detection(
    detection: &ClusterDetection,
    config: BallProjectionConfig,
) -> Option<BallMeasurement3d> {
    let pixel_diameter = pixel_diameter(detection, config.diameter_source);
    if pixel_diameter <= 0.0 {
        return None;
    }

    let z_m = config.focal_length_x_px * config.diameter_m / pixel_diameter;
    let x_m = (detection.centroid_x - config.principal_x_px) * z_m / config.focal_length_x_px;
    let y_m = (detection.centroid_y - config.principal_y_px) * z_m / config.focal_length_y_px;

    Some(BallMeasurement3d {
        timestamp_us: detection.timestamp_us,
        position_m: [x_m, y_m, z_m],
        pixel_diameter,
    })
}

fn pixel_diameter(detection: &ClusterDetection, source: BallDiameterSource) -> f32 {
    let width = f32::from(detection.bbox.width());
    let height = f32::from(detection.bbox.height());
    match source {
        BallDiameterSource::CircleFit => detection
            .circle_fit
            .map_or((width + height) / 2.0, |circle| circle.radius_px * 2.0),
        BallDiameterSource::BboxWidth => width,
        BallDiameterSource::BboxHeight => height,
        BallDiameterSource::BboxAverage => (width + height) / 2.0,
        BallDiameterSource::BboxMax => width.max(height),
        BallDiameterSource::BboxMin => width.min(height),
    }
}

#[derive(Debug, Clone, Copy)]
struct Kalman1d {
    position: f32,
    velocity: f32,
    p00: f32,
    p01: f32,
    p10: f32,
    p11: f32,
}

impl Default for Kalman1d {
    fn default() -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            p00: 1.0,
            p01: 0.0,
            p10: 0.0,
            p11: 1.0,
        }
    }
}

impl Kalman1d {
    fn new(position: f32) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    fn step(&mut self, dt: f32, measurement: f32) {
        let dt = dt.max(0.000_001);
        self.predict(dt);
        self.update(measurement);
    }

    fn predict(&mut self, dt: f32) {
        self.position += self.velocity * dt;

        let p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11;
        let p01 = self.p01 + dt * self.p11;
        let p10 = self.p10 + dt * self.p11;
        let p11 = self.p11;
        let process_noise = 0.05;

        self.p00 = p00 + process_noise * dt * dt;
        self.p01 = p01;
        self.p10 = p10;
        self.p11 = p11 + process_noise;
    }

    fn update(&mut self, measurement: f32) {
        let measurement_noise = 0.02;
        let residual = measurement - self.position;
        let residual_covariance = self.p00 + measurement_noise;
        let k0 = self.p00 / residual_covariance;
        let k1 = self.p10 / residual_covariance;

        self.position += k0 * residual;
        self.velocity += k1 * residual;

        let p00 = self.p00;
        let p01 = self.p01;
        let p10 = self.p10;
        let p11 = self.p11;

        self.p00 = (1.0 - k0) * p00;
        self.p01 = (1.0 - k0) * p01;
        self.p10 = p10 - k1 * p00;
        self.p11 = p11 - k1 * p01;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundingBox;

    #[test]
    fn projects_detection_from_known_ball_size() {
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
        let config = BallProjectionConfig {
            enabled: true,
            diameter_m: 0.1,
            focal_length_x_px: 1_000.0,
            focal_length_y_px: 1_000.0,
            principal_x_px: 320.0,
            principal_y_px: 240.0,
            diameter_source: BallDiameterSource::BboxAverage,
        };
        let measurement = measurement_from_detection(&detection, config).unwrap();

        assert_eq!(measurement.pixel_diameter, 20.0);
        assert_eq!(measurement.position_m, [0.0, 0.0, 5.0]);
    }
}
