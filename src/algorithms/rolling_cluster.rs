use crate::algorithms::EventAlgorithm;
use crate::event::{BoundingBox, Event};
use crate::{EventError, Result};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct RollingClusterTrackerConfig {
    pub window_us: u64,
    pub step_us: u64,
    pub cell_size: u16,
    pub min_events: usize,
    pub min_cells: usize,
    pub max_bbox_width: u16,
    pub max_bbox_height: u16,
    pub circle_fit: bool,
    pub circle_inlier_tolerance_px: f32,
}

impl Default for RollingClusterTrackerConfig {
    fn default() -> Self {
        Self {
            window_us: 20_000,
            step_us: 5_000,
            cell_size: 2,
            min_events: 20,
            min_cells: 3,
            max_bbox_width: 200,
            max_bbox_height: 200,
            circle_fit: true,
            circle_inlier_tolerance_px: 2.5,
        }
    }
}

impl RollingClusterTrackerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.window_us == 0 {
            return Err(EventError::InvalidConfig {
                field: "window_us",
                message: "must be positive",
            });
        }
        if self.step_us == 0 {
            return Err(EventError::InvalidConfig {
                field: "step_us",
                message: "must be positive",
            });
        }
        if self.cell_size == 0 {
            return Err(EventError::InvalidConfig {
                field: "cell_size",
                message: "must be positive",
            });
        }
        if self.circle_inlier_tolerance_px <= 0.0 {
            return Err(EventError::InvalidConfig {
                field: "circle_inlier_tolerance_px",
                message: "must be positive",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleFit {
    pub center_x: f32,
    pub center_y: f32,
    pub radius_px: f32,
    pub inlier_count: usize,
    pub inlier_ratio: f32,
    pub mean_error_px: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterDetection {
    pub timestamp_us: u64,
    pub window_start_us: u64,
    pub window_end_us: u64,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub bbox: BoundingBox,
    pub event_count: usize,
    pub confidence: f32,
    pub circle_fit: Option<CircleFit>,
}

pub struct RollingClusterTracker {
    config: RollingClusterTrackerConfig,
    events: VecDeque<Event>,
    next_emit_us: Option<u64>,
    previous_centroid: Option<(f32, f32)>,
}

#[derive(Debug, Clone)]
struct CellAccum {
    count: usize,
    sum_x: u64,
    sum_y: u64,
    bbox: BoundingBox,
    events: Vec<Event>,
}

#[derive(Debug, Clone)]
struct ClusterAccum {
    count: usize,
    sum_x: u64,
    sum_y: u64,
    bbox: BoundingBox,
    cell_count: usize,
    events: Vec<Event>,
}

impl RollingClusterTracker {
    pub fn new(config: RollingClusterTrackerConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            events: VecDeque::new(),
            next_emit_us: None,
            previous_centroid: None,
        })
    }

    pub fn process_event(&mut self, event: Event) -> Vec<ClusterDetection> {
        self.events.push_back(event);
        self.drop_old_events(event.timestamp_us);

        let next_emit_us = self
            .next_emit_us
            .get_or_insert(event.timestamp_us + self.config.step_us);
        if event.timestamp_us < *next_emit_us {
            return Vec::new();
        }

        let mut detections = Vec::new();
        while event.timestamp_us >= self.next_emit_us.unwrap() {
            let timestamp_us = self.next_emit_us.unwrap();
            if let Some(detection) = self.detect_at(timestamp_us) {
                self.previous_centroid = Some((detection.centroid_x, detection.centroid_y));
                detections.push(detection);
            }
            self.next_emit_us = Some(timestamp_us + self.config.step_us);
        }

        detections
    }

    pub fn finish(&mut self) -> Option<ClusterDetection> {
        let timestamp_us = self.events.back()?.timestamp_us;
        let detection = self.detect_at(timestamp_us)?;
        self.previous_centroid = Some((detection.centroid_x, detection.centroid_y));
        Some(detection)
    }

    pub fn set_config(&mut self, config: RollingClusterTrackerConfig) -> Result<()> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    fn drop_old_events(&mut self, now_us: u64) {
        let min_timestamp_us = now_us.saturating_sub(self.config.window_us);
        while self
            .events
            .front()
            .is_some_and(|event| event.timestamp_us < min_timestamp_us)
        {
            self.events.pop_front();
        }
    }

    fn detect_at(&self, timestamp_us: u64) -> Option<ClusterDetection> {
        let window_start_us = timestamp_us.saturating_sub(self.config.window_us);
        let mut cells: HashMap<(u16, u16), CellAccum> = HashMap::new();

        for event in self.events.iter().filter(|event| {
            event.timestamp_us >= window_start_us && event.timestamp_us <= timestamp_us
        }) {
            let key = (
                event.x / self.config.cell_size,
                event.y / self.config.cell_size,
            );
            cells
                .entry(key)
                .and_modify(|cell| cell.add(*event))
                .or_insert_with(|| CellAccum::from_event(*event));
        }

        if cells.is_empty() {
            return None;
        }

        self.find_best_cluster(&cells).map(|cluster| {
            let centroid_x = cluster.sum_x as f32 / cluster.count as f32;
            let centroid_y = cluster.sum_y as f32 / cluster.count as f32;
            let circle_fit = if self.config.circle_fit {
                fit_circle_ransac(&cluster.events, self.config.circle_inlier_tolerance_px)
            } else {
                None
            };
            let bbox_area = usize::from(cluster.bbox.width()) * usize::from(cluster.bbox.height());
            let density = if bbox_area == 0 {
                0.0
            } else {
                (cluster.count as f32 / bbox_area as f32).min(1.0)
            };
            let confidence = (cluster.count as f32 / self.config.min_events.max(1) as f32).min(1.0)
                * density.sqrt();

            ClusterDetection {
                timestamp_us,
                window_start_us,
                window_end_us: timestamp_us,
                centroid_x,
                centroid_y,
                bbox: cluster.bbox,
                event_count: cluster.count,
                confidence,
                circle_fit,
            }
        })
    }

    fn find_best_cluster(&self, cells: &HashMap<(u16, u16), CellAccum>) -> Option<ClusterAccum> {
        let mut visited = HashSet::new();
        let mut best: Option<(f32, ClusterAccum)> = None;

        for &key in cells.keys() {
            if visited.contains(&key) {
                continue;
            }

            let cluster = flood_fill_cluster(key, cells, &mut visited)?;
            if cluster.count < self.config.min_events {
                continue;
            }
            if cluster.cell_count < self.config.min_cells {
                continue;
            }
            if cluster.bbox.width() > self.config.max_bbox_width
                || cluster.bbox.height() > self.config.max_bbox_height
            {
                continue;
            }

            let centroid_x = cluster.sum_x as f32 / cluster.count as f32;
            let centroid_y = cluster.sum_y as f32 / cluster.count as f32;
            let continuity_bonus = self.previous_centroid.map_or(1.0, |(prev_x, prev_y)| {
                let dx = centroid_x - prev_x;
                let dy = centroid_y - prev_y;
                let distance = (dx * dx + dy * dy).sqrt();
                1.0 / (1.0 + distance / 100.0)
            });
            let score = cluster.count as f32 * continuity_bonus;

            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score > *best_score)
            {
                best = Some((score, cluster));
            }
        }

        best.map(|(_, cluster)| cluster)
    }
}

impl EventAlgorithm for RollingClusterTracker {
    type Output = ClusterDetection;

    fn process_event(&mut self, event: Event) -> Vec<Self::Output> {
        RollingClusterTracker::process_event(self, event)
    }

    fn finish(&mut self) -> Vec<Self::Output> {
        RollingClusterTracker::finish(self).into_iter().collect()
    }
}

impl CellAccum {
    fn from_event(event: Event) -> Self {
        Self {
            count: 1,
            sum_x: u64::from(event.x),
            sum_y: u64::from(event.y),
            bbox: BoundingBox {
                min_x: event.x,
                min_y: event.y,
                max_x: event.x,
                max_y: event.y,
            },
            events: vec![event],
        }
    }

    fn add(&mut self, event: Event) {
        self.count += 1;
        self.sum_x += u64::from(event.x);
        self.sum_y += u64::from(event.y);
        self.bbox.min_x = self.bbox.min_x.min(event.x);
        self.bbox.min_y = self.bbox.min_y.min(event.y);
        self.bbox.max_x = self.bbox.max_x.max(event.x);
        self.bbox.max_y = self.bbox.max_y.max(event.y);
        self.events.push(event);
    }
}

impl ClusterAccum {
    fn add_cell(&mut self, cell: &CellAccum) {
        self.count += cell.count;
        self.sum_x += cell.sum_x;
        self.sum_y += cell.sum_y;
        self.bbox.min_x = self.bbox.min_x.min(cell.bbox.min_x);
        self.bbox.min_y = self.bbox.min_y.min(cell.bbox.min_y);
        self.bbox.max_x = self.bbox.max_x.max(cell.bbox.max_x);
        self.bbox.max_y = self.bbox.max_y.max(cell.bbox.max_y);
        self.cell_count += 1;
        self.events.extend(cell.events.iter().copied());
    }
}

fn flood_fill_cluster(
    start: (u16, u16),
    cells: &HashMap<(u16, u16), CellAccum>,
    visited: &mut HashSet<(u16, u16)>,
) -> Option<ClusterAccum> {
    let start_cell = cells.get(&start)?;
    let mut cluster = ClusterAccum {
        count: 0,
        sum_x: 0,
        sum_y: 0,
        bbox: start_cell.bbox,
        cell_count: 0,
        events: Vec::new(),
    };

    let mut queue = VecDeque::from([start]);
    visited.insert(start);

    while let Some(key) = queue.pop_front() {
        let cell = cells.get(&key)?;
        cluster.add_cell(cell);

        let (cell_x, cell_y) = key;
        for dx in -1_i32..=1 {
            for dy in -1_i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let neighbor_x = i32::from(cell_x) + dx;
                let neighbor_y = i32::from(cell_y) + dy;
                if neighbor_x < 0 || neighbor_y < 0 {
                    continue;
                }

                let neighbor = (neighbor_x as u16, neighbor_y as u16);
                if cells.contains_key(&neighbor) && visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    Some(cluster)
}

fn fit_circle_ransac(events: &[Event], inlier_tolerance_px: f32) -> Option<CircleFit> {
    if events.len() < 6 {
        return None;
    }

    let points = sample_points(events, 96);
    if points.len() < 6 {
        return None;
    }

    let mut best: Option<CircleFit> = None;
    let step = (points.len() / 24).max(1);

    for i in (0..points.len()).step_by(step) {
        for j in ((i + 1)..points.len()).step_by(step) {
            for k in ((j + 1)..points.len()).step_by(step) {
                let Some((center_x, center_y, radius_px)) =
                    circle_from_three_points(points[i], points[j], points[k])
                else {
                    continue;
                };
                if !radius_px.is_finite() || radius_px < 1.0 {
                    continue;
                }

                let fit = score_circle(&points, center_x, center_y, radius_px, inlier_tolerance_px);
                if best.as_ref().is_none_or(|best| {
                    fit.inlier_count > best.inlier_count
                        || (fit.inlier_count == best.inlier_count
                            && fit.mean_error_px < best.mean_error_px)
                }) {
                    best = Some(fit);
                }
            }
        }
    }

    let fit = best?;
    if fit.inlier_count < 6 || fit.inlier_ratio < 0.25 {
        return None;
    }
    Some(fit)
}

fn sample_points(events: &[Event], max_points: usize) -> Vec<(f32, f32)> {
    let step = (events.len() / max_points).max(1);
    events
        .iter()
        .step_by(step)
        .take(max_points)
        .map(|event| (f32::from(event.x), f32::from(event.y)))
        .collect()
}

fn circle_from_three_points(
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
) -> Option<(f32, f32, f32)> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 0.001 {
        return None;
    }

    let a_sq = a.0 * a.0 + a.1 * a.1;
    let b_sq = b.0 * b.0 + b.1 * b.1;
    let c_sq = c.0 * c.0 + c.1 * c.1;
    let center_x = (a_sq * (b.1 - c.1) + b_sq * (c.1 - a.1) + c_sq * (a.1 - b.1)) / d;
    let center_y = (a_sq * (c.0 - b.0) + b_sq * (a.0 - c.0) + c_sq * (b.0 - a.0)) / d;
    let radius_px = ((a.0 - center_x).powi(2) + (a.1 - center_y).powi(2)).sqrt();

    Some((center_x, center_y, radius_px))
}

fn score_circle(
    points: &[(f32, f32)],
    center_x: f32,
    center_y: f32,
    radius_px: f32,
    inlier_tolerance_px: f32,
) -> CircleFit {
    let mut inlier_count = 0;
    let mut total_error = 0.0;

    for &(x, y) in points {
        let distance = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
        let error = (distance - radius_px).abs();
        if error <= inlier_tolerance_px {
            inlier_count += 1;
            total_error += error;
        }
    }

    CircleFit {
        center_x,
        center_y,
        radius_px,
        inlier_count,
        inlier_ratio: inlier_count as f32 / points.len() as f32,
        mean_error_px: if inlier_count == 0 {
            f32::INFINITY
        } else {
            total_error / inlier_count as f32
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_largest_cluster_in_rolling_window() {
        let mut tracker = RollingClusterTracker::new(RollingClusterTrackerConfig {
            window_us: 10_000,
            step_us: 1_000,
            cell_size: 1,
            min_events: 3,
            min_cells: 3,
            max_bbox_width: 20,
            max_bbox_height: 20,
            circle_fit: true,
            circle_inlier_tolerance_px: 2.5,
        })
        .unwrap();

        let events = [
            Event {
                timestamp_us: 0,
                x: 1,
                y: 1,
                polarity: true,
            },
            Event {
                timestamp_us: 100,
                x: 50,
                y: 50,
                polarity: true,
            },
            Event {
                timestamp_us: 200,
                x: 51,
                y: 50,
                polarity: true,
            },
            Event {
                timestamp_us: 300,
                x: 50,
                y: 51,
                polarity: false,
            },
            Event {
                timestamp_us: 1_000,
                x: 52,
                y: 51,
                polarity: true,
            },
        ];

        let detections = events
            .into_iter()
            .flat_map(|event| tracker.process_event(event))
            .collect::<Vec<_>>();

        assert_eq!(detections.len(), 1);
        let detection = &detections[0];
        assert_eq!(detection.event_count, 4);
        assert_eq!(detection.bbox.min_x, 50);
        assert_eq!(detection.bbox.min_y, 50);
        assert_eq!(detection.bbox.max_x, 52);
        assert_eq!(detection.bbox.max_y, 51);
        assert!((detection.centroid_x - 50.75).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_invalid_config() {
        let mut config = RollingClusterTrackerConfig::default();
        config.cell_size = 0;

        assert!(RollingClusterTracker::new(config).is_err());
    }

    #[test]
    fn fits_circle_from_cluster_events() {
        let mut events = Vec::new();
        for angle_deg in (0..360).step_by(15) {
            let angle = (angle_deg as f32).to_radians();
            events.push(Event {
                timestamp_us: angle_deg as u64,
                x: (100.0 + angle.cos() * 20.0).round() as u16,
                y: (80.0 + angle.sin() * 20.0).round() as u16,
                polarity: true,
            });
        }
        events.push(Event {
            timestamp_us: 1_000,
            x: 180,
            y: 160,
            polarity: true,
        });

        let fit = fit_circle_ransac(&events, 1.5).unwrap();

        assert!((fit.center_x - 100.0).abs() < 1.0);
        assert!((fit.center_y - 80.0).abs() < 1.0);
        assert!((fit.radius_px - 20.0).abs() < 1.0);
        assert!(fit.inlier_ratio > 0.8);
    }
}
