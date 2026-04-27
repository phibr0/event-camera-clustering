use crate::event::{BoundingBox, Event};
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
    pub polarity_filter: PolarityFilter,
    pub invert_polarity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarityFilter {
    All,
    Positive,
    Negative,
}

impl PolarityFilter {
    fn accepts(self, event: Event, invert_polarity: bool) -> bool {
        let polarity = if invert_polarity {
            !event.polarity
        } else {
            event.polarity
        };

        match self {
            Self::All => true,
            Self::Positive => polarity,
            Self::Negative => !polarity,
        }
    }
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
            polarity_filter: PolarityFilter::All,
            invert_polarity: true,
        }
    }
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
}

#[derive(Debug, Clone)]
struct ClusterAccum {
    count: usize,
    sum_x: u64,
    sum_y: u64,
    bbox: BoundingBox,
    cell_count: usize,
}

impl RollingClusterTracker {
    pub fn new(config: RollingClusterTrackerConfig) -> Self {
        assert!(config.window_us > 0, "window_us must be positive");
        assert!(config.step_us > 0, "step_us must be positive");
        assert!(config.cell_size > 0, "cell_size must be positive");

        Self {
            config,
            events: VecDeque::new(),
            next_emit_us: None,
            previous_centroid: None,
        }
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

    pub fn set_config(&mut self, config: RollingClusterTrackerConfig) {
        assert!(config.window_us > 0, "window_us must be positive");
        assert!(config.step_us > 0, "step_us must be positive");
        assert!(config.cell_size > 0, "cell_size must be positive");
        self.config = config;
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
            event.timestamp_us >= window_start_us
                && event.timestamp_us <= timestamp_us
                && self
                    .config
                    .polarity_filter
                    .accepts(**event, self.config.invert_polarity)
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
            polarity_filter: PolarityFilter::All,
            invert_polarity: false,
        });

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
    fn can_cluster_only_positive_events() {
        let mut tracker = RollingClusterTracker::new(RollingClusterTrackerConfig {
            window_us: 10_000,
            step_us: 1_000,
            cell_size: 1,
            min_events: 3,
            min_cells: 3,
            max_bbox_width: 20,
            max_bbox_height: 20,
            polarity_filter: PolarityFilter::Positive,
            invert_polarity: false,
        });

        let events = [
            Event {
                timestamp_us: 0,
                x: 10,
                y: 10,
                polarity: false,
            },
            Event {
                timestamp_us: 100,
                x: 11,
                y: 10,
                polarity: false,
            },
            Event {
                timestamp_us: 200,
                x: 12,
                y: 10,
                polarity: false,
            },
            Event {
                timestamp_us: 300,
                x: 50,
                y: 50,
                polarity: true,
            },
            Event {
                timestamp_us: 400,
                x: 51,
                y: 50,
                polarity: true,
            },
            Event {
                timestamp_us: 1_000,
                x: 50,
                y: 51,
                polarity: true,
            },
        ];

        let detections = events
            .into_iter()
            .flat_map(|event| tracker.process_event(event))
            .collect::<Vec<_>>();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].event_count, 3);
        assert_eq!(detections[0].bbox.min_x, 50);
    }

    #[test]
    fn can_invert_polarity_filtering() {
        let mut tracker = RollingClusterTracker::new(RollingClusterTrackerConfig {
            window_us: 10_000,
            step_us: 1_000,
            cell_size: 1,
            min_events: 3,
            min_cells: 3,
            max_bbox_width: 20,
            max_bbox_height: 20,
            polarity_filter: PolarityFilter::Positive,
            invert_polarity: true,
        });

        let events = [
            Event {
                timestamp_us: 0,
                x: 10,
                y: 10,
                polarity: false,
            },
            Event {
                timestamp_us: 100,
                x: 11,
                y: 10,
                polarity: false,
            },
            Event {
                timestamp_us: 1_000,
                x: 10,
                y: 11,
                polarity: false,
            },
            Event {
                timestamp_us: 1_100,
                x: 50,
                y: 50,
                polarity: true,
            },
        ];

        let detections = events
            .into_iter()
            .flat_map(|event| tracker.process_event(event))
            .collect::<Vec<_>>();

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].bbox.min_x, 10);
    }
}
