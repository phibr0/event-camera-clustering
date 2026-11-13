//! Causal 3D ball tracking on rotation-compensated events.
//!
//! Each update takes one event window and uses no later data:
//! 1. `motion::compensate_foreground` removes camera rotation and keeps regions
//!    that moved against the background.
//! 2. The newest foreground events are grouped into a few cluster proposals.
//! 3. Each proposal is checked for a circular outline in the newest events,
//!    foreground or not, because the foreground often keeps only part of a ball.
//! 4. Proposals are linked over time with a constant-velocity model in a
//!    rotation-stabilized image.
//! 5. The outline diameter and the known ball diameter give the distance.
//! 6. A Kalman filter with gravity in its motion model turns these single
//!    measurements into a flight path and a velocity.
//!
//! Camera translation is not measured, so positions are relative to the camera.
use crate::Event;
use crate::motion::{self, Camera, ForegroundFrame, IDENTITY, ImuTrack, MotionConfig, Rotation};
use std::collections::VecDeque;
use std::f64::consts::{PI, TAU};
use std::time::Instant;

/// Only this newest part of a window forms clusters; it shortens the streak of a fast ball.
const DETECT_US: u64 = 10_000;
/// Only this newest part of a window is searched for the outline.
const OUTLINE_US: u64 = 4_000;
const MAX_CLUSTERS: usize = 6;
const CELL_PX: usize = 3;
const MIN_CLUSTER_EVENTS: usize = 15;
const MIN_CLUSTER_CELLS: usize = 3;
const MAX_CLUSTER_SIZE_PX: u16 = 800;

const OUTLINE_SAMPLES: usize = 256;
const OUTLINE_POINTS: usize = 192;
const ANGLE_BINS: usize = 16;
/// A circle needs events in this many of its sixteen sectors.
const MIN_COVERAGE: u32 = 9;

/// Observations needed to start a track; half of them need a validated outline.
const START_OBSERVATIONS: usize = 6;
const START_SPEED_PX_S: f64 = 120.;
/// A track without a matching cluster is predicted for this long, then dropped.
const COAST_S: f64 = 0.08;
const PENDING_TIMEOUT_S: f64 = 0.035;
const MIN_DEPTH_M: f64 = 0.25;
const MAX_DEPTH_M: f64 = 15.;
const SIZE_HISTORY: usize = 5;

const GRAVITY_MPS2: f64 = 9.81;
/// Standard deviation of the acceleration that free flight does not explain,
/// mainly translation of a handheld camera. Smaller values give a stiffer
/// parabola that predicts these recordings worse.
const FLIGHT_ACCELERATION_MPS2: f64 = 8.;
/// Relative error assumed for the depth from one outline. It is larger than
/// the scatter between outlines because their errors persist over several updates.
const DEPTH_SIGMA: f64 = 0.3;
/// An outline whose size is this many standard deviations from the flight
/// so far is trusted half as much; see `FlightFilter::measure`.
const SIZE_TOLERANCE_SIGMAS: f64 = 1.5;

/// A validated circular outline in the current camera image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outline {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    /// Sectors of the circle, out of sixteen, that contain events.
    pub coverage: u32,
    /// Sampled events on the circle.
    pub support: u32,
}

/// A cluster proposal, refined to its outline when one was found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Center in the current camera image, undistorted pixels.
    pub x: f64,
    pub y: f64,
    /// The same center in the tracker's reference orientation.
    pub reference_x: f64,
    pub reference_y: f64,
    /// Bounding box of the cluster, or the outline diameter twice.
    pub width: f64,
    pub height: f64,
    /// Foreground events in the cluster.
    pub events: usize,
    pub outline: Option<Outline>,
}

/// Pinhole model of the current image and its rotation to the reference orientation.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// fx, fy, cx, cy in pixels.
    pub intrinsics: [f64; 4],
    pub width: f64,
    pub height: f64,
    /// Rotates rays of the current camera orientation into the reference orientation.
    pub to_reference: Rotation,
}

impl View {
    fn ray(&self, p: [f64; 2]) -> [f64; 3] {
        let [fx, fy, cx, cy] = self.intrinsics;
        [(p[0] - cx) / fx, (p[1] - cy) / fy, 1.]
    }

    /// NaN for a ray that does not point into the image half-space.
    fn pixel(&self, ray: [f64; 3]) -> [f64; 2] {
        if ray[2] <= 0. {
            return [f64::NAN; 2];
        }
        let [fx, fy, cx, cy] = self.intrinsics;
        [fx * ray[0] / ray[2] + cx, fy * ray[1] / ray[2] + cy]
    }

    pub fn reference_px(&self, camera_px: [f64; 2]) -> [f64; 2] {
        self.pixel(motion::transform(self.to_reference, self.ray(camera_px)))
    }

    pub fn camera_px(&self, reference_px: [f64; 2]) -> [f64; 2] {
        self.pixel(motion::transform(
            motion::transpose(self.to_reference),
            self.ray(reference_px),
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    /// No track. A new one may be forming.
    Lost,
    /// A cluster matched the track in this update.
    Measured,
    /// No cluster matched; the position is extrapolated.
    Predicted,
}

/// Image-space track after one association step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackPosition {
    /// Filtered center in the reference orientation and in the current image.
    pub reference_px: [f64; 2],
    pub camera_px: [f64; 2],
    /// Size of the last matched candidate.
    pub size_px: [f64; 2],
    /// Median of the last validated outline diameters.
    pub diameter_px: f64,
    /// Center of the candidate matched in this update, in the current image.
    pub measured_px: Option<[f64; 2]>,
    /// Image velocity of the track in the reference orientation, pixels per second.
    pub velocity_px_s: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Association {
    pub state: TrackState,
    /// Counts tracks from one; zero before the first track.
    pub track_id: u32,
    /// The matched candidate had a validated outline in this update.
    pub outline_supported: bool,
    pub track: Option<TrackPosition>,
}

#[derive(Clone, Copy)]
struct Observation {
    x: f64,
    y: f64,
    camera_x: f64,
    camera_y: f64,
    w: f64,
    h: f64,
    n: f64,
    outline: bool,
}

#[derive(Clone, Copy)]
struct Active {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    w: f64,
    h: f64,
    last: f64,
    t: f64,
    outline: bool,
}

/// Links candidates over time. Positions are kept in the reference orientation,
/// so camera rotation between updates does not appear as ball motion.
#[derive(Default)]
pub struct Associator {
    active: Option<Active>,
    pending: Vec<(Observation, f64)>,
    track_id: u32,
    sizes: VecDeque<f64>,
}

/// First element with the largest key, like Python's `max`.
fn first_max<'a>(
    items: impl IntoIterator<Item = &'a Observation>,
    key: impl Fn(&Observation) -> f64,
) -> Option<&'a Observation> {
    let mut best: Option<(&Observation, f64)> = None;
    for item in items {
        let value = key(item);
        if best.is_none_or(|(_, best)| value > best) {
            best = Some((item, value));
        }
    }
    best.map(|(item, _)| item)
}

impl Associator {
    pub fn has_track(&self) -> bool {
        self.active.is_some()
    }

    /// Drops the track and any forming track; track numbering continues.
    pub fn clear(&mut self) {
        self.active = None;
        self.pending.clear();
    }

    /// Moves the observations of a forming track to a new reference orientation.
    /// Only valid without a track, whose state would otherwise become inconsistent.
    fn rebase_pending(&mut self, to_new_reference: impl Fn([f64; 2]) -> [f64; 2]) {
        debug_assert!(self.active.is_none());
        for (observation, _) in &mut self.pending {
            [observation.x, observation.y] = to_new_reference([observation.x, observation.y]);
        }
        if self
            .pending
            .iter()
            .any(|(o, _)| !o.x.is_finite() || !o.y.is_finite())
        {
            self.pending.clear();
        }
    }

    /// Remembers a validated outline diameter; only the newest few are kept.
    fn push_size(&mut self, size: f64) {
        if self.sizes.len() == SIZE_HISTORY {
            self.sizes.pop_front();
        }
        self.sizes.push_back(size);
    }

    fn median_size(&self) -> f64 {
        let mut sizes: Vec<_> = self.sizes.iter().copied().collect();
        sizes.sort_by(f64::total_cmp);
        sizes[sizes.len() / 2]
    }

    /// `time_s` must increase between calls.
    pub fn step(
        &mut self,
        time_s: f64,
        candidates: &[Candidate],
        view: &View,
        ball_diameter_m: f64,
    ) -> Association {
        let t = time_s;
        let available: Vec<Observation> = candidates
            .iter()
            .filter(|c| c.reference_x.is_finite() && c.reference_y.is_finite())
            .map(|c| Observation {
                x: c.reference_x,
                y: c.reference_y,
                camera_x: c.x,
                camera_y: c.y,
                w: c.width,
                h: c.height,
                n: c.events as f64,
                outline: c.outline.is_some(),
            })
            .filter(|c| {
                let depth = view.intrinsics[0] * ball_diameter_m / c.w.min(c.h);
                c.n >= 30.
                    && c.w.min(c.h) / c.w.max(c.h) >= 0.25
                    && MIN_DEPTH_M < depth
                    && depth < MAX_DEPTH_M
                    && c.w.max(c.h) < 700.
            })
            .collect();
        let mut state = TrackState::Lost;
        let mut matched: Option<Observation> = None;

        if let Some(mut active) = self.active.take() {
            let dt = t - active.t;
            let px = active.x + active.vx * dt;
            let py = active.y + active.vy * dt;
            let size = self.median_size();
            let [camera_px, camera_py] = view.camera_px([px, py]);
            let [camera_ax, camera_ay] = view.camera_px([active.x, active.y]);
            // The gate is in image pixels, so it keeps its meaning after a large rotation.
            let gate = 25f64.max(1.5 * size.max(active.w).max(active.h))
                + 0.25 * (camera_px - camera_ax).hypot(camera_py - camera_ay);
            let distance = |c: &Observation| (c.camera_x - camera_px).hypot(c.camera_y - camera_py);
            let mut nearby: Vec<&Observation> = available
                .iter()
                .filter(|c| distance(c) <= gate && (0.35..=2.5).contains(&(c.w.min(c.h) / size)))
                .collect();
            if !nearby.is_empty() {
                if nearby.iter().any(|c| c.outline) {
                    nearby.retain(|c| c.outline);
                }
                let c = first_max(nearby, |c| c.n / (1. + distance(c) / (gate * 0.5))).unwrap();
                let (rx, ry) = (c.x - px, c.y - py);
                active = Active {
                    x: px + 0.8 * rx,
                    y: py + 0.8 * ry,
                    vx: active.vx + 0.2 * rx / dt,
                    vy: active.vy + 0.2 * ry / dt,
                    w: c.w,
                    h: c.h,
                    last: t,
                    t,
                    outline: c.outline,
                };
                if c.outline {
                    self.push_size(c.w.min(c.h));
                }
                state = TrackState::Measured;
                matched = Some(*c);
                self.active = Some(active);
            } else if t - active.last <= COAST_S + 1e-9 {
                active.x = px;
                active.y = py;
                active.t = t;
                state = TrackState::Predicted;
                self.active = Some(active);
            } else {
                self.pending.clear();
            }
        }

        if self.active.is_none() {
            // A forming track starts on a validated outline and may continue without one.
            // This is decided before a stale forming track is discarded below, as in the
            // offline experiment this was validated with.
            let forming = !self.pending.is_empty();
            let seeds: Vec<&Observation> = available
                .iter()
                .filter(|c| {
                    (forming || c.outline)
                        && c.w.min(c.h) / c.w.max(c.h) >= 0.5
                        && c.n / (c.w * c.h) >= 0.1
                })
                .collect();
            if self
                .pending
                .last()
                .is_some_and(|&(_, last)| t - last > PENDING_TIMEOUT_S)
            {
                self.pending.clear();
            }
            let chosen = if let Some(&(last, _)) = self.pending.last() {
                let [last_x, last_y] = view.camera_px([last.x, last.y]);
                let distance = |c: &Observation| (c.camera_x - last_x).hypot(c.camera_y - last_y);
                first_max(
                    seeds
                        .into_iter()
                        .filter(|c| distance(c) <= 25f64.max(2. * c.w.min(c.h))),
                    |c| c.n / (1. + distance(c) / 25.),
                )
            } else {
                first_max(seeds, |c| c.n)
            };
            if let Some(&c) = chosen {
                self.pending.push((c, t));
                if self.pending.len() > START_OBSERVATIONS {
                    self.pending.remove(0);
                }
                let outlines = self.pending.iter().filter(|(p, _)| p.outline).count();
                if self.pending.len() == START_OBSERVATIONS && outlines * 2 >= START_OBSERVATIONS {
                    let (first, first_t) = self.pending[0];
                    let dt = t - first_t;
                    let [first_x, first_y] = view.camera_px([first.x, first.y]);
                    if (c.camera_x - first_x).hypot(c.camera_y - first_y) / dt >= START_SPEED_PX_S {
                        self.track_id += 1;
                        self.active = Some(Active {
                            x: c.x,
                            y: c.y,
                            vx: (c.x - first.x) / dt,
                            vy: (c.y - first.y) / dt,
                            w: c.w,
                            h: c.h,
                            last: t,
                            t,
                            outline: c.outline,
                        });
                        self.sizes.clear();
                        for i in 0..self.pending.len() {
                            let (p, _) = self.pending[i];
                            if p.outline {
                                self.push_size(p.w.min(p.h));
                            }
                        }
                        state = TrackState::Measured;
                        matched = Some(c);
                    }
                }
            }
        }

        let mut outline_supported = false;
        let mut track = None;
        if let Some(active) = self.active {
            let [x, y] = view.camera_px([active.x, active.y]);
            if x.is_finite()
                && y.is_finite()
                && -10. < x
                && x < view.width + 10.
                && -10. < y
                && y < view.height + 10.
            {
                outline_supported = state == TrackState::Measured && active.outline;
                track = Some(TrackPosition {
                    reference_px: [active.x, active.y],
                    camera_px: [x, y],
                    size_px: [active.w, active.h],
                    diameter_px: self.median_size(),
                    measured_px: matched.map(|c| [c.camera_x, c.camera_y]),
                    velocity_px_s: [active.vx, active.vy],
                });
            } else {
                // The ball left the view.
                self.clear();
                state = TrackState::Lost;
            }
        }
        Association {
            state,
            track_id: self.track_id,
            outline_supported,
            track,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Cluster {
    x: f64,
    y: f64,
    min: [u16; 2],
    max: [u16; 2],
    events: usize,
}

impl Cluster {
    fn width(&self) -> f64 {
        f64::from(self.max[0] - self.min[0] + 1)
    }
    fn height(&self) -> f64 {
        f64::from(self.max[1] - self.min[1] + 1)
    }
}

#[derive(Clone, Copy, Default)]
struct Cell {
    count: u32,
    sum: [u32; 2],
    min: [u16; 2],
    max: [u16; 2],
    visited: bool,
}

/// Connected event clusters on a coarse grid; allocations are reused between updates.
#[derive(Default)]
struct ClusterGrid {
    columns: usize,
    rows: usize,
    cells: Vec<Cell>,
    touched: Vec<usize>,
    stack: Vec<usize>,
}

impl ClusterGrid {
    fn resize(&mut self, width: usize, height: usize) {
        let columns = width.div_ceil(CELL_PX);
        let rows = height.div_ceil(CELL_PX);
        if columns != self.columns || rows != self.rows {
            self.columns = columns;
            self.rows = rows;
            self.cells = vec![Cell::default(); columns * rows];
        }
    }

    /// The connected cluster with the most events that passes the size limits.
    fn largest(&mut self, events: &[[u16; 2]]) -> Option<Cluster> {
        for &[x, y] in events {
            let index = y as usize / CELL_PX * self.columns + x as usize / CELL_PX;
            let cell = &mut self.cells[index];
            if cell.count == 0 {
                self.touched.push(index);
                cell.min = [x, y];
                cell.max = [x, y];
            }
            cell.count += 1;
            cell.sum[0] += u32::from(x);
            cell.sum[1] += u32::from(y);
            cell.min = [cell.min[0].min(x), cell.min[1].min(y)];
            cell.max = [cell.max[0].max(x), cell.max[1].max(y)];
        }
        let mut best: Option<Cluster> = None;
        for &seed in &self.touched {
            if self.cells[seed].visited {
                continue;
            }
            self.cells[seed].visited = true;
            self.stack.push(seed);
            let mut count = 0usize;
            let mut sum = [0u64; 2];
            let mut min = [u16::MAX; 2];
            let mut max = [0u16; 2];
            let mut cells = 0;
            while let Some(index) = self.stack.pop() {
                let cell = self.cells[index];
                count += cell.count as usize;
                cells += 1;
                for axis in 0..2 {
                    sum[axis] += u64::from(cell.sum[axis]);
                    min[axis] = min[axis].min(cell.min[axis]);
                    max[axis] = max[axis].max(cell.max[axis]);
                }
                let (column, row) = (index % self.columns, index / self.columns);
                for ny in row.saturating_sub(1)..=(row + 1).min(self.rows - 1) {
                    for nx in column.saturating_sub(1)..=(column + 1).min(self.columns - 1) {
                        let neighbor = &mut self.cells[ny * self.columns + nx];
                        if neighbor.count > 0 && !neighbor.visited {
                            neighbor.visited = true;
                            self.stack.push(ny * self.columns + nx);
                        }
                    }
                }
            }
            if count >= MIN_CLUSTER_EVENTS
                && cells >= MIN_CLUSTER_CELLS
                && max[0] - min[0] < MAX_CLUSTER_SIZE_PX
                && max[1] - min[1] < MAX_CLUSTER_SIZE_PX
                && best.is_none_or(|best| count > best.events)
            {
                best = Some(Cluster {
                    x: sum[0] as f64 / count as f64,
                    y: sum[1] as f64 / count as f64,
                    min,
                    max,
                    events: count,
                });
            }
        }
        for index in self.touched.drain(..) {
            self.cells[index] = Cell::default();
        }
        best
    }

    /// Up to `MAX_CLUSTERS` clusters, largest first. Each one removes the events in its
    /// bounding box before the next search, so an object overlapping a larger one can be missed.
    fn propose(&mut self, mut events: Vec<[u16; 2]>) -> Vec<Cluster> {
        let mut clusters = Vec::new();
        while clusters.len() < MAX_CLUSTERS && events.len() >= MIN_CLUSTER_EVENTS {
            let Some(cluster) = self.largest(&events) else {
                break;
            };
            events.retain(|&[x, y]| {
                x < cluster.min[0] || x > cluster.max[0] || y < cluster.min[1] || y > cluster.max[1]
            });
            clusters.push(cluster);
        }
        clusters
    }
}

#[derive(Clone, Copy)]
struct RecentPoint {
    x: f32,
    y: f32,
    foreground: bool,
}

/// Fixed-seed generator: the same events always give the same outline.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (((self.next() >> 32) * n as u64) >> 32) as usize
    }
}

/// Keeps at most `limit` points, evenly spaced over the input.
fn subsample(points: &mut Vec<[f64; 2]>, limit: usize) {
    if points.len() <= limit {
        return;
    }
    let last = points.len() - 1;
    let step = last as f64 / (limit - 1) as f64;
    *points = (0..limit)
        .map(|i| {
            points[if i + 1 == limit {
                last
            } else {
                (i as f64 * step) as usize
            }]
        })
        .collect();
}

/// Searches the newest events around a cluster for a circle. Foreground events
/// inside the cluster seed the search; all nearby events may support the circle.
fn fit_outline(
    points: &[RecentPoint],
    cluster: &Cluster,
    width: usize,
    height: usize,
) -> Option<Outline> {
    let (x, y) = (cluster.x, cluster.y);
    let (w, h) = (cluster.width(), cluster.height());
    let extent = w.max(h);
    if extent > 400. || cluster.events < 30 {
        return None;
    }
    // A clipped arc can resemble a much smaller circle. Keep it for tracking only.
    if x.min(width as f64 - x).min(y).min(height as f64 - y) < extent * 0.5 + 3. {
        return None;
    }
    let pad = 72f64.max(extent * 1.1 + 12.);
    let mut cells: Vec<(i32, i32, usize)> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| (f64::from(p.x) - x).abs() < pad && (f64::from(p.y) - y).abs() < pad)
        .map(|(i, p)| {
            (
                (p.x / 1.5).round_ties_even() as i32,
                (p.y / 1.5).round_ties_even() as i32,
                i,
            )
        })
        .collect();
    if cells.len() < 15 {
        return None;
    }
    // One event per 1.5 px cell, so dense texture does not outvote a thin outline.
    cells.sort_unstable();
    cells.dedup_by_key(|&mut (cell_x, cell_y, _)| (cell_x, cell_y));
    let mut seeds = Vec::new();
    let mut others = Vec::new();
    for &(_, _, i) in &cells {
        let p = points[i];
        let position = [f64::from(p.x), f64::from(p.y)];
        if p.foreground
            && (position[0] - x).abs() < w * 0.75 + 3.
            && (position[1] - y).abs() < h * 0.75 + 3.
        {
            seeds.push(position);
        } else {
            others.push(position);
        }
    }
    subsample(&mut seeds, OUTLINE_POINTS);
    subsample(&mut others, OUTLINE_POINTS);
    if seeds.len() < 4 {
        return None;
    }
    let seed_count = seeds.len();
    let mut sample = seeds;
    sample.extend(others);

    let max_radius = 64f64.max(extent * 1.4 + 6.);
    // Events on the circle, weighted by how much of its circumference they cover.
    let score = |center: [f64; 2], radius: f64| -> Option<(f64, Outline)> {
        if !(3. ..=max_radius).contains(&radius)
            || (center[0] - x).hypot(center[1] - y) >= radius + 6.
        {
            return None;
        }
        let tolerance = 1.5f64.max(0.05 * radius);
        let inner = (radius - tolerance).powi(2);
        let outer = (radius + tolerance).powi(2);
        let mut sectors = 0u32;
        let mut support = 0u32;
        let mut seed_support = 0u32;
        for (i, p) in sample.iter().enumerate() {
            let (ex, ey) = (p[0] - center[0], p[1] - center[1]);
            let distance2 = ex * ex + ey * ey;
            if distance2 < inner || distance2 > outer {
                continue;
            }
            support += 1;
            seed_support += u32::from(i < seed_count);
            let sector = ((ey.atan2(ex) + PI) * ANGLE_BINS as f64 / TAU) as usize % ANGLE_BINS;
            sectors |= 1 << sector;
        }
        let coverage = sectors.count_ones();
        if coverage < MIN_COVERAGE || support < 12 || seed_support < 4 {
            return None;
        }
        // Coverage dominates: a full small circle beats a partial arc of a large one.
        let score = f64::from(support + 2 * seed_support)
            * (f64::from(coverage) / ANGLE_BINS as f64).powi(4)
            * radius.sqrt();
        Some((
            score,
            Outline {
                x: center[0],
                y: center[1],
                radius,
                coverage,
                support,
            },
        ))
    };
    let mut rng = SplitMix64(123);
    let mut best: Option<(f64, Outline)> = None;
    for _ in 0..OUTLINE_SAMPLES {
        // Circle through one seed event and two events of either kind.
        let a = sample[rng.below(seed_count)];
        let b = sample[rng.below(sample.len())];
        let c = sample[rng.below(sample.len())];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - a[0], c[1] - a[1]];
        let den = 2. * (u[0] * v[1] - u[1] * v[0]);
        if den.abs() <= 1. {
            continue;
        }
        let u2 = u[0] * u[0] + u[1] * u[1];
        let v2 = v[0] * v[0] + v[1] * v[1];
        let dx = (u2 * v[1] - v2 * u[1]) / den;
        let dy = (u[0] * v2 - v[0] * u2) / den;
        if let Some(candidate) = score([a[0] + dx, a[1] + dy], dx.hypot(dy)) {
            if best.is_none_or(|(best, _)| candidate.0 > best) {
                best = Some(candidate);
            }
        }
    }
    best.map(|(_, outline)| outline)
}

/// Kalman estimate of a ball in free flight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flight {
    /// Meters and meters per second in the camera frame at the end of the window.
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    /// The same state in the camera orientation of `BallUpdate::stabilized_frame`,
    /// where gravity is constant.
    pub stabilized_m: [f64; 3],
    pub stabilized_velocity_mps: [f64; 3],
    pub gravity_mps2: [f64; 3],
    /// One standard deviation of the depth.
    pub depth_sigma_m: f64,
}

impl Flight {
    /// Position on the flight path `seconds` after this update, or before it
    /// when negative, in the stabilized orientation.
    pub fn stabilized_after(&self, seconds: f64) -> [f64; 3] {
        std::array::from_fn(|k| {
            self.stabilized_m[k]
                + self.stabilized_velocity_mps[k] * seconds
                + 0.5 * self.gravity_mps2[k] * seconds * seconds
        })
    }
}

/// A tracked ball: the measurement of this update and the filtered flight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    /// Center of the image track in the current undistorted image.
    pub pixel: [f64; 2],
    /// Center of the candidate matched in this update, before any smoothing.
    pub measured_px: Option<[f64; 2]>,
    /// Size of the last matched candidate.
    pub size_px: [f64; 2],
    /// Median of the last validated outline diameters.
    pub diameter_px: f64,
    /// Position from `pixel` and `diameter_px` alone, without the flight model.
    /// Camera frame at the end of the window: x right, y down, z forward.
    pub measured_m: [f64; 3],
    /// `measured_m` in the camera orientation of `BallUpdate::stabilized_frame`.
    pub measured_stabilized_m: [f64; 3],
    /// `None` when the IMU provides no gravity direction.
    pub flight: Option<Flight>,
    /// Center in the track's reference orientation; see `BallTracker::camera_px`.
    pub reference_px: [f64; 2],
}

impl Ball {
    /// The filtered position, or the measurement without a flight estimate.
    pub fn position_m(&self) -> [f64; 3] {
        self.flight
            .map_or(self.measured_m, |flight| flight.position_m)
    }

    pub fn stabilized_m(&self) -> [f64; 3] {
        self.flight
            .map_or(self.measured_stabilized_m, |flight| flight.stabilized_m)
    }
}

/// Kalman filter of one flight. The state is position and velocity in the
/// stabilized orientation; the prediction adds gravity. Measurements are the
/// pixel of the ball center and the depth from the outline diameter. Gravity
/// bends the image path by an amount that depends on the distance, so the
/// pixels alone also constrain the depth once a flight is long enough.
struct FlightFilter {
    track_id: u32,
    time_s: f64,
    state: [f64; 6],
    covariance: [[f64; 6]; 6],
}

impl FlightFilter {
    /// Starts at a measured position. `to_stabilized` rotates camera axes into
    /// the stabilized orientation.
    fn new(
        track_id: u32,
        time_s: f64,
        position_m: [f64; 3],
        velocity_mps: [f64; 3],
        to_stabilized: Rotation,
    ) -> Self {
        let depth = position_m[2];
        // The bearing is known far better than the depth and its rate.
        let sigma = [0.1 * depth, 0.1 * depth, 0.3 * depth, 2., 2., 8.];
        let mut covariance = [[0.; 6]; 6];
        for block in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    covariance[block * 3 + i][block * 3 + j] = (0..3)
                        .map(|k| {
                            to_stabilized[i][k] * sigma[block * 3 + k].powi(2) * to_stabilized[j][k]
                        })
                        .sum();
                }
            }
        }
        let position = motion::transform(to_stabilized, position_m);
        let velocity = motion::transform(to_stabilized, velocity_mps);
        Self {
            track_id,
            time_s,
            state: std::array::from_fn(|i| if i < 3 { position[i] } else { velocity[i - 3] }),
            covariance,
        }
    }

    /// Free flight to `time_s` under `gravity` in the stabilized orientation.
    fn predict(&mut self, time_s: f64, gravity: [f64; 3]) {
        let dt = time_s - self.time_s;
        self.time_s = time_s;
        for k in 0..3 {
            self.state[k] += self.state[k + 3] * dt + 0.5 * gravity[k] * dt * dt;
            self.state[k + 3] += gravity[k] * dt;
        }
        let p = self.covariance;
        let q = FLIGHT_ACCELERATION_MPS2.powi(2);
        for i in 0..3 {
            for j in 0..3 {
                let (pp, pv, vp, vv) = (p[i][j], p[i][j + 3], p[i + 3][j], p[i + 3][j + 3]);
                self.covariance[i][j] = pp + dt * (pv + vp) + dt * dt * vv;
                self.covariance[i][j + 3] = pv + dt * vv;
                self.covariance[i + 3][j] = vp + dt * vv;
            }
            self.covariance[i][i] += q * dt.powi(4) / 4.;
            self.covariance[i][i + 3] += q * dt.powi(3) / 2.;
            self.covariance[i + 3][i] += q * dt.powi(3) / 2.;
            self.covariance[i + 3][i + 3] += q * dt * dt;
        }
    }

    /// Variance of a scalar with Jacobian `h` over the position.
    fn variance(&self, h: [f64; 3]) -> f64 {
        (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| h[i] * self.covariance[i][j] * h[j])
                    .sum::<f64>()
            })
            .sum()
    }

    /// One scalar measurement with Jacobian `h` over the position.
    fn correct(&mut self, h: [f64; 3], innovation: f64, variance: f64) {
        let ph: [f64; 6] =
            std::array::from_fn(|i| (0..3).map(|k| self.covariance[i][k] * h[k]).sum());
        let s = self.variance(h) + variance;
        for i in 0..6 {
            self.state[i] += ph[i] / s * innovation;
            for j in 0..6 {
                self.covariance[i][j] -= ph[i] * ph[j] / s;
            }
        }
    }

    /// `to_camera` rotates the stabilized orientation into the current camera axes.
    fn camera_position(&self, to_camera: Rotation) -> [f64; 3] {
        motion::transform(to_camera, [self.state[0], self.state[1], self.state[2]])
    }

    /// Corrects with the center of the matched candidate and, when it has an
    /// outline, with the depth that its diameter implies.
    fn measure(
        &mut self,
        to_camera: Rotation,
        intrinsics: [f64; 4],
        pixel: [f64; 2],
        pixel_sigma: f64,
        depth_m: Option<f64>,
    ) {
        let [fx, fy, cx, cy] = intrinsics;
        let depth_row =
            |c: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| to_camera[2][k] / c[2]) };
        // The circle search sometimes locks onto a structure of another size,
        // and such a circle is also off-center. The more the size of an outline
        // contradicts the flight so far, the less its center and depth count.
        let mut trust = 1.;
        let c = self.camera_position(to_camera);
        if let Some(depth_m) = depth_m.filter(|_| c[2] >= 0.05) {
            let ratio = (depth_m / c[2]).ln();
            let expected = self.variance(depth_row(c)) + DEPTH_SIGMA * DEPTH_SIGMA;
            trust = 1. / (1. + ratio * ratio / (SIZE_TOLERANCE_SIGMAS.powi(2) * expected));
        }
        // Each scalar is linearized at the state the previous one left.
        for axis in 0..2 {
            let c = self.camera_position(to_camera);
            if c[2] < 0.05 {
                return;
            }
            let (focal, center) = if axis == 0 { (fx, cx) } else { (fy, cy) };
            let h = std::array::from_fn(|k| {
                focal / c[2] * to_camera[axis][k]
                    - focal * c[axis] / (c[2] * c[2]) * to_camera[2][k]
            });
            let predicted = focal * c[axis] / c[2] + center;
            self.correct(
                h,
                pixel[axis] - predicted,
                pixel_sigma * pixel_sigma / trust,
            );
        }
        let c = self.camera_position(to_camera);
        if let Some(depth_m) = depth_m.filter(|_| c[2] >= 0.05) {
            // The logarithm, because the outline diameter has a relative error.
            self.correct(
                depth_row(c),
                (depth_m / c[2]).ln(),
                DEPTH_SIGMA * DEPTH_SIGMA / trust,
            );
        }
    }

    fn flight(&self, to_camera: Rotation, gravity: [f64; 3]) -> Flight {
        let position = [self.state[0], self.state[1], self.state[2]];
        let velocity = [self.state[3], self.state[4], self.state[5]];
        let depth_variance: f64 = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| to_camera[2][i] * self.covariance[i][j] * to_camera[2][j])
                    .sum::<f64>()
            })
            .sum();
        Flight {
            position_m: motion::transform(to_camera, position),
            velocity_mps: motion::transform(to_camera, velocity),
            stabilized_m: position,
            stabilized_velocity_mps: velocity,
            gravity_mps2: gravity,
            depth_sigma_m: depth_variance.max(0.).sqrt(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BallUpdate {
    /// End of the window in seconds after `origin_us`.
    pub time_s: f64,
    pub state: TrackState,
    pub track_id: u32,
    /// The outline was validated in this update. Otherwise `Ball::measured_m`
    /// still uses the diameter of earlier validated outlines.
    pub outline_supported: bool,
    pub ball: Option<Ball>,
    pub imu_covered: bool,
    /// Counts the orientations that stabilized positions have referred to. It
    /// changes after `reset` and after an IMU gap.
    pub stabilized_frame: u32,
    /// Processing time of this update: compensation and foreground, clusters
    /// and outlines, association and flight filter.
    pub stage_ms: [f64; 3],
}

impl BallUpdate {
    pub fn compute_ms(&self) -> f64 {
        self.stage_ms.iter().sum()
    }
}

/// The complete per-window pipeline. Feed windows in time order.
pub struct BallTracker {
    ball_diameter_m: f64,
    associator: Associator,
    grid: ClusterGrid,
    frame: ForegroundFrame,
    candidates: Vec<Candidate>,
    view: Option<View>,
    reference_us: u64,
    stabilized_us: Option<u64>,
    stabilized_frame: u32,
    to_stabilized: Rotation,
    /// Unit vector toward the ground in the stabilized orientation.
    gravity: Option<[f64; 3]>,
    flight: Option<FlightFilter>,
    last_end_us: Option<u64>,
}

impl BallTracker {
    pub fn new(ball_diameter_m: f64) -> Self {
        Self {
            ball_diameter_m,
            associator: Associator::default(),
            grid: ClusterGrid::default(),
            frame: ForegroundFrame {
                points: Vec::new(),
                foreground: Vec::new(),
                imu_covered: false,
            },
            candidates: Vec::new(),
            view: None,
            reference_us: 0,
            stabilized_us: None,
            stabilized_frame: 0,
            to_stabilized: IDENTITY,
            gravity: None,
            flight: None,
            last_end_us: None,
        }
    }

    pub fn ball_diameter_m(&self) -> f64 {
        self.ball_diameter_m
    }

    pub fn set_ball_diameter_m(&mut self, ball_diameter_m: f64) {
        self.ball_diameter_m = ball_diameter_m;
    }

    /// Forgets all tracks, for a seek or a changed setting. Track numbering continues.
    pub fn reset(&mut self) {
        self.associator.clear();
        self.candidates.clear();
        self.view = None;
        self.stabilized_us = None;
        self.gravity = None;
        self.flight = None;
        self.last_end_us = None;
    }

    /// Compensated events and foreground mask of the last update.
    pub fn frame(&self) -> &ForegroundFrame {
        &self.frame
    }

    /// Cluster proposals of the last update.
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// Maps `Ball::reference_px` of the current track into the image of the last update.
    pub fn camera_px(&self, reference_px: [f64; 2]) -> Option<[f64; 2]> {
        let p = self.view?.camera_px(reference_px);
        (p[0].is_finite() && p[1].is_finite()).then_some(p)
    }

    /// Maps a point in the stabilized orientation, such as `Flight::stabilized_after`,
    /// into the image of the last update. `None` behind the camera.
    pub fn stabilized_px(&self, stabilized_m: [f64; 3]) -> Option<[f64; 2]> {
        let p = self.view?.pixel(motion::transform(
            motion::transpose(self.to_stabilized),
            stabilized_m,
        ));
        (p[0].is_finite() && p[1].is_finite()).then_some(p)
    }

    /// `events` are the sensor events of the window `start_us..=end_us`. Windows
    /// overlap: only the newest 10 ms form clusters, the rest is background evidence.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        events: &[Event],
        origin_us: u64,
        start_us: u64,
        end_us: u64,
        camera: &Camera,
        imu: &ImuTrack,
        config: MotionConfig,
    ) -> BallUpdate {
        let tick = Instant::now();
        self.frame =
            motion::compensate_foreground(events, origin_us, start_us, end_us, camera, imu, config);
        let compensate_ms = tick.elapsed().as_secs_f64() * 1000.;
        let tick = Instant::now();
        if self.last_end_us.is_some_and(|last| end_us <= last) {
            self.reset();
        }
        self.last_end_us = Some(end_us);
        let imu_time = |us: u64| us.saturating_sub(origin_us) as f64 * 1e-6 + config.offset_s;
        let now = imu_time(end_us);
        let rotation_to = |us: u64| imu.relative_rotation(now, imu_time(us), config.mount);
        let mut view = View {
            intrinsics: camera.intrinsics(),
            width: camera.width as f64,
            height: camera.height as f64,
            to_reference: IDENTITY,
        };
        let mut update = BallUpdate {
            time_s: end_us.saturating_sub(origin_us) as f64 * 1e-6,
            state: TrackState::Lost,
            track_id: self.associator.track_id,
            outline_supported: false,
            ball: None,
            imu_covered: self.frame.imu_covered,
            stabilized_frame: self.stabilized_frame,
            stage_ms: [compensate_ms, 0., 0.],
        };

        // A track keeps the orientation it started in as its reference. Without a
        // track the reference follows the camera, so it never drifts out of view.
        let to_reference = if !self.frame.imu_covered {
            None
        } else if self.associator.has_track() {
            rotation_to(self.reference_us)
        } else {
            if !self.associator.pending.is_empty() && self.reference_us != end_us {
                match rotation_to(self.reference_us) {
                    Some(to_reference) => {
                        let old = View {
                            to_reference,
                            ..view
                        };
                        self.associator.rebase_pending(|p| old.camera_px(p));
                    }
                    None => self.associator.clear(),
                }
            }
            self.reference_us = end_us;
            Some(IDENTITY)
        };
        let Some(to_reference) = to_reference else {
            // Without rotation the track cannot be related to this image.
            self.associator.clear();
            self.candidates.clear();
            self.view = None;
            self.flight = None;
            return update;
        };
        view.to_reference = to_reference;
        self.view = Some(view);
        let to_stabilized = match self.stabilized_us.and_then(rotation_to) {
            Some(rotation) => rotation,
            None => {
                self.stabilized_us = Some(end_us);
                self.stabilized_frame += 1;
                self.gravity = None;
                self.flight = None;
                IDENTITY
            }
        };
        self.to_stabilized = to_stabilized;
        update.stabilized_frame = self.stabilized_frame;
        // Gravity is constant in the stabilized orientation, so average the
        // IMU's estimate over about 0.2 s.
        self.gravity = match imu.gravity_at(now) {
            Some(down) => {
                let down = motion::transform(to_stabilized, motion::transform(config.mount, down));
                let mean: [f64; 3] = match self.gravity {
                    Some(mean) => std::array::from_fn(|k| 0.95 * mean[k] + 0.05 * down[k]),
                    None => down,
                };
                let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
                Some(mean.map(|v| v / norm))
            }
            None => None,
        };

        let detect_from = end_us.saturating_sub(DETECT_US);
        let outline_from = end_us.saturating_sub(OUTLINE_US);
        let (max_x, max_y) = ((camera.width - 1) as f32, (camera.height - 1) as f32);
        let mut foreground = Vec::new();
        let mut recent = Vec::new();
        for (i, event) in events.iter().enumerate() {
            if event.timestamp_us < detect_from {
                continue;
            }
            let p = self.frame.points[i];
            if self.frame.foreground[i] {
                foreground.push([
                    p[0].round().clamp(0., max_x) as u16,
                    p[1].round().clamp(0., max_y) as u16,
                ]);
            }
            if event.timestamp_us >= outline_from
                && p[0] >= 0.
                && p[1] >= 0.
                && p[0] <= max_x
                && p[1] <= max_y
            {
                recent.push(RecentPoint {
                    x: p[0],
                    y: p[1],
                    foreground: self.frame.foreground[i],
                });
            }
        }
        self.grid.resize(camera.width, camera.height);
        self.candidates = self
            .grid
            .propose(foreground)
            .iter()
            .map(|cluster| {
                let outline = fit_outline(&recent, cluster, camera.width, camera.height);
                let (x, y, width, height) = match outline {
                    Some(o) => (o.x, o.y, 2. * o.radius, 2. * o.radius),
                    None => (cluster.x, cluster.y, cluster.width(), cluster.height()),
                };
                let [reference_x, reference_y] = view.reference_px([x, y]);
                Candidate {
                    x,
                    y,
                    reference_x,
                    reference_y,
                    width,
                    height,
                    events: cluster.events,
                    outline,
                }
            })
            .collect();
        update.stage_ms[1] = tick.elapsed().as_secs_f64() * 1000.;

        let tick = Instant::now();
        let association =
            self.associator
                .step(update.time_s, &self.candidates, &view, self.ball_diameter_m);
        update.state = association.state;
        update.track_id = association.track_id;
        update.outline_supported = association.outline_supported;
        let [fx, fy, cx, cy] = view.intrinsics;
        let to_camera = motion::transpose(to_stabilized);
        let gravity = self.gravity.map(|down| down.map(|v| v * GRAVITY_MPS2));
        update.ball = association.track.map(|track| {
            let depth = |diameter_px: f64| fx * self.ball_diameter_m / diameter_px;
            let position =
                |pixel: [f64; 2], z: f64| [(pixel[0] - cx) * z / fx, (pixel[1] - cy) * z / fy, z];
            let measured_m = position(track.camera_px, depth(track.diameter_px));
            if self
                .flight
                .as_ref()
                .is_some_and(|filter| filter.track_id != association.track_id)
            {
                self.flight = None;
            }
            let flight = gravity.and_then(|gravity| {
                match (&mut self.flight, track.measured_px) {
                    (Some(filter), measured_px) => {
                        filter.predict(update.time_s, gravity);
                        if let Some(pixel) = measured_px {
                            let (sigma, depth_m) = if association.outline_supported {
                                // The outline center is good to a fraction of its size.
                                (2. + 0.06 * track.size_px[0], Some(depth(track.size_px[0])))
                            } else {
                                // A cluster can be any part of the ball.
                                (0.5 * track.size_px[0].max(track.size_px[1]), None)
                            };
                            filter.measure(to_camera, view.intrinsics, pixel, sigma, depth_m);
                        }
                    }
                    (None, Some(pixel)) => {
                        // A new track: start at its measurement, moving with the image
                        // track at constant depth.
                        let z = depth(track.diameter_px);
                        self.flight = Some(FlightFilter::new(
                            association.track_id,
                            update.time_s,
                            position(pixel, z),
                            [
                                track.velocity_px_s[0] * z / fx,
                                track.velocity_px_s[1] * z / fy,
                                0.,
                            ],
                            to_stabilized,
                        ));
                    }
                    (None, None) => {}
                }
                self.flight
                    .as_ref()
                    .map(|filter| filter.flight(to_camera, gravity))
            });
            Ball {
                pixel: track.camera_px,
                measured_px: track.measured_px,
                size_px: track.size_px,
                diameter_px: track.diameter_px,
                measured_m,
                measured_stabilized_m: motion::transform(to_stabilized, measured_m),
                flight,
                reference_px: track.reference_px,
            }
        });
        if update.ball.is_none() {
            self.flight = None;
        }
        update.stage_ms[2] = tick.elapsed().as_secs_f64() * 1000.;
        update
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parabola::CameraCalibration;

    fn cluster(x: f64, y: f64, width: u16, height: u16) -> Cluster {
        let min = [x as u16 - width / 2, y as u16 - height / 2];
        Cluster {
            x,
            y,
            min,
            max: [min[0] + width - 1, min[1] + height - 1],
            events: 100,
        }
    }

    fn ring(seed_until: f64) -> Vec<RecentPoint> {
        (0..160)
            .map(|i| {
                let angle = i as f64 * TAU / 160.;
                RecentPoint {
                    x: (400. + 20. * angle.cos()) as f32,
                    y: (300. + 20. * angle.sin()) as f32,
                    foreground: angle < seed_until,
                }
            })
            .collect()
    }

    #[test]
    fn outline_fit_finds_a_partly_masked_ring_and_rejects_lines_and_clipped_arcs() {
        // Only a quarter of the ring is foreground, and the cluster is off-center.
        let mut points = ring(PI / 2.);
        let mut rng = SplitMix64(42);
        for _ in 0..100 {
            points.push(RecentPoint {
                x: 335. + rng.below(130) as f32,
                y: 235. + rng.below(130) as f32,
                foreground: false,
            });
        }
        let outline = fit_outline(&points, &cluster(416., 310., 12, 25), 1280, 720).unwrap();
        assert!(
            (outline.x - 400.).hypot(outline.y - 300.) < 1.,
            "{outline:?}"
        );
        assert!((outline.radius - 20.).abs() < 1., "{outline:?}");

        let line: Vec<_> = (0..100)
            .map(|i| RecentPoint {
                x: 370. + i as f32 * 0.6,
                y: 300.,
                foreground: true,
            })
            .collect();
        assert!(fit_outline(&line, &cluster(400., 300., 60, 3), 1280, 720).is_none());
        assert!(fit_outline(&ring(TAU), &cluster(400., 300., 40, 40), 410, 720).is_none());
    }

    fn view(to_reference: Rotation) -> View {
        View {
            intrinsics: [864., 864., 587., 350.],
            width: 1280.,
            height: 720.,
            to_reference,
        }
    }

    fn candidate(view: &View, x: f64, y: f64) -> Candidate {
        let [reference_x, reference_y] = view.reference_px([x, y]);
        Candidate {
            x,
            y,
            reference_x,
            reference_y,
            width: 20.,
            height: 20.,
            events: 120,
            outline: Some(Outline {
                x,
                y,
                radius: 10.,
                coverage: 16,
                support: 60,
            }),
        }
    }

    #[test]
    fn association_needs_motion_then_coasts_briefly() {
        let view = view(IDENTITY);
        let mut stationary = Associator::default();
        for i in 0..20 {
            let still = [candidate(&view, 400., 300.)];
            let step = stationary.step(i as f64 / 100., &still, &view, 0.07);
            assert_eq!(step.state, TrackState::Lost);
        }

        let mut moving = Associator::default();
        // A ball this large would be farther away than the tracker accepts.
        let mut implausible = Associator::default();
        let mut states = Vec::new();
        for i in 0..24 {
            let candidates: Vec<_> = (i < 12)
                .then(|| candidate(&view, 400. + 5. * i as f64, 300.))
                .into_iter()
                .collect();
            let t = i as f64 / 100.;
            states.push(moving.step(t, &candidates, &view, 0.07).state);
            assert_eq!(
                implausible.step(t, &candidates, &view, 0.5).state,
                TrackState::Lost
            );
        }
        assert_eq!(states[4], TrackState::Lost);
        assert!(states[5..12].iter().all(|&s| s == TrackState::Measured));
        assert!(states[12..20].iter().all(|&s| s == TrackState::Predicted));
        assert_eq!(states[20], TrackState::Lost);
    }

    #[test]
    fn association_derives_the_diameter_from_the_newest_outlines_only() {
        let view = view(IDENTITY);
        let mut associator = Associator::default();
        let mut diameters = Vec::new();
        for i in 0..14 {
            // The ball approaches: its outline grows from 20 px to 24 px at update 8.
            let mut c = candidate(&view, 400. + 5. * i as f64, 300.);
            if i >= 8 {
                c.width = 24.;
                c.height = 24.;
            }
            let step = associator.step(i as f64 / 100., &[c], &view, 0.07);
            diameters.push(step.track.map(|track| track.diameter_px));
        }
        // All six starting observations had an outline; five are remembered.
        assert_eq!(diameters[5], Some(20.));
        assert_eq!(diameters[9], Some(20.));
        // Three of the newest five outlines are large from update 10 on.
        assert_eq!(diameters[10], Some(24.));
        assert_eq!(diameters[13], Some(24.));
    }

    #[test]
    fn association_gate_stays_in_image_pixels_after_a_large_rotation() {
        let rotated = view(motion::rotation_vector([-PI / 3., 0., 0.]));
        let mut associator = Associator::default();
        for i in 0..12 {
            let step = associator.step(
                i as f64 / 100.,
                &[candidate(&rotated, 400., 300. + 15. * i as f64)],
                &rotated,
                0.07,
            );
            assert_eq!(step.state == TrackState::Measured, i >= 5, "update {i}");
        }
    }

    /// Standard normal samples from the fixed-seed generator.
    fn normal(rng: &mut SplitMix64) -> f64 {
        let uniform = |rng: &mut SplitMix64| (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2. * (1. - uniform(rng)).ln()).sqrt() * (TAU * uniform(rng)).cos()
    }

    #[test]
    fn flight_filter_recovers_a_ballistic_path_from_noisy_and_wrong_outlines() {
        let intrinsics = [864., 864., 587., 350.];
        let [fx, fy, cx, cy] = intrinsics;
        // A camera that is tilted against gravity and pans during the flight.
        let gravity = [0.9, 9.6, -1.8];
        let start = [-0.8, 0.5, 4.];
        let launch = [1.5, -4., -2.];
        let truth = |t: f64| -> [f64; 3] {
            std::array::from_fn(|k| start[k] + launch[k] * t + 0.5 * gravity[k] * t * t)
        };
        let to_camera = |t: f64| motion::rotation_vector([0.05 * t, 0.4 * t, -0.1 * t]);
        let mut rng = SplitMix64(7);
        let mut filter: Option<FlightFilter> = None;
        let (mut measured_error, mut filtered_error) = (0., 0.);
        for i in 0..=50 {
            let t = i as f64 * 0.01;
            let c = motion::transform(to_camera(t), truth(t));
            // Every sixth outline is a circle of half the size, 8 px off-center.
            let wrong = i % 6 == 5;
            let offset = if wrong { 8. } else { 0. };
            let pixel = [
                fx * c[0] / c[2] + cx + 2. * normal(&mut rng) + offset,
                fy * c[1] / c[2] + cy + 2. * normal(&mut rng) - offset,
            ];
            let depth = c[2] * (0.1 * normal(&mut rng)).exp() * if wrong { 2. } else { 1. };
            match &mut filter {
                None => {
                    let position = [
                        (pixel[0] - cx) * depth / fx,
                        (pixel[1] - cy) * depth / fy,
                        depth,
                    ];
                    let to_stabilized = motion::transpose(to_camera(t));
                    filter = Some(FlightFilter::new(1, t, position, [0.; 3], to_stabilized));
                }
                Some(filter) => {
                    filter.predict(t, gravity);
                    filter.measure(to_camera(t), intrinsics, pixel, 3., Some(depth));
                    if i > 25 {
                        measured_error += (depth - c[2]).abs() / 25.;
                        filtered_error +=
                            (filter.camera_position(to_camera(t))[2] - c[2]).abs() / 25.;
                    }
                }
            }
        }
        let flight = filter.unwrap().flight(to_camera(0.5), gravity);
        let distance =
            |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
        let velocity: [f64; 3] = std::array::from_fn(|k| launch[k] + gravity[k] * 0.5);
        // The ball is 2.8 m away at the end; single outlines are off by 0.7 m on average.
        assert!(
            distance(flight.stabilized_m, truth(0.5)) < 0.4,
            "{flight:?}"
        );
        assert!(
            distance(flight.stabilized_velocity_mps, velocity) < 0.6,
            "{flight:?}"
        );
        assert!(
            distance(flight.stabilized_after(0.2), truth(0.7)) < 0.45,
            "{flight:?}"
        );
        assert!(
            filtered_error < 0.6 * measured_error,
            "{measured_error} {filtered_error}"
        );
    }

    /// A ring crossing a static, flickering background in front of a still camera.
    #[test]
    fn tracker_reports_the_position_of_a_moving_ring() {
        let (fx, cx, cy) = (500., 320., 240.);
        let camera = Camera::new(
            CameraCalibration {
                image_width: Some(640),
                image_height: Some(480),
                focal_length_x_px: fx as f32,
                focal_length_y_px: fx as f32,
                principal_x_px: cx as f32,
                principal_y_px: cy as f32,
                distortion_coefficients: [0.; 5],
            },
            640,
            480,
        )
        .unwrap();
        let mut csv = String::from("timestamp,gx,gy,gz\n");
        for i in 0..=100 {
            csv.push_str(&format!("{},0,0,0\n", i as f64 * 0.01));
        }
        let imu = ImuTrack::from_csv(&csv).unwrap();
        let (radius, speed_px_s, diameter_m) = (20., 800., 0.07);
        let center = |us: u64| [100. + speed_px_s * us as f64 * 1e-6, 200.];
        let mut events = Vec::new();
        for ms in 0..400u64 {
            let timestamp_us = ms * 1000;
            for y in (30..450).step_by(3) {
                for x in [40, 41, 600, 601] {
                    events.push(Event {
                        timestamp_us,
                        x,
                        y,
                        polarity: true,
                    });
                }
            }
            let [x, y] = center(timestamp_us);
            for i in 0..180 {
                let angle = i as f64 * TAU / 180.;
                events.push(Event {
                    timestamp_us,
                    x: (x + radius * angle.cos()).round() as u16,
                    y: (y + radius * angle.sin()).round() as u16,
                    polarity: true,
                });
            }
        }
        let mut tracker = BallTracker::new(diameter_m);
        let mut last = None;
        for end_us in (40_000..400_000u64).step_by(10_000) {
            let start_us = end_us - 30_000;
            let first = events.partition_point(|e| e.timestamp_us < start_us);
            let after = events.partition_point(|e| e.timestamp_us <= end_us);
            let update = tracker.update(
                &events[first..after],
                0,
                start_us,
                end_us,
                &camera,
                &imu,
                MotionConfig::default(),
            );
            assert!(update.imu_covered);
            last = Some((end_us, update));
        }
        let (end_us, update) = last.unwrap();
        assert_eq!(update.state, TrackState::Measured, "{update:?}");
        assert!(update.outline_supported);
        assert_eq!(update.track_id, 1);
        let ball = update.ball.unwrap();
        let [x, y] = center(end_us);
        assert!(
            (ball.pixel[0] - x).abs() < 6. && (ball.pixel[1] - y).abs() < 3.,
            "{ball:?}"
        );
        assert!((ball.diameter_px - 2. * radius).abs() < 4., "{ball:?}");
        let depth = fx * diameter_m / (2. * radius);
        assert!((ball.measured_m[2] - depth).abs() < 0.1 * depth, "{ball:?}");
        assert!(
            (ball.measured_m[0] - (x - cx) * depth / fx).abs() < 0.02,
            "{ball:?}"
        );
        assert_eq!(ball.measured_m, ball.measured_stabilized_m);
        // This IMU file has no gravity columns, so there is no flight estimate.
        assert_eq!(ball.flight, None);
        assert_eq!(ball.position_m(), ball.measured_m);
    }
}
