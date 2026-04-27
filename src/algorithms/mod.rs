pub mod rolling_cluster;

pub use rolling_cluster::{ClusterDetection, RollingClusterTracker, RollingClusterTrackerConfig};

use crate::Event;

pub trait EventAlgorithm {
    type Output;

    fn process_event(&mut self, event: Event) -> Vec<Self::Output>;

    fn finish(&mut self) -> Vec<Self::Output> {
        Vec::new()
    }
}
