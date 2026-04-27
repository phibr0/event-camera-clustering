#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub timestamp_us: u64,
    pub x: u16,
    pub y: u16,
    pub polarity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox {
    pub min_x: u16,
    pub min_y: u16,
    pub max_x: u16,
    pub max_y: u16,
}

impl BoundingBox {
    pub fn width(&self) -> u16 {
        self.max_x - self.min_x + 1
    }

    pub fn height(&self) -> u16 {
        self.max_y - self.min_y + 1
    }
}
