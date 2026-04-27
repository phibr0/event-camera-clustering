use event_clustering::Event;
use event_clustering::algorithms::ClusterDetection;
use std::collections::VecDeque;

pub(crate) fn fill_rgb_bytes(buffer: &[u32], bytes: &mut [u8]) {
    for (color, rgb) in buffer.iter().zip(bytes.chunks_exact_mut(3)) {
        rgb[0] = ((color >> 16) & 0xff) as u8;
        rgb[1] = ((color >> 8) & 0xff) as u8;
        rgb[2] = (color & 0xff) as u8;
    }
}

pub(crate) fn render_frame(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    events: &VecDeque<Event>,
    detection: Option<&ClusterDetection>,
    invert_polarity: bool,
) {
    buffer.fill(0x000000);

    for event in events {
        let x = usize::from(event.x);
        let y = usize::from(event.y);
        if x >= width || y >= height {
            continue;
        }

        let index = y * width + x;
        let polarity = if invert_polarity {
            !event.polarity
        } else {
            event.polarity
        };
        buffer[index] = if polarity { 0xffffff } else { 0x3060ff };
    }

    if let Some(detection) = detection {
        draw_bbox(buffer, width, height, detection, 0xff2020);
        draw_cross(
            buffer,
            width,
            height,
            detection.centroid_x.round() as i32,
            detection.centroid_y.round() as i32,
            6,
            0xffff00,
        );
    }
}

fn draw_bbox(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    detection: &ClusterDetection,
    color: u32,
) {
    let min_x = usize::from(detection.bbox.min_x).min(width.saturating_sub(1));
    let min_y = usize::from(detection.bbox.min_y).min(height.saturating_sub(1));
    let max_x = usize::from(detection.bbox.max_x).min(width.saturating_sub(1));
    let max_y = usize::from(detection.bbox.max_y).min(height.saturating_sub(1));

    for x in min_x..=max_x {
        set_pixel(buffer, width, height, x as i32, min_y as i32, color);
        set_pixel(buffer, width, height, x as i32, max_y as i32, color);
    }
    for y in min_y..=max_y {
        set_pixel(buffer, width, height, min_x as i32, y as i32, color);
        set_pixel(buffer, width, height, max_x as i32, y as i32, color);
    }
}

fn draw_cross(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    radius: i32,
    color: u32,
) {
    for offset in -radius..=radius {
        set_pixel(buffer, width, height, x + offset, y, color);
        set_pixel(buffer, width, height, x, y + offset, color);
    }
}

fn set_pixel(buffer: &mut [u32], width: usize, height: usize, x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x < width && y < height {
        buffer[y * width + x] = color;
    }
}
