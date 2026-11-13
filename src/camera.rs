//! Live events from an event camera over USB.
//!
//! The camera support is optional: build with `--features camera`. It uses the
//! `neuromorphic-drivers` crate, which talks to IMX636 cameras such as the
//! Prophesee EVK4 and the IDS uEye XCP-E through libusb.
use event_clustering::Event;
use std::time::Instant;

pub(crate) const WIDTH: usize = 1280;
pub(crate) const HEIGHT: usize = 720;
/// The sensor can switch off this many single pixels.
pub(crate) const MAX_MASKED_PIXELS: usize = 64;

/// The events of one USB buffer. Timestamps count from the start of the camera.
pub(crate) struct Batch {
    pub(crate) events: Vec<Event>,
    /// When the buffer arrived from the camera.
    pub(crate) received: Instant,
}

/// Parses `x:y,x:y,...` in the pixel coordinates of the events.
pub(crate) fn parse_pixels(text: &str) -> Result<Vec<[u16; 2]>, String> {
    let pixels = text
        .split(',')
        .map(|pixel| {
            let (x, y) = pixel.trim().split_once(':')?;
            let pixel = [x.parse::<u16>().ok()?, y.parse::<u16>().ok()?];
            ((pixel[0] as usize) < WIDTH && (pixel[1] as usize) < HEIGHT).then_some(pixel)
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(format!(
            "--mask-pixels needs x:y,x:y,... inside {WIDTH}x{HEIGHT}"
        ))?;
    if pixels.len() > MAX_MASKED_PIXELS {
        return Err(format!(
            "the sensor can mask at most {MAX_MASKED_PIXELS} pixels"
        ));
    }
    Ok(pixels)
}

#[cfg(feature = "camera")]
mod usb {
    use super::{Batch, HEIGHT, WIDTH};
    use event_clustering::parser::Evt3Decoder;
    use neuromorphic_drivers::{GenericDevice, prophesee_evk4};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// The first connected camera, read on its own thread.
    pub(crate) struct LiveCamera {
        pub(crate) serial: String,
        batches: mpsc::Receiver<Result<Batch, String>>,
        stop: Arc<AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl LiveCamera {
        /// `masked` pixels are switched off in the sensor, for defective pixels
        /// that fire without pause.
        pub(crate) fn open(masked: &[[u16; 2]]) -> Result<Self, String> {
            let mut configuration = prophesee_evk4::DEFAULT_CONFIGURATION;
            for (slot, &[x, y]) in masked.iter().enumerate() {
                // 21 bits per pixel, three per word; rows count from the bottom.
                let code = 1 + x as u64 + WIDTH as u64 * (HEIGHT as u64 - 1 - y as u64);
                configuration.pixel_mask[slot / 3] |= code << ((slot % 3) * 21);
            }
            let (opened, wait_opened) = mpsc::channel();
            let (sender, batches) = mpsc::channel();
            let stop = Arc::new(AtomicBool::new(false));
            let thread = std::thread::spawn({
                let stop = stop.clone();
                move || {
                    let run = || -> Result<(), String> {
                        let (flag, event_loop) = neuromorphic_drivers::flag_and_event_loop()
                            .map_err(|error| error.to_string())?;
                        let device = prophesee_evk4::open(
                            neuromorphic_drivers::Identifier::None,
                            configuration,
                            &prophesee_evk4::RING_CONFIGURATION,
                            event_loop,
                            flag.clone(),
                        )
                        .map_err(|error| format!("cannot open a camera: {error}"))?;
                        let _ = opened.send(Ok(device.serial()));
                        let mut decoder = Evt3Decoder::new();
                        while !stop.load(Ordering::Relaxed) {
                            if let Some(view) =
                                device.next_with_timeout(&Duration::from_millis(100))
                            {
                                let mut events = Vec::new();
                                decoder.decode(view.slice, &mut events);
                                let batch = Batch {
                                    events,
                                    received: view.instant,
                                };
                                if sender.send(Ok(batch)).is_err() {
                                    break;
                                }
                            }
                            flag.load_error().map_err(|error| error.to_string())?;
                        }
                        Ok(())
                    };
                    if let Err(error) = run() {
                        // Whichever side still listens gets the error.
                        let _ = opened.send(Err(error.clone()));
                        let _ = sender.send(Err(error));
                    }
                }
            });
            let serial = wait_opened
                .recv()
                .map_err(|_| "the camera thread stopped".to_string())??;
            Ok(Self {
                serial,
                batches,
                stop,
                thread: Some(thread),
            })
        }

        /// The next batch that has arrived, without waiting.
        pub(crate) fn try_next(&mut self) -> Result<Option<Batch>, String> {
            match self.batches.try_recv() {
                Ok(batch) => batch.map(Some),
                Err(mpsc::TryRecvError::Empty) => Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => Err("the camera stopped".into()),
            }
        }
    }

    impl Drop for LiveCamera {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(not(feature = "camera"))]
mod usb {
    use super::Batch;

    /// Stand-in for builds without camera support.
    pub(crate) struct LiveCamera {
        pub(crate) serial: String,
    }

    impl LiveCamera {
        pub(crate) fn open(_masked: &[[u16; 2]]) -> Result<Self, String> {
            Err(
                "this build has no camera support; run with `cargo run --release --features camera -- ...`"
                    .into(),
            )
        }

        pub(crate) fn try_next(&mut self) -> Result<Option<Batch>, String> {
            Ok(None)
        }
    }
}

pub(crate) use usb::LiveCamera;
