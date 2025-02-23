
#[cfg(target_os = "macos")]
pub mod mac;

#[cfg(target_os = "windows")]
pub mod windows;


#[cfg(target_os = "macos")]
pub use mac as current;

#[cfg(target_os = "windows")]
pub use windows as current;


#[cfg(target_os = "macos")]
pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";

#[cfg(target_os = "windows")]
pub static FFMPEG_ENCODER: &str = "libx264";


use std::{fs::File, io::BufWriter, sync::{atomic::AtomicBool, Arc, Mutex}};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DisplayInfo {
    pub id: u32,
    pub title: String,
    pub is_primary: bool,

    pub x: i32,
    pub y: i32,

    pub original_width: u32,
    pub original_height: u32,
    pub capture_width: u32,
    pub capture_height: u32,
}

pub fn unified_event_listener_thread(
    should_run: Arc<AtomicBool>,
    keypress_log: Arc<Mutex<BufWriter<File>>>,
    mouse_log: Arc<Mutex<BufWriter<File>>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,
) {
    current::unified_event_listener_thread(should_run, keypress_log, mouse_log, pressed_keys);
}

pub fn get_display_info() -> Vec<DisplayInfo> {
    current::get_display_info()
}

