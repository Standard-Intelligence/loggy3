use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use colored::*;
use anyhow::Result;

// Global atomic sequence counter for all input events
static EVENT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

// Global timer start point for monotonic clock reference
lazy_static::lazy_static! {
    static ref MONOTONIC_START: Instant = Instant::now();
}

// Helper function to get the next sequence number
pub fn get_next_sequence() -> u64 {
    EVENT_SEQUENCE.fetch_add(1, Ordering::SeqCst)
}

// Get multiple timestamps for more robust time tracking
pub fn get_multi_timestamp() -> (u128, u64) {
    let wall_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    
    let monotonic_time = MONOTONIC_START.elapsed().as_millis() as u64;
    
    (wall_time, monotonic_time)
}

#[cfg(target_os = "macos")]
pub mod mac;
#[cfg(target_os = "macos")]
pub use mac::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT, FFMPEG_DOWNLOAD_URL};

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "windows")]
pub use windows::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT, FFMPEG_DOWNLOAD_URL};

pub static IS_WINDOWS: bool = cfg!(target_os = "windows");

pub struct LogWriterCache {
    session_dir: PathBuf,
    keypress_writers: HashMap<usize, Arc<Mutex<BufWriter<File>>>>,
    mouse_writers: HashMap<usize, Arc<Mutex<BufWriter<File>>>>,
    raw_keypress_writers: HashMap<usize, Arc<Mutex<BufWriter<File>>>>,
}

impl LogWriterCache {
    pub fn new(session_dir: PathBuf) -> Self {
        Self {
            session_dir,
            keypress_writers: HashMap::new(),
            mouse_writers: HashMap::new(),
            raw_keypress_writers: HashMap::new(),
        }
    }

    fn get_keypress_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.keypress_writers.contains_key(&chunk_index) {
            let chunk_dir = self.session_dir.join(format!("chunk_{:05}", chunk_index));
            create_dir_all(&chunk_dir)?;
            
            let log_path = chunk_dir.join("keypresses.log");
            let writer = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
            println!("{}", format!("Created new keypress log for chunk {}", chunk_index).yellow());
            self.keypress_writers.insert(chunk_index, writer);
        }
        Ok(self.keypress_writers.get(&chunk_index).unwrap().clone())
    }
    
    fn get_raw_keypress_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.raw_keypress_writers.contains_key(&chunk_index) {
            let chunk_dir = self.session_dir.join(format!("chunk_{:05}", chunk_index));
            create_dir_all(&chunk_dir)?;
            
            let log_path = chunk_dir.join("raw_keypresses.log");
            let writer = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
            println!("{}", format!("Created new raw keypress log for chunk {}", chunk_index).yellow());
            self.raw_keypress_writers.insert(chunk_index, writer);
        }
        Ok(self.raw_keypress_writers.get(&chunk_index).unwrap().clone())
    }
    
    fn get_mouse_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.mouse_writers.contains_key(&chunk_index) {
            let chunk_dir = self.session_dir.join(format!("chunk_{:05}", chunk_index));
            create_dir_all(&chunk_dir)?;
            
            let log_path = chunk_dir.join("mouse.log");
            let writer = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
            println!("{}", format!("Created new mouse log for chunk {}", chunk_index).yellow());
            self.mouse_writers.insert(chunk_index, writer);
        }
        Ok(self.mouse_writers.get(&chunk_index).unwrap().clone())
    }
}

fn log_mouse_event_with_cache(_timestamp: u128, cache: &Arc<Mutex<LogWriterCache>>, data: &str) {
    let seq = get_next_sequence();
    let (wall_time, monotonic_time) = get_multi_timestamp();
    let line = format!("({}, {}, {}, {})\n", seq, wall_time, monotonic_time, data);
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_mouse_writer(wall_time) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = writer_lock.write_all(line.as_bytes());
                let _ = writer_lock.flush();
            }
        }
    }
}

fn handle_key_event_with_cache(
    is_press: bool,
    key: rdev::Key,
    _timestamp: u128,
    cache: &Arc<Mutex<LogWriterCache>>,
    pressed_keys: &Mutex<Vec<String>>,
) {
    let key_str = format!("{:?}", key);
    
    // Get multi-timestamp for more robust logging
    let (wall_time, monotonic_time) = get_multi_timestamp();
    
    // Log the raw key event first with sequence number and multi-timestamp
    let action = if is_press { "press" } else { "release" };
    let raw_seq = get_next_sequence();
    let raw_line = format!("({}, {}, {}, '{}', '{}')\n", raw_seq, wall_time, monotonic_time, action, key_str);
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(raw_writer) = cache_lock.get_raw_keypress_writer(wall_time) {
            if let Ok(mut raw_writer_lock) = raw_writer.lock() {
                let _ = raw_writer_lock.write_all(raw_line.as_bytes());
                let _ = raw_writer_lock.flush();
            }
        }
    }
    
    // Continue with the existing state-tracking logic
    let mut keys = pressed_keys.lock().unwrap();

    if is_press {
        if !keys.contains(&key_str) {
            keys.push(key_str.clone());
        }
    } else {
        keys.retain(|k| k != &key_str);
    }

    let state = if keys.is_empty() {
        "none".to_string()
    } else {
        format!("+{}", keys.join("+"))
    };

    // Use a different sequence number for the state event
    let state_seq = get_next_sequence();
    let line = format!("({}, {}, {}, '{}')\n", state_seq, wall_time, monotonic_time, state);
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_keypress_writer(wall_time) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = writer_lock.write_all(line.as_bytes());
                let _ = writer_lock.flush();
            }
        }
    }
}