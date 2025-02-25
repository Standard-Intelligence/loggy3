use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use colored::*;
use anyhow::{Context, Result};

#[cfg(target_os = "macos")]
pub mod mac;
#[cfg(target_os = "macos")]
pub use mac::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT};

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "windows")]
pub use windows::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT};


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

pub struct LogWriterCache {
    session_dir: PathBuf,
    keypress_writers: HashMap<usize, Arc<Mutex<BufWriter<File>>>>,
    mouse_writers: HashMap<usize, Arc<Mutex<BufWriter<File>>>>,
}

impl LogWriterCache {
    pub fn new(session_dir: PathBuf) -> Self {
        Self {
            session_dir,
            keypress_writers: HashMap::new(),
            mouse_writers: HashMap::new(),
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

fn log_mouse_event_with_cache(timestamp: u128, cache: &Arc<Mutex<LogWriterCache>>, data: &str) {
    let line = format!("({}, {})\n", timestamp, data);
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_mouse_writer(timestamp) {
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
    timestamp: u128,
    cache: &Arc<Mutex<LogWriterCache>>,
    pressed_keys: &Mutex<Vec<String>>,
) {
    let key_str = format!("{:?}", key);
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

    let line = format!("({}, '{}')\n", timestamp, state);
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_keypress_writer(timestamp) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = writer_lock.write_all(line.as_bytes());
                let _ = writer_lock.flush();
            }
        }
    }
}