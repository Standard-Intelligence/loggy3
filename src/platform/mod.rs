use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use colored::*;
use anyhow::Result;

// Add import for VERBOSE from main
use crate::VERBOSE;

#[cfg(target_os = "macos")]
pub mod mac;
#[cfg(target_os = "macos")]
pub use mac::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, set_path_or_start_menu_shortcut, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT, FFMPEG_FILENAME, FFMPEG_DOWNLOAD_URL};

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "windows")]
pub use windows::{get_display_info, unified_event_listener_thread_with_cache, check_and_request_permissions, get_target_matching_display_info, set_path_or_start_menu_shortcut, get_windows_version_type, check_windows_version_compatibility, WindowsVersionType, FFMPEG_ENCODER, FFMPEG_PIXEL_FORMAT, FFMPEG_FILENAME, FFMPEG_DOWNLOAD_URL};

static EVENT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    static ref MONOTONIC_START: Instant = Instant::now();
}

pub struct MultiTimestamp {
    pub seq: u64,
    pub wall_time: u128,
    pub monotonic_time: u64,
}


pub fn get_multi_timestamp() -> MultiTimestamp {
    let seq = EVENT_SEQUENCE.fetch_add(1, Ordering::SeqCst);

    let wall_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    
    let monotonic_time = MONOTONIC_START.elapsed().as_millis() as u64;
    
    MultiTimestamp { seq, wall_time, monotonic_time }
}


struct LogWriter {
    file: File,
    chunk_index: usize,
}

pub struct LogWriterCache {
    session_dir: PathBuf,
    current_writers: HashMap<String, LogWriter>,
}

impl LogWriterCache {
    pub fn new(session_dir: PathBuf) -> Self {
        Self {
            session_dir,
            current_writers: HashMap::new()
        }
    }

    fn get_writer(&mut self, timestamp_ms: u128, log_type: &str) -> Result<&mut File> {
        let chunk_index = (timestamp_ms / 60000) as usize;

        let create_new_writer = || -> LogWriter {
            let chunk_dir = &self.session_dir.join(format!("chunk_{:05}", chunk_index));
            let log_path = chunk_dir.join(format!("{}.log", log_type));
            create_dir_all(&chunk_dir).unwrap();
            
            println!("{}", format!("Created new {} log for chunk {}", log_type, chunk_index).yellow());

            LogWriter {
                file: File::create(log_path).unwrap(),
                chunk_index
            }
        };

        let logwriter = self.current_writers.entry(log_type.to_string()).or_insert_with(create_new_writer);
        if logwriter.chunk_index != chunk_index {
            *logwriter = create_new_writer();
        }
        Ok(&mut logwriter.file)
    }
}

pub fn handle_event_with_cache(
    multi_timestamp: &MultiTimestamp,
    data: String,
    log_type: &str,
    cache: &Arc<Mutex<LogWriterCache>>,
) {
    let raw_line = format!("[({}, {}, {}), '{}']\n", 
        multi_timestamp.seq, 
        multi_timestamp.wall_time, 
        multi_timestamp.monotonic_time, 
        data);
    
    if VERBOSE.load(Ordering::SeqCst) {
        match log_type {
            "mouse" => println!("{}", raw_line.blue()),
            "keypresses" => println!("{}", raw_line.green()),
            _ => println!("{}", raw_line),
        }
    }
    
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(raw_writer) = cache_lock.get_writer(multi_timestamp.wall_time, log_type) {
            if let Err(e) = raw_writer.write_all(raw_line.as_bytes()) {
                eprintln!("Warning: Failed to write to {} log: {}", log_type, e);
            }
        } else {
            eprintln!("Warning: Failed to get writer for {} log", log_type);
        }
    } else {
        eprintln!("Warning: Failed to acquire lock on log writer cache");
    }
}