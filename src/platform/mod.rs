#[cfg(target_os = "macos")]
pub mod mac;

#[cfg(target_os = "windows")]
pub mod windows;

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
    fn new(session_dir: PathBuf) -> Self {
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
            
            // Create the log file
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

fn get_display_fingerprint() -> String {
    let displays = get_display_info();
    let mut display_strings: Vec<String> = displays
        .iter()
        .map(|d| format!("{}:{}x{}", d.id, d.original_width, d.original_height))
        .collect();
    display_strings.sort();
    display_strings.join(",")
}





mod platform {
    #[cfg(target_os = "macos")]
    pub use mac::*;

    #[cfg(target_os = "windows")]
    pub use windows::*;
}
