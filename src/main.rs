use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use platform::{DisplayInfo, FFMPEG_ENCODER};
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File},
    io::{BufWriter, Write, BufReader, BufRead},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType, YUVFrame},
    Target,
};
use std::fs;

mod platform;


struct Session {
    should_run: Arc<AtomicBool>,
    session_dir: PathBuf,

    event_thread: Option<thread::JoinHandle<()>>,

    capture_threads: Vec<(Arc<AtomicBool>, thread::JoinHandle<()>)>,

    keypress_log: Arc<Mutex<BufWriter<File>>>,
    mouse_log: Arc<Mutex<BufWriter<File>>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,

    error_rx: Receiver<()>,
    error_tx: Sender<()>,
    
    displays: Vec<DisplayInfo>,
    progress_threads: Vec<thread::JoinHandle<()>>,

    restart_rx: Receiver<()>,
    restart_tx: Sender<()>,
}

impl Session {
    fn new(should_run: Arc<AtomicBool>) -> Result<Option<Self>> {
        let displays = platform::get_display_info();
        if displays.is_empty() {
            return Ok(None);
        }

        let home_dir = dirs::home_dir().context("Could not determine home directory")?;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let session_dir = home_dir.join(format!("loggy3/session_{}", timestamp));
        create_dir_all(&session_dir)?;

        println!("Starting new session at: {}", session_dir.display());

        let json_path = session_dir.join("display_info.json");
        let mut f = File::create(&json_path)?;
        serde_json::to_writer_pretty(&mut f, &displays)?;

        let keypress_log_path = session_dir.join("keypresses.log");
        let mouse_log_path = session_dir.join("mouse.log");
        let keypress_log = Arc::new(Mutex::new(BufWriter::new(File::create(keypress_log_path)?)));
        let mouse_log = Arc::new(Mutex::new(BufWriter::new(File::create(mouse_log_path)?)));
        let pressed_keys = Arc::new(Mutex::new(vec![]));

        let (error_tx, error_rx) = mpsc::channel();

        let (restart_tx, restart_rx) = mpsc::channel();

        Ok(Some(Self {
            should_run,
            session_dir,
            event_thread: None,
            capture_threads: Vec::new(),
            keypress_log,
            mouse_log,
            pressed_keys,
            error_rx,
            error_tx,
            displays,
            progress_threads: Vec::new(),
            restart_rx,
            restart_tx,
        }))
    }

    fn start(&mut self) {
        let sr_clone_el = self.should_run.clone();
        let kp_log = self.keypress_log.clone();
        let m_log = self.mouse_log.clone();
        let keys = self.pressed_keys.clone();
        let restart_tx = self.restart_tx.clone();
        self.event_thread = Some(thread::spawn(move || {
            platform::unified_event_listener_thread(sr_clone_el, kp_log, m_log, keys, restart_tx)
        }));

        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }
    }

    fn stop(self) {
        for (flag, handle) in self.capture_threads {
            flag.store(false, Ordering::SeqCst);
            let _ = handle.join();
        }

        if let Some(event_thread) = self.event_thread {
            let start = Instant::now();
            let timeout = Duration::from_secs(5);

            while start.elapsed() < timeout {
                if event_thread.is_finished() {
                    let _ = event_thread.join();
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }

        // Stop progress indicator threads
        for handle in self.progress_threads {
            let _ = handle.join();
        }

        println!("Session stopped: {}", self.session_dir.display());
    }

    fn check_for_errors(&mut self) -> bool {
        let mut full_restart = false;
        while let Ok(_) = self.error_rx.try_recv() {
            full_restart = true;
        }
        full_restart
    }

    fn check_for_restart(&mut self) -> bool {
        self.restart_rx.try_recv().is_ok()
    }

    fn get_frame_count(&self, mp4_path: &PathBuf) -> Option<usize> {
        let ffprobe_path = get_ffmpeg_path().parent().unwrap().join("ffprobe");
        
        let output = Command::new(ffprobe_path)
            .args(&[
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                mp4_path.to_str().unwrap()
            ])
            .output()
            .ok()?;

        String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
    }

    fn get_timestamp_range(&self, path: &PathBuf, lines_to_remove: usize) -> Option<(u128, u128)> {
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            let mut first_timestamp = None;
            let mut last_timestamp = None;
            
            // Read only the lines we're going to remove
            for (i, line) in reader.lines().take(lines_to_remove).enumerate() {
                if let Ok(line) = line {
                    if let Ok(timestamp) = line.trim().parse::<u128>() {
                        if i == 0 {
                            first_timestamp = Some(timestamp);
                        }
                        last_timestamp = Some(timestamp);
                    }
                }
            }
            
            if let (Some(first), Some(last)) = (first_timestamp, last_timestamp) {
                return Some((first, last));
            }
        }
        None
    }

    fn clean_log_file_by_timestamp_range(&self, path: &PathBuf, start_time: u128, end_time: u128) {
        if !path.exists() {
            println!("    {} doesn't exist, skipping", path.file_name().unwrap().to_string_lossy());
            return;
        }

        let temp_path = path.with_extension("tmp");
        if let Ok(file) = File::open(path) {
            if let Ok(temp_file) = File::create(&temp_path) {
                let reader = BufReader::new(file);
                let mut writer = BufWriter::new(temp_file);
                let mut total_lines = 0;
                let mut kept_lines = 0;

                for line in reader.lines().filter_map(|l| l.ok()) {
                    total_lines += 1;
                    if let Some(timestamp_str) = line.split(',').next() {
                        if let Ok(timestamp) = timestamp_str.trim_matches(|c| c == '(' || c == ')')
                            .parse::<u128>() 
                        {
                            if timestamp < start_time || timestamp > end_time {
                                writeln!(writer, "{}", line).ok();
                                kept_lines += 1;
                            }
                        }
                    }
                }
                
                writer.flush().ok();
                println!("    Processed {} lines, kept {} lines", total_lines, kept_lines);
            }
        }
        
        fs::rename(&temp_path, path).ok();
    }

    fn clean_log_file_by_lines(&self, path: &PathBuf, lines_to_remove: usize) {
        if !path.exists() {
            println!("    {} doesn't exist, skipping", path.file_name().unwrap().to_string_lossy());
            return;
        }

        let temp_path = path.with_extension("tmp");
        if let Ok(file) = File::open(path) {
            if let Ok(temp_file) = File::create(&temp_path) {
                let reader = BufReader::new(file);
                let mut writer = BufWriter::new(temp_file);
                let mut total_lines = 0;
                let mut kept_lines = 0;

                // Skip the lines we want to remove
                for line in reader.lines().skip(lines_to_remove).filter_map(|l| l.ok()) {
                    total_lines += 1;
                    writeln!(writer, "{}", line).ok();
                    kept_lines += 1;
                }
                
                writer.flush().ok();
                println!("    Processed {} lines, kept {} lines", total_lines + lines_to_remove, kept_lines);
            }
        }
        
        fs::rename(&temp_path, path).ok();
    }

    fn handle_restart(&mut self) {
        println!("\nHandling Command+Shift+L restart...");
        
        let mut latest_timestamp_range = None;
        
        // Process each display
        for display in &self.displays {
            let display_dir = self.session_dir.join(format!("display_{}_{}", display.id, display.title));
            if !display_dir.exists() {
                println!("Skipping display {} - directory doesn't exist", display.id);
                continue;
            }

            println!("\nProcessing display {} ({}):", display.id, display.title);

            // Find and delete the two most recent MP4 files
            let mut mp4_files: Vec<_> = fs::read_dir(&display_dir)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path()
                        .extension()
                        .map_or(false, |ext| ext == "mp4")
                })
                .collect();

            if mp4_files.is_empty() {
                println!("  No MP4 files found");
                continue;
            }

            mp4_files.sort_by_key(|entry| entry.metadata().unwrap().modified().unwrap());
            
            let files_to_remove: Vec<_> = mp4_files.iter().rev().take(2).collect();
            
            println!("  Found {} MP4 files, removing {} most recent:", 
                mp4_files.len(), files_to_remove.len());

            let mut frames_in_files = 0;
            for file in &files_to_remove {
                let path = file.path();
                if let Some(frame_count) = self.get_frame_count(&path) {
                    println!("    Removing: {} ({} frames)", 
                        path.file_name().unwrap().to_string_lossy(),
                        frame_count);
                    frames_in_files += frame_count;
                    let _ = fs::remove_file(&path);
                }
            }
            
            // Clean up frames.log and get timestamp range
            let frames_log_path = display_dir.join("frames.log");
            if frames_log_path.exists() {
                println!("  Cleaning frames.log, removing {} frames", frames_in_files);
                
                // Get timestamp range before removing lines
                if let Some((start, end)) = self.get_timestamp_range(&frames_log_path, frames_in_files) {
                    match latest_timestamp_range {
                        None => latest_timestamp_range = Some((start, end)),
                        Some((curr_start, curr_end)) => {
                            latest_timestamp_range = Some((
                                curr_start.min(start),
                                curr_end.max(end)
                            ));
                        }
                    }
                    println!("    Frame timestamp range: {} to {}", start, end);
                }
                
                self.clean_log_file_by_lines(&frames_log_path, frames_in_files);
            }
        }

        // Clean up event logs based on timestamp range
        if let Some((start_time, end_time)) = latest_timestamp_range {
            println!("\nCleaning event logs for timestamp range {} to {}", start_time, end_time);
            
            let keypress_log_path = self.session_dir.join("keypresses.log");
            let mouse_log_path = self.session_dir.join("mouse.log");
            
            println!("  Cleaning keypresses.log");
            self.clean_log_file_by_timestamp_range(&keypress_log_path, start_time, end_time);
            println!("  Cleaning mouse.log");
            self.clean_log_file_by_timestamp_range(&mouse_log_path, start_time, end_time);
        } else {
            println!("\nNo timestamp range found, skipping event log cleanup");
        }

        println!("\nRestarting all threads and logs...");

        // First stop all threads
        for (flag, handle) in self.capture_threads.drain(..) {
            flag.store(false, Ordering::SeqCst);
            let _ = handle.join();
        }

        // Stop event thread
        if let Some(event_thread) = self.event_thread.take() {
            let start = Instant::now();
            let timeout = Duration::from_secs(5);
            while start.elapsed() < timeout {
                if event_thread.is_finished() {
                    let _ = event_thread.join();
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }

        // Recreate log files
        let keypress_log_path = self.session_dir.join("keypresses.log");
        let mouse_log_path = self.session_dir.join("mouse.log");
        
        // Replace the log files with new ones
        if let Ok(file) = File::create(&keypress_log_path) {
            self.keypress_log = Arc::new(Mutex::new(BufWriter::new(file)));
        }
        if let Ok(file) = File::create(&mouse_log_path) {
            self.mouse_log = Arc::new(Mutex::new(BufWriter::new(file)));
        }

        // Clear pressed keys
        if let Ok(mut keys) = self.pressed_keys.lock() {
            keys.clear();
        }

        // Restart event thread
        let sr_clone_el = self.should_run.clone();
        let kp_log = self.keypress_log.clone();
        let m_log = self.mouse_log.clone();
        let keys = self.pressed_keys.clone();
        let restart_tx = self.restart_tx.clone();
        self.event_thread = Some(thread::spawn(move || {
            platform::unified_event_listener_thread(sr_clone_el, kp_log, m_log, keys, restart_tx)
        }));

        // Restart capture threads
        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }

        println!("Restart complete!\n");
    }

    fn start_capture_for_display(&mut self, display: DisplayInfo) {
        let sr_for_thread = Arc::new(AtomicBool::new(true));
        let sr_clone = sr_for_thread.clone();
        let session_dir = self.session_dir.clone();
        let error_tx = self.error_tx.clone();

        let handle = thread::spawn(move || {
            capture_display_thread(sr_clone, display, session_dir, error_tx);
        });
        self.capture_threads.push((sr_for_thread, handle));
    }
}

fn main() -> Result<()> {
    // Clear terminal window
    print!("\x1B[2J\x1B[1;1H");
    std::io::stdout().flush().unwrap();

    // Log ffmpeg path once at startup
    let ffmpeg_path = get_ffmpeg_path();
    println!("Using ffmpeg at: {}", ffmpeg_path.display());

    let should_run = Arc::new(AtomicBool::new(true));

    let sr_for_signals = should_run.clone();
    thread::spawn(move || {
        let (tx, rx) = channel();
        
        set_handler(move || tx.send(()).expect("Could not send signal on channel."))
            .expect("Error setting Ctrl-C handler");
        
        println!("Waiting for Ctrl-C...");
        rx.recv().expect("Could not receive from channel.");
        println!("Got it! Exiting..."); 
        
        sr_for_signals.store(false, Ordering::SeqCst);
    });
    let mut last_display_fingerprint = String::new();

    while should_run.load(Ordering::SeqCst) {
        let current_fingerprint = get_display_fingerprint();
        let displays_changed = current_fingerprint != last_display_fingerprint;
        last_display_fingerprint = current_fingerprint.clone();

        match Session::new(should_run.clone())? {
            Some(mut session) => {
                session.start();

                while should_run.load(Ordering::SeqCst) {
                    if session.check_for_restart() {
                        session.handle_restart();
                    }

                    let need_restart = session.check_for_errors();
                    if need_restart {
                        println!("Session signaled a critical error. Restarting session.");
                        break;
                    }

                    let current = get_display_fingerprint();
                    if current != current_fingerprint {
                        println!("Display configuration changed. Starting new session.");
                        break;
                    }

                    thread::sleep(Duration::from_secs(1));
                }

                session.stop();
            }
            None => {
                if displays_changed {
                    println!("All displays disconnected. Waiting for displays to be connected...");
                }
                thread::sleep(Duration::from_secs(10));
            }
        }
    }

    Ok(())
}

fn get_display_fingerprint() -> String {
    let displays = platform::get_display_info();
    let mut display_strings: Vec<String> = displays
        .iter()
        .map(|d| format!("{}:{}x{}", d.id, d.original_width, d.original_height))
        .collect();
    display_strings.sort();
    display_strings.join(",")
}

fn capture_display_thread(
    should_run: Arc<AtomicBool>,
    display_info: DisplayInfo,
    session_dir: PathBuf,
    error_tx: Sender<()>,
) {
    
    let targets = scap::get_all_targets().into_iter().filter(|t| matches!(t, Target::Display(_))).collect::<Vec<_>>();
    
    let target = match targets.iter()
        .find(|t| match t {
            Target::Display(d) => d.id == display_info.id,
            _ => false
        })
        .cloned() {
            Some(t) => t,
            None => {
                eprintln!("Could not find matching display target for ID: {}", display_info.id);
                return;
            }
        };

    let capturer = match initialize_capturer(&target) {
        Some(c) => c,
        None => return,
    };

    let (width, height) = match capturer.lock() {
        Ok(mut c) => {
            let sz = c.get_output_frame_size();
            (sz[0], sz[1])
        }
        Err(_) => return,
    };


    let display_dir = session_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
    if let Err(e) = create_dir_all(&display_dir) {
        eprintln!("Failed to create display directory: {}", e);
        return;
    }

    let (mut ffmpeg_child, mut ffmpeg_stdin) = match initialize_ffmpeg(
        &display_dir,
        width.try_into().unwrap(),
        height.try_into().unwrap(),
    ) {
        Ok(child_and_stdin) => child_and_stdin,
        Err(e) => {
            eprintln!("Failed to launch ffmpeg: {}", e);
            return;
        }
    };

    // Just redirect stdout to /dev/null since we don't need it
    if let Some(stdout) = ffmpeg_child.stdout.take() {
        thread::spawn(move || {
            let _ = std::io::copy(&mut BufReader::new(stdout), &mut std::io::sink());
        });
    }

    // Print any error messages from ffmpeg
    if let Some(stderr) = ffmpeg_child.stderr.take() {
        let display_id = display_info.id;
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    eprintln!("FFmpeg (display {}): {}", display_id, line);
                }
            }
        });
    }
    
    let frames_log_path = display_dir.join("frames.log");
    let frames_log_file = match File::create(&frames_log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create frames log: {}", e);
            return;
        }
    };
    let mut frames_log = BufWriter::new(frames_log_file);
    
    while should_run.load(Ordering::SeqCst) {
        let (tx, rx) = mpsc::channel();
        let capturer_clone = capturer.clone();

        thread::spawn(move || {
            if let Ok(c) = capturer_clone.lock() {
                let frame = c.get_next_frame();
                let _ = tx.send(frame);
            }
        });

        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(Frame::YUVFrame(frame))) => {
                if let Err(e) = write_frame(&mut ffmpeg_stdin, &frame, &mut frames_log) {
                    eprintln!("Write error for display {}: {}", display_info.id, e);
                    break;
                }
            }
            Ok(Ok(_)) => {}

            Ok(Err(e)) => {
                eprintln!("Frame error on display {}: {}", display_info.id, e);
                handle_capture_error(&error_tx);
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                eprintln!("Frame timeout on display {}", display_info.id);
                handle_capture_error(&error_tx);
                break;
            }
            Err(e) => {
                eprintln!("Channel error on display {}: {}", display_info.id, e);
                break;
            }
        }
    }

    drop(ffmpeg_stdin);
    let _ = ffmpeg_child.wait();
    println!("Stopped capture for display {}", display_info.id);
}

fn handle_capture_error(error_tx: &Sender<()>) {
    let _ = error_tx.send(());
}

fn initialize_capturer(target: &Target) -> Option<Arc<Mutex<Capturer>>> {
    let opts = Options {
        fps: 30,
        output_type: FrameType::YUVFrame,
        output_resolution: Resolution::_720p,
        target: Some(target.clone()),
        show_cursor: true,
        ..Default::default()
    };
    match Capturer::build(opts) {
        Ok(mut c) => {
            c.start_capture();
            Some(Arc::new(Mutex::new(c)))
        }
        Err(e) => {
            eprintln!("Capturer init failed: {}", e);
            None
        }
    }
}

fn get_ffmpeg_path() -> PathBuf {
    let ffmpeg_paths = vec![
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg", 
        "/usr/bin/ffmpeg",
    ];

    for path in ffmpeg_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            return path_buf;
        }
    }
    
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(app_bundle) = exe_path.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
            let bundled_ffmpeg = app_bundle.join("Contents/Frameworks/ffmpeg");
            if bundled_ffmpeg.exists() {
                return bundled_ffmpeg;
            }
        }
    }

    PathBuf::from("ffmpeg")
}

fn initialize_ffmpeg(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
) -> Result<(Child, ChildStdin)> {
    let output_path = display_dir.join("output_%Y%m%d_%H%M%S.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let ffmpeg_path = get_ffmpeg_path();

    let mut child = Command::new(ffmpeg_path)
        .args(&[
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-color_range", "tv",
            "-s", &format!("{}x{}", width, height),
            "-r", "30",
            "-i", "pipe:0",
            "-c:v", FFMPEG_ENCODER,
            "-movflags", "+faststart",
            "-g", "60",
            "-f", "segment",
            "-segment_time", "60",
            "-reset_timestamps", "1",
            "-strftime", "1",
            "-loglevel", "error",
            &output_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let stdin = child.stdin.take().unwrap();
    Ok((child, stdin))
}

fn write_frame(
    ffmpeg_stdin: &mut ChildStdin,
    frame: &YUVFrame,
    frames_log: &mut BufWriter<File>,
) -> Result<()> {
    ffmpeg_stdin.write_all(&frame.luminance_bytes)?;
    ffmpeg_stdin.write_all(&frame.chrominance_bytes)?;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis();
    writeln!(frames_log, "{}", timestamp)?;
    frames_log.flush()?;

    Ok(())
}