mod platform;

use anyhow::{Context, Result};
use chrono::Local;
use colored::*;
use ctrlc::set_handler;
use dirs;
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType, YUVFrame, BGRAFrame},
    Target,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{create_dir_all, File},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::PathBuf,
    process::{exit, Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use sysinfo::System;
use ureq;
use lazy_static::lazy_static;
use uuid::Uuid;


static VERBOSE: AtomicBool = AtomicBool::new(false);
const GITHUB_REPO: &str = "Standard-Intelligence/loggy3";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const USER_ID_FILENAME: &str = "user_id.txt";

lazy_static! {
    static ref FFMPEG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
    static ref FFMPEG_DOWNLOAD_MUTEX: Mutex<()> = Mutex::new(());
    static ref USER_ID: Mutex<String> = Mutex::new(String::new());
}

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

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
    html_url: String,
}

#[derive(Debug, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

struct Session {
    should_run: Arc<AtomicBool>,
    session_dir: PathBuf,

    event_thread: Option<thread::JoinHandle<()>>,
    capture_threads: Vec<(Arc<AtomicBool>, thread::JoinHandle<()>)>,

    writer_cache: Arc<Mutex<platform::LogWriterCache>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,

    error_rx: Receiver<()>,
    error_tx: Sender<()>,
    
    displays: Vec<DisplayInfo>,
}

impl Session {
    fn new(should_run: Arc<AtomicBool>) -> Result<Option<Self>> {
        let displays = platform::get_display_info();
        if displays.is_empty() {
            return Ok(None);
        }

        let home_dir = dirs::home_dir().context("Could not determine home directory")?;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let session_dir = home_dir.join("Documents/loggy3").join(format!("session_{}", timestamp));
        create_dir_all(&session_dir)?;

        println!("\n{}", "=== Starting new recording session ===".cyan().bold());
        println!("Session directory: {}", session_dir.display().to_string().cyan());

        if let Err(e) = log_session_metadata(&session_dir) {
            eprintln!("Warning: Failed to log session metadata: {}", e);
        }

        let json_path = session_dir.join("display_info.json");
        let mut f = File::create(&json_path)?;
        serde_json::to_writer_pretty(&mut f, &displays)?;
        
        let writer_cache = Arc::new(Mutex::new(platform::LogWriterCache::new(session_dir.clone())));
        let pressed_keys = Arc::new(Mutex::new(vec![]));

        let (error_tx, error_rx) = mpsc::channel();

        Ok(Some(Self {
            should_run,
            session_dir,
            event_thread: None,
            capture_threads: Vec::new(),
            writer_cache,
            pressed_keys,
            error_rx,
            error_tx,
            displays,
        }))
    }
    
    fn start(&mut self) {
        let should_run = self.should_run.clone();
        let writer_cache = self.writer_cache.clone();
        let pressed_keys = self.pressed_keys.clone();
        self.event_thread = Some(thread::spawn(move || {
            platform::unified_event_listener_thread_with_cache(
                should_run,
                writer_cache,
                pressed_keys,
            )
        }));

        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }
    }

    fn stop(self) {
        println!("{}", "Stopping recording session...".yellow());

        let session_dir = self.session_dir.clone();
        
        // First set all flags to false to signal threads to stop
        for (flag, _) in &self.capture_threads {
            flag.store(false, Ordering::SeqCst);
        }
        
        // Wait a moment to allow capturer threads to notice the stop flag
        // This is critical to avoid the SendError panic
        thread::sleep(Duration::from_millis(500));
        
        // Now attempt to join threads safely
        for (_, handle) in self.capture_threads {
            let start = Instant::now();
            let timeout = Duration::from_secs(3);
            
            while start.elapsed() < timeout {
                if handle.is_finished() {
                    match handle.join() {
                        Ok(_) => break,
                        Err(e) => {
                            // Safely handle thread panic without propagating it
                            eprintln!("Screen capture thread ended with error: {:?}", e);
                            break;
                        }
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
            
            if start.elapsed() >= timeout {
                eprintln!("Screen capture thread did not exit cleanly within timeout");
            }
        }

        if let Some(event_thread) = self.event_thread {
            let start = Instant::now();
            let timeout = Duration::from_secs(3);

            while start.elapsed() < timeout {
                if event_thread.is_finished() {
                    match event_thread.join() {
                        Ok(_) => break,
                        Err(e) => {
                            eprintln!("Event listener thread ended with error: {:?}", e);
                            break;
                        }
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
            
            if start.elapsed() >= timeout {
                eprintln!("Event listener thread did not exit cleanly within timeout");
            }
        }
        
        println!("Session stopped: {}", session_dir.display());
    }

    fn check_for_errors(&mut self) -> bool {
        let mut full_restart = false;
        while let Ok(_) = self.error_rx.try_recv() {
            full_restart = true;
        }
        full_restart
    }

    fn start_capture_for_display(&mut self, display: DisplayInfo) {
        // TODO: Not sure that it really listens to the signal.
        let sr_for_thread = Arc::new(AtomicBool::new(true));
        let sr_clone = sr_for_thread.clone();
        let session_dir = self.session_dir.clone();
        let error_tx = self.error_tx.clone();

        let handle = capture_display_thread(sr_clone, display, session_dir, error_tx);
        self.capture_threads.push((sr_for_thread, handle));
    }
}

fn get_or_create_user_id() -> Result<String> {
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let loggy_dir = home_dir.join(".loggy3");
    create_dir_all(&loggy_dir)?;
    
    let user_id_path = loggy_dir.join(USER_ID_FILENAME);
    
    // Check if user ID file exists
    if user_id_path.exists() {
        // Read existing user ID
        let mut file = File::open(&user_id_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let user_id = contents.trim().to_string();
        if !user_id.is_empty() {
            return Ok(user_id);
        }
    }
    
    // Generate new user ID if none exists or is empty
    let new_user_id = Uuid::new_v4().to_string();
    
    // Save to file
    let mut file = File::create(&user_id_path)?;
    file.write_all(new_user_id.as_bytes())?;
    
    Ok(new_user_id)
}

pub fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let verbose_mode = args.iter().any(|arg| arg == "--verbose" || arg == "-v");
    
    if verbose_mode {
        VERBOSE.store(true, Ordering::SeqCst);
    }

    println!("{} {}", "\nLoggy3 Screen Recorder".bright_green().bold(), 
              format!("v{}", CURRENT_VERSION).bright_cyan());
    println!("{}", "======================".bright_green());

    // Get or create user ID
    let user_id = match get_or_create_user_id() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("Warning: Failed to get/create user ID: {}", e);
            "unknown".to_string()
        }
    };
    
    // Store user ID in lazy_static for use in other functions
    if let Ok(mut user_id_guard) = USER_ID.lock() {
        *user_id_guard = user_id.clone();
    }
    
    println!("{} {}", "User ID:".bright_yellow(), user_id.bright_cyan());

    if VERBOSE.load(Ordering::SeqCst) {
        println!("{}", "Verbose output enabled".yellow());
    }
    
    println!("{}", "Checking for updates...".cyan());
    
    if let Some((version, download_url, release_url)) = check_for_updates() {
        println!("{} {} {} {}", 
            "A new version".bright_yellow(),
            version.bright_green().bold(),
            "is available!".bright_yellow(),
            format!("(current: {})", CURRENT_VERSION).bright_black()
        );
        
        println!("Release page: {}", release_url.bright_blue().underline());
        
        println!("\nWould you like to update now? [Y/n] ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        match input.trim().to_lowercase().as_str() {
            "y" | "yes" | "" => {
                update_to_new_version(&download_url)?;
            }
            _ => {
                println!("{}", "Update skipped. The application will continue to run.".yellow());
            }
        }
    } else if VERBOSE.load(Ordering::SeqCst) {
        println!("{}", "You're running the latest version!".green());
    }

    println!("\n{}", "Checking system permissions...".bright_black());
    if let Err(e) = platform::check_and_request_permissions() {
        eprintln!("Error checking permissions: {}", e);
    }

    let should_run = Arc::new(AtomicBool::new(true));

    let sr_for_signals = should_run.clone();
    thread::spawn(move || {
        let (tx, rx) = channel();
        
        set_handler(move || tx.send(()).expect("Could not send signal on channel."))
            .expect("Error setting Ctrl-C handler");
        
        println!("\n{}", "Press Ctrl-C to stop recording...".bright_yellow());
        rx.recv().expect("Could not receive from channel.");
        println!("\n{}", "Stopping recording, wait a few seconds...".yellow()); 
        
        sr_for_signals.store(false, Ordering::SeqCst);
    });
    
    let mut last_display_fingerprint = String::new();

    while should_run.load(Ordering::SeqCst) {
        let current_fingerprint = get_display_fingerprint();
        let displays_changed = current_fingerprint != last_display_fingerprint;
        
        if displays_changed && !last_display_fingerprint.is_empty() {
            println!("{}", "Display configuration changed!".bright_yellow().bold());
        }
        
        last_display_fingerprint = current_fingerprint.clone();

        match Session::new(should_run.clone())? {
            Some(mut session) => {
                session.start();

                while should_run.load(Ordering::SeqCst) {
                    let need_restart = session.check_for_errors();
                    if need_restart {
                        println!("{}", "Session signaled a critical error. Restarting session.".red());
                        break;
                    }

                    let current = get_display_fingerprint();
                    if current != current_fingerprint {
                        println!("{}", "Display configuration changed. Stopping current session...".yellow());
                        break;
                    }

                    thread::sleep(Duration::from_secs(1));
                }

                session.stop();
                
                if !should_run.load(Ordering::SeqCst) {
                    break;
                }
                
                thread::sleep(Duration::from_secs(1));
            }
            None => {
                if displays_changed {
                    println!("{}", "All displays disconnected. Waiting for displays to be connected...".yellow());
                }
                thread::sleep(Duration::from_secs(10));
            }
        }
    }

    println!("{}", "Recording stopped. Thank you for using Loggy3!".green().bold());
    Ok(())
}
fn get_display_fingerprint() -> String {
    let displays = platform::get_display_info();
    match serde_json::to_string(&displays) {
        Ok(json) => json,
        Err(_) => {
            let mut display_strings: Vec<String> = displays
                .iter()
                .map(|d| format!("{}:{}x{} at x={},y={}", d.id, d.original_width, d.original_height, d.x, d.y))
                .collect();
            display_strings.sort();
            display_strings.join(",")
        }
    }
}

fn capture_display_thread(
    should_run: Arc<AtomicBool>,
    display_info: DisplayInfo,
    session_dir: PathBuf,
    error_tx: Sender<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        capture_display_impl(should_run, display_info, session_dir, error_tx);
    })
}

fn capture_display_impl(
    should_run: Arc<AtomicBool>,
    display_info: DisplayInfo,
    session_dir: PathBuf,
    error_tx: Sender<()>,
) {
    println!("{} {} ({} x {})", 
        "Starting capture for display".green(),
        display_info.title.cyan(),
        display_info.capture_width.to_string().yellow(),
        display_info.capture_height.to_string().yellow()
    );
    
    let targets = scap::get_all_targets().into_iter().filter(|t| matches!(t, Target::Display(_))).collect::<Vec<_>>();

    let target = platform::get_target_matching_display_info(targets, display_info.clone()).unwrap();

    let capturer = match initialize_capturer(&target) {
        Some(c) => c,
        None => return,
    };

    let (capturer_width, capturer_height) = if platform::IS_WINDOWS {
        (display_info.original_width, display_info.original_height) // No matter what we ask scap, Windows will capture at the original resolution.
    } else {
        match capturer.lock() {
            Ok(mut c) => {
                let sz = c.get_output_frame_size();
                (sz[0], sz[1])
            }
            Err(_) => return,
        }
    };

    let start_time_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut current_chunk_index = (start_time_ms / 60000) as usize;
    
    let current_chunk_dir = session_dir.join(format!("chunk_{:05}", current_chunk_index));
    let display_dir = current_chunk_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
    if let Err(e) = create_dir_all(&display_dir) {
        eprintln!("Failed to create display directory: {}", e);
        return;
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
    
    let start_time = Instant::now();
    let mut total_frame_count = 0;
    let mut chunk_frame_count = 0;
    let mut last_status = Instant::now();
    
    let mut ffmpeg_process = start_new_ffmpeg_process(&display_dir, capturer_width.try_into().unwrap(), capturer_height.try_into().unwrap(), display_info.id);
    if ffmpeg_process.is_none() {
        eprintln!("Failed to start initial ffmpeg process for display {}", display_info.id);
        return;
    }
    
    let status_indicator = format!("[Display {}]", display_info.title);
    println!("{} Started recording", status_indicator.cyan());
    
    let signal_check_interval = Duration::from_millis(100);
    
    while should_run.load(Ordering::SeqCst) {
        let frame_result = {
            match capturer.try_lock() {
                Ok(capturer_guard) => {
                    // Check if we should still be running before requesting a frame
                    if !should_run.load(Ordering::SeqCst) {
                        break;
                    }
                    capturer_guard.get_next_frame().map_err(|e| {
                        if e.to_string().contains("SendError") {
                            anyhow::anyhow!("Display disconnected or capture interrupted")
                        } else {
                            anyhow::anyhow!("Frame error: {}", e)
                        }
                    })
                },
                Err(_) => {
                    thread::sleep(signal_check_interval);
                    // Check if we should exit while waiting
                    if !should_run.load(Ordering::SeqCst) {
                        break;
                    }
                    Err(anyhow::anyhow!("Capturer busy"))
                }
            }
        };

        match frame_result {
            Ok(Frame::YUVFrame(frame)) => {
                total_frame_count += 1;
                chunk_frame_count += 1;
                
                let current_timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                
                let new_chunk_index = (current_timestamp / 60000) as usize;
                
                if new_chunk_index > current_chunk_index {
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Finalizing chunk {} based on epoch time ({} frames)", 
                            current_chunk_index,
                            chunk_frame_count).yellow()
                    );
                    
                    if let Some((mut child, stdin)) = ffmpeg_process.take() {
                        drop(stdin);
                        if let Err(e) = child.wait() {
                            eprintln!("Error waiting for ffmpeg to complete: {}", e);
                        }
                    }
                    
                    current_chunk_index = new_chunk_index;
                    chunk_frame_count = 0;
                    
                    let current_chunk_dir = session_dir.join(format!("chunk_{:05}", current_chunk_index));
                    let display_dir = current_chunk_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
                    if let Err(e) = create_dir_all(&display_dir) {
                        eprintln!("Failed to create display directory for new chunk: {}", e);
                        handle_capture_error(&error_tx);
                        break;
                    }

                    let frames_log_path = display_dir.join("frames.log");
                    match File::create(&frames_log_path) {
                        Ok(file) => {
                            frames_log = BufWriter::new(file);
                        },
                        Err(e) => {
                            eprintln!("Failed to create frames log for new chunk: {}", e);
                            handle_capture_error(&error_tx);
                            break;
                        }
                    }
                    
                    ffmpeg_process = start_new_ffmpeg_process(&display_dir, capturer_width.try_into().unwrap(), capturer_height.try_into().unwrap(), display_info.id);
                    if ffmpeg_process.is_none() {
                        eprintln!("Failed to start new ffmpeg process for display {}", display_info.id);
                        handle_capture_error(&error_tx);
                        break;
                    }
                    
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Started new chunk {}", current_chunk_index).green()
                    );
                }
            
                if last_status.elapsed() >= Duration::from_secs(5) {
                    let fps = total_frame_count as f64 / start_time.elapsed().as_secs_f64();
                    
                    let seconds_in_current_chunk = (current_timestamp % 60000) / 1000;
                    let seconds_remaining = 60 - seconds_in_current_chunk as u64;

                    println!("{} Recording at {} fps (chunk {}, frames: {}, seconds remaining: {})", 
                        status_indicator.cyan(),
                        format!("{:.1}", fps).bright_green(),
                        current_chunk_index.to_string().yellow(),
                        chunk_frame_count.to_string().yellow(),
                        seconds_remaining.to_string().bright_yellow()
                    );

                    last_status = Instant::now();
                }
                
                if let Some((_, ref mut stdin)) = ffmpeg_process {
                    if let Err(e) = write_yuv_frame(stdin, &frame, &mut frames_log) {
                        eprintln!("Write error for display {}: {}", display_info.id, e);
                        break;
                    }
                } else {
                    eprintln!("No active ffmpeg process to write frame for display {}", display_info.id);
                    break;
                }
            }
            Ok(Frame::BGRA(frame)) => {
                total_frame_count += 1;
                chunk_frame_count += 1;
                
                let current_timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                
                let new_chunk_index = (current_timestamp / 60000) as usize;
                
                if new_chunk_index > current_chunk_index {
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Finalizing chunk {} based on epoch time ({} frames)", 
                            current_chunk_index,
                            chunk_frame_count).yellow()
                    );
                    
                    if let Some((mut child, stdin)) = ffmpeg_process.take() {
                        drop(stdin);
                        if let Err(e) = child.wait() {
                            eprintln!("Error waiting for ffmpeg to complete: {}", e);
                        }
                    }
                    
                    current_chunk_index = new_chunk_index;
                    chunk_frame_count = 0;
                    
                    let current_chunk_dir = session_dir.join(format!("chunk_{:05}", current_chunk_index));
                    let display_dir = current_chunk_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
                    if let Err(e) = create_dir_all(&display_dir) {
                        eprintln!("Failed to create display directory for new chunk: {}", e);
                        handle_capture_error(&error_tx);
                        break;
                    }

                    let frames_log_path = display_dir.join("frames.log");
                    match File::create(&frames_log_path) {
                        Ok(file) => {
                            frames_log = BufWriter::new(file);
                        },
                        Err(e) => {
                            eprintln!("Failed to create frames log for new chunk: {}", e);
                            handle_capture_error(&error_tx);
                            break;
                        }
                    }
                    
                    ffmpeg_process = start_new_ffmpeg_process(&display_dir, capturer_width.try_into().unwrap(), capturer_height.try_into().unwrap(), display_info.id);
                    if ffmpeg_process.is_none() {
                        eprintln!("Failed to start new ffmpeg process for display {}", display_info.id);
                        handle_capture_error(&error_tx);
                        break;
                    }
                    
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Started new chunk {}", current_chunk_index).green()
                    );
                }
            
                if last_status.elapsed() >= Duration::from_secs(5) {
                    let fps = total_frame_count as f64 / start_time.elapsed().as_secs_f64();
                    
                    let seconds_in_current_chunk = (current_timestamp % 60000) / 1000;
                    let seconds_remaining = 60 - seconds_in_current_chunk as u64;

                    println!("{} Recording at {} fps (chunk {}, frames: {}, seconds remaining: {})", 
                        status_indicator.cyan(),
                        format!("{:.1}", fps).bright_green(),
                        current_chunk_index.to_string().yellow(),
                        chunk_frame_count.to_string().yellow(),
                        seconds_remaining.to_string().bright_yellow()
                    );

                    last_status = Instant::now();
                }
                
                if let Some((_, ref mut stdin)) = ffmpeg_process {
                    if let Err(e) = write_bgra_frame(stdin, &frame, &mut frames_log) {
                        eprintln!("Write error for display {}: {}", display_info.id, e);
                        break;
                    }
                } else {
                    eprintln!("No active ffmpeg process to write frame for display {}", display_info.id);
                    break;
                }
            }
            Ok(_) => {
                eprintln!("Unknown frame type. Not writing frame.");
                continue;
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Capturer busy") {
                    continue;
                } else {
                    eprintln!("Frame error on display {}: {}", display_info.id, e);
                    
                    if !should_run.load(Ordering::SeqCst) {
                        break;
                    }
                    
                    thread::sleep(signal_check_interval);
                    
                    if total_frame_count == 0 && start_time.elapsed() > Duration::from_secs(10) {
                        eprintln!("No frames captured after 10 seconds. Signaling error.");
                        handle_capture_error(&error_tx);
                        break;
                    }
                    continue;
                }
            }
        }
    }

    // Safe shutdown of ffmpeg
    if let Some((mut child, stdin)) = ffmpeg_process.take() {
        // First explicitly drop stdin to close the pipe
        drop(stdin);
        
        // Give ffmpeg time to process remaining frames and shut down
        let start = Instant::now();
        let timeout = Duration::from_secs(2);
        
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process already exited cleanly
            },
            Ok(None) => {
                // Process still running, wait for it with timeout
                let mut exited = false;
                
                while start.elapsed() < timeout && !exited {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            exited = true;
                        },
                        Ok(None) => {
                            thread::sleep(Duration::from_millis(100));
                        },
                        Err(e) => {
                            eprintln!("Error checking ffmpeg status: {}", e);
                            break;
                        },
                    }
                }
                
                if !exited {
                    // Force kill if timeout exceeded
                    eprintln!("Timeout waiting for ffmpeg to finish - killing process");
                    if let Err(e) = child.kill() {
                        eprintln!("Failed to kill ffmpeg process: {}", e);
                    }
                }
            },
            Err(e) => {
                eprintln!("Error waiting for ffmpeg: {}", e);
            },
        }
    }
    
    // Clean up the capturer to prevent SendError panics
    stop_capturer_safely(&capturer);
    thread::sleep(Duration::from_millis(100));
    
    println!("{} Recording stopped", status_indicator.cyan());
}

fn start_new_ffmpeg_process(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
    display_id: u32,
) -> Option<(Child, ChildStdin)> {
    let output_path = display_dir.join("video.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let ffmpeg_path = match download_ffmpeg() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("Failed to download ffmpeg: {}", e);
            return None;
        }
    };

    let log_level = if VERBOSE.load(Ordering::SeqCst) {
        "info"
    } else {
        "error"
    };

    let mut child = match Command::new(ffmpeg_path)
        .args(&[
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", platform::FFMPEG_PIXEL_FORMAT,
            "-color_range", "tv",
            "-s", &format!("{}x{}", width, height),
            "-r", "30",
            "-i", "pipe:0",
            "-vf", "scale=1280:-1",
            "-c:v", platform::FFMPEG_ENCODER,
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof+faststart",
            "-frag_duration", "1000000",
            "-g", "60",
            "-loglevel", log_level,
            &output_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to start ffmpeg: {}", e);
                return None;
            }
        };

    let stdin = match child.stdin.take() {
        Some(s) => s,
        None => {
            eprintln!("Failed to get ffmpeg stdin");
            return None;
        }
    };

    if let Some(stdout) = child.stdout.take() {
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if VERBOSE.load(Ordering::SeqCst) {
                        println!("FFmpeg stdout (display {}): {}", display_id, line);
                    }
                }
            }
        });
    }

    if let Some(stderr) = child.stderr.take() {
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if VERBOSE.load(Ordering::SeqCst) || line.contains("error") {
                        eprintln!("FFmpeg (display {}): {}", display_id, line);
                    }
                }
            }
        });
    }

    Some((child, stdin))
}

fn handle_capture_error(error_tx: &Sender<()>) {
    let _ = error_tx.send(());
}

fn initialize_capturer(target: &Target) -> Option<Arc<Mutex<Capturer>>> {
    let opts = Options {
        fps: 30,
        output_type: if platform::IS_WINDOWS { FrameType::BGRAFrame } else { FrameType::YUVFrame },
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

fn stop_capturer_safely(capturer: &Arc<Mutex<Capturer>>) {
    match capturer.try_lock() {
        Ok(mut guard) => {
            if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                guard.stop_capture();
            })) {
                eprintln!("Error during capturer cleanup: {:?}", e);
            }
        },
        Err(_) => {
            eprintln!("Could not obtain lock to safely stop capturer");
        }
    }
}

fn download_ffmpeg() -> Result<PathBuf> {
    if let Ok(ffmpeg_path_guard) = FFMPEG_PATH.lock() {
        if let Some(path) = ffmpeg_path_guard.clone() {
            if path.exists() {
                return Ok(path);
            }
        }
    }

    let _download_lock = match FFMPEG_DOWNLOAD_MUTEX.lock() {
        Ok(lock) => lock,
        Err(e) => return Err(anyhow::anyhow!("Failed to acquire download mutex: {}", e)),
    };

    if let Ok(ffmpeg_path_guard) = FFMPEG_PATH.lock() {
        if let Some(path) = ffmpeg_path_guard.clone() {
            if path.exists() {
                return Ok(path);
            }
        }
    }

    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let loggy_dir = home_dir.join(".loggy3");
    create_dir_all(&loggy_dir)?;
    
    let ffmpeg_path = loggy_dir.join("ffmpeg");
    
    if !ffmpeg_path.exists() {
        println!("Downloading ffmpeg binary...");
        
        let temp_path = loggy_dir.join("ffmpeg.downloading");
        
        println!("Downloading from {}", platform::FFMPEG_DOWNLOAD_URL);
        
        let response = ureq::get(platform::FFMPEG_DOWNLOAD_URL)
            .call()
            .context("Failed to download ffmpeg binary")?;
        
        let mut file = File::create(&temp_path).context("Failed to create temporary file")?;
        let mut buffer = Vec::new();
        response.into_reader().read_to_end(&mut buffer).context("Failed to read response")?;
        file.write_all(&buffer).context("Failed to write to temporary file")?;
            
        std::fs::rename(&temp_path, &ffmpeg_path)?;
        println!("Download complete");
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&ffmpeg_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&ffmpeg_path, perms)?;
        }
    }
    
    // Update the global cache with the path
    if let Ok(mut ffmpeg_path_guard) = FFMPEG_PATH.lock() {
        *ffmpeg_path_guard = Some(ffmpeg_path.clone());
    }
    
    Ok(ffmpeg_path)
}

fn check_for_updates() -> Option<(String, String, String)> {
    let api_url = format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO);
    
    // Determine which binary we need for this platform
    let binary_suffix = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "macos-arm64"
        } else {
            // We don't support Intel Macs according to loggy3.sh
            return None;
        }
    } else if cfg!(target_os = "windows") {
        "windows.exe"
    } else {
        // Unsupported platform
        return None;
    };
    
    let target_asset_name = format!("loggy3-{}", binary_suffix);
    
    if VERBOSE.load(Ordering::SeqCst) {
        println!("Looking for updates for asset: {}", target_asset_name);
    }
    
    match ureq::get(&api_url).call() {
        Ok(response) => {
            if let Ok(release) = response.into_json::<GitHubRelease>() {
                let latest_version = release.tag_name.trim_start_matches('v').to_string();
                
                if is_newer_version(&latest_version, CURRENT_VERSION) {
                    // Find the correct asset for the current platform
                    if let Some(asset) = release.assets.iter().find(|a| a.name == target_asset_name) {
                        return Some((latest_version, asset.browser_download_url.clone(), release.html_url));
                    } else {
                        if VERBOSE.load(Ordering::SeqCst) {
                            println!("No matching asset found for the current platform. Available assets:");
                            for asset in &release.assets {
                                println!("- {}", asset.name);
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            if VERBOSE.load(Ordering::SeqCst) {
                eprintln!("Failed to check for updates: {}", e);
            }
        }
    }
    
    None
}

fn is_newer_version(new_version: &str, current_version: &str) -> bool {
    let parse_version = |v: &str| -> Vec<u32> {
        v.split('.')
         .map(|s| s.parse::<u32>().unwrap_or(0))
         .collect()
    };
    
    let new_parts = parse_version(new_version);
    let current_parts = parse_version(current_version);
    
    for i in 0..3 {
        let new_part = new_parts.get(i).copied().unwrap_or(0);
        let current_part = current_parts.get(i).copied().unwrap_or(0);
        
        if new_part > current_part {
            return true;
        } else if new_part < current_part {
            return false;
        }
    }
    
    false
}

fn update_to_new_version(download_url: &str) -> Result<()> {
    println!("{}", "Downloading the latest version...".cyan());
    
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let install_dir = home_dir.join(".local/bin");
    create_dir_all(&install_dir)?;
    
    // Determine the correct binary name based on platform
    let target_binary_name = if cfg!(target_os = "windows") {
        "loggy3.exe"
    } else {
        "loggy3"
    };
    
    let target_path = install_dir.join(target_binary_name);
    let temp_path = target_path.with_extension("new");
    
    println!("Downloading from {}", download_url);
    let response = ureq::get(download_url)
        .call()
        .context("Failed to download update")?;
    
    let mut file = File::create(&temp_path).context("Failed to create temporary file")?;
    let mut buffer = Vec::new();
    response.into_reader().read_to_end(&mut buffer).context("Failed to read response")?;
    file.write_all(&buffer).context("Failed to write to temporary file")?;
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&temp_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&temp_path, perms)?;
    }
    
    println!("{}", "Installing update...".cyan());
    
    std::fs::rename(&temp_path, &target_path)?;
    println!("{}", "✓ Update installed successfully!".green());
    println!("{}", "Please restart loggy3 to use the new version.".bright_green().bold());
    exit(0);
}


fn write_frame_timestamp(frames_log: &mut BufWriter<File>) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis();
    writeln!(frames_log, "{}", timestamp)?;
    frames_log.flush()?;

    Ok(())
}

fn write_yuv_frame(
    ffmpeg_stdin: &mut ChildStdin,
    frame: &YUVFrame,
    frames_log: &mut BufWriter<File>,
) -> Result<()> {
    ffmpeg_stdin.write_all(&frame.luminance_bytes)?;
    ffmpeg_stdin.write_all(&frame.chrominance_bytes)?;
    write_frame_timestamp(frames_log)?;
    Ok(())
}

fn write_bgra_frame(
    ffmpeg_stdin: &mut ChildStdin,
    frame: &BGRAFrame,
    frames_log: &mut BufWriter<File>,
) -> Result<()> {
    ffmpeg_stdin.write_all(&frame.data)?;
    write_frame_timestamp(frames_log)?;
    Ok(())
}

fn log_session_metadata(session_dir: &PathBuf) -> Result<()> {
    let mut system = System::new_all();
    system.refresh_all();
    
    // Get the user ID from lazy_static
    let user_id = USER_ID.lock()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| "unknown".to_string());
    
    let metadata = serde_json::json!({
        "app_version": CURRENT_VERSION,
        "timestamp": Local::now().to_rfc3339(),
        "user_id": user_id,
        "system_info": {
            "os_name": System::name().unwrap_or_else(|| "Unknown".to_string()),
            "os_version": System::os_version().unwrap_or_else(|| "Unknown".to_string()),
            "kernel_version": System::kernel_version().unwrap_or_else(|| "Unknown".to_string()),
            "hostname": System::host_name().unwrap_or_else(|| "Unknown".to_string()),
            "cpu": {
                "num_physical_cores": system.physical_core_count().unwrap_or(0),
                "num_total_cores": system.cpus().len(),
                "model": system.global_cpu_info().name().to_string(),
            },
            "memory": {
                "total_memory_kb": system.total_memory(),
                "available_memory_kb": system.available_memory(),
                "used_memory_kb": system.used_memory(),
            },
        }
    });
    
    let metadata_path = session_dir.join("session_metadata.json");
    let file = File::create(&metadata_path)?;
    serde_json::to_writer_pretty(file, &metadata)?;
    
    println!("{}", "✓ Session metadata logged successfully".green());
    Ok(())
}