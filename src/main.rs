mod platform;

use anyhow::{Context, Result};
use chrono::Local;
use colored::*;
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
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use sysinfo::System;
use ureq;
use lazy_static::lazy_static;
use uuid::Uuid;
use indicatif::{ProgressBar, ProgressStyle};
use fs2;


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

        let documents_dir = dirs::document_dir().context("Could not determine documents directory")?;
        
        let random_id = uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("").to_string();
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let session_dir = documents_dir.join("loggy3").join(format!("session_{}_{}", timestamp, random_id));
        
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

        let (error_tx, error_rx) = mpsc::channel();

        Ok(Some(Self {
            should_run,
            session_dir,
            event_thread: None,
            capture_threads: Vec::new(),
            writer_cache,
            error_rx,
            error_tx,
            displays,
        }))
    }
    
    fn start(&mut self) {
        let should_run = self.should_run.clone();
        let writer_cache = self.writer_cache.clone();
        self.event_thread = Some(thread::spawn(move || {
            platform::unified_event_listener_thread_with_cache(
                should_run,
                writer_cache,
            )
        }));

        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }
    }

    fn stop(self) {
        println!("{}", "Stopping recording session...".yellow());
        self.should_run.store(false, Ordering::SeqCst);

        let session_dir = self.session_dir.clone();
        
        for (flag, _) in &self.capture_threads {
            flag.store(false, Ordering::SeqCst);
        }
        
        thread::sleep(Duration::from_millis(500));
        
        for (_, handle) in self.capture_threads {
            let start = Instant::now();
            let timeout = Duration::from_secs(3);
            
            while start.elapsed() < timeout {
                if handle.is_finished() {
                    match handle.join() {
                        Ok(_) => break,
                        Err(e) => {
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
    
    if user_id_path.exists() {
        let mut file = File::open(&user_id_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        
        let user_id = contents.trim().to_string();
        if !user_id.is_empty() {
            return Ok(user_id);
        }
    }
    
    let new_user_id = Uuid::new_v4().to_string();
    
    let mut file = File::create(&user_id_path)?;
    file.write_all(new_user_id.as_bytes())?;
    
    Ok(new_user_id)
}

fn check_for_running_instance() -> Result<std::fs::File, String> {
    let home_dir = match dirs::home_dir() {
        Some(dir) => dir,
        None => return Err("Could not determine home directory".to_string()),
    };
    
    let lock_dir = home_dir.join(".loggy3");
    if let Err(e) = std::fs::create_dir_all(&lock_dir) {
        return Err(format!("Could not create lock directory: {}", e));
    }
    
    let lock_path = lock_dir.join("loggy3.lock");
    
    match std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(&lock_path)
    {
        Ok(file) => {
            let mut file = file;
            match fs2::FileExt::try_lock_exclusive(&file) {
                Ok(_) => {
                    if let Err(e) = std::io::Write::write_all(&mut file, std::process::id().to_string().as_bytes()) {
                        return Err(format!("Could not write to lock file: {}", e));
                    }
                    
                    Ok(file)
                },
                Err(_) => {
                    return Err("Another instance of Loggy3 is already running.".to_string());
                }
            }
        }
        Err(e) => Err(format!("Could not open lock file: {}", e)),
    }
}

pub fn main() -> Result<()> {
    loop {
        let _lock_file = match check_for_running_instance() {
            Ok(file) => file,
            Err(e) => {
                eprintln!("{}", format!("Error: {}", e).bright_red().bold());
                eprintln!("{}", "Please close any other running instances of Loggy3 before starting a new one.".yellow());
                std::process::exit(1);
            }
        };
        
        let args: Vec<String> = std::env::args().collect();
        let verbose_mode = args.iter().any(|arg| arg == "--verbose" || arg == "-v");
        let force_update = args.iter().any(|arg| arg == "--force-update" || arg == "-u");
        
        if verbose_mode {
            VERBOSE.store(true, Ordering::SeqCst);
        }
        #[cfg(target_os = "windows")] {
            if let Err(e) = platform::check_windows_version_compatibility() {
                eprintln!("Error: {}", e);
                return Err(anyhow::anyhow!("Incompatible Windows version: {}", e));
            }
            
            if let Ok(version_type) = platform::get_windows_version_type() {
                match version_type {
                    platform::WindowsVersionType::Windows10 => {
                        println!("Running on Windows 10");
                        colored::control::set_override(false);
                    },
                    platform::WindowsVersionType::Windows11 => {
                        println!("Running on Windows 11");
                    },
                    _ => {}
                }
            } else {
                eprintln!("Error: Could not determine Windows version");
            }
        }
        
        println!("{} {}", "\nLoggy3 Screen Recorder".bright_green().bold(), 
                format!("v{}", CURRENT_VERSION).bright_cyan());
        println!("{}", "======================".bright_green());
        println!("{}", "Usage:".bright_yellow());
        println!("{}", "  --verbose, -v       Enable verbose output".bright_black());
        println!("{}", "  --force-update, -u  Force update to latest version".bright_black());
        println!("");

        let user_id = match get_or_create_user_id() {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Warning: Failed to get/create user ID: {}", e);
                "unknown".to_string()
            }
        };
        
        if let Ok(mut user_id_guard) = USER_ID.lock() {
            *user_id_guard = user_id.clone();
        }
        
        println!("{} {}", "User ID:".bright_yellow(), user_id.bright_cyan());
        
        if VERBOSE.load(Ordering::SeqCst) {
            println!("{}", "Verbose output enabled".yellow());
        }
        
        println!("{}", "Checking for updates...".cyan());
        
        if let Some((version, download_url, release_url)) = check_for_updates(force_update) {
            if force_update && version == CURRENT_VERSION {
                println!("{} {} {} {}", 
                    "Forcing update to version".bright_yellow(),
                    version.bright_green().bold(),
                    "(same as current version)".bright_yellow(),
                    ""
                );
            } else {
                println!("{} {} {} {}", 
                    "A new version".bright_yellow(),
                    version.bright_green().bold(),
                    "is available!".bright_yellow(),
                    format!("(current: {})", CURRENT_VERSION).bright_black()
                );
            }
            
            println!("Release page: {}", release_url.bright_blue().underline());
            
            if force_update {
                println!("\nForce update flag is set. Updating automatically...");
                update_to_new_version(&download_url)?;
            } else {
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
            }
        } else if force_update {
            println!("{}", "Force update requested but no update is available.".yellow());
        } else if VERBOSE.load(Ordering::SeqCst) {
            println!("{}", "You're running the latest version!".green());
        }

        println!("\n{}", "Checking system permissions...".bright_black());
        if let Err(e) = platform::check_and_request_permissions() {
            eprintln!("Error checking permissions: {}", e);
            println!("{}", "Loggy3 cannot run without the required permissions. Please restart after granting permissions.".bright_red().bold());
            return Err(anyhow::anyhow!("Missing required permissions: {}", e));
        }

        if let Err(e) = platform::set_path_or_start_menu_shortcut() {
            eprintln!("{}", e);
        }

        let should_run = Arc::new(AtomicBool::new(true));
        
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
    }
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

    let (capturer_width, capturer_height) = if cfg!(target_os = "macos") {
        match capturer.lock() {
            Ok(mut c) => {
                let sz = c.get_output_frame_size();
                (sz[0], sz[1])
            }
            Err(_) => return,
        }
    } else {
        (display_info.original_width, display_info.original_height)
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

    if let Some((mut child, stdin)) = ffmpeg_process.take() {
        drop(stdin);
        
        let start = Instant::now();
        let timeout = Duration::from_secs(2);
        
        match child.try_wait() {
            Ok(Some(_)) => {
            },
            Ok(None) => {
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
        output_type: if cfg!(target_os = "macos") { FrameType::YUVFrame } else { FrameType::BGRAFrame },
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
    
    let ffmpeg_path = loggy_dir.join(platform::FFMPEG_FILENAME);
    
    // Check if ffmpeg exists but might not have executable permissions
    if ffmpeg_path.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            
            let is_executable = match std::fs::metadata(&ffmpeg_path) {
                Ok(meta) => {
                    let perms = meta.permissions();
                    let mode = perms.mode();
                    (mode & 0o111) != 0
                },
                Err(_) => false,
            };
            
            if !is_executable {
                println!("{}", "Found existing FFmpeg binary without executable permissions. Fixing...".yellow());
                
                if let Err(e) = std::process::Command::new("chmod")
                    .arg("+x")
                    .arg(&ffmpeg_path)
                    .status() {
                        eprintln!("Warning: Failed to set executable permissions with chmod: {}", e);
                }
            }
        }
    }
    
    if !ffmpeg_path.exists() {
        println!("{}", "Downloading ffmpeg binary (required)...".cyan());
        
        let temp_path = loggy_dir.join("ffmpeg.downloading");
        
        if VERBOSE.load(Ordering::SeqCst) {
            println!("Downloading from {}", platform::FFMPEG_DOWNLOAD_URL);
        }
        
        let response = match ureq::get(platform::FFMPEG_DOWNLOAD_URL).call() {
            Ok(resp) => resp,
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to download ffmpeg binary: {}. Cannot continue without ffmpeg.", e));
            }
        };
        
        let content_length = response
            .header("Content-Length")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        
        let pb = if content_length > 0 {
            let pb = ProgressBar::new(content_length);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"));
            pb
        } else {
            let pb = ProgressBar::new_spinner();
            pb.set_style(ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {bytes} downloaded")
                .unwrap());
            pb
        };
        
        let mut file = match File::create(&temp_path) {
            Ok(file) => file,
            Err(e) => {
                pb.finish_with_message("Download failed");
                return Err(anyhow::anyhow!("Failed to create temporary file: {}. Cannot continue without ffmpeg.", e));
            }
        };
        
        let mut reader = response.into_reader();
        let mut buffer = [0; 8192];
        let mut downloaded = 0;
        
        while let Ok(n) = reader.read(&mut buffer) {
            if n == 0 { break }
            
            if let Err(e) = file.write_all(&buffer[..n]) {
                pb.finish_with_message("Download failed");
                return Err(anyhow::anyhow!("Failed to write to temporary file: {}. Cannot continue without ffmpeg.", e));
            }
            
            downloaded += n as u64;
            pb.set_position(downloaded);
        }
        
        if let Err(e) = file.flush() {
            pb.finish_with_message("Download failed");
            return Err(anyhow::anyhow!("Failed to flush file: {}. Cannot continue without ffmpeg.", e));
        }
        
        drop(file);
        
        pb.finish_with_message("Download complete");
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            
            if let Err(e) = std::process::Command::new("chmod")
                .arg("+x")
                .arg(&temp_path)
                .status() {
                    eprintln!("Warning: Failed to set executable permissions with chmod: {}", e);
                    
                    let mut perms = match std::fs::metadata(&temp_path) {
                        Ok(meta) => meta.permissions(),
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to get file permissions: {}. Cannot continue without ffmpeg.", e));
                        }
                    };
                    perms.set_mode(0o755);
                    if let Err(e) = std::fs::set_permissions(&temp_path, perms) {
                        return Err(anyhow::anyhow!("Failed to set file permissions: {}. Cannot continue without ffmpeg.", e));
                    }
            }
            
            println!("{}", "✓ Set executable permissions on ffmpeg".green());
        }
        
        if let Err(e) = std::fs::rename(&temp_path, &ffmpeg_path) {
            return Err(anyhow::anyhow!("Failed to rename temporary file: {}. Cannot continue without ffmpeg.", e));
        }
        
        #[cfg(unix)]
        {
            if let Err(e) = std::process::Command::new("chmod")
                .arg("+x")
                .arg(&ffmpeg_path)
                .status() {
                    eprintln!("Warning: Failed to set executable permissions after rename: {}", e);
            }
        }
    }
    
    if !ffmpeg_path.exists() {
        return Err(anyhow::anyhow!("FFmpeg binary does not exist after download. Cannot continue."));
    }
    
    if let Ok(mut ffmpeg_path_guard) = FFMPEG_PATH.lock() {
        *ffmpeg_path_guard = Some(ffmpeg_path.clone());
    }
    
    Ok(ffmpeg_path)
}

fn check_for_updates(force_update: bool) -> Option<(String, String, String)> {
    let api_url = format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO);
    
    let binary_suffix = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "macos-arm64"
        } else {
            return None;
        }
    } else if cfg!(target_os = "windows") {
        "windows.exe"
    } else {
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
                
                if force_update || is_newer_version(&latest_version, CURRENT_VERSION) {
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
                } else if VERBOSE.load(Ordering::SeqCst) {
                    println!("{}", "You're running the latest version!".green());
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
    
    let target_path = std::env::current_exe().context("Failed to get current executable path")?;
    let temp_path = target_path.with_extension("new");
    
    if VERBOSE.load(Ordering::SeqCst) {
        println!("Downloading from {}", download_url);
    }
    
    let response = match ureq::get(download_url).call() {
        Ok(resp) => resp,
        Err(e) => {
            return Err(anyhow::anyhow!("Failed to download update: {}", e));
        }
    };
    
    let content_length = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    
    let pb = if content_length > 0 {
        let pb = ProgressBar::new(content_length);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {bytes} downloaded")
            .unwrap());
        pb
    };
    
    let mut file = match File::create(&temp_path) {
        Ok(file) => file,
        Err(e) => {
            pb.finish_with_message("Download failed");
            return Err(anyhow::anyhow!("Failed to create temporary file: {}", e));
        }
    };
    
    let mut reader = response.into_reader();
    let mut buffer = [0; 8192];
    let mut downloaded = 0;
    
    while let Ok(n) = reader.read(&mut buffer) {
        if n == 0 { break }
        
        if let Err(e) = file.write_all(&buffer[..n]) {
            pb.finish_with_message("Download failed");
            return Err(anyhow::anyhow!("Failed to write to temporary file: {}", e));
        }
        
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    
    if let Err(e) = file.flush() {
        pb.finish_with_message("Download failed");
        return Err(anyhow::anyhow!("Failed to flush file: {}", e));
    }
    
    drop(file);
    
    pb.finish_with_message("Download complete");
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        
        // Try chmod first (more reliable on macOS)
        if let Err(e) = std::process::Command::new("chmod")
            .arg("+x")
            .arg(&temp_path)
            .status() {
                eprintln!("Warning: Failed to set executable permissions with chmod: {}", e);
                
                // Fallback to Rust's permission system
                let mut perms = match std::fs::metadata(&temp_path) {
                    Ok(meta) => meta.permissions(),
                    Err(e) => {
                        return Err(anyhow::anyhow!("Failed to get file permissions: {}", e));
                    }
                };
                perms.set_mode(0o755); // rwxr-xr-x
                if let Err(e) = std::fs::set_permissions(&temp_path, perms) {
                    return Err(anyhow::anyhow!("Failed to set file permissions: {}", e));
                }
        }
        
        println!("{}", "✓ Set executable permissions on update".green());
    }
    
    println!("{}", format!("Installing update to {}...", target_path.display()).cyan());
    
    #[cfg(windows)]
    {
        // On Windows, we can't replace a running executable
        // Create a batch script to replace the file after the application exits
        let batch_path = target_path.with_extension("bat");
        let pid = std::process::id();
        
        // Create batch script content
        // The script will:
        // 1. Wait for the current process to exit
        // 2. Try to copy the new executable over the old one
        // 3. Delete the temporary files
        // 4. Start the new executable
        let batch_content = format!(
            "@echo off\n\
             :wait\n\
             timeout /t 1 /nobreak > nul\n\
             tasklist | find \"{} {}\" > nul\n\
             if not errorlevel 1 goto wait\n\
             echo Updating Loggy3...\n\
             :copy\n\
             copy /y \"{}\" \"{}\" > nul\n\
             if errorlevel 1 (\n\
             timeout /t 1 /nobreak > nul\n\
             goto copy\n\
             )\n\
             echo Update complete!\n\
             del \"{}\" > nul\n\
             del \"%~f0\" > nul\n\
             start \"\" \"{}\"",
            "loggy3.exe", pid,
            temp_path.to_string_lossy(), target_path.to_string_lossy(),
            temp_path.to_string_lossy(), target_path.to_string_lossy()
        );
        
        // Write the batch script
        match std::fs::write(&batch_path, batch_content) {
            Ok(_) => {
                // Execute the batch script
                match std::process::Command::new("cmd")
                    .args(&["/c", "start", "/min", "", batch_path.to_string_lossy().as_ref()])
                    .spawn() {
                    Ok(_) => {
                        println!("\n{}", "Press Enter to exit and apply the update...".bright_cyan());
                        let mut input = String::new();
                        if let Err(e) = std::io::stdin().read_line(&mut input) {
                            eprintln!("Error reading input: {}", e);
                        }
                        println!("{}", "Exiting to apply update...".bright_green());
                        exit(0);
                    },
                    Err(e) => {
                        return Err(anyhow::anyhow!("Failed to start update script: {}", e));
                    }
                }
            },
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to create update script: {}", e));
            }
        }
    }
    
    #[cfg(not(windows))]
    {
        if let Err(e) = std::fs::rename(&temp_path, &target_path) {
            return Err(anyhow::anyhow!("Failed to install update: {}", e));
        }
        
        // Double-check permissions after rename on Unix systems
        #[cfg(unix)]
        {
            if let Err(e) = std::process::Command::new("chmod")
                .arg("+x")
                .arg(&target_path)
                .status() {
                    eprintln!("Warning: Failed to set executable permissions after rename: {}", e);
                    
                    // Fallback to Rust's permission system
                    use std::os::unix::fs::PermissionsExt;
                    match std::fs::metadata(&target_path) {
                        Ok(meta) => {
                            let mut perms = meta.permissions();
                            perms.set_mode(0o755); // rwxr-xr-x
                            if let Err(e) = std::fs::set_permissions(&target_path, perms) {
                                eprintln!("Warning: Failed to set file permissions after rename: {}", e);
                            }
                        },
                        Err(e) => {
                            eprintln!("Warning: Failed to get file permissions after rename: {}", e);
                        }
                    };
            }
        }
        
        println!("{}", "✓ Update installed successfully!".green());
        println!("{}", "Please restart loggy3 to use the new version.".bright_green().bold());
        exit(0);
    }
}


fn write_frame_timestamp(frames_log: &mut BufWriter<File>) -> Result<()> {
    let multi_timestamp = platform::get_multi_timestamp();
    
    writeln!(frames_log, "{}, {}, {}", 
        multi_timestamp.seq, 
        multi_timestamp.wall_time, 
        multi_timestamp.monotonic_time)?;
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
