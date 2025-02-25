use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File},
    io::{BufWriter, Write, BufReader, BufRead, Read},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio, exit},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
    collections::HashMap,
};
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType, YUVFrame},
    Target,
};

use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use rdev::{listen, Event, EventType};
use ureq;

// Permission checking code for macOS
// IOKit bindings for Input Monitoring permissions
#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOHIDCheckAccess(request: u32) -> u32;
    fn IOHIDRequestAccess(request: u32) -> bool;
}

// CoreGraphics bindings for Screen Recording permissions
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
}

// Input Monitoring constants
// According to IOKit/IOHIDLib.h, for 10.15+:
//  kIOHIDRequestTypePostEvent   = 0, // Accessibility
//  kIOHIDRequestTypeListenEvent = 1, // Input Monitoring
const KIOHID_REQUEST_TYPE_LISTEN_EVENT: u32 = 1;

// Functions for checking and requesting permissions

/// Checks the current input monitoring status:
///  - Some(true) = Granted
///  - Some(false) = Denied
///  - None = Not determined yet
fn check_input_monitoring_access() -> Option<bool> {
    unsafe {
        let status = IOHIDCheckAccess(KIOHID_REQUEST_TYPE_LISTEN_EVENT);
        match status {
            0 => Some(true),  // Granted
            1 => Some(false), // Denied
            2 => None,        // Not determined yet
            _ => None,        // Any unexpected value -> treat like "unknown"
        }
    }
}

/// Requests input monitoring access (prompts user if not determined).
/// Returns `true` if permission is (now) granted, `false` otherwise.
fn request_input_monitoring_access() -> bool {
    unsafe { IOHIDRequestAccess(KIOHID_REQUEST_TYPE_LISTEN_EVENT) }
}

/// Checks if screen recording permission is granted
/// Returns true if granted, false if denied or not determined
fn check_screen_recording_access() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

/// Requests screen recording access (prompts user if not determined)
/// Returns true if permission is granted, false otherwise
fn request_screen_recording_access() -> bool {
    unsafe { CGRequestScreenCaptureAccess() }
}

use serde::{Deserialize, Serialize};
use colored::*;

pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";
pub static VERBOSE: AtomicBool = AtomicBool::new(false);
pub static AUTO_UPDATES_DISABLED: AtomicBool = AtomicBool::new(false);

const GITHUB_REPO: &str = "Standard-Intelligence/loggy3";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

// GitHub API response structures
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

// Writer cache to manage log files across chunk boundaries
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

    // Get or create a writer for keypresses based on timestamp
    fn get_keypress_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.keypress_writers.contains_key(&chunk_index) {
            // Create the chunk directory if needed
            let chunk_dir = self.session_dir.join(format!("chunk_{:05}", chunk_index));
            create_dir_all(&chunk_dir)?;
            
            // Create the log file
            let log_path = chunk_dir.join("keypresses.log");
            let writer = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
            println!("{}", format!("Created new keypress log for chunk {}", chunk_index).yellow());
            self.keypress_writers.insert(chunk_index, writer);
        }
        Ok(self.keypress_writers.get(&chunk_index).unwrap().clone())
    }
    
    // Get or create a writer for mouse events based on timestamp
    fn get_mouse_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.mouse_writers.contains_key(&chunk_index) {
            // Create the chunk directory if needed
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

// New function to log mouse events with automatic chunk management
fn log_mouse_event_with_cache(timestamp: u128, cache: &Arc<Mutex<LogWriterCache>>, data: &str) {
    let line = format!("({}, {})\n", timestamp, data);
    
    // Get the appropriate writer for this timestamp
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_mouse_writer(timestamp) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = writer_lock.write_all(line.as_bytes());
                let _ = writer_lock.flush();
            }
        }
    }
}

// New function to handle key events with automatic chunk management
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
    
    // Get the appropriate writer for this timestamp
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_keypress_writer(timestamp) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = writer_lock.write_all(line.as_bytes());
                let _ = writer_lock.flush();
            }
        }
    }
}

// New unified event listener that uses the writer cache
pub fn unified_event_listener_thread_with_cache(
    should_run: Arc<AtomicBool>,
    writer_cache: Arc<Mutex<LogWriterCache>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,
) {
    println!("{}", "Starting input event logging with automatic chunk rotation...".green());
    let tap = CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![
            CGEventType::MouseMoved,
            CGEventType::LeftMouseDragged,
            CGEventType::RightMouseDragged,
            CGEventType::OtherMouseDragged,
        ],
        {
            let writer_cache = writer_cache.clone();
            let should_run = should_run.clone();
            move |_, event_type, event| {
                if !should_run.load(Ordering::SeqCst) {
                    return None;
                }
                let timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();

                match event_type {
                    CGEventType::MouseMoved |
                    CGEventType::LeftMouseDragged |
                    CGEventType::RightMouseDragged |
                    CGEventType::OtherMouseDragged => {
                        let dx = event.get_integer_value_field(EventField::MOUSE_EVENT_DELTA_X);
                        let dy = event.get_integer_value_field(EventField::MOUSE_EVENT_DELTA_Y);
                        log_mouse_event_with_cache(timestamp, &writer_cache, &format!("{{'type': 'delta', 'deltaX': {}, 'deltaY': {}}}", dx, dy));
                    }
                    _ => {}
                }
                None
            }
        },
    ).expect("Unable to create CGEvent tap. Did you enable Accessibility (Input Monitoring)?");

    let run_loop_source = tap.mach_port.create_runloop_source(0).unwrap();
    
    let event_thread = thread::spawn({
        let should_run = should_run.clone();
        let writer_cache = writer_cache.clone();
        let pressed_keys = pressed_keys.clone();
        move || {
            match listen(move |event: Event| {
                if !should_run.load(Ordering::SeqCst) {
                    return;
                }

                let timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();

                match event.event_type {
                    EventType::KeyPress(k) => {
                        handle_key_event_with_cache(true, k, timestamp, &writer_cache, &pressed_keys);
                    }
                    EventType::KeyRelease(k) => {
                        handle_key_event_with_cache(false, k, timestamp, &writer_cache, &pressed_keys);
                    }
                    EventType::MouseMove { x, y } => {
                        log_mouse_event_with_cache(timestamp, &writer_cache, &format!("{{'type': 'move', 'x': {}, 'y': {}}}", x, y));
                    }
                    EventType::ButtonPress(btn) => {
                        log_mouse_event_with_cache(timestamp, &writer_cache, &format!("{{'type': 'button', 'action': 'press', 'button': '{:?}'}}", btn));
                    }
                    EventType::ButtonRelease(btn) => {
                        log_mouse_event_with_cache(timestamp, &writer_cache, &format!("{{'type': 'button', 'action': 'release', 'button': '{:?}'}}", btn));
                    }
                    EventType::Wheel { delta_x, delta_y } => {
                        log_mouse_event_with_cache(timestamp, &writer_cache, &format!("{{'type': 'wheel', 'deltaX': {}, 'deltaY': {}}}", delta_x, delta_y));
                    }
                }
            }) {
                Ok(_) => {
                    println!("{}", "Input event listener stopped normally".yellow());
                },
                Err(e) => {
                    eprintln!("{}", format!("Input event listener error: {:?}. Input events will not be logged.", e).red());
                    eprintln!("{}", "This is likely due to missing Input Monitoring permission.".red());
                    eprintln!("{}", "Please ensure Input Monitoring permission is granted in System Settings.".yellow());
                }
            }
        }
    });

    CFRunLoop::get_current().add_source(&run_loop_source, unsafe { kCFRunLoopCommonModes });
    tap.enable();
    CFRunLoop::run_current();

    let _ = event_thread.join();
}

pub fn get_display_info() -> Vec<DisplayInfo> {
    let mut results = Vec::new();
    match CGDisplay::active_displays() {
        Ok(display_ids) => {
            for id in display_ids {
                let cg_display = CGDisplay::new(id);
                let bounds = cg_display.bounds();
                let x = bounds.origin.x as i32;
                let y = bounds.origin.y as i32;
                let width = bounds.size.width as u32;
                let height = bounds.size.height as u32;

                results.push(DisplayInfo {
                    id,
                    title: format!("Display {}", id),
                    is_primary: cg_display.is_main(),
                    x,
                    y,
                    original_width: width,
                    original_height: height,
                    capture_width: 1280,
                    capture_height: (height as f32 * (1280.0 / width as f32)) as u32,
                });
            }
        }
        Err(e) => eprintln!("Error retrieving active displays: {:?}", e),
    }
    results
}

struct Session {
    should_run: Arc<AtomicBool>,
    session_dir: PathBuf,

    event_thread: Option<thread::JoinHandle<()>>,

    capture_threads: Vec<(Arc<AtomicBool>, thread::JoinHandle<()>)>,

    // Replace individual log files with a writer cache
    writer_cache: Arc<Mutex<LogWriterCache>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,

    error_rx: Receiver<()>,
    error_tx: Sender<()>,
    
    displays: Vec<DisplayInfo>,
    progress_threads: Vec<thread::JoinHandle<()>>,
    
    // We still keep track of the current chunk index for display/info purposes
    current_chunk_index: usize,
    last_chunk_timestamp: u128,
}

impl Session {
    fn new(should_run: Arc<AtomicBool>) -> Result<Option<Self>> {
        let displays = get_display_info();
        if displays.is_empty() {
            return Ok(None);
        }

        let home_dir = dirs::home_dir().context("Could not determine home directory")?;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let session_dir = home_dir.join("Documents/loggy3").join(format!("session_{}", timestamp));
        create_dir_all(&session_dir)?;

        println!("\n{}", "=== Starting new recording session ===".cyan().bold());
        println!("Session directory: {}", session_dir.display().to_string().cyan());
        println!("{} {}", "Found".bright_white(), format!("{} display(s) to record:", displays.len()).bright_white());
        for display in &displays {
            println!("- {} ({} x {})", 
                display.title.cyan(),
                display.capture_width.to_string().yellow(),
                display.capture_height.to_string().yellow()
            );
        }
        println!("{}\n", "=====================================".cyan());

        let json_path = session_dir.join("display_info.json");
        let mut f = File::create(&json_path)?;
        serde_json::to_writer_pretty(&mut f, &displays)?;

        // Initialize chunk index based on current epoch time
        let current_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        
        // Calculate chunk number based on epoch time, in 60-second increments
        let current_chunk_index = (current_timestamp / 60000) as usize;
        
        // Create the writer cache instead of individual log files
        let writer_cache = Arc::new(Mutex::new(LogWriterCache::new(session_dir.clone())));
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
            progress_threads: Vec::new(),
            current_chunk_index: current_chunk_index,
            last_chunk_timestamp: current_timestamp,
        }))
    }
    
    fn start(&mut self) {
        let sr_clone_el = self.should_run.clone();
        let writer_cache = self.writer_cache.clone();
        let keys = self.pressed_keys.clone();
        self.event_thread = Some(thread::spawn(move || {
            unified_event_listener_thread_with_cache(
                sr_clone_el,
                writer_cache,
                keys,
            )
        }));

        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }
    }

    fn stop(self) {
        // No need for the cleanup_short_sessions parameter
        let session_dir = self.session_dir.clone();
        
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
        // We need to take ownership of each handle to join it
        for handle in self.progress_threads {
            let _ = handle.join();
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

        let handle = thread::spawn(move || {
            capture_display_thread(sr_clone, display, session_dir, error_tx);
        });
        self.capture_threads.push((sr_for_thread, handle));
    }

    // Update the rotate_to_new_chunk method to just update tracking variables
    // This no longer needs to create new files - the writer cache does that automatically
    fn rotate_to_new_chunk(&mut self) -> Result<()> {
        // Get current timestamp and calculate chunk index directly from epoch time
        let current_timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        
        // Calculate chunk index based on epoch time in 60-second increments
        self.current_chunk_index = (current_timestamp / 60000) as usize;
        
        // Update timestamp for the new chunk
        self.last_chunk_timestamp = current_timestamp;
            
        println!("{}", format!("Current chunk index is now {}", self.current_chunk_index).green());
        
        Ok(())
    }
}

pub fn main() -> Result<()> {
    // Check for command-line flags
    let args: Vec<String> = std::env::args().collect();
    let verbose_mode = args.iter().any(|arg| arg == "--verbose" || arg == "-v");
    let no_update_check = args.iter().any(|arg| arg == "--no-update-check");
    let disable_auto_update = args.iter().any(|arg| arg == "--disable-auto-update");
    let enable_auto_update = args.iter().any(|arg| arg == "--enable-auto-update");
    
    if verbose_mode {
        VERBOSE.store(true, Ordering::SeqCst);
    }
    
    // Load auto-update preferences
    match load_update_preferences() {
        Ok(disabled) => {
            AUTO_UPDATES_DISABLED.store(disabled, Ordering::SeqCst);
        }
        Err(_) => {
            // First run, auto-updates are enabled by default
            AUTO_UPDATES_DISABLED.store(false, Ordering::SeqCst);
            
            // Create config file with default settings
            let _ = save_update_preferences(false);
        }
    }
    
    // Override with command-line flags if provided
    if disable_auto_update {
        AUTO_UPDATES_DISABLED.store(true, Ordering::SeqCst);
        let _ = save_update_preferences(true);
    } else if enable_auto_update {
        AUTO_UPDATES_DISABLED.store(false, Ordering::SeqCst);
        let _ = save_update_preferences(false);
    }

    println!("{} {}", "\nLoggy3 Screen Recorder".bright_green().bold(), 
              format!("v{}", CURRENT_VERSION).bright_cyan());
    println!("{}", "======================".bright_green());

    if VERBOSE.load(Ordering::SeqCst) {
        println!("{}", "Verbose output enabled".yellow());
    }
    
    // Check for updates unless explicitly disabled
    if !no_update_check && !AUTO_UPDATES_DISABLED.load(Ordering::SeqCst) {
        println!("{}", "Checking for updates...".cyan());
        
        if let Some((version, download_url, release_url)) = check_for_updates() {
            println!("{} {} {} {}", 
                "A new version".bright_yellow(),
                version.bright_green().bold(),
                "is available!".bright_yellow(),
                format!("(current: {})", CURRENT_VERSION).bright_black()
            );
            
            println!("Release page: {}", release_url.bright_blue().underline());
            
            // Prompt user for action
            println!("\nWould you like to update now? [Y/n/never] ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            
            match input.trim().to_lowercase().as_str() {
                "y" | "yes" | "" => {
                    // User wants to update
                    update_to_new_version(&download_url)?;
                }
                "never" => {
                    // User wants to disable auto-updates
                    println!("{}", "Auto-updates disabled. You can re-enable them with --enable-auto-update".yellow());
                    AUTO_UPDATES_DISABLED.store(true, Ordering::SeqCst);
                    save_update_preferences(true)?;
                }
                _ => {
                    // User doesn't want to update now
                    println!("{}", "Update skipped. The application will continue to run.".yellow());
                }
            }
        } else if VERBOSE.load(Ordering::SeqCst) {
            println!("{}", "You're running the latest version!".green());
        }
    }

    // Check permissions at startup
    println!("\n{}", "Checking system permissions...".bright_black());
    
    // Check Screen Recording permission
    println!("{}", "Checking Screen Recording Permission...".bright_black());
    
    // Use proper screen recording permission check function
    if check_screen_recording_access() {
        println!("{}", "✓ Screen Recording permission is already granted.".green().bold());
    } else {
        println!("{}", "✗ Screen Recording permission is denied.".red());
        println!("{}", "Please enable it manually in:".yellow());
        println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
        
        // Request permission to trigger the system dialog
        let granted = request_screen_recording_access();
        if granted {
            println!("{}", "✓ Screen Recording permission just granted! Thank you!".green().bold());
        } else {
            println!("{}", "Permission not granted. You may need to go to:".red());
            println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
            println!("{}", "...and enable it for this application.".yellow());
            println!("{}", "Note: You may need to quit and restart Terminal after granting permission".yellow());
            return Ok(());
        }
    }

    // Check Input Monitoring permission
    println!("\n{}", "Checking Input Monitoring Permission...".bright_black());
    
    // Use the proper input monitoring permission check
    match check_input_monitoring_access() {
        Some(true) => {
            println!("{}", "✓ Input Monitoring permission is already granted.".green().bold());
        }
        Some(false) => {
            println!("{}", "✗ Input Monitoring permission is denied.".red());
            println!("{}", "Please enable it manually in:".yellow());
            println!("{}", "System Settings → Privacy & Security → Input Monitoring".yellow());
            
            // Try to open System Settings directly 
            let open_settings_result = Command::new("open")
                .args(["-a", "System Settings"])
                .spawn();
                
            match open_settings_result {
                Ok(_) => {
                    println!("\n{}", "System Settings has been opened for you.".bright_white());
                    println!("{}", "Please navigate to: Privacy & Security > Input Monitoring".bright_white());
                }
                Err(_) => {
                    println!("\n{}", "Could not automatically open System Settings.".red());
                    println!("{}", "Please open it manually from the Dock or Applications folder.".yellow());
                }
            }
            
            // Also try more direct methods for different macOS versions
            let _ = Command::new("open")
                .args(["x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"])
                .spawn();
                
            println!("\n{}", "After enabling the permission, please restart this app.".bright_green());
            return Ok(());
        }
        None => {
            println!("{}", "🟡 Input Monitoring permission is not determined. Requesting now...".yellow());
            println!("{}", "If prompted, please click \"Allow\" to grant Input Monitoring permission.".bright_green().bold());
            
            let granted = request_input_monitoring_access();
            if granted {
                println!("{}", "✓ Permission just granted! Thank you!".green().bold());
            } else {
                println!("{}", "✗ Permission not granted.".red());
                println!("{}", "You may need to go to:".yellow());
                println!("{}", "System Settings → Privacy & Security → Input Monitoring".yellow());
                println!("{}", "...and enable it for this application.".yellow());
                return Ok(());
            }
        }
    }

    println!("\n{}", "All permissions granted! Starting recorder...".green().bold());

    // Add a note about permissions being granted to Terminal
    println!("{}", "Note: Permissions are granted to Terminal, not Loggy3 itself. Running elsewhere requires re-granting permissions.".bright_black());

    let ffmpeg_path = get_ffmpeg_path();
    println!("Using ffmpeg at: {}", ffmpeg_path.display().to_string().cyan());

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
        last_display_fingerprint = current_fingerprint.clone();

        match Session::new(should_run.clone())? {
            Some(mut session) => {
                session.start();

                while should_run.load(Ordering::SeqCst) {
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
                    
                    // Check if we need to rotate to a new chunk based on epoch time
                    let current_timestamp = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    
                    // Calculate what the current chunk should be
                    let current_epoch_chunk = (current_timestamp / 60000) as usize;
                    
                    // If our chunk index is behind the epoch time chunk, rotate to catch up
                    if current_epoch_chunk > session.current_chunk_index {
                        if let Err(e) = session.rotate_to_new_chunk() {
                            eprintln!("Error rotating to new chunk: {}", e);
                        }
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
    let displays = get_display_info();
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
    println!("{} {} ({} x {})", 
        "Starting capture for display".green(),
        display_info.title.cyan(),
        display_info.capture_width.to_string().yellow(),
        display_info.capture_height.to_string().yellow()
    );
    
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

    // Initialize chunk index based on current epoch time
    let start_time_ms = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut current_chunk_index = (start_time_ms / 60000) as usize;
    
    // Create first chunk directory for this display
    let current_chunk_dir = session_dir.join(format!("chunk_{:05}", current_chunk_index));
    let display_dir = current_chunk_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
    if let Err(e) = create_dir_all(&display_dir) {
        eprintln!("Failed to create display directory: {}", e);
        return;
    }

    // Set up frames log in the current chunk's display directory
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
    
    // Start first ffmpeg process
    let mut ffmpeg_process = start_new_ffmpeg_process(&display_dir, width.try_into().unwrap(), height.try_into().unwrap(), display_info.id);
    if ffmpeg_process.is_none() {
        eprintln!("Failed to start initial ffmpeg process for display {}", display_info.id);
        return;
    }
    
    // Print initial status message
    let status_indicator = format!("[Display {}]", display_info.title);
    println!("{} Started recording", status_indicator.cyan());
    
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
                total_frame_count += 1;
                chunk_frame_count += 1;
                
                // Get current timestamp
                let current_timestamp = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis();
                
                // Calculate chunk index based on epoch time
                let new_chunk_index = (current_timestamp / 60000) as usize;
                
                // Check if we need to rotate to a new chunk
                if new_chunk_index > current_chunk_index {
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Finalizing chunk {} based on epoch time ({} frames)", 
                            current_chunk_index,
                            chunk_frame_count).yellow()
                    );
                    
                    // Close current ffmpeg process and start a new one
                    if let Some((mut child, stdin)) = ffmpeg_process.take() {
                        // Drop stdin first to close the pipe
                        drop(stdin);
                        
                        // Then wait for the child process to finish
                        if let Err(e) = child.wait() {
                            eprintln!("Error waiting for ffmpeg to complete: {}", e);
                        }
                    }
                    
                    // Update to the new chunk index
                    current_chunk_index = new_chunk_index;
                    chunk_frame_count = 0;
                    
                    // Create directory for the new chunk
                    let current_chunk_dir = session_dir.join(format!("chunk_{:05}", current_chunk_index));
                    let display_dir = current_chunk_dir.join(format!("display_{}_{}", display_info.id, display_info.title));
                    if let Err(e) = create_dir_all(&display_dir) {
                        eprintln!("Failed to create display directory for new chunk: {}", e);
                        handle_capture_error(&error_tx);
                        break;
                    }

                    // Create new frames log for the new chunk
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
                    
                    ffmpeg_process = start_new_ffmpeg_process(&display_dir, width.try_into().unwrap(), height.try_into().unwrap(), display_info.id);
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
                    
                    // Calculate seconds remaining in current chunk based on epoch time
                    let seconds_in_current_chunk = (current_timestamp % 60000) / 1000;
                    let seconds_remaining = 60 - seconds_in_current_chunk as u64;

                    // Use a simple mutex for synchronizing terminal output across threads
                    static STATUS_MUTEX: Mutex<()> = Mutex::new(());
                    let _status_lock = STATUS_MUTEX.lock().unwrap();
                    
                    // Print an inline status update
                    println!("{} Recording at {} fps (chunk {}, frames: {}, seconds remaining: {})", 
                        status_indicator.cyan(),
                        format!("{:.1}", fps).bright_green(),
                        current_chunk_index.to_string().yellow(),
                        chunk_frame_count.to_string().yellow(),
                        seconds_remaining.to_string().bright_yellow()
                    );

                    last_status = Instant::now();
                }
                
                // Write frame to current ffmpeg process
                if let Some((_, ref mut stdin)) = ffmpeg_process {
                    if let Err(e) = write_frame(stdin, &frame, &mut frames_log) {
                        eprintln!("Write error for display {}: {}", display_info.id, e);
                        break;
                    }
                } else {
                    eprintln!("No active ffmpeg process to write frame for display {}", display_info.id);
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
                // eprintln!("Frame timeout on display {} - ignoring due to idle display", display_info.id);
                continue;
            }
            Err(e) => {
                eprintln!("Channel error on display {}: {}", display_info.id, e);
                break;
            }
        }
    }

    // Clean up the ffmpeg process
    if let Some((mut child, stdin)) = ffmpeg_process {
        drop(stdin);
        let _ = child.wait();
    }
    
    println!("{} Recording stopped", status_indicator.cyan());
}

// Helper function to start a new ffmpeg process
fn start_new_ffmpeg_process(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
    display_id: u32,
) -> Option<(Child, ChildStdin)> {
    // Create a new ffmpeg process with single output
    let output_path = display_dir.join("video.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let ffmpeg_path = get_ffmpeg_path();

    let log_level = if VERBOSE.load(Ordering::SeqCst) {
        "info"
    } else {
        "error"
    };

    // Start the ffmpeg process
    let mut child = match Command::new(ffmpeg_path)
        .args(&[
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-color_range", "tv",
            "-s", &format!("{}x{}", width, height),
            "-r", "30",
            "-i", "pipe:0",
            "-c:v", FFMPEG_ENCODER,
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof+faststart",
            "-frag_duration", "1000000",  // Fragment every 1 second
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

    // Get stdin handle
    let stdin = match child.stdin.take() {
        Some(s) => s,
        None => {
            eprintln!("Failed to get ffmpeg stdin");
            return None;
        }
    };

    // Set up logging
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

fn download_ffmpeg() -> Result<PathBuf> {
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let loggy_dir = home_dir.join(".loggy3");
    create_dir_all(&loggy_dir)?;
    
    let ffmpeg_path = loggy_dir.join("ffmpeg");
    
    if !ffmpeg_path.exists() {
        println!("Downloading ffmpeg binary...");
        
        let temp_path = loggy_dir.join("ffmpeg.downloading");
        
        let command = format!(
            "curl -L -o {} https://publicr2.standardinternal.com/ffmpeg_binaries/macos_arm/ffmpeg",
            temp_path.display()
        );
        
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .status()
            .context("Failed to execute curl command")?;
            
        if !status.success() {
            return Err(anyhow::anyhow!("Failed to download ffmpeg binary"));
        }
        
        std::fs::rename(&temp_path, &ffmpeg_path)?;
        println!("Download complete");
        
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&ffmpeg_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&ffmpeg_path, perms)?;
    }
    
    Ok(ffmpeg_path)
}

// Check if there is a newer version available
fn check_for_updates() -> Option<(String, String, String)> {
    // If auto-updates are disabled, return None
    if AUTO_UPDATES_DISABLED.load(Ordering::SeqCst) {
        return None;
    }

    let api_url = format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO);
    
    match ureq::get(&api_url).call() {
        Ok(response) => {
            if let Ok(release) = response.into_json::<GitHubRelease>() {
                // Remove 'v' prefix if present for version comparison
                let latest_version = release.tag_name.trim_start_matches('v').to_string();
                
                // Compare versions
                if is_newer_version(&latest_version, CURRENT_VERSION) {
                    // Find the binary asset
                    if let Some(asset) = release.assets.iter().find(|a| a.name == "loggy3") {
                        return Some((latest_version, asset.browser_download_url.clone(), release.html_url));
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

// Simple version comparison (assumes semver-like versions: x.y.z)
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
    
    false  // Versions are equal
}

// Update to a newer version
fn update_to_new_version(download_url: &str) -> Result<()> {
    println!("{}", "Downloading the latest version...".cyan());
    
    // Get the home dir to install to
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let install_dir = home_dir.join(".local/bin");
    create_dir_all(&install_dir)?;
    let target_path = install_dir.join("loggy3");
    
    // Create a temporary file for the download
    let temp_path = target_path.with_extension("new");
    
    // Download the new version
    let response = ureq::get(download_url)
        .call()
        .context("Failed to download update")?;
    
    let mut file = File::create(&temp_path).context("Failed to create temporary file")?;
    let mut buffer = Vec::new();
    response.into_reader().read_to_end(&mut buffer).context("Failed to read response")?;
    file.write_all(&buffer).context("Failed to write to temporary file")?;
    
    // Make the new version executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&temp_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&temp_path, perms)?;
    }
    
    // Try direct file replacement first
    println!("{}", "Installing update...".cyan());
    
    // On unix, we can just replace the executable directly since we have permission 
    // to files in our own home directory
    if let Err(e) = std::fs::rename(&temp_path, &target_path) {
        if VERBOSE.load(Ordering::SeqCst) {
            eprintln!("Failed to rename file directly: {}", e);
            eprintln!("Falling back to delayed update");
        }
        
        // Create a bash script to replace the executable on next run
        let script_path = temp_path.with_extension("update.sh");
        let script_content = format!(
            r#"#!/bin/bash
# Wait for 1 second
sleep 1
# Replace the executable
mv "{}" "{}"
echo "Update complete! Please run 'loggy3' to start the updated version."
# Clean up
rm -f "$0"
"#,
            temp_path.display(),
            target_path.display()
        );
        
        let mut script_file = File::create(&script_path)?;
        script_file.write_all(script_content.as_bytes())?;
        
        // Make the script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        // Execute the update script
        Command::new(&script_path).spawn()?;
        
        println!("{}", "Update staged! The update will complete when this program exits.".green());
        println!("{}", "After you close this application, run 'loggy3' to start the updated version.".cyan());
    } else {
        println!("{}", "✓ Update installed successfully!".green());
        println!("{}", "Please restart the application to use the new version.".cyan());
    }
    
    // Exit the program after a successful update
    println!("{}", "Please restart loggy3 to use the new version.".bright_green().bold());
    exit(0);
}

// Save auto-update preferences
fn save_update_preferences(disabled: bool) -> Result<()> {
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let config_dir = home_dir.join(".loggy3");
    create_dir_all(&config_dir)?;
    
    let config_path = config_dir.join("config.json");
    let config = serde_json::json!({
        "auto_updates_disabled": disabled
    });
    
    let file = File::create(&config_path)?;
    serde_json::to_writer_pretty(file, &config)?;
    
    Ok(())
}

// Load auto-update preferences
fn load_update_preferences() -> Result<bool> {
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let config_path = home_dir.join(".loggy3/config.json");
    
    if config_path.exists() {
        let file = File::open(&config_path)?;
        let config: serde_json::Value = serde_json::from_reader(file)?;
        
        if let Some(disabled) = config.get("auto_updates_disabled").and_then(|v| v.as_bool()) {
            return Ok(disabled);
        }
    }
    
    // Default to auto-updates enabled
    Ok(false)
}

fn get_ffmpeg_path() -> PathBuf {
    if let Ok(ffmpeg_path) = download_ffmpeg() {
        return ffmpeg_path;
    }
    
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