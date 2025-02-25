use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File, OpenOptions},
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


fn log_mouse_event(timestamp: u128, mouse_log: &Mutex<BufWriter<File>>, data: &str) {
    let line = format!("({}, {})\n", timestamp, data);
    if let Ok(mut writer) = mouse_log.lock() {
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.flush();
    }
}


fn handle_key_event(
    is_press: bool,
    key: rdev::Key,
    timestamp: u128,
    key_log: &Mutex<BufWriter<File>>,
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
    if let Ok(mut writer) = key_log.lock() {
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.flush();
    }
}

pub fn unified_event_listener_thread(
    should_run: Arc<AtomicBool>,
    keypress_log: Arc<Mutex<BufWriter<File>>>,
    mouse_log: Arc<Mutex<BufWriter<File>>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,
) {
    println!("{}", "Starting input event logging...".green());
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
            let mouse_log = mouse_log.clone();
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
                        log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'delta', 'deltaX': {}, 'deltaY': {}}}", dx, dy));
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
        let keypress_log = keypress_log.clone();
        let mouse_log = mouse_log.clone();
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
                        handle_key_event(true, k, timestamp, &keypress_log, &pressed_keys);
                    }
                    EventType::KeyRelease(k) => {
                        handle_key_event(false, k, timestamp, &keypress_log, &pressed_keys);
                    }
                    EventType::MouseMove { x, y } => {
                        log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'move', 'x': {}, 'y': {}}}", x, y));
                    }
                    EventType::ButtonPress(btn) => {
                        log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'press', 'button': '{:?}'}}", btn));
                    }
                    EventType::ButtonRelease(btn) => {
                        log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'release', 'button': '{:?}'}}", btn));
                    }
                    EventType::Wheel { delta_x, delta_y } => {
                        log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'wheel', 'deltaX': {}, 'deltaY': {}}}", delta_x, delta_y));
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

    keypress_log: Arc<Mutex<BufWriter<File>>>,
    mouse_log: Arc<Mutex<BufWriter<File>>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,

    error_rx: Receiver<()>,
    error_tx: Sender<()>,
    
    displays: Vec<DisplayInfo>,
    progress_threads: Vec<thread::JoinHandle<()>>,
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

        let keypress_log_path = session_dir.join("keypresses.log");
        let mouse_log_path = session_dir.join("mouse.log");
        let keypress_log = Arc::new(Mutex::new(BufWriter::new(File::create(keypress_log_path)?)));
        let mouse_log = Arc::new(Mutex::new(BufWriter::new(File::create(mouse_log_path)?)));
        let pressed_keys = Arc::new(Mutex::new(vec![]));

        let (error_tx, error_rx) = mpsc::channel();

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
        }))
    }
    
    // Check if this session has at least one complete chunk (1 minute of recording)
    fn has_complete_chunks(&self) -> bool {
        let mut has_chunks = false;
        
        // Iterate through each display directory
        for display in &self.displays {
            let display_dir = self.session_dir.join(format!("display_{}_{}", display.id, display.title));
            if !display_dir.exists() {
                continue;
            }
            
            // Check if there are any chunk files
            if let Ok(entries) = std::fs::read_dir(&display_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        let file_name = entry.file_name().to_string_lossy().to_string();
                        
                        // Check if this is a completed chunk file (not being written)
                        if file_name.starts_with("chunk_") && file_name.ends_with(".mp4") {
                            // Check if file size is at least 1MB (reasonable for a complete chunk)
                            if let Ok(metadata) = entry.metadata() {
                                if metadata.len() > 1_000_000 {  // 1MB minimum size
                                    has_chunks = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            
            if has_chunks {
                break;
            }
        }
        
        has_chunks
    }

    fn start(&mut self) {
        let sr_clone_el = self.should_run.clone();
        let kp_log = self.keypress_log.clone();
        let m_log = self.mouse_log.clone();
        let keys = self.pressed_keys.clone();
        self.event_thread = Some(thread::spawn(move || {
            unified_event_listener_thread(
                sr_clone_el,
                kp_log,
                m_log,
                keys,
            )
        }));

        for display in self.displays.clone() {
            self.start_capture_for_display(display);
        }
    }

    fn stop(self, cleanup_short_sessions: bool) {
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
        
        // Check if this is a short/glitched session that should be cleaned up
        if cleanup_short_sessions && !self.has_complete_chunks() {
            println!("{}", "Short recording session detected - cleaning up...".yellow());
            // Remove the session directory and all its contents
            if let Err(e) = std::fs::remove_dir_all(&self.session_dir) {
                if VERBOSE.load(Ordering::SeqCst) {
                    eprintln!("Failed to clean up short session: {}", e);
                }
            } else {
                println!("{}", "✓ Short session cleaned up".green());
                return;
            }
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
    println!("\n{}", "Checking system permissions...".yellow());
    
    // Check Screen Recording permission
    println!("{}", "Checking Screen Recording Permission...".yellow());
    
    // Use proper screen recording permission check function
    if check_screen_recording_access() {
        println!("{}", "✓ Screen Recording permission is already granted.".green());
    } else {
        println!("{}", "✗ Screen Recording permission is denied.".red());
        println!("{}", "Please enable it manually in:".yellow());
        println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
        
        // Request permission to trigger the system dialog
        let granted = request_screen_recording_access();
        if granted {
            println!("{}", "✓ Screen Recording permission just granted! Thank you!".green());
        } else {
            println!("{}", "Permission not granted. You may need to go to:".red());
            println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
            println!("{}", "...and enable it for this application.".yellow());
            println!("{}", "Note: You may need to quit and restart Terminal after granting permission".yellow());
            return Ok(());
        }
    }

    // Check Input Monitoring permission
    println!("\n{}", "Checking Input Monitoring Permission...".yellow());
    
    // Use the proper input monitoring permission check
    match check_input_monitoring_access() {
        Some(true) => {
            println!("{}", "✓ Input Monitoring permission is already granted.".green());
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
                println!("{}", "✓ Permission just granted! Thank you!".green());
            } else {
                println!("{}", "✗ Permission not granted.".red());
                println!("{}", "You may need to go to:".yellow());
                println!("{}", "System Settings → Privacy & Security → Input Monitoring".yellow());
                println!("{}", "...and enable it for this application.".yellow());
                return Ok(());
            }
        }
    }

    println!("\n{}", "All permissions granted! Starting recorder...".green());

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

                    thread::sleep(Duration::from_secs(1));
                }

                // Only cleanup short sessions when we're restarting due to errors or display changes
                // not when user explicitly stops with Ctrl-C
                let cleanup_short_sessions = !should_run.load(Ordering::SeqCst);
                session.stop(cleanup_short_sessions);
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

    if let Some(stdout) = ffmpeg_child.stdout.take() {
        let display_id = display_info.id;
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

    if let Some(stderr) = ffmpeg_child.stderr.take() {
        let display_id = display_info.id;
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
    let mut frame_count = 0;
    let mut last_status = Instant::now();
    
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
                frame_count += 1;
            
                if last_status.elapsed() >= Duration::from_secs(5) {
                    let fps = frame_count as f64 / start_time.elapsed().as_secs_f64();

                    // Overwrite the same line: "\r\x1b[2K" resets and clears the current line
                    print!("\r\x1b[2KDisplay {}: Recording at {} fps", 
                        display_info.title.cyan(),
                        format!("{:.1}", fps).bright_green()
                    );

                    // Flush to ensure the line appears immediately
                    std::io::stdout().flush().unwrap();

                    last_status = Instant::now();
                }
                
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
                // eprintln!("Frame timeout on display {} - ignoring due to idle display", display_info.id);
                continue;
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
    let mut response = ureq::get(download_url)
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
    
    // Continue running the current version
    return Ok(());
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

fn initialize_ffmpeg(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
) -> Result<(Child, ChildStdin)> {
    let output_path = display_dir.join("chunk_%05d.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let ffmpeg_path = get_ffmpeg_path();

    let log_level = if VERBOSE.load(Ordering::SeqCst) {
        "info"
    } else {
        "error"
    };

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
            "-loglevel", log_level,
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