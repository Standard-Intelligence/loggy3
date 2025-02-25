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
use sysinfo::System;

use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use rdev::{listen, Event, EventType};
use ureq;

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOHIDCheckAccess(request: u32) -> u32;
    fn IOHIDRequestAccess(request: u32) -> bool;
}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
    fn CGRequestScreenCaptureAccess() -> bool;
}

const KIOHID_REQUEST_TYPE_LISTEN_EVENT: u32 = 1;

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

fn request_input_monitoring_access() -> bool {
    unsafe { IOHIDRequestAccess(KIOHID_REQUEST_TYPE_LISTEN_EVENT) }
}

fn check_screen_recording_access() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

fn request_screen_recording_access() -> bool {
    unsafe { CGRequestScreenCaptureAccess() }
}

pub fn check_permissions() -> Result<()> {
    

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
}

use serde::{Deserialize, Serialize};
use colored::*;

pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";

use super::LogWriterCache;

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