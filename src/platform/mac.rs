use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use scap::Target;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use rdev::{listen, Event, EventType};
use super::{LogWriterCache, log_mouse_event_with_cache, handle_key_event_with_cache};
use crate::DisplayInfo;
use colored::*;

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

pub fn check_input_monitoring_access() -> Option<bool> {
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

pub fn check_screen_recording_access() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

fn request_screen_recording_access() -> bool {
    unsafe { CGRequestScreenCaptureAccess() }
}

pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";
pub static FFMPEG_PIXEL_FORMAT: &str = "nv12";

pub fn check_and_request_permissions() -> Result<(), &'static str> {
    println!("{}", "Checking Screen Recording Permission...".bright_black());
    
    if check_screen_recording_access() {
        println!("{}", "✓ Screen Recording permission is already granted.".green().bold());
    } else {
        println!("{}", "✗ Screen Recording permission is denied.".red());
        println!("{}", "Please enable it manually in:".yellow());
        println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
        
        let granted = request_screen_recording_access();
        if granted {
            println!("{}", "✓ Screen Recording permission just granted! Thank you!".green().bold());
        } else {
            println!("{}", "Permission not granted. You may need to go to:".red());
            println!("{}", "System Settings → Privacy & Security → Screen Recording".yellow());
            println!("{}", "...and enable it for this application.".yellow());
            println!("{}", "Note: You may need to quit and restart Terminal after granting permission".yellow());
            return Err("Screen recording permission not granted");
        }
    }

    println!("\n{}", "Checking Input Monitoring Permission...".bright_black());
    
    match check_input_monitoring_access() {
        Some(true) => {
            println!("{}", "✓ Input Monitoring permission is already granted.".green().bold());
        }
        Some(false) => {
            println!("{}", "✗ Input Monitoring permission is denied.".red());
            println!("{}", "Please enable it manually in:".yellow());
            println!("{}", "System Settings → Privacy & Security → Input Monitoring".yellow());
            
            let open_settings_result = std::process::Command::new("open")
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
            
            let _ = std::process::Command::new("open")
                .args(["x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"])
                .spawn();
                
            println!("\n{}", "After enabling the permission, please restart this app.".bright_green());
            return Err("Input monitoring permission not granted");
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
                return Err("Input monitoring permission not granted");
            }
        }
    }

    println!("\n{}", "All permissions granted! Starting recorder...".green().bold());

    println!("{}", "Note: Permissions are granted to Terminal, not Loggy3 itself. Running elsewhere requires re-granting permissions.".bright_black());

    Ok(())
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


pub fn get_target_matching_display_info(targets: Vec<Target>, display_info: DisplayInfo) -> Option<Target> {
    let target = match targets.iter()
        .find(|t| match t {
            Target::Display(d) => d.id == display_info.id,
            _ => false
        })
        .cloned() {
            Some(t) => t,
            None => {
                eprintln!("Could not find matching display target for ID: {}", display_info.id);
                return None;
            }
        };
    Some(target)
}