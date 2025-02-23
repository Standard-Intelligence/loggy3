use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use rdev::{listen, Event, EventType};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::fs::File;


use super::DisplayInfo;

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
            
            if keys.contains(&"MetaLeft".to_string()) && 
               keys.contains(&"ShiftLeft".to_string()) && 
               key_str == "KeyW" {
                println!("Command + Shift + L detected!");
            }
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
            let _ = listen(move |event: Event| {
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
            });
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