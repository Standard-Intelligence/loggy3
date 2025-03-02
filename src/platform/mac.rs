use colored::*;
use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField, CGEventFlags, CGKeyCode};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use scap::Target;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use super::{LogWriterCache, handle_event_with_cache, get_multi_timestamp};
use crate::DisplayInfo;

pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";
pub static FFMPEG_PIXEL_FORMAT: &str = "nv12";
pub static FFMPEG_FILENAME: &str = "ffmpeg";
pub static FFMPEG_DOWNLOAD_URL: &str = "https://publicr2.standardinternal.com/ffmpeg_binaries/macos_arm/ffmpeg";


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
            0 => Some(true),
            1 => Some(false),
            2 => None,
            _ => None,
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
        _ => {
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


// The following code implements key code mapping for macOS virtual key codes
// These values are sourced from Carbon.framework's Events.h file
// Originally defined in Inside Mac Volume V, pg. V-191

fn keycode_to_string(keycode: CGKeyCode) -> String {
    // Map keycode to human-readable name based on macOS key constants
    let key_name = match keycode {
        // Letters
        0x00 => "A",
        0x01 => "S",
        0x02 => "D",
        0x03 => "F",
        0x04 => "H",
        0x05 => "G",
        0x06 => "Z",
        0x07 => "X",
        0x08 => "C",
        0x09 => "V",
        0x0B => "B",
        0x0C => "Q",
        0x0D => "W",
        0x0E => "E",
        0x0F => "R",
        0x10 => "Y",
        0x11 => "T",
        0x1F => "O",
        0x20 => "U",
        0x22 => "I",
        0x23 => "P",
        0x25 => "L",
        0x26 => "J",
        0x28 => "K",
        0x2D => "N",
        0x2E => "M",
        
        // Numbers
        0x12 => "1",
        0x13 => "2",
        0x14 => "3",
        0x15 => "4",
        0x17 => "5",
        0x16 => "6",
        0x1A => "7",
        0x1C => "8",
        0x19 => "9",
        0x1D => "0",
        
        // Special Characters
        0x18 => "Equal",
        0x1B => "Minus",
        0x21 => "LeftBracket",
        0x1E => "RightBracket",
        0x27 => "Quote",
        0x29 => "Semicolon",
        0x2A => "Backslash",
        0x2B => "Comma",
        0x2C => "Slash",
        0x2F => "Period",
        0x32 => "Grave",
        
        // Function Keys
        0x7A => "F1",
        0x78 => "F2",
        0x63 => "F3",
        0x76 => "F4",
        0x60 => "F5",
        0x61 => "F6",
        0x62 => "F7",
        0x64 => "F8",
        0x65 => "F9",
        0x6D => "F10",
        0x67 => "F11",
        0x6F => "F12",
        0x69 => "F13",
        0x6B => "F14",
        0x71 => "F15",
        0x6A => "F16",
        0x40 => "F17",
        0x4F => "F18",
        0x50 => "F19",
        0x5A => "F20",
        
        // Control Keys
        0x24 => "Return",
        0x30 => "Tab",
        0x31 => "Space",
        0x33 => "Delete",
        0x35 => "Escape",
        0x37 => "Command",
        0x38 => "Shift",
        0x39 => "CapsLock",
        0x3A => "Option",
        0x3B => "Control",
        0x36 => "RightCommand",
        0x3C => "RightShift",
        0x3D => "RightOption",
        0x3E => "RightControl",
        0x3F => "Function",
        0x47 => "KeypadClear",
        0x4C => "KeypadEnter",
        
        // Arrow Keys
        0x7B => "LeftArrow",
        0x7C => "RightArrow",
        0x7D => "DownArrow",
        0x7E => "UpArrow",
        
        // Navigation Keys
        0x72 => "Help",
        0x73 => "Home",
        0x74 => "PageUp",
        0x75 => "ForwardDelete",
        0x77 => "End",
        0x79 => "PageDown",
        
        // Keypad
        0x52 => "Keypad0",
        0x53 => "Keypad1",
        0x54 => "Keypad2",
        0x55 => "Keypad3",
        0x56 => "Keypad4",
        0x57 => "Keypad5",
        0x58 => "Keypad6",
        0x59 => "Keypad7",
        0x5B => "Keypad8",
        0x5C => "Keypad9",
        0x41 => "KeypadDecimal",
        0x43 => "KeypadMultiply",
        0x45 => "KeypadPlus",
        0x4B => "KeypadDivide",
        0x4E => "KeypadMinus",
        0x51 => "KeypadEquals",
        
        // Media Keys
        0x48 => "VolumeUp",
        0x49 => "VolumeDown",
        0x4A => "Mute",
        
        // ISO/JIS Specific
        0x0A => "ISO_Section",
        0x5D => "JIS_Yen",
        0x5E => "JIS_Underscore",
        0x5F => "JIS_KeypadComma",
        0x66 => "JIS_Eisu",
        0x68 => "JIS_Kana",
        
        // For unrecognized keycodes, show the hex value
        _ => return format!("Unknown(0x{:X})", keycode),
    };
    
    key_name.to_string()
}

fn flags_to_json(flags: CGEventFlags) -> String {
    format!(
        "{{\"alphaShift\": {}, \"shift\": {}, \"control\": {}, \"alternate\": {}, \"command\": {}, \"help\": {}, \"secondaryFn\": {}, \"numericPad\": {}, \"nonCoalesced\": {}}}",
        flags.contains(CGEventFlags::CGEventFlagAlphaShift),
        flags.contains(CGEventFlags::CGEventFlagShift),
        flags.contains(CGEventFlags::CGEventFlagControl),
        flags.contains(CGEventFlags::CGEventFlagAlternate),
        flags.contains(CGEventFlags::CGEventFlagCommand),
        flags.contains(CGEventFlags::CGEventFlagHelp),
        flags.contains(CGEventFlags::CGEventFlagSecondaryFn),
        flags.contains(CGEventFlags::CGEventFlagNumericPad),
        flags.contains(CGEventFlags::CGEventFlagNonCoalesced)
    )
}

// Keep track of the last seen flags to detect changes
static mut LAST_FLAGS: CGEventFlags = CGEventFlags::CGEventFlagNull;

fn detect_flag_changes(current_flags: CGEventFlags) -> String {
    let last_flags = unsafe { LAST_FLAGS };
    
    // Update the last flags for next time
    unsafe { LAST_FLAGS = current_flags; }
    
    // Check which flags changed
    let mut changes = Vec::new();
    
    // Check each flag
    if current_flags.contains(CGEventFlags::CGEventFlagAlphaShift) != last_flags.contains(CGEventFlags::CGEventFlagAlphaShift) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagAlphaShift) { "pressed" } else { "released" };
        changes.push(format!("\"alphaShift\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagShift) != last_flags.contains(CGEventFlags::CGEventFlagShift) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagShift) { "pressed" } else { "released" };
        changes.push(format!("\"shift\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagControl) != last_flags.contains(CGEventFlags::CGEventFlagControl) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagControl) { "pressed" } else { "released" };
        changes.push(format!("\"control\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagAlternate) != last_flags.contains(CGEventFlags::CGEventFlagAlternate) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagAlternate) { "pressed" } else { "released" };
        changes.push(format!("\"alternate\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagCommand) != last_flags.contains(CGEventFlags::CGEventFlagCommand) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagCommand) { "pressed" } else { "released" };
        changes.push(format!("\"command\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagHelp) != last_flags.contains(CGEventFlags::CGEventFlagHelp) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagHelp) { "pressed" } else { "released" };
        changes.push(format!("\"help\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagSecondaryFn) != last_flags.contains(CGEventFlags::CGEventFlagSecondaryFn) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagSecondaryFn) { "pressed" } else { "released" };
        changes.push(format!("\"secondaryFn\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagNumericPad) != last_flags.contains(CGEventFlags::CGEventFlagNumericPad) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagNumericPad) { "pressed" } else { "released" };
        changes.push(format!("\"numericPad\": \"{}\"", state));
    }
    
    if current_flags.contains(CGEventFlags::CGEventFlagNonCoalesced) != last_flags.contains(CGEventFlags::CGEventFlagNonCoalesced) {
        let state = if current_flags.contains(CGEventFlags::CGEventFlagNonCoalesced) { "pressed" } else { "released" };
        changes.push(format!("\"nonCoalesced\": \"{}\"", state));
    }
    
    // If no changes detected, return empty object
    if changes.is_empty() {
        return "{}".to_string();
    }
    
    // Return as JSON object
    format!("{{{}}}", changes.join(", "))
}

pub fn unified_event_listener_thread_with_cache(
    should_run: Arc<AtomicBool>,
    writer_cache: Arc<Mutex<LogWriterCache>>,
) {
    println!("{}", "Starting input event logging with automatic chunk rotation...".green());
    let tap = CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![
            // Mouse events.
            CGEventType::LeftMouseDown,
            CGEventType::LeftMouseUp,
            CGEventType::RightMouseDown,
            CGEventType::RightMouseUp,
            CGEventType::MouseMoved,
            CGEventType::LeftMouseDragged,
            CGEventType::RightMouseDragged,

            // Keyboard events.
            CGEventType::KeyDown,
            CGEventType::KeyUp,
            CGEventType::FlagsChanged,

            // Specialized control devices.
            CGEventType::ScrollWheel,
            CGEventType::TabletPointer,
            CGEventType::TabletProximity,
            CGEventType::OtherMouseDown,
            CGEventType::OtherMouseUp,
            CGEventType::OtherMouseDragged,

            // Out of band event types. These are delivered to the event tap callback
            // to notify it of unusual conditions that disable the event tap.
            CGEventType::TapDisabledByTimeout,
            CGEventType::TapDisabledByUserInput,
        ],
        {
            let writer_cache = writer_cache.clone();
            let should_run = should_run.clone();
            move |_, event_type, event| {
                if !should_run.load(Ordering::SeqCst) {
                    return None;
                }
                let multi_timestamp = get_multi_timestamp();
                let location = event.location();
                let flags = event.get_flags();

                // Common fields for all events
                let common_fields = format!(
                    "\"eventType\": \"{:?}\", \"location\": {{\"x\": {}, \"y\": {}}}, \"flags\": {}, \"flagsDetail\": {}, \"flagsChanged\": {}",
                    event_type, location.x, location.y, flags.bits(), flags_to_json(flags), detect_flag_changes(flags)
                );

                match event_type {
                    // Mouse movement events
                    CGEventType::MouseMoved |
                    CGEventType::LeftMouseDragged |
                    CGEventType::RightMouseDragged |
                    CGEventType::OtherMouseDragged => {
                        let dx = event.get_integer_value_field(EventField::MOUSE_EVENT_DELTA_X);
                        let dy = event.get_integer_value_field(EventField::MOUSE_EVENT_DELTA_Y);
                        let button_number = event.get_integer_value_field(EventField::MOUSE_EVENT_BUTTON_NUMBER);
                        let pressure = event.get_double_value_field(EventField::MOUSE_EVENT_PRESSURE);
                        
                        let json = format!(
                            "{{\"type\": \"mouse_movement\", {}, \"deltaX\": {}, \"deltaY\": {}, \"buttonNumber\": {}, \"pressure\": {}}}",
                            common_fields, dx, dy, button_number, pressure
                        );
                        handle_event_with_cache(&multi_timestamp, json, "mouse", &writer_cache);
                    },
                    
                    // Mouse button events
                    CGEventType::LeftMouseDown | CGEventType::OtherMouseDown | CGEventType::RightMouseDown => {
                        let button_number = event.get_integer_value_field(EventField::MOUSE_EVENT_BUTTON_NUMBER);
                        let click_state = event.get_integer_value_field(EventField::MOUSE_EVENT_CLICK_STATE);
                        let pressure = event.get_double_value_field(EventField::MOUSE_EVENT_PRESSURE);
                        
                        let json = format!(
                            "{{\"type\": \"mouse_down\", {}, \"buttonNumber\": {}, \"clickState\": {}, \"pressure\": {}}}",
                            common_fields, button_number, click_state, pressure
                        );
                        handle_event_with_cache(&multi_timestamp, json, "mouse", &writer_cache);
                    },
                    
                    CGEventType::LeftMouseUp | CGEventType::OtherMouseUp | CGEventType::RightMouseUp => {
                        let button_number = event.get_integer_value_field(EventField::MOUSE_EVENT_BUTTON_NUMBER);
                        
                        let json = format!(
                            "{{\"type\": \"mouse_up\", {}, \"buttonNumber\": {}}}",
                            common_fields, button_number
                        );
                        handle_event_with_cache(&multi_timestamp, json, "mouse", &writer_cache);
                    },
                    
                    // Keyboard events
                    CGEventType::KeyDown => {
                        let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                        let is_autorepeat = event.get_integer_value_field(EventField::KEYBOARD_EVENT_AUTOREPEAT);
                        let keyboard_type = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYBOARD_TYPE);
                        
                        let json = format!(
                            "{{\"type\": \"key_down\", {}, \"keycode\": {}, \"keycodeStr\": \"{}\", \"isAutorepeat\": {}, \"keyboardType\": {}}}",
                            common_fields, keycode, keycode_to_string(keycode as CGKeyCode), is_autorepeat, keyboard_type
                        );
                        handle_event_with_cache(&multi_timestamp, json, "keypresses", &writer_cache);
                    },
                    
                    CGEventType::KeyUp => {
                        let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                        let keyboard_type = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYBOARD_TYPE);
                        
                        let json = format!(
                            "{{\"type\": \"key_up\", {}, \"keycode\": {}, \"keycodeStr\": \"{}\", \"keyboardType\": {}}}",
                            common_fields, keycode, keycode_to_string(keycode as CGKeyCode), keyboard_type
                        );
                        handle_event_with_cache(&multi_timestamp, json, "keypresses", &writer_cache);
                    },
                    
                    CGEventType::FlagsChanged => {
                        // For modifier key events (shift, control, etc.)
                        // Try to determine which key was pressed/released based on the flags
                        let flag_changes = detect_flag_changes(flags);
                        
                        let json = format!(
                            "{{\"type\": \"flags_changed\", {}, \"flagChanges\": {}}}",
                            common_fields, flag_changes
                        );
                        handle_event_with_cache(&multi_timestamp, json, "keypresses", &writer_cache);
                    },
                    
                    // Scroll wheel events
                    CGEventType::ScrollWheel => {
                        let delta_axis_1 = event.get_integer_value_field(EventField::SCROLL_WHEEL_EVENT_DELTA_AXIS_1);
                        let delta_axis_2 = event.get_integer_value_field(EventField::SCROLL_WHEEL_EVENT_DELTA_AXIS_2);
                        let is_continuous = event.get_integer_value_field(EventField::SCROLL_WHEEL_EVENT_IS_CONTINUOUS);
                        
                        // Get more precise values if available
                        let fixed_delta_axis_1 = event.get_double_value_field(EventField::SCROLL_WHEEL_EVENT_FIXED_POINT_DELTA_AXIS_1);
                        let fixed_delta_axis_2 = event.get_double_value_field(EventField::SCROLL_WHEEL_EVENT_FIXED_POINT_DELTA_AXIS_2);
                        let point_delta_axis_1 = event.get_integer_value_field(EventField::SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_1);
                        let point_delta_axis_2 = event.get_integer_value_field(EventField::SCROLL_WHEEL_EVENT_POINT_DELTA_AXIS_2);
                        
                        let json = format!(
                            "{{\"type\": \"scroll_wheel\", {}, \"deltaAxis1\": {}, \"deltaAxis2\": {}, \"isContinuous\": {}, \"fixedDeltaAxis1\": {}, \"fixedDeltaAxis2\": {}, \"pointDeltaAxis1\": {}, \"pointDeltaAxis2\": {}}}",
                            common_fields, delta_axis_1, delta_axis_2, is_continuous, fixed_delta_axis_1, fixed_delta_axis_2, point_delta_axis_1, point_delta_axis_2
                        );
                        handle_event_with_cache(&multi_timestamp, json, "mouse", &writer_cache);
                    },
                    
                    // Tablet events
                    CGEventType::TabletPointer => {
                        let point_x = event.get_integer_value_field(EventField::TABLET_EVENT_POINT_X);
                        let point_y = event.get_integer_value_field(EventField::TABLET_EVENT_POINT_Y);
                        let point_z = event.get_integer_value_field(EventField::TABLET_EVENT_POINT_Z);
                        let point_buttons = event.get_integer_value_field(EventField::TABLET_EVENT_POINT_BUTTONS);
                        let pressure = event.get_double_value_field(EventField::TABLET_EVENT_POINT_PRESSURE);
                        let tilt_x = event.get_double_value_field(EventField::TABLET_EVENT_TILT_X);
                        let tilt_y = event.get_double_value_field(EventField::TABLET_EVENT_TILT_Y);
                        let rotation = event.get_double_value_field(EventField::TABLET_EVENT_ROTATION);
                        let tangential_pressure = event.get_double_value_field(EventField::TABLET_EVENT_TANGENTIAL_PRESSURE);
                        let device_id = event.get_integer_value_field(EventField::TABLET_EVENT_DEVICE_ID);
                        
                        let json = format!(
                            "{{\"type\": \"tablet_pointer\", {}, \"pointX\": {}, \"pointY\": {}, \"pointZ\": {}, \"pointButtons\": {}, \"pressure\": {}, \"tiltX\": {}, \"tiltY\": {}, \"rotation\": {}, \"tangentialPressure\": {}, \"deviceId\": {}}}",
                            common_fields, point_x, point_y, point_z, point_buttons, pressure, tilt_x, tilt_y, rotation, tangential_pressure, device_id
                        );
                        handle_event_with_cache(&multi_timestamp, json, "tablet", &writer_cache);
                    },
                    
                    CGEventType::TabletProximity => {
                        let vendor_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_VENDOR_ID);
                        let tablet_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_TABLET_ID);
                        let pointer_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_POINTER_ID);
                        let device_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_DEVICE_ID);
                        let system_tablet_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_SYSTEM_TABLET_ID);
                        let vendor_pointer_type = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_VENDOR_POINTER_TYPE);
                        let pointer_serial = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_VENDOR_POINTER_SERIAL_NUMBER);
                        let unique_id = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_VENDOR_UNIQUE_ID);
                        let capability_mask = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_CAPABILITY_MASK);
                        let pointer_type = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_POINTER_TYPE);
                        let enter_proximity = event.get_integer_value_field(EventField::TABLET_PROXIMITY_EVENT_ENTER_PROXIMITY);
                        
                        let json = format!(
                            "{{\"type\": \"tablet_proximity\", {}, \"vendorId\": {}, \"tabletId\": {}, \"pointerId\": {}, \"deviceId\": {}, \"systemTabletId\": {}, \"vendorPointerType\": {}, \"pointerSerial\": {}, \"uniqueId\": {}, \"capabilityMask\": {}, \"pointerType\": {}, \"enterProximity\": {}}}",
                            common_fields, vendor_id, tablet_id, pointer_id, device_id, system_tablet_id, vendor_pointer_type, pointer_serial, unique_id, capability_mask, pointer_type, enter_proximity
                        );
                        handle_event_with_cache(&multi_timestamp, json, "tablet", &writer_cache);
                    },
                    
                    // Tap disabled events
                    CGEventType::TapDisabledByTimeout => {
                        let json = format!(
                            "{{\"type\": \"tap_disabled_by_timeout\", {}}}",
                            common_fields
                        );
                        handle_event_with_cache(&multi_timestamp, json, "system", &writer_cache);
                    },
                    
                    CGEventType::TapDisabledByUserInput => {
                        let json = format!(
                            "{{\"type\": \"tap_disabled_by_user_input\", {}}}",
                            common_fields
                        );
                        handle_event_with_cache(&multi_timestamp, json, "system", &writer_cache);
                    },
                    
                    // Null event or any other event type
                    _ => {
                        let json = format!(
                            "{{\"type\": \"unknown_event\", {}}}",
                            common_fields
                        );
                        handle_event_with_cache(&multi_timestamp, json, "other", &writer_cache);
                    }
                }
                None
            }
        },
    ).expect("Unable to create mouse CGEvent tap. Did you enable Accessibility (Input Monitoring)?");

    let tap_source = tap.mach_port.create_runloop_source(0).unwrap();
    tap.enable();
    
    CFRunLoop::get_current().add_source(&tap_source, unsafe { kCFRunLoopCommonModes });
    CFRunLoop::run_current();
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

pub fn set_path_or_start_menu_shortcut() -> Result<(), String> {
    Ok(())
}