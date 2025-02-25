#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(target_os = "macos")]
mod macos_input_monitoring {
    // We'll manually declare the FFI signatures for the needed IOKit functions:
    //  - IOHIDCheckAccess(request: u32) -> u32  (returns 0=granted,1=denied,2=not determined)
    //  - IOHIDRequestAccess(request: u32) -> bool (returns true if granted)
    // and link against IOKit.

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOHIDCheckAccess(request: u32) -> u32;
        fn IOHIDRequestAccess(request: u32) -> bool;
    }

    // According to IOKit/IOHIDLib.h, for 10.15+:
    //  kIOHIDRequestTypePostEvent   = 0, // Accessibility
    //  kIOHIDRequestTypeListenEvent = 1, // Input Monitoring
    const KIOHID_REQUEST_TYPE_LISTEN_EVENT: u32 = 1;

    /// Checks the current input monitoring status:
    ///  - 0 = Granted
    ///  - 1 = Denied
    ///  - 2 = NotDetermined
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

    /// Requests input monitoring access (prompts user if not determined).
    /// Returns `true` if permission is (now) granted, `false` otherwise.
    pub fn request_input_monitoring_access() -> bool {
        unsafe { IOHIDRequestAccess(KIOHID_REQUEST_TYPE_LISTEN_EVENT) }
    }
}

#[cfg(target_os = "macos")]
mod macos_screen_recording {
    // We need to link against CoreGraphics for screen recording permissions
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
        fn CGRequestScreenCaptureAccess() -> bool;
    }

    /// Checks if screen recording permission is granted
    /// Returns true if granted, false if denied or not determined
    pub fn check_screen_recording_access() -> bool {
        unsafe { CGPreflightScreenCaptureAccess() }
    }

    /// Requests screen recording access (prompts user if not determined)
    /// Returns true if permission is granted, false otherwise
    pub fn request_screen_recording_access() -> bool {
        unsafe { CGRequestScreenCaptureAccess() }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    use macos_input_monitoring::*;
    use macos_screen_recording::*;

    // Check Input Monitoring permissions
    println!("Checking Input Monitoring permissions...");
    match check_input_monitoring_access() {
        Some(true) => {
            println!("✅ Input Monitoring permission is already granted.");
        }
        Some(false) => {
            println!("❌ Input Monitoring permission is denied.");
            println!("Please enable it manually in:");
            println!("System Settings → Privacy & Security → Input Monitoring");
        }
        None => {
            println!("🟡 Input Monitoring permission is not determined. Requesting now...");
            let granted = request_input_monitoring_access();
            if granted {
                println!("✅ Permission just granted! Thank you!");
            } else {
                println!("❌ Permission not granted. You may need to go to:");
                println!("System Settings → Privacy & Security → Input Monitoring");
                println!("...and enable it for this application.");
            }
        }
    }

    // Check Screen Recording permissions
    println!("\nChecking Screen Recording permissions...");
    if check_screen_recording_access() {
        println!("✅ Screen Recording permission is already granted.");
    } else {
        println!("🟡 Screen Recording permission is not granted. Requesting now...");
        let granted = request_screen_recording_access();
        if granted {
            println!("✅ Screen Recording permission just granted! Thank you!");
        } else {
            println!("❌ Screen Recording permission not granted. You may need to go to:");
            println!("System Settings → Privacy & Security → Screen Recording");
            println!("...and enable it for this application.");
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("This example only works on macOS (10.15+).");
}
