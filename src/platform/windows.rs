use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use lazy_static::lazy_static;
use serde::Serialize;
use colored::*;

use winapi::shared::hidusage::{
    HID_USAGE_GENERIC_KEYBOARD, HID_USAGE_GENERIC_MOUSE, HID_USAGE_PAGE_GENERIC,
};
use winapi::shared::minwindef::{BOOL, DWORD, TRUE, UINT, WPARAM, LPARAM};
use winapi::shared::ntdef::LPCWSTR;
use winapi::shared::windef::{HDC, HMONITOR, HWND, RECT};
use winapi::um::libloaderapi::GetModuleHandleW;
use winapi::um::shellscalingapi::{SetProcessDpiAwareness, PROCESS_PER_MONITOR_DPI_AWARE};
use winapi::um::winuser::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, EnumDisplayMonitors,
    GetMessageW, GetMonitorInfoW, GetRawInputData, LoadCursorW, PostQuitMessage, RegisterClassExW,
    RegisterRawInputDevices, TranslateMessage, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT,
    HRAWINPUT, IDC_ARROW, MONITORINFO, MONITORINFOF_PRIMARY, MSG,
    RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RIDEV_INPUTSINK, RID_INPUT,
    RIM_TYPEKEYBOARD, RIM_TYPEMOUSE, RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP,
    RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_MIDDLE_BUTTON_UP, RI_MOUSE_RIGHT_BUTTON_DOWN,
    RI_MOUSE_RIGHT_BUTTON_UP, RI_MOUSE_WHEEL, WM_DESTROY, WM_INPUT, WNDCLASSEXW,
    WS_DISABLED, WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use scap::Target;

use crate::DisplayInfo;
use super::{LogWriterCache, handle_event_with_cache, get_multi_timestamp, MultiTimestamp};

pub static FFMPEG_ENCODER: &str = "libx264";
pub static FFMPEG_PIXEL_FORMAT: &str = "bgra";
pub static FFMPEG_FILENAME: &str = "ffmpeg.exe";
pub static FFMPEG_DOWNLOAD_URL: &str = "https://publicr2.standardinternal.com/ffmpeg_binaries/windows_x64/ffmpeg.exe";


// A small helper to store the raw monitor data after enumeration
struct Monitor {
    rect: RECT,
    is_primary: bool,
}

// Helper to get width/height from a RECT
trait RectExt {
    fn width(&self) -> i32;
    fn height(&self) -> i32;
}

impl RectExt for RECT {
    fn width(&self) -> i32 {
        self.right - self.left
    }
    fn height(&self) -> i32 {
        self.bottom - self.top
    }
}

// Our callback data for enumerating monitors
struct MonitorCollection(Vec<Monitor>);

unsafe extern "system" fn monitor_enum_proc(
    hmonitor: HMONITOR,
    _hdc: HDC,
    _lprc_clip: *mut RECT,
    lparam: isize,
) -> BOOL {
    let mut mi: MONITORINFO = mem::zeroed();
    mi.cbSize = mem::size_of::<MONITORINFO>() as DWORD;

    if GetMonitorInfoW(hmonitor, &mut mi) != 0 {
        let is_primary = (mi.dwFlags & MONITORINFOF_PRIMARY) != 0;
        let rect = mi.rcMonitor;
        let monitors = &mut *(lparam as *mut MonitorCollection);
        monitors.0.push(Monitor { rect, is_primary });
    }
    TRUE
}

// Enumerate all monitors into a Vec<Monitor>
fn enumerate_monitors() -> Vec<Monitor> {
    // Mark the process as DPI-aware for consistent coords
    unsafe { SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE) };

    let mut monitors = MonitorCollection(Vec::new());
    let monitors_ptr = &mut monitors as *mut MonitorCollection as isize;

    unsafe {
        EnumDisplayMonitors(
            ptr::null_mut(),
            ptr::null(),
            Some(monitor_enum_proc),
            monitors_ptr,
        );
    }

    monitors.0
}

/// The main function returning DisplayInfo for each monitor.
pub fn get_display_info() -> Vec<DisplayInfo> {
    let monitors = enumerate_monitors();
    let mut results = Vec::new();

    for (i, m) in monitors.iter().enumerate() {
        let x = m.rect.left;
        let y = m.rect.top;
        let width = m.rect.width() as u32;
        let height = m.rect.height() as u32;
        let is_primary = m.is_primary;

        results.push(DisplayInfo {
            id: i as u32,
            title: format!("Display {}", i),
            is_primary,
            x,
            y,
            original_width: width,
            original_height: height,
            capture_width: 1280,
            capture_height: (1280 * height) / width,
        });
    }

    results
}

// ---------------------------------------------------------------------------
// Now, for the Raw Input listener code:

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum RawEvent {
    #[serde(rename_all = "camelCase")]
    Delta {
        delta_x: i32,
        delta_y: i32,
        timestamp: u128,
    },
    #[serde(rename_all = "camelCase")]
    Wheel {
        delta_x: i32,
        delta_y: i32,
        timestamp: u128,
    },
    #[serde(rename_all = "camelCase")]
    Button {
        action: String,
        button: String,
        timestamp: u128,
    },
    #[serde(rename_all = "camelCase")]
    Key {
        action: String,
        key_code: u32,
        timestamp: u128,
    },
}

fn to_wstring(str: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(str)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

lazy_static! {
    static ref WRITER_CACHE: Mutex<Option<Arc<Mutex<LogWriterCache>>>> = Mutex::new(None);
    static ref SHOULD_RUN: AtomicBool = AtomicBool::new(true);
}

/// Log a mouse event using the writer cache
fn log_mouse_event_cached(multi_timestamp: &MultiTimestamp, event: &RawEvent, writer_cache: &Arc<Mutex<LogWriterCache>>) {
    // let timestamp = match event {
    //     RawEvent::Delta { timestamp, .. } => *timestamp,
    //     RawEvent::Wheel { timestamp, .. } => *timestamp,
    //     RawEvent::Button { timestamp, .. } => *timestamp,
    //     RawEvent::Key { timestamp, .. } => *timestamp,
    // };
    
    // Convert the event to a JSON string and add sequence number
    if let Ok(json_string) = serde_json::to_string(event) {
        handle_event_with_cache(multi_timestamp, json_string, "mouse", writer_cache);
    }   
}

fn log_key_event_cached(event: &RawEvent, writer_cache: &Arc<Mutex<LogWriterCache>>) {
    if let RawEvent::Key { action, key_code, timestamp: _ } = event {
        let multi_timestamp = get_multi_timestamp();

        let key_str = format!("VK_{}", key_code);
        
        let raw_line = format!("'{}', '{}'", action, key_str);
        
        // if let Ok(mut cache_lock) = writer_cache.lock() {
        //     if let Ok(raw_writer) = cache_lock.get_raw_keypress_writer(wall_time) {
        //         if let Ok(mut raw_writer_lock) = raw_writer.lock() {
        //             let _ = writeln!(raw_writer_lock, "{}", raw_line);
        //             let _ = raw_writer_lock.flush();
        //         }
        //     }
        // }
        handle_event_with_cache(&multi_timestamp, raw_line, "keypresses", writer_cache);
    }
}

/// For convenience, a single function to handle a press/release with cache.
fn handle_key_event_cached(
    pressed: bool,
    vkey: u32,
    timestamp: u128,
    writer_cache: &Arc<Mutex<LogWriterCache>>,
) {
    let event = RawEvent::Key {
        action: if pressed {
            "press".to_string()
        } else {
            "release".to_string()
        },
        key_code: vkey,
        timestamp,
    };

    log_key_event_cached(&event, writer_cache);
}

unsafe fn handle_raw_input_with_cache(
    lparam: LPARAM,
    writer_cache: &Arc<Mutex<LogWriterCache>>,
) {
    let mut raw: RAWINPUT = mem::zeroed();
    let mut size = mem::size_of::<RAWINPUT>() as u32;
    let header_size = mem::size_of::<RAWINPUTHEADER>() as u32;

    let res = GetRawInputData(
        lparam as HRAWINPUT,
        RID_INPUT,
        &mut raw as *mut RAWINPUT as *mut _,
        &mut size,
        header_size,
    );
    if res == std::u32::MAX {
        return; // error
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    match raw.header.dwType {
        RIM_TYPEMOUSE => {
            let multi_timestamp = get_multi_timestamp();
            let mouse = raw.data.mouse();
            let flags = mouse.usButtonFlags;
            let wheel_delta = mouse.usButtonData as i16;
            let last_x = mouse.lLastX;
            let last_y = mouse.lLastY;

            // Movement:
            if last_x != 0 || last_y != 0 {
                let event = RawEvent::Delta {
                    delta_x: last_x,
                    delta_y: last_y,
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }

            // Wheel:
            if (flags & RI_MOUSE_WHEEL) != 0 {
                let event = RawEvent::Wheel {
                    delta_x: 0,
                    delta_y: wheel_delta as i32,
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }

            // Buttons:
            if (flags & RI_MOUSE_LEFT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
            if (flags & RI_MOUSE_LEFT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event_cached(&multi_timestamp, &event, writer_cache);
            }
        }
        RIM_TYPEKEYBOARD => {
            let kb = raw.data.keyboard();
            // typical: 0 => press, 1 => release
            let pressed = (kb.Flags & 0x01) == 0;

            handle_key_event_cached(pressed, kb.VKey as u32, timestamp, writer_cache);
        }
        _ => {}
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: UINT,
    wparam: WPARAM,
    lparam: LPARAM,
) -> isize {
    match msg {
        WM_INPUT => {
            let wc = WRITER_CACHE.lock().unwrap();

            if let Some(writer_cache) = &*wc {
                if SHOULD_RUN.load(Ordering::SeqCst) {
                    handle_raw_input_with_cache(lparam, writer_cache);
                }
            }
            0
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

fn create_hidden_window() -> HWND {
    let class_name = to_wstring("RawInputHiddenClass");
    let hinstance = unsafe { GetModuleHandleW(ptr::null()) };

    let wc = WNDCLASSEXW {
        cbSize: mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(window_proc),
        cbClsExtra: 0,
        cbWndExtra: 0,
        hInstance: hinstance,
        hIcon: ptr::null_mut(),
        hCursor: unsafe { LoadCursorW(ptr::null_mut(), IDC_ARROW) },
        hbrBackground: ptr::null_mut(),
        lpszMenuName: ptr::null_mut(),
        lpszClassName: class_name.as_ptr(),
        hIconSm: ptr::null_mut(),
    };

    let atom = unsafe { RegisterClassExW(&wc) };
    if atom == 0 {
        panic!("Failed to register window class");
    }

    let hwnd = unsafe {
        CreateWindowExW(
            0,
            atom as LPCWSTR,
            to_wstring("RawInputHidden").as_ptr(),
            // Hidden & disabled so we don't show a window
            WS_OVERLAPPEDWINDOW & !WS_VISIBLE | WS_DISABLED,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            100,
            100,
            ptr::null_mut(),
            ptr::null_mut(),
            hinstance,
            ptr::null_mut(),
        )
    };

    if hwnd.is_null() {
        panic!("Failed to create hidden window");
    }

    hwnd
}

fn register_raw_input(hwnd: HWND) -> bool {
    let rid = [
        RAWINPUTDEVICE {
            usUsagePage: HID_USAGE_PAGE_GENERIC,
            usUsage: HID_USAGE_GENERIC_MOUSE,
            dwFlags: RIDEV_INPUTSINK,
            hwndTarget: hwnd,
        },
        RAWINPUTDEVICE {
            usUsagePage: HID_USAGE_PAGE_GENERIC,
            usUsage: HID_USAGE_GENERIC_KEYBOARD,
            dwFlags: RIDEV_INPUTSINK,
            hwndTarget: hwnd,
        },
    ];

    let ret = unsafe {
        RegisterRawInputDevices(
            rid.as_ptr(),
            rid.len() as u32,
            mem::size_of::<RAWINPUTDEVICE>() as u32,
        )
    };
    ret == TRUE
}

/// The Windows version of unified_event_listener_thread using Raw Input with cache support.
pub fn unified_event_listener_thread_with_cache(
    should_run: Arc<AtomicBool>,
    writer_cache: Arc<Mutex<LogWriterCache>>,
) {
    println!("{}", "Starting input event logging with automatic chunk rotation...".green());
    
    // Set up lazy_static references so our window proc can use them:
    {
        let mut wc = WRITER_CACHE.lock().unwrap();
        *wc = Some(writer_cache.clone());

        SHOULD_RUN.store(true, Ordering::SeqCst);
    }

    thread::spawn(move || {
        let hwnd = create_hidden_window();
        if !register_raw_input(hwnd) {
            eprintln!("Failed to register raw input devices");
            return;
        }

        unsafe {
            let mut msg: MSG = mem::zeroed();
            while should_run.load(Ordering::SeqCst) {
                let ret = GetMessageW(&mut msg, ptr::null_mut(), 0, 0);
                if ret == 0 {
                    // WM_QUIT
                    break;
                } else if ret == -1 {
                    // error
                    break;
                } else {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
            // Done, destroy window
            DestroyWindow(hwnd);
        }
    });
}

pub fn get_target_matching_display_info(targets: Vec<Target>, display_info: DisplayInfo) -> Result<Target, String> {
    match targets.iter()
        .find(|t| match t {
            Target::Display(d) => {
                unsafe {
                    let hmonitor = d.id as u32 as HMONITOR;
                    let mut mi: MONITORINFO = mem::zeroed();
                    mi.cbSize = mem::size_of::<MONITORINFO>() as DWORD;

                    if GetMonitorInfoW(hmonitor, &mut mi) == 0 {return false; }

                    let rect = mi.rcMonitor;
                    display_info.x == rect.left && display_info.y == rect.top && display_info.original_width == rect.width() as u32 && display_info.original_height == rect.height() as u32
                }
            },
            _ => false
        })
        .cloned() {
            Some(t) => Ok(t),
            None => Err(format!("Could not find matching display target for display with title: {}, location: ({},{}), size: {}x{}", display_info.title, display_info.x, display_info.y, display_info.original_width, display_info.original_height))
        }
}

pub fn check_and_request_permissions() -> Result<(), String> {
    println!("{}", "Windows will display a bright yellow border around the screen when recording is active.".bright_black());

    Ok(())
}


use std::path::Path;
use std::env;
use winapi::um::shellapi::ShellExecuteW;
use std::os::windows::ffi::OsStrExt;
use std::ffi::OsStr;
use std::iter::once;

fn create_start_menu_shortcut(target_path: &str, shortcut_name: &str) -> Result<(), String> {
    let appdata = env::var("APPDATA")
        .map_err(|e| format!("Failed to get APPDATA directory while creating start menu shortcut: {}", e))?;
    let start_menu = Path::new(&appdata).join("Microsoft\\Windows\\Start Menu\\Programs");
    
    let temp_dir = env::temp_dir();
    let vbs_path = temp_dir.join("create_shortcut.vbs");
    
    let shortcut_path = start_menu.join(format!("{}.lnk", shortcut_name));
    
    let vbs_content = format!(
        "Set WshShell = CreateObject(\"WScript.Shell\")\n\
         Set Shortcut = WshShell.CreateShortcut(\"{}\")\n\
         Shortcut.TargetPath = \"{}\"\n\
         Shortcut.Description = \"{}\"\n\
         Shortcut.WorkingDirectory = \"{}\"\n\
         Shortcut.Save",
        shortcut_path.to_string_lossy(),
        target_path,
        shortcut_name,
        Path::new(target_path).parent().unwrap().to_string_lossy()
    );
    
    std::fs::write(&vbs_path, vbs_content)
        .map_err(|e| format!("Failed to write VBS script while creating start menu shortcut: {}", e))?;
    
    // Execute the VBScript
    let vbs_path_wide: Vec<u16> = OsStr::new(vbs_path.to_str().unwrap())
        .encode_wide()
        .chain(once(0))
        .collect();
    
    let operation: Vec<u16> = OsStr::new("open")
        .encode_wide()
        .chain(once(0))
        .collect();
        
    unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            vbs_path_wide.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1, // SW_SHOWNORMAL
        );
    }
    
    println!("{}", format!("Shortcut created at: {}", shortcut_path.display()).bright_black());
    Ok(())
}

pub fn set_path_or_start_menu_shortcut() -> Result<(), String> {
    println!("{}", "Adding start menu shortcut...".bright_black());
    let target_path = std::env::current_exe()
        .map_err(|e| format!("Failed to get current executable path: {}", e))?;
    let shortcut_name = "Loggy3";
    create_start_menu_shortcut(target_path.to_str().unwrap(), shortcut_name)?;
    Ok(())
}

#[derive(Debug)]
pub struct WindowsVersion {
    pub major: u32,
    pub minor: u32,
    pub build: u32,
}

pub enum WindowsVersionType {
    Windows10,
    Windows11,
    Unsupported,
}

pub fn get_windows_version() -> Result<WindowsVersion, String> {
    // Use RtlGetVersion which is more reliable than GetVersionExW
    use winapi::shared::ntdef::{NTSTATUS, NT_SUCCESS};
    use winapi::um::winnt::OSVERSIONINFOW;
    use ntapi::ntrtl::RtlGetVersion;
    use std::mem::zeroed;

    unsafe {
        let mut osvi: OSVERSIONINFOW = zeroed();
        osvi.dwOSVersionInfoSize = std::mem::size_of::<OSVERSIONINFOW>() as u32;
        
        let status: NTSTATUS = RtlGetVersion(&mut osvi as *mut _);
        if !NT_SUCCESS(status) {
            return Err(format!("RtlGetVersion failed with status: {}", status));
        }
        
        Ok(WindowsVersion {
            major: osvi.dwMajorVersion,
            minor: osvi.dwMinorVersion,
            build: osvi.dwBuildNumber,
        })
    }
}

pub fn get_windows_version_type() -> Result<WindowsVersionType, String> {
    let version = get_windows_version()?;
    
    match (version.major, version.minor, version.build) {
        // Windows 10
        (10, 0, _) if version.build < 22000 => Ok(WindowsVersionType::Windows10),
        
        // Windows 11 (build number >= 22000)
        (10, 0, build) if build >= 22000 => Ok(WindowsVersionType::Windows11),
        
        // Anything else is not supported
        _ => Ok(WindowsVersionType::Unsupported),
    }
}

pub fn check_windows_version_compatibility() -> Result<(), String> {
    match get_windows_version_type()? {
        WindowsVersionType::Windows10 | WindowsVersionType::Windows11 => Ok(()),
        WindowsVersionType::Unsupported => {
            let version = get_windows_version()?;
            Err(format!("Unsupported Windows version: {}.{}.{}. Loggy3 requires Windows 10 or newer.", 
                version.major, version.minor, version.build))
        }
    }
}
