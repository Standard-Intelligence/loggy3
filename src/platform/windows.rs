use std::fs::{File, create_dir_all, OpenOptions};
use std::io::{BufWriter, Write, Read, BufReader, BufRead};
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio, exit};

use anyhow::{Context, Result};
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use colored::*;
use dirs;

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

use super::DisplayInfo;

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

struct Monitor {
    rect: RECT,
    is_primary: bool,
}

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

fn enumerate_monitors() -> Vec<Monitor> {
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

pub fn get_display_info() -> Vec<DisplayInfo> {
    let monitors = enumerate_monitors();
    let mut results = Vec::new();

    for (i, m) in monitors.iter().enumerate() {
        let x = m.rect.left;
        let y = m.rect.top;
        let width = m.rect.width() as u32;
        let height = m.rect.height() as u32;
        let is_primary = m.is_primary;

        let capture_width = 1280;
        let capture_height = (height as f32 * (capture_width as f32 / width as f32)) as u32;

        results.push(DisplayInfo {
            id: i as u32,
            title: format!("Display {}", i),
            is_primary,
            x,
            y,
            original_width: width,
            original_height: height,
            capture_width,
            capture_height,
        });
    }

    results
}

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
    static ref MOUSE_LOG: Mutex<Option<Arc<Mutex<BufWriter<File>>>>> = Mutex::new(None);
    static ref KEY_LOG: Mutex<Option<Arc<Mutex<BufWriter<File>>>>> = Mutex::new(None);
    static ref SHOULD_RUN: AtomicBool = AtomicBool::new(true);

    
    static ref PRESSED_KEYS: Mutex<Option<Arc<Mutex<Vec<String>>>>> = Mutex::new(None);
}


fn log_mouse_event(event: &RawEvent, mouse_log: &Arc<Mutex<BufWriter<File>>>) {
    if let Ok(mut writer) = mouse_log.lock() {
        let _ = serde_json::to_writer(&mut *writer, event);
        let _ = writeln!(&mut *writer);
        let _ = writer.flush();
    }
}


fn log_key_event(event: &RawEvent, keypress_log: &Arc<Mutex<BufWriter<File>>>) {
    if let Ok(mut writer) = keypress_log.lock() {
        let _ = serde_json::to_writer(&mut *writer, event);
        let _ = writeln!(&mut *writer);
        let _ = writer.flush();
    }
}


fn update_pressed_keys(pressed: bool, key_code: u32, pressed_keys: &Arc<Mutex<Vec<String>>>) {
    
    let key_str = format!("VK_{}", key_code);
    let mut pk = pressed_keys.lock().unwrap();

    if pressed {
        if !pk.contains(&key_str) {
            pk.push(key_str);
        }
    } else {
        pk.retain(|k| k != &key_str);
    }
}


fn handle_key_event(
    pressed: bool,
    vkey: u32,
    timestamp: u128,
    keypress_log: &Arc<Mutex<BufWriter<File>>>,
    pressed_keys: &Arc<Mutex<Vec<String>>>,
) {
    update_pressed_keys(pressed, vkey, pressed_keys);

    let event = RawEvent::Key {
        action: if pressed {
            "press".to_string()
        } else {
            "release".to_string()
        },
        key_code: vkey,
        timestamp,
    };

    log_key_event(&event, keypress_log);
}

unsafe fn handle_raw_input(
    lparam: LPARAM,
    mouse_log: &Arc<Mutex<BufWriter<File>>>,
    keypress_log: &Arc<Mutex<BufWriter<File>>>,
    pressed_keys: &Arc<Mutex<Vec<String>>>,
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
        return; 
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    match raw.header.dwType {
        RIM_TYPEMOUSE => {
            let mouse = raw.data.mouse();
            let flags = mouse.usButtonFlags;
            let wheel_delta = mouse.usButtonData as i16;
            let last_x = mouse.lLastX;
            let last_y = mouse.lLastY;

            
            if last_x != 0 || last_y != 0 {
                let event = RawEvent::Delta {
                    delta_x: last_x,
                    delta_y: last_y,
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }

            
            if (flags & RI_MOUSE_WHEEL) != 0 {
                let event = RawEvent::Wheel {
                    delta_x: 0,
                    delta_y: wheel_delta as i32,
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }

            
            if (flags & RI_MOUSE_LEFT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
            if (flags & RI_MOUSE_LEFT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event(&event, mouse_log);
            }
        }
        RIM_TYPEKEYBOARD => {
            let kb = raw.data.keyboard();
            
            let pressed = (kb.Flags & 0x01) == 0;

            handle_key_event(pressed, kb.VKey as u32, timestamp, keypress_log, pressed_keys);
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
            let ml = MOUSE_LOG.lock().unwrap();
            let kl = KEY_LOG.lock().unwrap();
            let pk = PRESSED_KEYS.lock().unwrap();

            if let (Some(m_log), Some(k_log), Some(keys)) = (&*ml, &*kl, &*pk) {
                if SHOULD_RUN.load(Ordering::SeqCst) {
                    handle_raw_input(lparam, m_log, k_log, keys);
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



pub fn unified_event_listener_thread(
    should_run: Arc<AtomicBool>,
    keypress_log: Arc<Mutex<BufWriter<File>>>,
    mouse_log: Arc<Mutex<BufWriter<File>>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,
) {
    
    {
        let mut ml = MOUSE_LOG.lock().unwrap();
        *ml = Some(mouse_log.clone());

        let mut kl = KEY_LOG.lock().unwrap();
        *kl = Some(keypress_log.clone());

        let mut pk = PRESSED_KEYS.lock().unwrap();
        *pk = Some(pressed_keys.clone());

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
                    
                    break;
                } else if ret == -1 {
                    
                    break;
                } else {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
            
            DestroyWindow(hwnd);
        }
    });
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
                    if let Some(asset) = release.assets.iter().find(|a| a.name == "loggy3.exe") {
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
    
    // Get the path to the current executable
    let current_exe = std::env::current_exe().context("Failed to get current executable path")?;
    
    // Create a temporary file for the download
    let temp_path = current_exe.with_extension("new.exe");
    
    // Download the new version
    let mut response = ureq::get(download_url)
        .call()
        .context("Failed to download update")?;
    
    let mut file = File::create(&temp_path).context("Failed to create temporary file")?;
    let mut buffer = Vec::new();
    response.into_reader().read_to_end(&mut buffer).context("Failed to read response")?;
    file.write_all(&buffer).context("Failed to write to temporary file")?;
    
    // Create a batch file to replace the current executable
    let script_path = current_exe.with_extension("update.bat");
    let script_content = format!(
        r#"@echo off
:: Wait for the original process to exit
timeout /t 1 /nobreak > nul
:: Replace the executable
copy /y "{}" "{}"
:: Execute the new version
start "" "{}" %*
:: Delete this batch file
del "%~f0"
"#,
        temp_path.display(),
        current_exe.display(),
        current_exe.display()
    );
    
    let mut script_file = File::create(&script_path)?;
    script_file.write_all(script_content.as_bytes())?;
    
    // Execute the update script
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = Command::new("cmd")
        .arg("/c")
        .arg(&script_path)
        .args(args)
        .spawn()?;
    
    // Exit the current process
    println!("{}", "Update downloaded! Restarting application...".green());
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

    println!("{}", "\nLoggy3 Screen Recorder".bright_green().bold());
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
    
    // TODO: Windows implementation goes here
    println!("Windows support is coming soon!");
    
    Ok(())
}
