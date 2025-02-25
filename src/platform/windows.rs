use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, Read, BufReader, BufRead};
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio, exit};
use std::collections::HashMap;
use std::sync::mpsc::{self, channel, Receiver, Sender};
use chrono::Local;

use anyhow::{Context, Result};
use lazy_static::lazy_static;
use serde::{Serialize, Deserialize};
use colored::*;
use dirs;
use ctrlc::set_handler;
use sysinfo::System;

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

// For screen capturing
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, 
    GetDC, ReleaseDC, SelectObject, SRCCOPY, GetDIBits, DIB_RGB_COLORS,
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, RGBQUAD
};
use windows::Win32::UI::WindowsAndMessaging::GetDesktopWindow;

pub static FFMPEG_ENCODER: &str = "h264_nvenc";
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
            title: format!("Display {}", i + 1),
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

    fn get_keypress_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.keypress_writers.contains_key(&chunk_index) {
            let chunk_dir = self.session_dir.join(format!("chunk_{:05}", chunk_index));
            create_dir_all(&chunk_dir)?;
            
            let log_path = chunk_dir.join("keypresses.log");
            let writer = Arc::new(Mutex::new(BufWriter::new(File::create(log_path)?)));
            println!("{}", format!("Created new keypress log for chunk {}", chunk_index).yellow());
            self.keypress_writers.insert(chunk_index, writer);
        }
        Ok(self.keypress_writers.get(&chunk_index).unwrap().clone())
    }
    
    fn get_mouse_writer(&mut self, timestamp_ms: u128) -> Result<Arc<Mutex<BufWriter<File>>>> {
        let chunk_index = (timestamp_ms / 60000) as usize;
        if !self.mouse_writers.contains_key(&chunk_index) {
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

fn log_mouse_event_with_cache(timestamp: u128, cache: &Arc<Mutex<LogWriterCache>>, event: &RawEvent) {
    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_mouse_writer(timestamp) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = serde_json::to_writer(&mut *writer_lock, &event);
                let _ = writeln!(&mut *writer_lock);
                let _ = writer_lock.flush();
            }
        }
    }
}

fn handle_key_event_with_cache(
    is_press: bool,
    key_code: u32,
    timestamp: u128,
    cache: &Arc<Mutex<LogWriterCache>>,
    pressed_keys: &Mutex<Vec<String>>,
) {
    let key_str = format!("VK_{}", key_code);
    let mut keys = pressed_keys.lock().unwrap();

    if is_press {
        if !keys.contains(&key_str) {
            keys.push(key_str.clone());
        }
    } else {
        keys.retain(|k| k != &key_str);
    }

    let event = RawEvent::Key {
        action: if is_press { "press".to_string() } else { "release".to_string() },
        key_code,
        timestamp,
    };

    if let Ok(mut cache_lock) = cache.lock() {
        if let Ok(writer) = cache_lock.get_keypress_writer(timestamp) {
            if let Ok(mut writer_lock) = writer.lock() {
                let _ = serde_json::to_writer(&mut *writer_lock, &event);
                let _ = writeln!(&mut *writer_lock);
                let _ = writer_lock.flush();
            }
        }
    }
}

lazy_static! {
    static ref MOUSE_LOG_CACHE: Mutex<Option<Arc<Mutex<LogWriterCache>>>> = Mutex::new(None);
    static ref PRESSED_KEYS: Mutex<Option<Arc<Mutex<Vec<String>>>>> = Mutex::new(None);
    static ref SHOULD_RUN: AtomicBool = AtomicBool::new(true);
}

unsafe fn handle_raw_input(
    lparam: LPARAM,
    cache: &Arc<Mutex<LogWriterCache>>,
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
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            
            if (flags & RI_MOUSE_WHEEL) != 0 {
                let event = RawEvent::Wheel {
                    delta_x: 0,
                    delta_y: wheel_delta as i32,
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            
            // Handle button presses
            if (flags & RI_MOUSE_LEFT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            if (flags & RI_MOUSE_LEFT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Left".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Right".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN) != 0 {
                let event = RawEvent::Button {
                    action: "press".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_UP) != 0 {
                let event = RawEvent::Button {
                    action: "release".to_string(),
                    button: "Middle".to_string(),
                    timestamp,
                };
                log_mouse_event_with_cache(timestamp, cache, &event);
            }
        }
        RIM_TYPEKEYBOARD => {
            let kb = raw.data.keyboard();
            let pressed = (kb.Flags & 0x01) == 0;
            handle_key_event_with_cache(pressed, kb.VKey as u32, timestamp, cache, pressed_keys);
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
            let ml = MOUSE_LOG_CACHE.lock().unwrap();
            let pk = PRESSED_KEYS.lock().unwrap();

            if let (Some(cache), Some(keys)) = (&*ml, &*pk) {
                if SHOULD_RUN.load(Ordering::SeqCst) {
                    handle_raw_input(lparam, cache, keys);
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

pub fn unified_event_listener_thread_with_cache(
    should_run: Arc<AtomicBool>,
    writer_cache: Arc<Mutex<LogWriterCache>>,
    pressed_keys: Arc<Mutex<Vec<String>>>,
) {
    println!("{}", "Starting input event logging with automatic chunk rotation...".green());
    
    {
        let mut ml = MOUSE_LOG_CACHE.lock().unwrap();
        *ml = Some(writer_cache.clone());

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
                    // WM_QUIT
                    break;
                } else if ret == -1 {
                    // Error
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

// Screen capture implementation for Windows
fn capture_screen(display_info: &DisplayInfo) -> Result<Vec<u8>> {
    unsafe {
        let desktop_hwnd = GetDesktopWindow();
        let src_dc = GetDC(desktop_hwnd);
        
        // Check if we got a valid DC
        if src_dc.is_invalid() {
            return Err(anyhow::anyhow!("Failed to get desktop DC"));
        }

        let dest_dc = CreateCompatibleDC(src_dc);
        
        // Check if we got a valid compatible DC
        if dest_dc.is_invalid() {
            ReleaseDC(desktop_hwnd, src_dc);
            return Err(anyhow::anyhow!("Failed to create compatible DC"));
        }

        // Create a bitmap compatible with the screen DC
        let bitmap = CreateCompatibleBitmap(
            src_dc, 
            display_info.capture_width as i32, 
            display_info.capture_height as i32
        );
        
        // Check if bitmap creation failed
        if bitmap.is_invalid() {
            DeleteDC(dest_dc);
            ReleaseDC(desktop_hwnd, src_dc);
            return Err(anyhow::anyhow!("Failed to create compatible bitmap"));
        }

        // Select the bitmap into the compatible DC
        let old_obj = SelectObject(dest_dc, bitmap);
        
        // Use BitBlt to capture the screen
        let result = BitBlt(
            dest_dc,
            0, 0,
            display_info.capture_width as i32,
            display_info.capture_height as i32,
            src_dc,
            display_info.x,
            display_info.y,
            SRCCOPY
        );
        
        if !result.as_bool() {
            SelectObject(dest_dc, old_obj);
            DeleteObject(bitmap);
            DeleteDC(dest_dc);
            ReleaseDC(desktop_hwnd, src_dc);
            return Err(anyhow::anyhow!("BitBlt failed"));
        }

        // Get the bitmap data
        let mut bitmap_info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: display_info.capture_width as i32,
                biHeight: -(display_info.capture_height as i32), // Negative for top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0 as u32,
                biSizeImage: 0,
                biXPelsPerMeter: 0,
                biYPelsPerMeter: 0,
                biClrUsed: 0,
                biClrImportant: 0,
            },
            bmiColors: [RGBQUAD {
                rgbBlue: 0,
                rgbGreen: 0,
                rgbRed: 0,
                rgbReserved: 0,
            }],
        };

        let buffer_size = (display_info.capture_width * display_info.capture_height * 4) as usize;
        let mut buffer = vec![0u8; buffer_size];

        let scan_lines = GetDIBits(
            dest_dc,
            bitmap,
            0,
            display_info.capture_height,
            Some(buffer.as_mut_ptr() as *mut std::ffi::c_void),
            &mut bitmap_info as *mut BITMAPINFO,
            DIB_RGB_COLORS
        );

        // Clean up GDI resources
        SelectObject(dest_dc, old_obj);
        DeleteObject(bitmap);
        DeleteDC(dest_dc);
        ReleaseDC(desktop_hwnd, src_dc);

        if scan_lines == 0 {
            return Err(anyhow::anyhow!("GetDIBits failed"));
        }

        // Convert from BGRA to RGB for FFmpeg
        let mut rgb_buffer = Vec::with_capacity((display_info.capture_width * display_info.capture_height * 3) as usize);
        for i in 0..(display_info.capture_width * display_info.capture_height) as usize {
            let base = i * 4;
            rgb_buffer.push(buffer[base + 2]); // R
            rgb_buffer.push(buffer[base + 1]); // G
            rgb_buffer.push(buffer[base]); // B
        }

        Ok(rgb_buffer)
    }
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
        let _ = error_tx.send(());
        return;
    }

    // Set up frames log in the current chunk's display directory
    let frames_log_path = display_dir.join("frames.log");
    let frames_log_file = match File::create(&frames_log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to create frames log: {}", e);
            let _ = error_tx.send(());
            return;
        }
    };
    let mut frames_log = BufWriter::new(frames_log_file);
    
    let start_time = Instant::now();
    let mut total_frame_count = 0;
    let mut chunk_frame_count = 0;
    let mut last_status = Instant::now();
    
    // Start first ffmpeg process
    let mut ffmpeg_process = start_new_ffmpeg_process(
        &display_dir, 
        display_info.capture_width as usize, 
        display_info.capture_height as usize, 
        display_info.id
    );
    
    if ffmpeg_process.is_none() {
        eprintln!("Failed to start initial ffmpeg process for display {}", display_info.id);
        let _ = error_tx.send(());
        return;
    }
    
    // Print initial status message
    let status_indicator = format!("[Display {}]", display_info.title);
    println!("{} Started recording", status_indicator.cyan());
    
    // Target frame rate and interval
    let target_fps = 30.0;
    let frame_interval = Duration::from_secs_f64(1.0 / target_fps);
    
    // Main capture loop
    while should_run.load(Ordering::SeqCst) {
        let frame_start = Instant::now();
        
        // Capture a frame
        match capture_screen(&display_info) {
            Ok(frame_data) => {
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
                        let _ = error_tx.send(());
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
                            let _ = error_tx.send(());
                            break;
                        }
                    }
                    
                    ffmpeg_process = start_new_ffmpeg_process(
                        &display_dir, 
                        display_info.capture_width as usize, 
                        display_info.capture_height as usize, 
                        display_info.id
                    );
                    
                    if ffmpeg_process.is_none() {
                        eprintln!("Failed to start new ffmpeg process for display {}", display_info.id);
                        let _ = error_tx.send(());
                        break;
                    }
                    
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Started new chunk {}", current_chunk_index).green()
                    );
                }
            
                // Print status periodically
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
                    if let Err(e) = write_frame(stdin, &frame_data, &mut frames_log, current_timestamp) {
                        eprintln!("Write error for display {}: {}", display_info.id, e);
                        let _ = error_tx.send(());
                        break;
                    }
                } else {
                    eprintln!("No active ffmpeg process to write frame for display {}", display_info.id);
                    let _ = error_tx.send(());
                    break;
                }
                
                // Sleep to maintain target frame rate
                let elapsed = frame_start.elapsed();
                if elapsed < frame_interval {
                    thread::sleep(frame_interval - elapsed);
                }
            },
            Err(e) => {
                eprintln!("Frame capture error on display {}: {}", display_info.id, e);
                
                // Check if we should still be running
                if !should_run.load(Ordering::SeqCst) {
                    break;
                }
                
                // Sleep briefly before trying again
                thread::sleep(Duration::from_millis(100));
                
                // Signal an error if we haven't gotten a frame for a long time
                if total_frame_count == 0 && start_time.elapsed() > Duration::from_secs(10) {
                    eprintln!("No frames captured after 10 seconds. Signaling error.");
                    let _ = error_tx.send(());
                    break;
                }
            }
        }
    }

    // Clean up the ffmpeg process
    if let Some((mut child, stdin)) = ffmpeg_process {
        // Try to gracefully close the pipe
        drop(stdin);
        
        // Wait with timeout
        let start = Instant::now();
        let timeout = Duration::from_secs(2);
        
        // First try to see if the process is already done
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process already exited
            },
            Ok(None) => {
                // Process still running, wait with timeout
                let mut exited = false;
                
                while start.elapsed() < timeout && !exited {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            exited = true;
                        },
                        Ok(None) => {
                            // Still running, sleep and try again
                            thread::sleep(Duration::from_millis(100));
                        },
                        Err(e) => {
                            eprintln!("Error checking ffmpeg status: {}", e);
                            break;
                        },
                    }
                }
                
                // If still not exited, force kill
                if !exited {
                    eprintln!("Timeout waiting for ffmpeg to finish - killing process");
                    let _ = child.kill();
                }
            },
            Err(e) => {
                eprintln!("Error waiting for ffmpeg: {}", e);
            },
        }
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
            "-pix_fmt", "rgb24",
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

fn get_ffmpeg_path() -> PathBuf {
    // On Windows, first check in the same directory as the executable
    if let Ok(exe_path) = std::env::current_exe() {
        let default_path = PathBuf::from(".");
        let exe_dir = exe_path.parent().unwrap_or(&default_path);
        let local_ffmpeg = exe_dir.join("ffmpeg.exe");
        if local_ffmpeg.exists() {
            return local_ffmpeg;
        }
    }

    // Check in a local application directory
    let default_path = PathBuf::from(".");
    let home_dir = dirs::home_dir().unwrap_or(default_path);
    let app_ffmpeg = home_dir.join(".loggy3").join("ffmpeg.exe");
    if app_ffmpeg.exists() {
        return app_ffmpeg;
    }

    // Check standard PATH locations
    PathBuf::from("ffmpeg.exe")
}

fn write_frame(
    ffmpeg_stdin: &mut ChildStdin,
    frame_data: &[u8],
    frames_log: &mut BufWriter<File>,
    timestamp: u128,
) -> Result<()> {
    // Write the raw frame data to ffmpeg's stdin
    ffmpeg_stdin.write_all(frame_data)?;
    
    // Log the frame timestamp
    writeln!(frames_log, "{}", timestamp)?;
    frames_log.flush()?;

    Ok(())
}

struct Session {
    should_run: Arc<AtomicBool>,
    session_dir: PathBuf,

    event_thread: Option<thread::JoinHandle<()>>,
    capture_threads: Vec<(Arc<AtomicBool>, thread::JoinHandle<()>)>,

    writer_cache: Arc<Mutex<LogWriterCache>>,
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

        // Log session metadata (version and system info)
        if let Err(e) = log_session_metadata(&session_dir) {
            eprintln!("Warning: Failed to log session metadata: {}", e);
        }

        let json_path = session_dir.join("display_info.json");
        let mut f = File::create(&json_path)?;
        serde_json::to_writer_pretty(&mut f, &displays)?;
        
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
        let session_dir = self.session_dir.clone();
        
        // Log that we're stopping the session
        println!("{}", "Stopping recording session...".yellow());
        
        // Stop all capture threads first, with a timeout mechanism
        for (flag, handle) in self.capture_threads {
            // Set should_run to false for this thread
            flag.store(false, Ordering::SeqCst);
            
            // Wait for thread to finish with timeout
            let start = Instant::now();
            let timeout = Duration::from_secs(3);
            
            // Try to join the thread with timeout
            while start.elapsed() < timeout {
                if handle.is_finished() {
                    let _ = handle.join();
                    break;
                }
                thread::sleep(Duration::from_millis(100));
            }
            
            // If thread didn't finish within timeout, just log and continue
            if start.elapsed() >= timeout {
                eprintln!("Warning: Screen capture thread did not terminate cleanly within timeout");
            }
        }

        // Stop event thread with a timeout
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
            
            if start.elapsed() >= timeout {
                eprintln!("Warning: Event listener thread did not terminate cleanly within timeout");
            }
        }

        // Clean up any progress threads
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
    let response = ureq::get(download_url)
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
    Command::new("cmd")
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

// Log system metadata at the session level
fn log_session_metadata(session_dir: &PathBuf) -> Result<()> {
    // Initialize system information
    let mut system = System::new_all();
    system.refresh_all();
    
    // Create a metadata structure
    let metadata = serde_json::json!({
        "app_version": CURRENT_VERSION,
        "timestamp": Local::now().to_rfc3339(),
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
            }
        }
    });
    
    // Create the metadata file
    let metadata_path = session_dir.join("session_metadata.json");
    let file = File::create(&metadata_path)?;
    serde_json::to_writer_pretty(file, &metadata)?;
    
    println!("{}", "✓ Session metadata logged successfully".green());
    Ok(())
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
    
    // Check for updates
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

                // Explicitly stopping the session to ensure clean shutdown
                session.stop();
                
                // If we're exiting due to a global shutdown (Ctrl-C), break the outer loop
                if !should_run.load(Ordering::SeqCst) {
                    break;
                }
                
                // For display changes or errors, wait a moment before restarting
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
    Ok(())
}