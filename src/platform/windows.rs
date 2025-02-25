use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File, OpenOptions},
    io::{BufWriter, Write, BufReader, BufRead, Read},
    mem,
    ptr,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio, exit},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType, BGRAFrame},
    Target,
};

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

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use colored::*;
use ureq;

// Used both by windows.rs and mod.rs
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

pub static FFMPEG_ENCODER: &str = "libx264";
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

fn log_mouse_event(timestamp: u128, mouse_log: &Mutex<BufWriter<File>>, data: &str) {
    let line = format!("({}, {})\n", timestamp, data);
    if let Ok(mut writer) = mouse_log.lock() {
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.flush();
    }
}

fn handle_key_event(
    is_press: bool,
    key: u32,
    timestamp: u128,
    key_log: &Mutex<BufWriter<File>>,
    pressed_keys: &Mutex<Vec<String>>,
) {
    let key_str = format!("VK_{}", key);
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
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'delta', 'deltaX': {}, 'deltaY': {}}}", last_x, last_y));
            }

            if (flags & RI_MOUSE_WHEEL) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'wheel', 'deltaX': 0, 'deltaY': {}}}", wheel_delta));
            }

            if (flags & RI_MOUSE_LEFT_BUTTON_DOWN) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'press', 'button': 'Left'}}"));
            }
            if (flags & RI_MOUSE_LEFT_BUTTON_UP) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'release', 'button': 'Left'}}"));
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_DOWN) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'press', 'button': 'Right'}}"));
            }
            if (flags & RI_MOUSE_RIGHT_BUTTON_UP) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'release', 'button': 'Right'}}"));
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_DOWN) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'press', 'button': 'Middle'}}"));
            }
            if (flags & RI_MOUSE_MIDDLE_BUTTON_UP) != 0 {
                log_mouse_event(timestamp, &mouse_log, &format!("{{'type': 'button', 'action': 'release', 'button': 'Middle'}}"));
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
    println!("{}", "Starting input event logging...".green());
    
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
        // Check for complete chunks before stopping threads 
        // (we need to do this before moving any part of self)
        let has_complete_chunks = !cleanup_short_sessions || self.has_complete_chunks();
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
        
        // Check if this is a short/glitched session that should be cleaned up
        if cleanup_short_sessions && !has_complete_chunks {
            println!("{}", "Short recording session detected - cleaning up...".yellow());
            // Remove the session directory and all its contents
            if let Err(e) = std::fs::remove_dir_all(&session_dir) {
                if VERBOSE.load(Ordering::SeqCst) {
                    eprintln!("Failed to clean up short session: {}", e);
                }
            } else {
                println!("{}", "✓ Short session cleaned up".green());
                return;
            }
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

fn get_target_matching_display_info(
    targets: Vec<Target>,
    display_info: DisplayInfo,
) -> Result<Target, String> {
    for target in targets {
        if let Target::Display(d) = &target {
            if d.id == display_info.id {
                return Ok(target);
            }
        }
    }
    Err(format!("No matching display found for ID: {}", display_info.id))
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
    
    let target = match get_target_matching_display_info(targets, display_info.clone()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    let capturer = match initialize_capturer(&target) {
        Some(c) => c,
        None => return,
    };

    // Here we assume width and height are already in DisplayInfo
    let width = display_info.original_width;
    let height = display_info.original_height;

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
            Ok(Ok(Frame::BGRA(frame))) => {
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
            Ok(Ok(_)) => {
                eprintln!("Unexpected frame type received for display {}", display_info.id);
                handle_capture_error(&error_tx);
            }

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
        output_type: FrameType::BGRAFrame,
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

fn initialize_ffmpeg(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
) -> Result<(Child, ChildStdin)> {
    let output_path = display_dir.join("chunk_%05d.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let mut child = Command::new("ffmpeg")
        .args(&[
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", &format!("{}x{}", width, height),
            "-r", "30",
            "-i", "pipe:0",
            "-c:v", FFMPEG_ENCODER,
            "-movflags", "+faststart",
            "-g", "60",
            "-f", "segment",
            "-segment_time", "60",
            "-reset_timestamps", "1",
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
    frame: &BGRAFrame,
    frames_log: &mut BufWriter<File>,
) -> Result<()> {
    ffmpeg_stdin.write_all(&frame.data)?;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis();
    writeln!(frames_log, "{}", timestamp)?;
    frames_log.flush()?;

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