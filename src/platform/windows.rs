use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use lazy_static::lazy_static;
use serde::Serialize;

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
