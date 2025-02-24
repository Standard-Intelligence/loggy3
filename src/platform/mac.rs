use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File},
    io::{BufWriter, Write, BufReader, BufRead, Read},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, channel, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType, YUVFrame},
    Target,
};

use core_graphics::display::CGDisplay;
use core_graphics::event::{CGEventTap, CGEventTapLocation, CGEventTapPlacement,
    CGEventTapOptions, CGEventType, EventField};
use core_foundation::runloop::{CFRunLoop, kCFRunLoopCommonModes};
use rdev::{listen, Event, EventType};

use serde::{Deserialize, Serialize};
use indicatif::{ProgressBar, ProgressStyle};
use colored::*;

pub static FFMPEG_ENCODER: &str = "h264_videotoolbox";
pub static VERBOSE: AtomicBool = AtomicBool::new(false);

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
    println!("{}", "Starting input event logging...".green());
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

    fn stop(self) {
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
        for handle in self.progress_threads {
            let _ = handle.join();
        }

        println!("Session stopped: {}", self.session_dir.display());
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

pub fn main() -> Result<()> {
    // Check for verbose flag
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--verbose" || arg == "-v") {
        VERBOSE.store(true, Ordering::SeqCst);
    }

    println!("{}", "\nLoggy3 Screen Recorder".bright_green().bold());
    println!("{}", "======================".bright_green());

    if VERBOSE.load(Ordering::SeqCst) {
        println!("{}", "Verbose output enabled".yellow());
    }

    // Check permissions at startup
    println!("\n{}", "Checking system permissions...".yellow());
    
    // Check Screen Recording permission
    let has_screen_permission = CGDisplay::active_displays().is_ok();
    print!("Screen Recording Permission: ");
    if has_screen_permission {
        println!("{}", "✓ Enabled".green());
    } else {
        println!("{}", "✗ Disabled".red());
        println!("{}", "Please enable Screen Recording permission in System Settings > Privacy & Security > Screen Recording".red());
        return Ok(());
    }

    // Check Input Monitoring by attempting to create an event tap
    print!("Input Monitoring Permission: ");
    let test_tap = CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![CGEventType::MouseMoved],
        |_, _, _| None,
    );
    
    if test_tap.is_ok() {
        println!("{}", "✓ Enabled".green());
    } else {
        println!("{}", "✗ Disabled".red());
        println!("{}", "Please enable Input Monitoring permission in System Settings > Privacy & Security > Input Monitoring".red());
        return Ok(());
    }

    println!("\n{}", "All permissions granted! Starting recorder...".green());

    let ffmpeg_path = get_ffmpeg_path();
    println!("Using ffmpeg at: {}", ffmpeg_path.display().to_string().cyan());

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

                session.stop();
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

fn get_display_fingerprint() -> String {
    let displays = get_display_info();
    let mut display_strings: Vec<String> = displays
        .iter()
        .map(|d| format!("{}:{}x{}", d.id, d.original_width, d.original_height))
        .collect();
    display_strings.sort();
    display_strings.join(",")
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
    
    let target = match targets.iter()
        .find(|t| match t {
            Target::Display(d) => d.id == display_info.id,
            _ => false
        })
        .cloned() {
            Some(t) => t,
            None => {
                eprintln!("Could not find matching display target for ID: {}", display_info.id);
                return;
            }
        };

    let capturer = match initialize_capturer(&target) {
        Some(c) => c,
        None => return,
    };

    let (width, height) = match capturer.lock() {
        Ok(mut c) => {
            let sz = c.get_output_frame_size();
            (sz[0], sz[1])
        }
        Err(_) => return,
    };


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
            Ok(Ok(Frame::YUVFrame(frame))) => {
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
            Ok(Ok(_)) => {}

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
        output_type: FrameType::YUVFrame,
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

fn download_ffmpeg() -> Result<PathBuf> {
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let loggy_dir = home_dir.join(".loggy3");
    create_dir_all(&loggy_dir)?;
    
    let ffmpeg_path = loggy_dir.join("ffmpeg");
    
    if !ffmpeg_path.exists() {
        println!("Downloading ffmpeg binary...");
        
        let temp_path = loggy_dir.join("ffmpeg.downloading");
        
        let command = format!(
            "curl -L -o {} https://publicr2.standardinternal.com/ffmpeg_binaries/macos_arm/ffmpeg",
            temp_path.display()
        );
        
        let status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .status()
            .context("Failed to execute curl command")?;
            
        if !status.success() {
            return Err(anyhow::anyhow!("Failed to download ffmpeg binary"));
        }
        
        std::fs::rename(&temp_path, &ffmpeg_path)?;
        println!("Download complete");
        
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&ffmpeg_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&ffmpeg_path, perms)?;
    }
    
    Ok(ffmpeg_path)
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

fn initialize_ffmpeg(
    display_dir: &std::path::Path,
    width: usize,
    height: usize,
) -> Result<(Child, ChildStdin)> {
    let output_path = display_dir.join("chunk_%05d.mp4");
    let output_str = output_path.to_string_lossy().to_string();

    let ffmpeg_path = get_ffmpeg_path();

    let log_level = if VERBOSE.load(Ordering::SeqCst) {
        "info"
    } else {
        "error"
    };

    let mut child = Command::new(ffmpeg_path)
        .args(&[
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-color_range", "tv",
            "-s", &format!("{}x{}", width, height),
            "-r", "30",
            "-i", "pipe:0",
            "-c:v", FFMPEG_ENCODER,
            "-movflags", "+faststart",
            "-g", "60",
            "-f", "segment",
            "-segment_time", "60",
            "-reset_timestamps", "1",
            "-loglevel", log_level,
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
    frame: &YUVFrame,
    frames_log: &mut BufWriter<File>,
) -> Result<()> {
    ffmpeg_stdin.write_all(&frame.luminance_bytes)?;
    ffmpeg_stdin.write_all(&frame.chrominance_bytes)?;

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_millis();
    writeln!(frames_log, "{}", timestamp)?;
    frames_log.flush()?;

    Ok(())
}