use anyhow::{Context, Result};
use chrono::Local;
use dirs;
use platform::{DisplayInfo, FFMPEG_ENCODER};
use ctrlc::set_handler;
use std::{
    fs::{create_dir_all, File},
    io::{BufWriter, Write, BufReader, BufRead},
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

mod platform;


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
        let displays = platform::get_display_info();
        if displays.is_empty() {
            return Ok(None);
        }

        let home_dir = dirs::home_dir().context("Could not determine home directory")?;
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let session_dir = home_dir.join(format!("loggy3/session_{}", timestamp));
        create_dir_all(&session_dir)?;

        println!("Starting new session at: {}", session_dir.display());

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
            platform::unified_event_listener_thread(
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

fn main() -> Result<()> {
    let ffmpeg_path = get_ffmpeg_path();
    println!("Using ffmpeg at: {}", ffmpeg_path.display());

    let should_run = Arc::new(AtomicBool::new(true));

    let sr_for_signals = should_run.clone();
    thread::spawn(move || {
        let (tx, rx) = channel();
        
        set_handler(move || tx.send(()).expect("Could not send signal on channel."))
            .expect("Error setting Ctrl-C handler");
        
        println!("Waiting for Ctrl-C...");
        rx.recv().expect("Could not receive from channel.");
        println!("Got it! Exiting..."); 
        
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
    let displays = platform::get_display_info();
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
        thread::spawn(move || {
            let _ = std::io::copy(&mut BufReader::new(stdout), &mut std::io::sink());
        });
    }

    if let Some(stderr) = ffmpeg_child.stderr.take() {
        let display_id = display_info.id;
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    eprintln!("FFmpeg (display {}): {}", display_id, line);
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
                eprintln!("Frame timeout on display {} - ignoring due to idle display", display_info.id);
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

fn get_ffmpeg_path() -> PathBuf {
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
            "-loglevel", "error",
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