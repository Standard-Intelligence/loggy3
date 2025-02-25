use platform;


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

    // Initialize capturer
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
        return;
    }

    // Set up frames log in the current chunk's display directory
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
    let mut total_frame_count = 0;
    let mut chunk_frame_count = 0;
    let mut last_status = Instant::now();
    
    // Start first ffmpeg process
    let mut ffmpeg_process = start_new_ffmpeg_process(&display_dir, width.try_into().unwrap(), height.try_into().unwrap(), display_info.id);
    if ffmpeg_process.is_none() {
        eprintln!("Failed to start initial ffmpeg process for display {}", display_info.id);
        return;
    }
    
    // Print initial status message
    let status_indicator = format!("[Display {}]", display_info.title);
    println!("{} Started recording", status_indicator.cyan());
    
    // How often to check for coordination signals
    let signal_check_interval = Duration::from_millis(100);
    
    while should_run.load(Ordering::SeqCst) {
        // Try to get a frame with a short timeout
        let frame_result = {
            // Create a time-limited attempt to get the lock
            match capturer.try_lock() {
                Ok(capturer_guard) => {
                    // We have the lock, get the frame
                    capturer_guard.get_next_frame().map_err(|e| anyhow::anyhow!("Frame error: {}", e))
                },
                Err(_) => {
                    // Couldn't get the lock, sleep briefly and check signals
                    thread::sleep(signal_check_interval);
                    // Return a special "busy" error that we can handle differently
                    Err(anyhow::anyhow!("Capturer busy"))
                }
            }
        };

        match frame_result {
            Ok(Frame::YUVFrame(frame)) => {
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
                        handle_capture_error(&error_tx);
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
                            handle_capture_error(&error_tx);
                            break;
                        }
                    }
                    
                    ffmpeg_process = start_new_ffmpeg_process(&display_dir, width.try_into().unwrap(), height.try_into().unwrap(), display_info.id);
                    if ffmpeg_process.is_none() {
                        eprintln!("Failed to start new ffmpeg process for display {}", display_info.id);
                        handle_capture_error(&error_tx);
                        break;
                    }
                    
                    println!("{} {}", 
                        status_indicator.cyan(),
                        format!("Started new chunk {}", current_chunk_index).green()
                    );
                }
            
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
                    if let Err(e) = write_frame(stdin, &frame, &mut frames_log) {
                        eprintln!("Write error for display {}: {}", display_info.id, e);
                        break;
                    }
                } else {
                    eprintln!("No active ffmpeg process to write frame for display {}", display_info.id);
                    break;
                }
            }
            Ok(_) => {
                // Non-YUV frame, can ignore
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Capturer busy") {
                    // Capturer is just busy, check signals and continue
                    continue;
                } else {
                    // For any other error, just log and continue
                    // If this is a persistent issue, the session manager will handle it
                    if error_msg.contains("Timeout") || error_msg.contains("No frames available") {
                        // These are common errors, no need to log
                    } else {
                        // Log uncommon errors
                        eprintln!("Frame error on display {}: {}", display_info.id, e);
                    }
                    
                    // Check if we should still be running
                    if !should_run.load(Ordering::SeqCst) {
                        break;
                    }
                    
                    // Sleep briefly before trying again
                    thread::sleep(signal_check_interval);
                    
                    // Any persistent error will cause the session manager to eventually restart
                    // Signal an error if we haven't gotten a frame for a long time
                    if total_frame_count == 0 && start_time.elapsed() > Duration::from_secs(10) {
                        eprintln!("No frames captured after 10 seconds. Signaling error.");
                        handle_capture_error(&error_tx);
                        break;
                    }
                    continue;
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
            "-pix_fmt", "nv12",
            "-color_range", "tv",
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
                    if let Some(asset) = release.assets.iter().find(|a| a.name == "loggy3") {
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
    
    // Get the home dir to install to
    let home_dir = dirs::home_dir().context("Could not determine home directory")?;
    let install_dir = home_dir.join(".local/bin");
    create_dir_all(&install_dir)?;
    let target_path = install_dir.join("loggy3");
    
    // Create a temporary file for the download
    let temp_path = target_path.with_extension("new");
    
    // Download the new version
    let response = ureq::get(download_url)
        .call()
        .context("Failed to download update")?;
    
    let mut file = File::create(&temp_path).context("Failed to create temporary file")?;
    let mut buffer = Vec::new();
    response.into_reader().read_to_end(&mut buffer).context("Failed to read response")?;
    file.write_all(&buffer).context("Failed to write to temporary file")?;
    
    // Make the new version executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&temp_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&temp_path, perms)?;
    }
    
    // Try direct file replacement first
    println!("{}", "Installing update...".cyan());
    
    // On unix, we can just replace the executable directly since we have permission 
    // to files in our own home directory
    if let Err(e) = std::fs::rename(&temp_path, &target_path) {
        if VERBOSE.load(Ordering::SeqCst) {
            eprintln!("Failed to rename file directly: {}", e);
            eprintln!("Falling back to delayed update");
        }
        
        // Create a bash script to replace the executable on next run
        let script_path = temp_path.with_extension("update.sh");
        let script_content = format!(
            r#"#!/bin/bash
# Wait for 1 second
sleep 1
# Replace the executable
mv "{}" "{}"
echo "Update complete! Please run 'loggy3' to start the updated version."
# Clean up
rm -f "$0"
"#,
            temp_path.display(),
            target_path.display()
        );
        
        let mut script_file = File::create(&script_path)?;
        script_file.write_all(script_content.as_bytes())?;
        
        // Make the script executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        // Execute the update script
        Command::new(&script_path).spawn()?;
        
        println!("{}", "Update staged! The update will complete when this program exits.".green());
        println!("{}", "After you close this application, run 'loggy3' to start the updated version.".cyan());
    } else {
        println!("{}", "✓ Update installed successfully!".green());
        println!("{}", "Please restart the application to use the new version.".cyan());
    }
    
    // Exit the program after a successful update
    println!("{}", "Please restart loggy3 to use the new version.".bright_green().bold());
    exit(0);
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
            },
            "screen_recording_permission": check_screen_recording_access(),
            "input_monitoring_permission": check_input_monitoring_access(),
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
    let args: Vec<String> = std::env::args().collect();
    let verbose_mode = args.iter().any(|arg| arg == "--verbose" || arg == "-v");
    
    if verbose_mode {
        VERBOSE.store(true, Ordering::SeqCst);
    }
    
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

    println!("{} {}", "\nLoggy3 Screen Recorder".bright_green().bold(), 
              format!("v{}", CURRENT_VERSION).bright_cyan());
    println!("{}", "======================".bright_green());

    if VERBOSE.load(Ordering::SeqCst) {
        println!("{}", "Verbose output enabled".yellow());
    }
    
    // Check for updates
    if !AUTO_UPDATES_DISABLED.load(Ordering::SeqCst) {
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
            println!("\nWould you like to update now? [Y/n] ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            
            match input.trim().to_lowercase().as_str() {
                "y" | "yes" | "" => {
                    // User wants to update
                    update_to_new_version(&download_url)?;
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

    platform::check_permissions()?;

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
