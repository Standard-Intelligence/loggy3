mod platform;

#[cfg(target_os = "macos")]
use platform::mac;

#[cfg(target_os = "windows")]
use platform::windows;

fn main() {
    #[cfg(target_os = "macos")]
    mac::main().unwrap();

    #[cfg(target_os = "windows")]
    windows::main().unwrap();
}