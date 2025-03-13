#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use pyo3::prelude::*;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use pyo3::wrap_pyfunction;
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
use std::path::Path;

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod tokens;

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[pyfunction]
fn chunk_tokenizer(chunk_path: &str, os_type: Option<&str>) -> PyResult<Vec<(usize, usize)>> {
    let path = Path::new(chunk_path);
    Ok(tokens::chunk_tokenizer(path, os_type))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[pyfunction]
fn token_to_readable(token: usize) -> PyResult<String> {
    Ok(tokens::token_to_readable(token))
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[pyfunction]
fn print_token_sequence(tokens: Vec<(usize, usize)>) -> PyResult<()> {
    tokens::print_token_sequence(&tokens);
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
#[pymodule]
fn loggy3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(token_to_readable, m)?)?;
    m.add_function(wrap_pyfunction!(print_token_sequence, m)?)?;
    Ok(())
}

// Provide an empty library implementation for macOS and Windows
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub fn empty_function() {
    // This is just to provide some exported symbol for non-Python platforms
    println!("This function does nothing and is never called.");
}

// For macOS and Windows, we need to have a cdylib export
#[cfg(any(target_os = "macos", target_os = "windows"))]
#[no_mangle]
pub extern "C" fn dummy_exported_symbol() {
    // This function exists solely to provide a symbol for the cdylib
} 