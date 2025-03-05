use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::path::Path;

mod tokens;

#[pyfunction]
fn chunk_tokenizer(chunk_path: &str, os_type: &str) -> PyResult<Vec<(usize, usize)>> {
    let path = Path::new(chunk_path);
    Ok(tokens::chunk_tokenizer(path, os_type))
}

#[pyfunction]
fn token_to_readable(token: usize) -> PyResult<String> {
    Ok(tokens::token_to_readable(token))
}

#[pyfunction]
fn print_token_sequence(tokens: Vec<(usize, usize)>) -> PyResult<()> {
    tokens::print_token_sequence(&tokens);
    Ok(())
}

#[pymodule]
fn loggy3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(token_to_readable, m)?)?;
    m.add_function(wrap_pyfunction!(print_token_sequence, m)?)?;
    Ok(())
} 