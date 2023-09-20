pub mod app_wrapper;

use app_wrapper::OnnxWrapper;
use pyo3::prelude::*;

#[pymodule]
fn app_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OnnxWrapper>()?;

    Ok(())
}
