use std::path::PathBuf;

use app::onnx_runner::OnnxRunner;
use pyo3::{prelude::*, types::PyType};

#[pyclass]
pub struct OnnxWrapper {
    onnx_model: OnnxRunner,

}

#[pymethods]
impl OnnxWrapper {
    #[classmethod]
    pub fn from_file(_cls: &PyType, config_file: String) -> PyResult<Self> {
        let config_path = PathBuf::from(config_file.clone());
        let onnx_model = OnnxRunner::new(&config_path);
        Ok(OnnxWrapper { onnx_model })
    }

    pub fn run(&self, x: f64, y: f64) -> PyResult<f64> {
        let result = self.onnx_model.run(x, y);
        Ok(result)
    }
}
