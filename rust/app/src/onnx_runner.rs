use std::path::PathBuf;

use ndarray::CowArray;
use ort::{
    tensor::OrtOwnedTensor, Environment, GraphOptimizationLevel, Session, SessionBuilder, Value,
};

pub struct OnnxRunner {
    pub session: Session,
}

impl OnnxRunner {
    pub fn new(onnx_model_path: &PathBuf) -> OnnxRunner {
        let env = Environment::builder().build().unwrap().into_arc();

        let session = SessionBuilder::new(&env)
            .unwrap()
            .with_intra_threads(1)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_model_from_file(onnx_model_path)
            .unwrap();

        OnnxRunner { session }
    }

    pub fn run(&self, x: f64, y: f64) -> f64 {
        let array = ndarray::Array1::from(vec![x as f32, y as f32])
            .into_shape((1, 2))
            .unwrap();

        let x = CowArray::from(array).into_dyn();
        let value = Value::from_array(self.session.allocator(), &x).unwrap();
        let input = vec![value];

        let result: OrtOwnedTensor<f32, _> =
            self.session.run(input).unwrap()[0].try_extract().unwrap();
        let output_f64 = result.view().to_owned().into_raw_vec()[0] as f64;
        output_f64
    }
}
