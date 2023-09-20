use std::path::PathBuf;

use app::onnx_runner::OnnxRunner;

use clap::Parser;

#[derive(Parser, Debug)]
pub struct AppArgs {
    pub onnx_file: PathBuf,
}
fn main() {
    let args = AppArgs::parse();
    let runner = OnnxRunner::new(&args.onnx_file);

    let result = runner.run(50.0, 0.0);
    println!("Result: {}", result);
}
