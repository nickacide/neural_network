mod layer;
mod neural_network;

use neural_network::*;
use simple_matrix::Matrix;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    fn softplus(value: &Matrix<f32>) -> Matrix<f32> {
        // let mut result: Vec<f32> = vec![];
        // value.apply(|v| result.push((1. + v.exp()).ln()));
        // Matrix::from_iter(value.rows(), value.cols(), result)
        let mut clone = value.clone();
        clone.apply_mut(|v| *v = (1. + v.exp()).ln());
        clone
    }
    fn softplus_derivative(value: &Matrix<f32>) -> Matrix<f32> {
        // let mut result: Vec<f32> = vec![];
        let mut clone = value.clone();
        clone.apply_mut(|v| *v = v.exp() / (1. + v.exp()));
        // Matrix::from_iter(value.rows(), value.cols(), result)
        clone
    }

    let net = NeuralNetwork::new(1, vec![2], 1, softplus, softplus_derivative);

    let out = net.out(&Matrix::from_iter(1, 1, vec![1.]));
    dbg!(out);

    let sample1 = (
        Matrix::from_iter(1, 1, vec![1.]),
        Matrix::from_iter(1, 1, vec![3.]),
    );
    // let sample2 = (
    //     Matrix::from_iter(1, 1, vec![1.]),
    //     Matrix::from_iter(1, 1, vec![2.]),
    // );
    // let sample3 = (
    //     Matrix::from_iter(1, 1, vec![1.]),
    //     Matrix::from_iter(1, 1, vec![2.]),
    // );
    // let sample4 = (
    //     Matrix::from_iter(1, 1, vec![1.]),
    //     Matrix::from_iter(1, 1, vec![2.]),
    // );

    let training_data = vec![sample1];

    let cost = net.cost(&training_data);

    dbg!(cost);
}
