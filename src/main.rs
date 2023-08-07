mod layer;
mod neural_network;
mod node;

use neural_network::*;

fn main() {
    fn softplus(value: f32) -> f32 {
        (1. + value.exp()).ln()
    }
    fn softplus_derivative(value: f32) -> f32 {
        value.exp() / (1. + value.exp())
    }

    let net = NeuralNetwork::new(1, vec![2], 1, softplus, softplus_derivative);

    let out = net.out(&vec![1.])[0];
    println!("{out}");

    let training_data: Vec<(Vec<f32>, Vec<f32>)> = vec![(vec![1.], vec![2.]), (vec![0.], vec![0.])];
    let cost = net.cost(&training_data);

    println!("Cost: {}", cost[0]);
}
