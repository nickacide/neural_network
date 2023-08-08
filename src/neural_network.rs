use crate::layer::*;
use simple_matrix::Matrix;

#[allow(dead_code)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    activation: fn(&Matrix<f32>) -> Matrix<f32>,
    activation_derivative: fn(&Matrix<f32>) -> Matrix<f32>,
}
impl NeuralNetwork {
    pub fn new(
        input_size: usize,
        hidden_size: Vec<usize>,
        output_size: usize,
        activation: fn(&Matrix<f32>) -> Matrix<f32>,
        activation_derivative: fn(&Matrix<f32>) -> Matrix<f32>,
    ) -> Self {
        let mut layers: Vec<Layer> = vec![];
        let input_layer = Layer::new(LayerType::InputLayer(Matrix::from_iter(1, input_size, vec![0.; input_size])));
        layers.push(input_layer);

        for (index, width) in hidden_size.iter().enumerate() {
            let height = layers[index].height();
            let weights = Matrix::from_iter(height, *width, vec![1.; height * width]);
            let biases = Matrix::from_iter(1, *width, vec![0.; *width]);
            let hidden_layer = Layer::new(LayerType::HiddenLayer(weights, biases));
            layers.push(hidden_layer);
        }

        let height = layers.last().expect("Invalid Neural Network").height();
        let output_layer = Layer::new(LayerType::OutputLayer(Matrix::from_iter(height, output_size, vec![1.; height * output_size]), Matrix::from_iter(1, output_size, vec![0.; output_size])));
        layers.push(output_layer);

        Self {
            layers,
            activation,
            activation_derivative,
        }
    }

    pub fn out(&self, inputs: &Matrix<f32>) -> Matrix<f32> {
        let mut previous: Matrix<f32> = inputs.clone();
        let hidden_layers = &self.layers[1..&self.layers.len() - 1];
        // dbg!(&self.layers);
        // dbg!(hidden_layers);
        for hidden_layer in hidden_layers {
            // let mut new: Vec<f32> = vec![];
            let (weights, biases) = hidden_layer.weights_biases();
            previous = previous * weights + biases;
            previous = (self.activation)(&previous);
            // for node in &hidden_layer.nodes {
            //     let mut sum = 0.;
            //     for (weight_index, weight) in node.weights.iter().enumerate() {
            //         sum += previous[weight_index] * weight;
            //     }
            //     sum += node.bias;
            //     new.push((self.activation)(sum));
            // }
            // dbg!(previous);
            // previous = new;
        }
        let output = self.layers.last().expect("Invalid Neural Network");
        let (weights, biases) = output.weights_biases();
        previous * weights + biases
        // let mut outputs: Vec<f32> = vec![];
        // for node in &self.layers.last().unwrap().nodes {
        //     let mut sum = 0.;
        //     for (weight_index, weight) in node.weights.iter().enumerate() {
        //         sum += previous[weight_index] * weight;
        //     }
        //     sum += node.bias;
        //     outputs.push(sum);
        // }
        // outputs
    }

    pub fn cost(&self, training_data: &Vec<(Matrix<f32>, Matrix<f32>)>) -> Matrix<f32> {
        let first = &training_data[0].1;
        let mut costs = Matrix::from_iter(first.rows(), first.cols(), vec![0.; first.len()]);
        for (x, y) in training_data {
            let predicted = self.out(x);
            let cost = (&predicted - y) * (&predicted - y);
            costs += cost;
        }
        costs.apply_mut(|v| *v /= training_data.len() as f32);        
        costs
    }

    // pub fn cost(&self, training_data: &Vec<(Matrix<f32>, Matrix<f32>)>) -> Vec<f32> {
    //     let (rows, c)
    //     let mut costs: Matrix<f32> = Matrix::from_iter(tra, cols, data);
    //     for (x, y) in training_data {
    //         let predicted = self.out(x);
    //         for (node_index, predicted_node) in predicted.iter().enumerate() {
    //             let node_cost = (predicted_node - y[node_index]).powi(2);
    //             costs[node_index] += node_cost;
    //         }
    //     }
    //     costs
    //         .iter()
    //         .map(|c| c / training_data.len() as f32)
    //         .collect()
    // }
}
