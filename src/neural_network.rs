use crate::layer::*;
use crate::node::*;

#[allow(dead_code)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    activation: fn(f32) -> f32,
    activation_derivative: fn(f32) -> f32,
}

impl NeuralNetwork {
    pub fn new(
        input_size: usize,
        hidden_size: Vec<usize>,
        output_size: usize,
        activation: fn(f32) -> f32,
        activation_derivative: fn(f32) -> f32,
    ) -> Self {
        let mut layers: Vec<Layer> = vec![];
        layers.push(Layer {
            nodes: Node::many(input_size, vec![], 0., 0),
        });
        for (index, hidden) in hidden_size.iter().enumerate() {
            layers.push(Layer {
                nodes: Node::many(*hidden, vec![1.; layers[index].nodes.len()], 0., index + 1),
            });
        }
        layers.push(Layer {
            nodes: Node::many(
                output_size,
                vec![1.; layers.last().expect("Invalid Neural Network").nodes.len()],
                0.,
                layers.len(),
            ),
        });

        Self {
            layers,
            activation,
            activation_derivative,
        }
    }

    pub fn out(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut previous: Vec<f32> = inputs.clone();
        let hidden_layers = &self.layers[1..&self.layers.len() - 1];
        // dbg!(&self.layers);
        // dbg!(hidden_layers);
        for hidden_layer in hidden_layers {
            let mut new: Vec<f32> = vec![];
            for node in &hidden_layer.nodes {
                let mut sum = 0.;
                for (weight_index, weight) in node.weights.iter().enumerate() {
                    sum += previous[weight_index] * weight;
                }
                sum += node.bias;
                new.push((self.activation)(sum));
            }
            // dbg!(previous);
            previous = new;
        }
        let mut outputs: Vec<f32> = vec![];
        for node in &self.layers.last().unwrap().nodes {
            let mut sum = 0.;
            for (weight_index, weight) in node.weights.iter().enumerate() {
                sum += previous[weight_index] * weight;
            }
            sum += node.bias;
            outputs.push(sum);
        }
        outputs
    }

    pub fn cost(&self, training_data: &Vec<(Vec<f32>, Vec<f32>)>) -> Vec<f32> {
        let mut costs: Vec<f32> = vec![0.; training_data[0].1.len()];
        for (x, y) in training_data {
            let predicted = self.out(x);
            for (node_index, predicted_node) in predicted.iter().enumerate() {
                let node_cost = (predicted_node - y[node_index]).powi(2);
                costs[node_index] += node_cost;
            }
        }
        costs
            .iter()
            .map(|c| c / training_data.len() as f32)
            .collect()
    }
}
