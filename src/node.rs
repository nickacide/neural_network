#[derive(Debug, Clone)]
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub layer: usize,
}

impl Node {
    pub fn new(weights: Vec<f32>, bias: f32, layer: usize) -> Self {
        Self {
            weights,
            bias,
            layer,
        }
    }

    pub fn many(count: usize, weights: Vec<f32>, bias: f32, layer: usize) -> Vec<Self> {
        vec![Node::new(weights, bias, layer); count]
    }
}
