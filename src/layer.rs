use simple_matrix::Matrix;

#[derive(Clone)]
pub enum LayerType {
    InputLayer(Matrix<f32>),
    HiddenLayer(Matrix<f32>, Matrix<f32>),
    OutputLayer(Matrix<f32>, Matrix<f32>)
}

pub struct Layer {
    layer_type: LayerType
}

impl Layer {
    pub fn new(layer_type: LayerType) -> Self {
        Layer { layer_type }
    }

    pub fn height(&self) -> usize {
        match &self.layer_type {
            LayerType::InputLayer(m) => m.cols(),
            LayerType::HiddenLayer(m, _) => m.cols(),
            LayerType::OutputLayer(m, _) => m.cols()
        }
    }

    pub fn weights_biases(&self) -> (Matrix<f32>, Matrix<f32>) {
        match &self.layer_type {
            LayerType::InputLayer(_) => panic!("Input Layer has no weights or biases"),
            LayerType::HiddenLayer(w, b) => (w.clone(), b.clone()),
            LayerType::OutputLayer(w, b) => (w.clone(), b.clone())
        }
    }
}