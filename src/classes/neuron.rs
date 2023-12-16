use crate::math::sigmoid::sigmoid;
use crate::misc::weight_funcs::dot_vecs;
#[derive(Debug)]
pub struct Neuron {
    pub weights:Vec<f32>,
    pub bias:f32
}
impl Neuron {
    #[allow(dead_code)]
    pub fn new(weights:Vec<f32>,bias:f32) -> Neuron {
        Neuron{
            weights:weights,
            bias:bias
        }
    }
    #[allow(dead_code)]
    pub fn out_base(&self,input:&Vec<f32>) -> f32 {
       let output = dot_vecs(&self.weights,input); 
       output+self.bias
    }
    #[allow(dead_code)]
    pub fn activate(&self,input:&Vec<f32>) -> f32 {
       let output = dot_vecs(&self.weights,input); 
       sigmoid(output+self.bias)
    }
}
