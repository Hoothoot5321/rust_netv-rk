use crate::math::sigmoid::sigmoid;
use crate::misc::weight_funcs::dot_vecs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Neuron {
    pub weights:Vec<f64>,
    pub bias:f64
}
impl Neuron {
    #[allow(dead_code)]
    pub fn new(weights:Vec<f64>,bias:f64) -> Neuron {
        Neuron{
            weights:weights,
            bias:bias
        }
    }
    #[allow(dead_code)]
    pub fn out_base(&self,input:&Vec<f64>) -> f64 {
       let output = dot_vecs(&self.weights,input); 
       output+self.bias
    }
    #[allow(dead_code)]
    pub fn activate(&self,input:&Vec<f64>) -> f64 {
       let output = dot_vecs(&self.weights,input); 
       sigmoid(output+self.bias)
    }
}
