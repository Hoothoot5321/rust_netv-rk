use rand::{Rng,rngs::ThreadRng};
use crate::classes::neuron::Neuron;

#[allow(dead_code)]
pub fn gen_weight(amount:i32,rng: &mut ThreadRng ) -> Vec<f64> {
    let mut returner:Vec<f64> = vec![];
    for _ in 0..amount {
            let rand:f64 = rng.gen_range(-0.5..0.5); 
            returner.push(rand);
    }
    return returner;
}

#[allow(dead_code)]
pub fn gen_layer(neurons:i32,weights:i32,rng:&mut ThreadRng)->Vec<Neuron> {
    let mut neu_vec:Vec<Neuron> = vec![];
    for _ in 0..neurons {
        let bias:f64 = rng.gen_range(-0.5..0.5);
        let weight = gen_weight(weights,rng);
        let neuron = Neuron::new(weight,bias);
        neu_vec.push(neuron);
    }    
    neu_vec
}


#[allow(dead_code)]
pub fn mult_weights(weight1:&Vec<f64>,weight2:&Vec<f64>) -> Vec<f64> {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1*val2).collect::<Vec<f64>>()
}


#[allow(dead_code)]
pub fn update_weight(weight1:&Vec<f64>,weight2:&Vec<f64>) -> Vec<f64> {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1-val2).collect::<Vec<f64>>()
}

#[allow(dead_code)]
pub fn dot_vecs(weight1:&Vec<f64>,weight2:&Vec<f64>) -> f64 {
    weight1.iter().zip(weight2).map(|(val1,val2)| val1*val2).fold(0.0,|acc,val|acc+val)
}
