use crate::classes::neuron::Neuron;
use crate::math::{sigmoid::{sigmoid,derived_sigmoid},mean_squared::{mean_squared,derived_mean_squared}};
use std::fs::File;
use std::io::prelude::*;
use rand::{Rng,rngs::ThreadRng};
pub struct FeedOut{
    pub out:Vec<f32>,
    pub activated:Vec<Vec<f32>>,
    pub base:Vec<Vec<f32>>
}
pub struct Network {
    layers:Vec<Vec<Neuron>>,
    learning_rate:f32
}

impl Network {
    pub fn new(layers:Vec<Vec<Neuron>>,learning_rate:f32) -> Network {
        Network {
            layers:layers,
            learning_rate:learning_rate
        }
    }
    pub fn feedforward(&self,input:&Vec<f32>) -> FeedOut{
        let mut data = input.clone();
        let mut activated = vec![];
        let mut base= vec![];
        for layer in self.layers.iter() {
            let mut temp_vec = vec![];
            let mut temp_base= vec![];
            for neuron in layer.iter() {
                let output = neuron.out_base(&data);
                let activ = sigmoid(output); 
                temp_base.push(output);
                temp_vec.push(activ);
            }
            data = temp_vec.clone();
            activated.push(temp_vec);
            base.push(temp_base);
        }
        FeedOut{
            out:data,
            activated:activated,
            base:base
        }
    }
    pub fn backprop(&mut self,errs:Vec<f32>,input:&Vec<f32>,activated:Vec<Vec<f32>>,base:Vec<Vec<f32>>) {
        let mut to_next:Vec<Vec<f32>> = vec![];

            for _ in 0..self.layers[0].len(){
                to_next.push(vec![]);
            } 
        for (num_n,neuron) in self.layers[1].iter_mut().enumerate() {
            let derived_out = derived_sigmoid(base[1][num_n]);
            let err = errs[num_n];
            let mut sum_next = vec![];
            for (num_w,weight) in neuron.weights.iter_mut().enumerate() {
                let to_next = *weight*derived_out*err; 
                sum_next.push(to_next);

                let change= derived_out*activated[0][num_w];
                *weight +=err*self.learning_rate*change; 
            }
            neuron.bias+=err*self.learning_rate*derived_out;
            for (i,val) in sum_next.iter().enumerate() {
                to_next[i].push(*val);
            }
        }

        for (num_n,neuron) in self.layers[0].iter_mut().enumerate() {
            let derived_out = derived_sigmoid(base[0][num_n]);
            let err = to_next[num_n].iter().fold(0.0,|acc,val|acc+val); 
            for (num_w,weight) in neuron.weights.iter_mut().enumerate() {
                let change= derived_out*input[num_w];
                *weight +=err*self.learning_rate*change; 
            }
            neuron.bias+=err*self.learning_rate*derived_out;
        }
    }
    pub fn train(&mut self,answers:Vec<Vec<f32>>,inputs:Vec<Vec<f32>>,cycles:usize,rng:&mut ThreadRng) {
        let mut csv_string = String::new();
        for cycle in 0..cycles {
            let round = rng.gen_range(0.. inputs.len());
            let answer = &answers[round];
            let input = &inputs[round];

            let output = self.feedforward(input); 
            let predictions = output.out;

           let mut errs = vec![]; 
           for (i,prediction) in predictions.iter().enumerate() {
               let err = derived_mean_squared(*prediction,answer[i]);
               errs.push(err);
           }

           let activated = output.activated;
           let base = output.base;
           self.backprop(errs,input,activated,base);

            if cycle % 1000 == 0 {
                let mut sum = 0.0;
                let input_length = inputs.len();
                let answer_length = answers[0].len();
                let f_ans_len = answer_length as f32;
                for i in 0..input_length{
                    let output = self.feedforward(&inputs[i]);
                    let predictions = output.out;

                    let mut temp_sum = 0.0;
                    for a in 0..answer_length{
                        temp_sum+= mean_squared(predictions[a],answers[i][a]);
                    }
                    sum+=temp_sum/f_ans_len;
                }
                let err = sum/(input_length as f32);
                csv_string+=&format!("{},{}\n",cycle,err);
                println!("Err: {}",err);
            }
        }
        let mut file = File::create("random.csv").unwrap();
        file.write_all(csv_string.as_bytes()).unwrap();
    }
}
