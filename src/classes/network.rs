use crate::classes::neuron::Neuron;
use serde_json;
use crate::math::{sigmoid::{sigmoid,derived_sigmoid}
    ,mean_squared::{mean_squared
        ,derived_mean_squared}
    ,get_maks::get_ind_max
        ,softmax::softmax};
use std::fs::File;
use std::io::prelude::*;
use rand::{Rng,rngs::ThreadRng,seq::SliceRandom};

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
        let input_layer = &self.layers[0];
        let mut input_activ = vec![];
        let mut input_base= vec![];

        for neuron in input_layer{
           let input_base_out = neuron.out_base(input); 
           let input_activ_out = sigmoid(input_base_out);
           input_activ.push(input_activ_out);
           input_base.push(input_base_out);
        }

        let output_layer = &self.layers[1];
        let mut output_base= vec![];
        for neuron in output_layer {
            let output_base_out = neuron.out_base(&input_activ);
           output_base.push(output_base_out);
        }
        let output_activ = softmax(&output_base); 

        let activated = vec![input_activ,output_activ.clone()];

        let base = vec![input_base,output_base];

        FeedOut{
            out:output_activ,
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
            //let derived_out = derived_sigmoid(base[1][num_n]);
            let err = errs[num_n];
            let mut sum_next = vec![];
            for (num_w,weight) in neuron.weights.iter_mut().enumerate() {
                let to_next = *weight*err; 
                sum_next.push(to_next);

                let change= activated[0][num_w];
                *weight -=err*self.learning_rate*change; 
            }
            neuron.bias-=self.learning_rate*err;
            for (i,val) in sum_next.iter().enumerate() {
                to_next[i].push(*val);
            }
        }

        for (num_n,neuron) in self.layers[0].iter_mut().enumerate() {
            let derived_out = derived_sigmoid(base[0][num_n]);
            let err = to_next[num_n].iter().fold(0.0,|acc,val|acc+val); 
            for (num_w,weight) in neuron.weights.iter_mut().enumerate() {
                let change= derived_out*input[num_w];
                *weight -=err*self.learning_rate*change; 
            }
            neuron.bias-=err*self.learning_rate*derived_out*err;
        }
    }
    pub fn train(&mut self,answers:Vec<Vec<f32>>,inputs:Vec<Vec<f32>>,cycles:usize,rng:&mut ThreadRng) {
        let mut csv_string = String::from("Cycles,Err,Accuracy\n");
        let batch_size = 10;
        let f_batch_size = batch_size as f32;
        let mut temp_random:Vec<usize> =(0..inputs.len()).collect(); 
        temp_random.shuffle(rng);
        let mut batch = &temp_random[0..batch_size];
        for cycle in 0..cycles {
            let mut errs = vec![0.0;10]; 
            let mut full_activated = vec![];
            let mut full_base = vec![];
            let mut full_input = vec![0.0;inputs[0].len()];
            for temp_l in 0..self.layers.len() {
                let mut temp_vec = vec![0.0;self.layers[temp_l].len()];
                full_activated.push(temp_vec.clone());
                full_base.push(temp_vec);
            }
            for batch_num in batch {

                let answer = &answers[*batch_num];
                let input = &inputs[*batch_num];

                let output = self.feedforward(input); 
                let predictions = output.out;
                let real_predictions = predictions;

                for (i,prediction) in real_predictions.iter().enumerate() {
                    let err = derived_mean_squared(*prediction,answer[i]);
                    errs[i]+=err/f_batch_size;
                }
                for inst in 0..input.len() {
                   full_input[inst]+=input[inst]/f_batch_size; 
                }

                let activated = output.activated;
                let base = output.base;
                for ind in 0..activated.len() {
                    for ind2 in 0..activated[ind].len() {
                        full_activated[ind][ind2]+=activated[ind][ind2]/f_batch_size;

                        full_base[ind][ind2]+=base[ind][ind2]/f_batch_size;
                    }
                }
           }

           self.backprop(errs,&full_input,full_activated,full_base);

           temp_random.shuffle(rng);
           batch = &temp_random[0..batch_size];

            if cycle % 1000 == 0 {
                let mut sum = 0.0;
                let input_length = inputs.len();
                let answer_length = answers[0].len();
                let f_ans_len = answer_length as f32;
                let mut amount_right = 0.0;
                for i in 0..input_length{
                    let output = self.feedforward(&inputs[i]);
                    let predictions = output.out;
                    let real_predictions = predictions;
                    

                    let mut temp_sum = 0.0;
                    for a in 0..answer_length{
                        temp_sum+= mean_squared(real_predictions[a],answers[i][a]);
                    }
                    let ans_ind = get_ind_max(&answers[i]);
                    let pred_ind = get_ind_max(&real_predictions);

                    if i == 0 {
                        println!("{:?}",ans_ind);
                        println!("{:?}",pred_ind);
                        println!("{:?}",real_predictions);

                        let pred_sum_sum = real_predictions.iter().fold(0.0,|acc,val|acc+val);

                        println!("Pred sum: {}",pred_sum_sum);
                    }
                    if ans_ind == pred_ind {
                        amount_right+=1.0;
                    }
                    sum+=temp_sum/f_ans_len;
                }
                let proc_err = (amount_right/(input_length as f32))*100.0; 
                let err = sum/(input_length as f32);
                let layer_string:String  = serde_json::to_string(&self.layers).unwrap();
                let mut json_file= File::create("weights.json").unwrap();
                json_file.write_all(layer_string.as_bytes()).unwrap(); 
                csv_string+=&format!("{},{},{}\n",cycle,err,proc_err);
                let mut file = File::create("random.csv").unwrap();
                file.write_all(csv_string.as_bytes()).unwrap();
                println!("Cycle: {}\nErr: {}\nAccuracy: {}\n",cycle,err,proc_err);
            }
        }
    }
}
