use crate::classes::neuron::Neuron;
use std::fs;
use std::time::Instant;
use serde_json;
use crate::math::{sigmoid::{sigmoid,derived_sigmoid}
    ,mean_squared::{mean_squared
        ,derived_mean_squared}
    ,get_maks::get_ind_max
        ,softmax::softmax,
    relu::{relu,derived_relu}};
use std::fs::File;
use std::io::prelude::*;
use rand::{Rng,rngs::ThreadRng,seq::SliceRandom};

pub struct FeedOut{
    pub out:Vec<f64>,
    pub activated:Vec<Vec<f64>>,
    pub base:Vec<Vec<f64>>
}
pub struct Network {
    layers:Vec<Vec<Neuron>>,
    start_learning_rate:f64,
    learning_rate:f64
}

impl Network {
    pub fn new(layers:Vec<Vec<Neuron>>,learning_rate:f64) -> Network {
        Network {
            layers:layers,
            start_learning_rate:learning_rate,
            learning_rate:learning_rate
        }
    }
    pub fn feedforward(&self,input:&Vec<f64>) -> FeedOut{

        let mut data = input.to_vec();
        let mut activ_out_arr= vec![];
        let mut activated = vec![];
        let mut base= vec![];
        for (num,layer) in self.layers.iter().enumerate()  {
            let mut temp_base = vec![]; 
            let mut temp_activ = vec![]; 
            for neuron in layer {

                let base_out = neuron.out_base(&data);
                temp_base.push(base_out);
                if num != &self.layers.len()-1 {

                    let activ_out= relu(base_out);

                    activ_out_arr.push(activ_out);

                    temp_activ.push(activ_out);
                }
                else {

                    let activ_out= base_out;

                    activ_out_arr.push(activ_out);

                    temp_activ.push(activ_out);
                    /*
                    let activ_out = softmax(&data);
                    activ_out_arr = activ_out.clone();
                    temp_activ= activ_out;
                    */
                }
            }
            let cloned = activ_out_arr.clone();
            data  = cloned; 
            activ_out_arr = vec![];
            base.push(temp_base);
            activated.push(temp_activ);
        }
        let out_data = softmax(&data);
        FeedOut{
            out:out_data,
            activated:activated,
            base:base
        }
    }
    pub fn backprop(&mut self,errs:Vec<f64>,input:&Vec<f64>,activated:Vec<Vec<f64>>,base:Vec<Vec<f64>>) {
        let mut to_next:Vec<Vec<f64>> = vec![];
        let mut temp_to_next:Vec<Vec<f64>> = vec![];
        let neuron_amounts:Vec<usize> = self.layers.iter().map(|val|val.len()).collect();


        for (layer_n,layer) in self.layers.iter_mut().enumerate().rev() {
            to_next = temp_to_next;
            if layer_n != 0{
            temp_to_next = vec![vec![];neuron_amounts[layer_n-1]];
            }
            else {
                temp_to_next = vec![];
            }
            for (num_n,neuron) in layer.iter_mut().enumerate() {
                let derived_out:f64;
                let err = if to_next.len() ==0 {
                    //derived_out = derived_sigmoid(base[layer_n][num_n]);
                    derived_out = 1.0;
                    errs[num_n]
                }
                else {
                    derived_out = derived_relu(base[layer_n][num_n]);
                    to_next[num_n].iter().fold(0.0,|acc,val|acc+val) 
                };
                let mut sum_next = vec![];
                for (weight_n,weight) in neuron.weights.iter_mut().enumerate() {
                    let to_next = *weight*err*derived_out;
                    sum_next.push(to_next);

                    let pre = if layer_n == 0{
                        input[weight_n]
                    } else {

                        activated[layer_n-1][weight_n]
                    };
                    let change = pre*derived_out;
                    *weight -=err*self.learning_rate*change;
                }

                neuron.bias-=self.learning_rate*err*derived_out;
                if layer_n != 0 {
                    for (i,val) in sum_next.iter().enumerate() {
                        temp_to_next[i].push(*val);
                    }
                }
            }
        }
    }
    pub fn train(&mut self,answers:&Vec<Vec<f64>>,inputs:&Vec<Vec<f64>>,test_ans:&Vec<Vec<f64>>,test_in:&Vec<Vec<f64>>,cycles:usize,rng:&mut ThreadRng,start_time:Instant,title:String,iteration:i32) {
        let file_name = format!("random-bogstav{}.csv",iteration);
        let mut start_csv = fs::read_to_string(&file_name).unwrap_or_else(|_|"".to_string())+",,,\n";

        let mut csv_string = start_csv+&title+",,,\nCycles,Time,Err,Accuracy\n";
        let batch_size = 36;
        let f_batch_size = batch_size as f64;
        let mut temp_random:Vec<usize> =(0..inputs.len()).collect(); 
        temp_random.shuffle(rng);
        let mut batch = &temp_random[0..batch_size];
        for cycle in 0..cycles {
            let mut errs = vec![0.0;36]; 
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

                let output = self.feedforward(&input); 
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

            if cycle % 1000== 0 {
                let mut sum = 0.0;
                let input_length = test_in.len();
                let answer_length = test_ans[0].len();
                let f_ans_len = answer_length as f64;
                let mut amount_right = 0.0;
                for i in 0..input_length{
                    let output = self.feedforward(&test_in[i]);
                    let predictions = output.out;
                    let real_predictions = predictions;
                    

                    let mut temp_sum = 0.0;
                    for a in 0..answer_length{
                        temp_sum+= mean_squared(real_predictions[a],test_ans[i][a]);
                    }
                    let ans_ind = get_ind_max(&test_ans[i]);
                    let pred_ind = get_ind_max(&real_predictions);

                    if i == 0 {
                        println!("{:?}",ans_ind);
                        println!("{:?}",pred_ind);
                        println!("{:?}",real_predictions);

                    }
                    if ans_ind == pred_ind {
                        amount_right+=1.0;
                    }
                    sum+=temp_sum/f_ans_len;
                }
                let proc_err = (amount_right/(input_length as f64))*100.0; 
                self.learning_rate=self.start_learning_rate*(1.0-proc_err/100.0);
                let err = sum/(input_length as f64);
                let layer_string:String  = serde_json::to_string(&self.layers).unwrap();
                let mut json_file= File::create("weights.json").unwrap();
                json_file.write_all(layer_string.as_bytes()).unwrap(); 
                let elapsed = (start_time.elapsed().as_millis() as f64)/1000.0;
                csv_string+=&format!("{},{},{},{}\n",cycle,elapsed,err,proc_err);
                let mut file = File::create(&file_name).unwrap();
                file.write_all(csv_string.as_bytes()).unwrap();
                println!("Title: {}\nOmgang: {}",title,iteration);
                println!("Cycle: {}\nErr: {}\nAccuracy: {}\nLearning rate: {}\nTime: {}\n",cycle,err,proc_err,&self.learning_rate,elapsed);
            }
        }
    }
}
