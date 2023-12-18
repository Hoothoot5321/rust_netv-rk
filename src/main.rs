use rand::prelude::SliceRandom;
use std::time::Instant;
mod misc;
use misc::weight_funcs::{gen_layer};

mod classes;
use classes::{network::Network,neuron::Neuron};

mod math;
use math::{get_maks::get_ind_max};

use std::fs;

use csv::{Reader};


fn parse_answer(num:f64) -> Vec<f64> {
    const SIZE:usize = 36;
    let mut temp_vec = vec![0.0;SIZE];
    let temp = num as usize;
    temp_vec[temp] = 1.0; 
    temp_vec
} 
fn split_answer(reader:&mut Reader<&[u8]>) ->[Vec<Vec<f64>>;2]   {

    let mut answers:Vec<Vec<f64>> = vec![];
    let mut input :Vec<Vec<f64>> = vec![];
    for record in reader.records() {
        let record = record.unwrap();
        let mut temp_input:Vec<f64> = vec![]; 
        for (i,val) in record.iter().enumerate()  {
            let parsed:f64 = val.parse().unwrap();
            if i == 0 {
               let temp_answer:Vec<f64> = parse_answer(parsed);  
               answers.push(temp_answer);
            }
            else {
                temp_input.push(parsed);
            }
        }
        input.push(temp_input);
    }
    return [answers,input]
}
fn load_test(reader:&mut Reader<&[u8]>) -> Vec<Vec<f64>> {

    let mut input :Vec<Vec<f64>> = vec![];
    for record in reader.records() {
        let record = record.unwrap();
        let mut temp_input:Vec<f64> = vec![]; 
        for (i,val) in record.iter().enumerate()  {
            let parsed:f64 = val.parse().unwrap();
            temp_input.push(parsed);
        }
        input.push(temp_input);
    }
    input
} 



fn main() {
    let mut rng = rand::thread_rng();
    let load_weights = false;

    let path = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\num_c_train.csv";
    let test_path = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\num_c_test.csv".to_owned();
    let test_sing = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\testes.csv"; 
    let weight_file = r"C:\Users\MartinNammat\Documents\Programming-2\all_tests\rust_network_sigmoid\weight 96-proc.json";

    let file_content = fs::read_to_string(path).unwrap(); 
    let mut reader = Reader::from_reader(file_content.as_bytes());
    let [answers,input] = split_answer(&mut reader);
    let file_test = fs::read_to_string(test_path).unwrap();
    let mut test_reader = Reader::from_reader(file_test.as_bytes());
    let [test_ans,test_in] = split_answer(&mut test_reader);
    //let input = load_test(&mut reader);
    let in_weights:Vec<Vec<Neuron>>;
   if load_weights {
    let weight_content= fs::read_to_string(weight_file).unwrap(); 
    let all_weights = serde_json::from_str::<Vec<Vec<Neuron>>>(&weight_content).unwrap(); 
    in_weights = all_weights;
   } 
   else {

    let layer_1 = gen_layer(40,784,&mut rng);



    let layer_output = gen_layer(36,40,&mut rng);

    let over_all = vec![layer_1,layer_output]; 
    in_weights = over_all;
   }
   let mut network = Network::new(in_weights,0.5); 

   let start = Instant::now();
   network.train(&answers,&input,&test_ans,&test_in,500000,&mut rng,start,"Teties".to_owned(),0);
    /*
   let layers = vec![1,2,3];
   let neurons = vec![5,10,15,20];
   for i in 0..10 {
       layers.iter().for_each(|l_n| {
           neurons.iter().for_each(|n_n| {
               let start = Instant::now();
               let input_layer = gen_layer(*n_n,784,&mut rng);

               let mut over_all = vec![input_layer];
               for a in 0..(l_n-1) {
                   let temp_layer = gen_layer(*n_n,*n_n,&mut rng);
                   over_all.push(temp_layer);
               }  

               let output_layer = gen_layer(36,*n_n,&mut rng);

               over_all.push(output_layer);

               let mut network = Network::new(over_all,1.0);
               let title = format!("Antal H lag: {}|Nueroner pr lag {}",l_n,n_n);
               network.train(&answers,&input,&test_ans,&test_in,10000,&mut rng,start,title,i);
           })
       })
    }  
    */
   /*
    let mut inds:Vec<usize> = (0..test_in.len()).collect();
    inds.shuffle(&mut rng);
    for i in 0..100 {
    let output = network.feedforward(&test_in[inds[i]]);
    let pre_ind = get_ind_max(&output.out);
    println!("{}",pre_ind);
    let ans_ind = get_ind_max(&test_ans[inds[i]]);
    println!("{}",ans_ind);
    println!("");
    }
    */
    
    





}

